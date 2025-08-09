#!/usr/bin/env python3
"""
Agentic SQL Optimizer (standalone)

- Agent loop (ReAct-style) that plans, calls tools, proposes safe rewrites, and verifies
- No built-in Databricks ai_functions and no hardcoded recommendations
- Tools: SQLGlot parsing, Databricks SQL EXPLAIN, DESCRIBE DETAIL, sample comparison, timing
- Provider-agnostic LLM adapter (OpenAI or Databricks Model Serving via HTTP)

Usage examples:
  python optimizer/agent_sql_optimizer.py --query "SELECT * FROM my_catalog.my_schema.table" --provider openai --model gpt-4o-mini --max-iters 4
  python optimizer/agent_sql_optimizer.py --query-file ./query.sql --provider databricks --endpoint my-endpoint --max-iters 3 --format json

Env (OpenAI):
  OPENAI_API_KEY

Env (Databricks Model Serving):
  DATABRICKS_HOST, DATABRICKS_TOKEN

Databricks SQL connection (warehouse):
  DATABRICKS_HOST, DATABRICKS_WAREHOUSE_ID, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional

import pandas as pd
import requests
import sqlglot
from sqlglot import expressions as exp
from databricks import sql as dbsql

# ----------------------------
# Databricks SQL helpers
# ----------------------------

def _dbx_conn():
    host = os.getenv("DATABRICKS_HOST")
    http_path = f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}"
    client_id = os.getenv("DATABRICKS_CLIENT_ID"); client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
    if not all([host, http_path, client_id, client_secret]):
        raise RuntimeError("Missing DATABRICKS_* env vars for SQL connection.")
    creds = dbsql.auth.OAuthClientCredentials(client_id=client_id, client_secret=client_secret)
    return dbsql.connect(server_hostname=host, http_path=http_path, credentials_provider=creds)


def run_sql_df(query: str) -> pd.DataFrame:
    with _dbx_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            return pd.DataFrame(rows, columns=cols)


# ----------------------------
# Tools available to the agent
# ----------------------------

def tool_parse_sql(query: str) -> Dict[str, Any]:
    try:
        e = sqlglot.parse_one(query)
        tables = sorted({ ".".join([p for p in [t.catalog, t.db, t.name] if p]) for t in e.find_all(exp.Table) })
        joins = [j.args.get("on").sql() for j in e.find_all(exp.Join) if j.args.get("on")]
        where_cols = [c.sql() for w in e.find_all(exp.Where) for c in w.find_all(exp.Column)]
        return {"ok": True, "tables": tables, "joins": joins, "filter_columns": where_cols}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


def tool_get_explain(query: str) -> Dict[str, Any]:
    try:
        df = run_sql_df(f"EXPLAIN {query}")
        plan = "\n".join(df.iloc[:,0].astype(str).tolist()) if df.shape[1] == 1 else df.to_string(index=False)
        return {"ok": True, "plan": plan}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


def tool_get_table_details(tables: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "details": {}}
    for t in tables:
        try:
            df = run_sql_df(f"DESCRIBE DETAIL {t}")
            out["details"][t] = df.iloc[0].to_dict() if not df.empty else {}
        except Exception as ex:
            out["details"][t] = {"error": str(ex)}
    return out


def tool_sample_compare(orig_query: str, cand_query: str, limit: int = 1000) -> Dict[str, Any]:
    try:
        df1 = run_sql_df(f"SELECT * FROM ({orig_query}) LIMIT {limit}")
        df2 = run_sql_df(f"SELECT * FROM ({cand_query}) LIMIT {limit}")
        schema_equal = list(df1.columns) == list(df2.columns)
        rate = 0.0
        if schema_equal and not df1.empty and not df2.empty:
            k = df1.columns[0]
            s1 = set(df1[k].astype(str).head(200)); s2 = set(df2[k].astype(str).head(200))
            rate = len(s1 & s2) / max(1, len(s1 | s2))
        return {"ok": True, "schema_equal": schema_equal, "sample_match_rate": rate, "rows1": len(df1), "rows2": len(df2)}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


def tool_time_query(query: str, limit: int = 50000) -> Dict[str, Any]:
    try:
        t0 = time.time()
        _ = run_sql_df(f"SELECT * FROM ({query}) LIMIT {limit}")
        ms = int((time.time() - t0) * 1000)
        return {"ok": True, "elapsed_ms": ms}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


TOOLS_SPEC = {
    "parse_sql": {"args": {"query": "str"}},
    "get_explain": {"args": {"query": "str"}},
    "get_table_details": {"args": {"tables": "list[str]"}},
    "sample_compare": {"args": {"original_query": "str", "candidate_query": "str"}},
    "time_query": {"args": {"query": "str"}},
    "propose_rewrite": {"args": {"sql": "str"}},
    "finish": {"args": {"reason": "str"}},
}

TOOLS_HELP = """You can call these tools by outputting a single JSON object:
{"action": "parse_sql", "args": {"query": "..."}}
Actions: parse_sql, get_explain, get_table_details, sample_compare, time_query, propose_rewrite, finish.
Rules:
- Always preserve semantics (same columns) when proposing rewrites.
- Validate candidates with sample_compare; then check time_query.
- Stop using finish when you have the best improved query or cannot improve.
"""

# ----------------------------
# LLM Adapters
# ----------------------------

class LLM:
    def ask(self, messages: List[Dict[str,str]]) -> str:
        raise NotImplementedError

class OpenAIChat(LLM):
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
    def ask(self, messages: List[Dict[str,str]]) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

class DatabricksServing(LLM):
    def __init__(self, endpoint: str):
        host = os.getenv("DATABRICKS_HOST"); token = os.getenv("DATABRICKS_TOKEN")
        if not (host and token):
            raise RuntimeError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set for Databricks Serving")
        self.url = f"https://{host}/serving-endpoints/{endpoint}/invocations"
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    def ask(self, messages: List[Dict[str,str]]) -> str:
        payload = {"messages": messages, "temperature": 0}
        r = requests.post(self.url, headers=self.headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Expect OpenAI-like response or adapt to your model schema
        if isinstance(data, dict) and "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        return json.dumps(data, ensure_ascii=False)

# ----------------------------
# Agent loop
# ----------------------------

SYSTEM = (
    "You are a Databricks SQL optimization agent. "
    "Use tools to gather evidence, propose safe rewrites that preserve schema/semantics, "
    "validate with sample_compare, compare cost with time_query, and finish with the best improved query.\n" + TOOLS_HELP
)

def parse_action(text: str) -> Dict[str, Any]:
    # Expect a single JSON object with keys: action, args
    try:
        obj = json.loads(text.strip())
        if not isinstance(obj, dict):
            return {"action": "invalid", "error": "Not a JSON object"}
        return obj
    except Exception as ex:
        return {"action": "invalid", "error": str(ex)}


def optimize(query: str, llm: LLM, max_iters: int = 4, sample_limit: int = 1000) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Optimize this Spark SQL without changing semantics (preserve same columns):\n{query}"},
    ]
    observations: List[str] = []
    best = {"query": query, "time_ms": tool_time_query(query).get("elapsed_ms")}

    for _ in range(max_iters):
        # Provide latest observations
        if observations:
            messages.append({"role": "user", "content": "OBSERVATIONS:\n" + "\n".join(observations)[-8000:]})
        # Ask LLM for next tool call
        messages.append({"role": "user", "content": "Return a JSON with {\"action\":..., \"args\":{...}} now."})
        action_raw = llm.ask(messages)
        act = parse_action(action_raw)
        observations.append(f"LLM_ACTION: {action_raw}")
        if act.get("action") == "invalid":
            observations.append("Invalid action JSON; ask again.")
            continue

        name = act.get("action"); args = act.get("args", {})
        if name == "parse_sql":
            res = tool_parse_sql(args.get("query", query))
            observations.append("parse_sql -> " + json.dumps(res, ensure_ascii=False)[:1200])
            continue
        if name == "get_explain":
            res = tool_get_explain(args.get("query", query))
            observations.append("get_explain ->\n" + res.get("plan", "")[:2400])
            continue
        if name == "get_table_details":
            res = tool_get_table_details(args.get("tables", []))
            observations.append("get_table_details -> " + json.dumps(res, ensure_ascii=False)[:2000])
            continue
        if name == "sample_compare":
            res = tool_sample_compare(args.get("original_query", query), args.get("candidate_query", query), limit=sample_limit)
            observations.append("sample_compare -> " + json.dumps(res, ensure_ascii=False))
            continue
        if name == "time_query":
            res = tool_time_query(args.get("query", query))
            observations.append("time_query -> " + json.dumps(res, ensure_ascii=False))
            continue
        if name == "propose_rewrite":
            cand = args.get("sql", "")
            if not cand:
                observations.append("propose_rewrite missing sql")
                continue
            cmp_res = tool_sample_compare(query, cand, limit=sample_limit)
            observations.append("sample_compare(auto) -> " + json.dumps(cmp_res, ensure_ascii=False))
            if cmp_res.get("ok") and cmp_res.get("schema_equal") and cmp_res.get("sample_match_rate",0) >= 0.7:
                t_res = tool_time_query(cand)
                observations.append("time_query(auto) -> " + json.dumps(t_res, ensure_ascii=False))
                new_ms = t_res.get("elapsed_ms")
                if new_ms is not None and (best["time_ms"] is None or new_ms < best["time_ms"]):
                    best = {"query": cand, "time_ms": new_ms}
                    observations.append(f"Candidate accepted with elapsed_ms={new_ms}")
                else:
                    observations.append(f"Candidate not faster (elapsed_ms={new_ms}); try another approach.")
            else:
                observations.append("Candidate failed validation (schema or sample mismatch)")
            continue
        if name == "finish":
            observations.append("Agent requested finish")
            break
        # Unknown -> request proper tool call next iteration
        observations.append("Unknown action; please output a valid tool call JSON.")

    return best


def main():
    ap = argparse.ArgumentParser(description="Agentic SQL optimizer (Databricks)")
    ap.add_argument("--query", type=str, default="", help="SQL string")
    ap.add_argument("--query-file", type=str, default="", help="Path to .sql file")
    ap.add_argument("--provider", choices=["openai","databricks"], required=True)
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model (if provider=openai)")
    ap.add_argument("--endpoint", type=str, default="", help="Databricks serving endpoint name (if provider=databricks)")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--format", choices=["text","json"], default="text")
    args = ap.parse_args()

    if not args.query and not args.query_file:
        print("Provide --query or --query-file", file=sys.stderr)
        sys.exit(2)
    query = args.query
    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as f:
            query = f.read()

    if args.provider == "openai":
        llm = OpenAIChat(model=args.model)
    else:
        if not args.endpoint:
            print("--endpoint required for provider=databricks", file=sys.stderr)
            sys.exit(2)
        llm = DatabricksServing(endpoint=args.endpoint)

    result = optimize(query, llm, max_iters=args.max_iters)
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("=== Best Query ===\n" + result["query"]) 
        print(f"\nElapsed (ms): {result.get('time_ms')}")


if __name__ == "__main__":
    main()