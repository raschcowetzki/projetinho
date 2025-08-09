#!/usr/bin/env python3
"""
Standalone SQL optimization assistant for Databricks SQL.

Features:
- Parse and analyze a SQL query using SQLGlot (tables, joins, filters)
- Run EXPLAIN to get the physical plan
- Fetch DESCRIBE DETAIL for each referenced table (format, partitions, size)
- Heuristic recommendations (projection pruning, predicate pushdown, join hints, file layout)
- Optional AI summary/rewrite suggestions using ai_gen (Databricks SQL)

Usage:
  python sql_optimizer.py --query "SELECT ..." --ai
  python sql_optimizer.py --query-file path/to/query.sql --ai --format json

Environment variables (same as the app):
  DATABRICKS_HOST
  DATABRICKS_CLIENT_ID
  DATABRICKS_CLIENT_SECRET
  DATABRICKS_WAREHOUSE_ID
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any, Optional

import pandas as pd
import sqlglot
from sqlglot import expressions as exp
from databricks import sql as dbsql

# ----------------------------
# Databricks SQL connection helpers
# ----------------------------

def get_connection():
    host = os.getenv("DATABRICKS_HOST")
    http_path = f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}"
    client_id = os.getenv("DATABRICKS_CLIENT_ID")
    client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
    if not all([host, http_path, client_id, client_secret]):
        raise RuntimeError("Missing required Databricks env vars.")
    cfg = dbsql.auth.OAuthClientCredentials(client_id=client_id, client_secret=client_secret)
    return dbsql.connect(server_hostname=host, http_path=http_path, credentials_provider=cfg)


def run_sql(query: str) -> pd.DataFrame:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            return pd.DataFrame(rows, columns=cols)


# ----------------------------
# Analysis helpers
# ----------------------------

def explain_query(query: str) -> str:
    try:
        df = run_sql(f"EXPLAIN {query}")
        # EXPLAIN often returns one text column with plan per line; join if many rows
        if df.shape[1] == 1:
            return "\n".join(str(x) for x in df.iloc[:, 0].tolist())
        return df.to_string(index=False)
    except Exception as e:
        return f"<EXPLAIN failed: {e}>"


def describe_table_detail(table: str) -> Dict[str, Any]:
    try:
        df = run_sql(f"DESCRIBE DETAIL {table}")
        if not df.empty:
            return df.iloc[0].to_dict()
        return {}
    except Exception:
        return {}


def parse_tables_with_sqlglot(query: str) -> List[str]:
    try:
        expression = sqlglot.parse_one(query)
        tables = set()
        for t in expression.find_all(exp.Table):
            tables.add(".".join([p for p in [t.catalog, t.db, t.name] if p]))
        return sorted(list(tables))
    except Exception:
        return []


def extract_filters_and_joins(query: str) -> Dict[str, Any]:
    info = {"filter_columns": [], "join_conditions": []}
    try:
        expression = sqlglot.parse_one(query)
        # WHERE filters
        for where in expression.find_all(exp.Where):
            for col in where.find_all(exp.Column):
                info["filter_columns"].append(col.sql())
        # JOIN conditions
        for j in expression.find_all(exp.Join):
            cond = j.args.get("on")
            if cond is not None:
                info["join_conditions"].append(cond.sql())
    except Exception:
        pass
    return info


def sqlglot_normalize(query: str) -> Optional[str]:
    try:
        expression = sqlglot.parse_one(query)
        # Lightweight normalization
        return expression.sql(dialect="spark")
    except Exception:
        return None


# ----------------------------
# Heuristic recommendations
# ----------------------------

def heuristic_recommendations(query: str, tables: List[str], details: Dict[str, Dict[str, Any]], plan: str, parse_info: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    qlow = query.lower()
    # Projection pruning
    if "select *" in qlow:
        recs.append("Evite SELECT *; projete apenas as colunas necessárias para reduzir varreduras e shuffle.")
    # Predicate pushdown
    if not parse_info.get("filter_columns"):
        recs.append("Considere adicionar filtros (WHERE) para reduzir dados o quanto antes (predicate pushdown).")
    # Join hints
    if "broadcast" not in qlow and "\nBroadcastHashJoin" in plan:
        recs.append("Verifique se o lado pequeno do join está sendo broadcast; considere usar HINT BROADCAST se fizer sentido.")
    # Table layout
    for t, d in details.items():
        parts = d.get("partitionColumns") or d.get("partitioncolumns")
        if isinstance(parts, list) and len(parts) > 0:
            recs.append(f"Tabela {t} particionada por {parts}; garanta filtros nessas colunas para podar partições.")
        if (d.get("numFiles") or 0) > 5000:
            recs.append(f"Tabela {t} contém muitos arquivos pequenos; considere OPTIMIZE para compactar.")
    # ZORDER suggestions based on filters
    fc = [c.split(".")[-1] for c in parse_info.get("filter_columns", [])]
    if fc:
        top = list({c for c in fc if c})[:3]
        recs.append(f"Considere ZORDER BY {top} nas tabelas consultadas para acelerar filtros frequentes.")
    # Aggregations/windows
    if any(k in qlow for k in ["over (", "window "]):
        recs.append("Avalie substituir janelas por joins/aggregations prévias quando possível para reduzir custo.")
    return recs


# ----------------------------
# AI optimizer (ai_gen)
# ----------------------------

def ai_summary(plan: str, tables: List[str], details: Dict[str, Dict[str, Any]], heuristics: List[str]) -> str:
    try:
        lines = [
            "Você é um otimizador de queries SQL em Databricks.",
            "Resumo das tabelas e detalhes:",
        ]
        for t in tables:
            d = details.get(t) or {}
            parts = d.get("partitionColumns") or d.get("partitioncolumns") or []
            lines.append(f"- {t}: formato={d.get('format')}, numFiles={d.get('numFiles')}, tamanho={d.get('sizeInBytes')}, partitions={parts}")
        lines.append("Plano (EXPLAIN):")
        lines.append(plan[:8000])
        lines.append("Recomendações heurísticas:")
        for r in heuristics:
            lines.append(f"- {r}")
        lines.append("Forneça recomendações concretas (em bullets), priorizadas por impacto, e opcionalmente um esboço de reescrita SQL.")
        prompt = "\n".join(lines).replace("'", "''")
        df = run_sql(f"SELECT ai_gen('{prompt}') AS insights")
        return df.iloc[0]['insights'] if not df.empty else ""
    except Exception as e:
        return f"<AI summary failed: {e}>"


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Databricks SQL optimizer assistant")
    ap.add_argument("--query", type=str, default="", help="SQL query string")
    ap.add_argument("--query-file", type=str, default="", help="Path to .sql file")
    ap.add_argument("--ai", action="store_true", help="Enable AI summary (ai_gen)")
    ap.add_argument("--format", choices=["text","json"], default="text", help="Output format")
    args = ap.parse_args()

    if not args.query and not args.query_file:
        print("Provide --query or --query-file", file=sys.stderr)
        sys.exit(2)
    query = args.query
    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as f:
            query = f.read()

    parsed_tables = parse_tables_with_sqlglot(query)
    plan = explain_query(query)
    details = {t: describe_table_detail(t) for t in parsed_tables}
    parse_info = extract_filters_and_joins(query)
    normalized = sqlglot_normalize(query)
    heur = heuristic_recommendations(query, parsed_tables, details, plan, parse_info)
    ai = ai_summary(plan, parsed_tables, details, heur) if args.ai else ""

    result = {
        "query": query,
        "normalized": normalized,
        "tables": parsed_tables,
        "explain": plan,
        "table_details": details,
        "parse_info": parse_info,
        "recommendations": heur,
        "ai_summary": ai,
    }

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\n=== Tables ===")
        print("\n".join(parsed_tables) if parsed_tables else "(none)")
        print("\n=== EXPLAIN ===")
        print(plan)
        print("\n=== Heuristics ===")
        for r in heur:
            print(f"- {r}")
        if ai:
            print("\n=== AI Summary ===")
            print(ai)


if __name__ == "__main__":
    main()