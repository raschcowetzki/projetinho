import os, yaml, json, requests
from code_editor import code_editor
import streamlit as st
import sqlparse
import pandas as pd
from databricks import sql
from databricks.sdk.core import Config, oauth_service_principal
from databricks.sdk import WorkspaceClient
from streamlit_option_menu import option_menu
from sql_metadata import Parser
from atlassian import Jira
from io import BytesIO
from streamlit_tags import st_tags
import plotly.express as px

w = WorkspaceClient()
server_hostname = os.getenv("DATABRICKS_HOST")


# # Reseta as variáveis em `st.session_state`
# def load_default_values(reset: False):
#     DEFAULT_VALUES = config["DEFAULT_VALUES"]
#     if reset not in (True, False):
#         raise ValueError("O argumento 'option' deve ser True ou False.")
    
#     for key, value in DEFAULT_VALUES.items():
#         if reset == False and key not in st.session_state:
#             st.session_state[key] = value
#         elif reset == True:
#             st.session_state[key] = value
#         else:
#             pass

    
# def get_origins(sql):
#     tabelas = list(set(
#                 Parser(sql).tables
#             ))
#     origins = eng_list(tabelas)
#     return origins
            
# Função para alterar etapas (navegação)
def alterar_etapa(valor: int):
    st.session_state.etapa_atual = st.session_state.etapa_atual + valor
    st.rerun()

def schema_validator(query: str):
    query = f"DESCRIBE {query} LIMIT 0"
    return sql_query(query)

def credential_provider():
    config = Config(
        host          = f"https://{server_hostname}",
        client_id     = os.getenv("DATABRICKS_CLIENT_ID"),
        client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
    )
    return oauth_service_principal(config)

def sql_query(query: str):
    formatted_query = sqlparse.format(query, reindent=True, keyword_case='upper')
    with sql.connect(
                    server_hostname      = server_hostname,
                    http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
                    credentials_provider = credential_provider
                    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(formatted_query)
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description] if cursor.description else []
            return pd.DataFrame(rows, columns=cols)

# ----------------------------
# Helpers e UX
# ----------------------------

def _download_button(df: pd.DataFrame, label: str, filename: str):
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label=label, data=csv, file_name=filename, mime='text/csv')

def export(model_query, type_export):
    pass
            
def etapa_1():
    st.session_state.msg_erro_2 = None
    st.title("Classificação - Dados iniciais")
    st.session_state.button_disabled = True
    query = code_editor("", lang="sql", height="300px", buttons=[
        {"name": "Run", "feather": "Play", "hasText": True, "showWithIcon": True, "commands": ["submit"], "alwaysOn": True, "style": {"bottom": "6px", "right": "0.4rem"}}
    ])
    
    if query['text'] != '':
        try:
            df_zero = sql_query(f"SELECT * FROM ({query['text']}) LIMIT 0")
            all_columns = list(df_zero.columns)
            string_like = [c for c in all_columns if str(df_zero.dtypes.get(c, "")).lower().startswith("object")]
            if not string_like:
                string_like = all_columns
            st.session_state.colunas = string_like
            coluna_input = st.selectbox("Selecione a coluna para classificar", options=st.session_state.colunas)
            keywords = st_tags(
                            label='Adicione as Categorias:',
                            text='Pressione ENTER para adicionar mais',
                            value=[],
                            key="aljnf")
            st.session_state.button_disabled = False
        except Exception as e:
            st.error(f"Não foi possível inferir colunas da query. Detalhes: {e}")
    else:
        st.write("Escreva sua query")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Voltar"):
            st.session_state.query = query['text']
            alterar_etapa(-1)
    with col2:
        if st.button("Avançar", disabled=st.session_state.button_disabled):
            st.session_state.query = query['text']
            st.session_state.keywords = keywords
            st.session_state.coluna_input = coluna_input
            alterar_etapa(1)
    
    if st.session_state.msg_erro_2:
        st.error(st.session_state.msg_erro_2)

def etapa_2():
    st.title("Classificação - Resultado")
    keywords_str = "'"+"','".join(st.session_state.keywords)+"'"
    st.session_state.model_query = f"""SELECT {st.session_state.coluna_input}, ai_classify({st.session_state.coluna_input}, ARRAY({keywords_str})) AS classificacao FROM ({st.session_state.query} limit 1000)"""
    df_analitico = sql_query(st.session_state.model_query)
    st.dataframe(df_analitico,use_container_width = True, hide_index=True)
    # Gráfico interativo
    counts = df_analitico.groupby("classificacao").size().reset_index(name="quantidade")
    fig = px.bar(counts, x="classificacao", y="quantidade", title="Distribuição por Classe")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Análise"):
        total = len(df_analitico)
        st.write(f"Total de registros: {total}")
        st.write(counts.sort_values('quantidade', ascending=False))
    _download_button(df_analitico, "Baixar classificações (CSV)", "classificacoes.csv")
    st.session_state.type_export = st.radio("Opções para exportar",["CSV", "EXP"], horizontal = True)
    st.session_state.name_export = st.text_input("Nome do arquivo ou exp", value="resultado_classificacao")
    if st.session_state.type_export == "EXP":
        st.session_state.exp_periodicity = st.selectbox("Recorrência",["Diário","Mensal","Estático"]) 
        st.session_state.exp_description = st.text_input("Descrição")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Voltar"):
            alterar_etapa(-1)
    with col2:
        if st.button("Exportar", disabled=st.session_state.button_disabled):
            st.info("Exportação ainda não implementada.")

# Funções auxiliares genéricas

def _input_query_and_column(title: str):
    st.title(title)
    st.session_state.button_disabled = True
    query_state = code_editor("", lang="sql", height="300px", buttons=[
        {"name": "Run", "feather": "Play", "hasText": True, "showWithIcon": True, "commands": ["submit"], "alwaysOn": True, "style": {"bottom": "6px", "right": "0.4rem"}}
    ])
    selected_column = None
    if query_state['text'] != '':
        try:
            # Tenta inferir esquema a partir da query
            df_zero = sql_query(f"SELECT * FROM ({query_state['text']}) LIMIT 0")
            all_columns = list(df_zero.columns)
            # Considera colunas do tipo 'object' como texto
            string_like = [c for c in all_columns if str(df_zero.dtypes.get(c, "")).lower().startswith("object")]
            if not string_like:
                # Fallback: permitir qualquer coluna se não conseguimos inferir
                string_like = all_columns
            st.session_state.colunas = string_like
            selected_column = st.selectbox("Selecione a coluna de texto", options=st.session_state.colunas)
            st.session_state.button_disabled = False
        except Exception as e:
            st.error(f"Não foi possível inferir o esquema da query. Detalhes: {e}")
            st.write("Dica: forneça uma consulta SQL válida, por exemplo, SELECT col_texto FROM minha_tabela")
    else:
        st.write("Escreva sua query")
    return query_state['text'], selected_column

# Análise de Sentimento

def analyze_sentiment():
    if 'step_sent' not in st.session_state:
        st.session_state.step_sent = 1

    if st.session_state.step_sent == 1:
        st.caption("Forneça uma consulta que retorne ao menos uma coluna de texto.")
        query_text, col = _input_query_and_column("Análise de Sentimento - Dados iniciais")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Avançar", disabled=st.session_state.button_disabled):
                st.session_state.sent_query = query_text
                st.session_state.sent_col = col
                st.session_state.step_sent = 2
                st.rerun()
        with col2:
            if st.button("Limpar"):
                st.session_state.step_sent = 1
                st.rerun()
    else:
        st.title("Análise de Sentimento - Resultado")
        with st.spinner("Executando análise no warehouse..."):
            model_query = f"SELECT {st.session_state.sent_col} AS texto, ai_analyze_sentiment({st.session_state.sent_col}) AS sentimento FROM ({st.session_state.sent_query} limit 1000)"
            df = sql_query(model_query)
        st.dataframe(df, use_container_width=True, hide_index=True)
        counts = df.groupby("sentimento").size().reset_index(name="quantidade")
        fig = px.bar(counts, x="sentimento", y="quantidade", title="Distribuição de Sentimentos")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Análise"):
            total = len(df)
            resumo = counts.sort_values('quantidade', ascending=False)
            st.write(f"Total de registros: {total}")
            st.write(resumo)
        _download_button(df, "Baixar resultados (CSV)", "sentimentos.csv")
        if st.button("Voltar"):
            st.session_state.step_sent = 1
            st.rerun()

def extract_entities():
    st.title("Extração de Entidade")
    st.caption("Forneça uma consulta que retorne uma coluna de texto e escolha os tipos de entidades a extrair.")

    # Editor de consulta
    query_state = code_editor(
        "",
        lang="sql",
        height="250px",
        buttons=[
            {"name": "Run", "feather": "Play", "hasText": True, "showWithIcon": True, "commands": ["submit"], "alwaysOn": True, "style": {"bottom": "6px", "right": "0.4rem"}}
        ]
    )
    if not query_state['text']:
        st.info("Escreva sua consulta acima e clique em Run para habilitar os parâmetros.")
        return

    # Inferir colunas e selecionar a coluna de texto
    try:
        df_zero = sql_query(f"SELECT * FROM ({query_state['text']}) LIMIT 0")
        cols = list(df_zero.columns)
        text_col = st.selectbox("Coluna de texto", options=cols, index=0 if cols else None)
    except Exception as e:
        st.error(f"Não foi possível inferir as colunas: {e}")
        return

    # Parâmetros
    st.subheader("Parâmetros")
    p1, p2, p3 = st.columns([2,1,1])
    with p1:
        entity_types = st_tags(
            label='Tipos de entidade (ex.: person, organization, location, date, email, phone, url, money):',
            text='Pressione ENTER para adicionar mais',
            value=['person','organization','location']
        )
    with p2:
        limit = st.number_input("Limite de linhas", min_value=10, max_value=10000, value=1000, step=10)
    with p3:
        top_n = st.number_input("Top N (agregados)", min_value=5, max_value=100, value=30, step=5)

    run = st.button("Executar extração", type="primary")
    if not run:
        return

    # Executar extração
    types = [t.strip() for t in entity_types if t and t.strip()]
    if not types:
        st.warning("Informe ao menos um tipo de entidade.")
        return

    types_str = "'" + "','".join(types) + "'"
    with st.spinner("Extraindo entidades no warehouse..."):
        q = f"SELECT {text_col} AS texto, ai_extract({text_col}, ARRAY({types_str})) AS entidades FROM ({query_state['text']}) LIMIT {int(limit)}"
        try:
            df_raw = sql_query(q)
        except Exception as e:
            st.error(f"Falha ao executar extração: {e}")
            return

    st.subheader("Resultados")
    tab_raw, tab_long, tab_aggs = st.tabs(["Tabela (bruta)", "Entidades (normalizadas)", "Agregados"]) 

    with tab_raw:
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
        _download_button(df_raw, "Baixar resultados (CSV)", "entidades_bruto.csv")

    # Normalização das entidades
    def _flatten_entities(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for idx, row in df.iterrows():
            ents = row.get('entidades')
            if ents is None:
                continue
            try:
                if isinstance(ents, str):
                    ents = json.loads(ents)
            except Exception:
                pass
            if isinstance(ents, dict):
                for ent_type, ent_values in ents.items():
                    if isinstance(ent_values, list):
                        for v in ent_values:
                            rows.append({"linha": idx, "tipo": ent_type, "valor": str(v)})
                    else:
                        rows.append({"linha": idx, "tipo": ent_type, "valor": str(ent_values)})
            elif isinstance(ents, list):
                for v in ents:
                    rows.append({"linha": idx, "tipo": "unknown", "valor": str(v)})
            else:
                rows.append({"linha": idx, "tipo": "unknown", "valor": str(ents)})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["linha","tipo","valor"])

    df_long = _flatten_entities(df_raw)

    with tab_long:
        if df_long.empty:
            st.info("Nenhuma entidade foi detectada.")
        else:
            st.dataframe(df_long, use_container_width=True, hide_index=True)
            _download_button(df_long, "Baixar entidades (CSV)", "entidades_normalizadas.csv")

    with tab_aggs:
        if df_long.empty:
            st.info("Sem dados para agregar.")
        else:
            counts = df_long.groupby(["tipo", "valor"]).size().reset_index(name="quantidade")
            top = counts.sort_values("quantidade", ascending=False).head(int(top_n))
            fig = px.bar(top, x="valor", y="quantidade", color="tipo", title="Top entidades extraídas")
            st.plotly_chart(fig, use_container_width=True)
            by_type = df_long.groupby("tipo").size().reset_index(name="quantidade")
            fig2 = px.pie(by_type, names='tipo', values='quantidade', title='Distribuição por tipo')
            st.plotly_chart(fig2, use_container_width=True)

# Geração de Texto (ai_gen)

def gen_text():
    st.title("Geração de Texto")
    st.caption("Use modelos generativos para criar textos a partir de um prompt.")
    prompt = st.text_area("Prompt", height=150, placeholder="Descreva o que deseja gerar...")
    if st.button("Gerar", disabled=(not prompt)):
        with st.spinner("Gerando texto..."):
            safe_prompt = prompt.replace("'", "''")
            df = sql_query(f"SELECT ai_gen('{safe_prompt}') AS texto_gerado")
        texto = df.iloc[0]["texto_gerado"] if not df.empty else ""
        st.text_area("Resultado", value=str(texto), height=300)
        if isinstance(texto, str):
            words = pd.Series([w.lower() for w in texto.split() if len(w) > 3])
            if not words.empty:
                freq = words.value_counts().reset_index()
                freq.columns = ["palavra","frequencia"]
                fig = px.bar(freq.head(30), x="palavra", y="frequencia", title="Top palavras")
                st.plotly_chart(fig, use_container_width=True)
        _download_button(df, "Baixar texto (CSV)", "texto_gerado.csv")

# Tradução (ai_translate)

def translate_text():
    if 'step_tr' not in st.session_state:
        st.session_state.step_tr = 1
    if st.session_state.step_tr == 1:
        st.caption("Forneça uma consulta que retorne uma coluna de texto a ser traduzida.")
        query_text, col = _input_query_and_column("Tradução - Dados iniciais")
        lang = st.selectbox("Idioma de destino", ["en","pt","es","fr","de","it","ja","zh"], index=0)
        if st.button("Avançar", disabled=st.session_state.button_disabled):
            st.session_state.tr_query = query_text
            st.session_state.tr_col = col
            st.session_state.tr_lang = lang
            st.session_state.step_tr = 2
            st.rerun()
    else:
        st.title("Tradução - Resultado")
        with st.spinner("Traduzindo textos..."):
            q = f"SELECT {st.session_state.tr_col} AS original, ai_translate({st.session_state.tr_col}, '{st.session_state.tr_lang}') AS traducao FROM ({st.session_state.tr_query} limit 1000)"
            df = sql_query(q)
        st.dataframe(df, use_container_width=True, hide_index=True)
        df['len'] = df['traducao'].astype(str).apply(len)
        fig = px.histogram(df, x='len', nbins=30, title='Distribuição do tamanho do texto traduzido')
        st.plotly_chart(fig, use_container_width=True)
        _download_button(df, "Baixar traduções (CSV)", "traducoes.csv")
        if st.button("Voltar"):
            st.session_state.step_tr = 1
            st.rerun()

# Sumarização (ai_summarize)

def summarize_text():
    if 'step_sum' not in st.session_state:
        st.session_state.step_sum = 1
    if st.session_state.step_sum == 1:
        st.caption("Forneça uma consulta que retorne uma coluna de texto a ser sumarizada.")
        query_text, col = _input_query_and_column("Sumarização - Dados iniciais")
        if st.button("Avançar", disabled=st.session_state.button_disabled):
            st.session_state.sum_query = query_text
            st.session_state.sum_col = col
            st.session_state.step_sum = 2
            st.rerun()
    else:
        st.title("Sumarização - Resultado")
        with st.spinner("Sumarizando..."):
            q = f"SELECT {st.session_state.sum_col} AS original, ai_summarize({st.session_state.sum_col}) AS resumo FROM ({st.session_state.sum_query} limit 1000)"
            df = sql_query(q)
        st.dataframe(df, use_container_width=True, hide_index=True)
        df['len'] = df['resumo'].astype(str).apply(len)
        fig = px.histogram(df, x='len', nbins=30, title='Distribuição do tamanho dos resumos')
        st.plotly_chart(fig, use_container_width=True)
        _download_button(df, "Baixar resumos (CSV)", "resumos.csv")
        if st.button("Voltar"):
            st.session_state.step_sum = 1
            st.rerun()

# Correção de Gramática (ai_fix_grammar)

def fix_grammar_page():
    if 'step_fix' not in st.session_state:
        st.session_state.step_fix = 1
    if st.session_state.step_fix == 1:
        st.caption("Forneça uma consulta que retorne uma coluna de texto para correção gramatical.")
        query_text, col = _input_query_and_column("Correção de Gramática - Dados iniciais")
        if st.button("Avançar", disabled=st.session_state.button_disabled):
            st.session_state.fix_query = query_text
            st.session_state.fix_col = col
            st.session_state.step_fix = 2
            st.rerun()
    else:
        st.title("Correção de Gramática - Resultado")
        with st.spinner("Corrigindo..."):
            q = f"SELECT {st.session_state.fix_col} AS original, ai_fix_grammar({st.session_state.fix_col}) AS corrigido FROM ({st.session_state.fix_query} limit 1000)"
            df = sql_query(q)
        st.dataframe(df, use_container_width=True, hide_index=True)
        df['len_original'] = df['original'].astype(str).apply(len)
        df['len_corrigido'] = df['corrigido'].astype(str).apply(len)
        melted = df.melt(value_vars=['len_original','len_corrigido'], var_name='tipo', value_name='tamanho')
        fig = px.histogram(melted, x='tamanho', color='tipo', barmode='overlay', nbins=40, title='Comprimento do texto: antes vs depois')
        st.plotly_chart(fig, use_container_width=True)
        _download_button(df, "Baixar correções (CSV)", "correcoes.csv")
        if st.button("Voltar"):
            st.session_state.step_fix = 1
            st.rerun()

# Detecção de Idioma (ai_detect_language)

def detect_language_page():
    if 'step_lang' not in st.session_state:
        st.session_state.step_lang = 1
    if st.session_state.step_lang == 1:
        st.caption("Forneça uma consulta que retorne uma coluna de texto para detecção de idioma.")
        query_text, col = _input_query_and_column("Detecção de Idioma - Dados iniciais")
        if st.button("Avançar", disabled=st.session_state.button_disabled):
            st.session_state.lang_query = query_text
            st.session_state.lang_col = col
            st.session_state.step_lang = 2
            st.rerun()
    else:
        st.title("Detecção de Idioma - Resultado")
        with st.spinner("Detectando idiomas..."):
            q = f"SELECT {st.session_state.lang_col} AS texto, ai_detect_language({st.session_state.lang_col}) AS idioma FROM ({st.session_state.lang_query} limit 1000)"
            df = sql_query(q)
        st.dataframe(df, use_container_width=True, hide_index=True)
        counts = df.groupby("idioma").size().reset_index(name="quantidade")
        fig = px.pie(counts, names='idioma', values='quantidade', title='Idiomas detectados')
        st.plotly_chart(fig, use_container_width=True)
        _download_button(df, "Baixar idiomas (CSV)", "idiomas.csv")
        if st.button("Voltar"):
            st.session_state.step_lang = 1
            st.rerun()

# ----------------------------
# Projeção (Forecast)
# ----------------------------

def forecast_projection():
    st.title("Projeção - Previsão de Séries Temporais")
    st.caption("Forneça uma consulta que retorne colunas de tempo e valor.")

    query_state = code_editor("", lang="sql", height="250px", buttons=[{"name": "Run", "feather": "Play", "hasText": True, "showWithIcon": True, "commands": ["submit"], "alwaysOn": True, "style": {"bottom": "6px", "right": "0.4rem"}}])
    if not query_state['text']:
        st.info("Escreva sua consulta à esquerda e clique em Run.")
        return

    # Inferência de colunas
    try:
        df_zero = sql_query(f"SELECT * FROM ({query_state['text']}) LIMIT 0")
        cols = list(df_zero.columns)
        time_col = st.selectbox("Coluna de tempo", options=cols, index=0 if cols else None)
        value_col = st.selectbox("Coluna alvo (valor)", options=[c for c in cols if c != time_col], index=0 if len(cols) > 1 else None)
    except Exception as e:
        st.error(f"Não foi possível inferir as colunas: {e}")
        return

    col_a, col_b = st.columns([1,1])
    with col_a:
        frequency = st.selectbox("Frequência", ["D","W","M"], index=0, help="Periodicidade dos dados")
    with col_b:
        horizon = st.number_input("Horizonte (períodos)", min_value=1, max_value=365, value=30)

    run = st.button("Executar previsão", type="primary")
    if not run:
        return

    # Consultar histórico primeiro (necessário para calcular horizon como TIMESTAMP)
    with st.spinner("Consultando dados históricos..."):
        df_hist = sql_query(f"SELECT {time_col} AS ds, {value_col} AS y FROM ({query_state['text']}) ORDER BY {time_col}")
    if df_hist.empty:
        st.warning("A consulta não retornou dados.")
        return

    # Calcular horizonte como TIMESTAMP/STRING aceito pela TVF
    try:
        df_hist['ds'] = pd.to_datetime(df_hist['ds'])
        last_date = df_hist['ds'].max()
        if frequency == 'D':
            horizon_end = last_date + pd.Timedelta(days=int(horizon))
        elif frequency == 'W':
            horizon_end = last_date + pd.Timedelta(weeks=int(horizon))
        else:
            horizon_end = last_date + pd.DateOffset(months=int(horizon))
        horizon_str = horizon_end.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        st.error(f"Falha ao calcular o horizonte da previsão: {e}")
        return

    # Executa ai_forecast no warehouse
    with st.spinner("Gerando previsão com ai_forecast..."):
        base_sql = f"SELECT {time_col} AS ds, {value_col} AS y FROM ({query_state['text']})"
        forecast_sql = (
            f"SELECT * FROM ai_forecast("
            f"TABLE({base_sql}), time_col => 'ds', value_col => 'y', horizon => '{horizon_str}'"
            f")"
        )
        try:
            df_fc = sql_query(forecast_sql)
        except Exception as e:
            st.error(f"Falha ao executar ai_forecast: {e}")
            return

    if df_fc.empty:
        st.warning("A função ai_forecast não retornou dados.")
        return

    # Mapear colunas retornadas
    col_map = {
        'ds': None,
        'yhat': None,
        'yhat_lower': None,
        'yhat_upper': None
    }
    candidates = {
        'ds': ['ds', 'date', 'timestamp', time_col],
        'yhat': ['y_forecast', 'yhat', 'forecast', 'prediction'],
        'yhat_lower': ['y_lower', 'yhat_lower', 'lower_bound', 'lo'],
        'yhat_upper': ['y_upper', 'yhat_upper', 'upper_bound', 'hi']
    }
    for key, names in candidates.items():
        for n in names:
            if n in df_fc.columns:
                col_map[key] = n
                break

    if not col_map['ds'] or not col_map['yhat']:
        st.error("Não foi possível identificar as colunas de data e previsão retornadas por ai_forecast.")
        st.write("Colunas retornadas:", list(df_fc.columns))
        return

    # Plot completo com intervalo
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.to_datetime(df_hist['ds']), y=pd.to_numeric(df_hist['y'], errors='coerce'), name='Histórico', mode='lines', line=dict(color='#2a9d8f')))
        if col_map['yhat_lower'] and col_map['yhat_upper']:
            fig.add_traces([
                go.Scatter(
                    x=pd.to_datetime(df_fc[col_map['ds']]),
                    y=pd.to_numeric(df_fc[col_map['yhat_upper']], errors='coerce'),
                    mode='lines', line=dict(width=0), name='Limite Superior', showlegend=False
                ),
                go.Scatter(
                    x=pd.to_datetime(df_fc[col_map['ds']]),
                    y=pd.to_numeric(df_fc[col_map['yhat_lower']], errors='coerce'),
                    mode='lines', line=dict(width=0), name='Limite Inferior', fill='tonexty', fillcolor='rgba(63,161,16,0.2)', showlegend=False
                )
            ])
        fig.add_trace(go.Scatter(x=pd.to_datetime(df_fc[col_map['ds']]), y=pd.to_numeric(df_fc[col_map['yhat']], errors='coerce'), name='Projeção', mode='lines', line=dict(color='#3FA110')))
        fig.update_layout(title='Histórico e Projeção com Intervalo (ai_forecast)', xaxis_title='Tempo', yaxis_title=value_col)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Falha ao montar o gráfico: {e}")

    # Tabelas e download
    st.subheader("Resultados")
    cols_show = [c for c in [col_map['ds'], col_map['yhat_lower'], col_map['yhat'], col_map['yhat_upper']] if c]
    st.dataframe(df_fc[cols_show], use_container_width=True, hide_index=True)
    _download_button(df_fc[cols_show], "Baixar projeção (CSV)", "projecao_ai_forecast.csv")

# Genie Chat (embed)

def genie_chat():
    st.title("Genie Chat")
    st.write("Converse com seus dados usando um espaço do Genie.")
    default_url = os.getenv('GENIE_SPACE_URL', '')
    space_url = st.text_input("URL do Space do Genie", value=default_url, placeholder="https://.../genie/spaces/<id>")
    if space_url:
        import streamlit.components.v1 as components
        components.iframe(space_url, height=750)
    else:
        st.info("Informe a URL do Space do Genie para iniciar o chat.")

# AutoML Page

def automl_page():
    st.title("AutoML - Treinamento Automatizado")
    try:
        import databricks.automl as automl
    except Exception:
        automl = None
    mode = st.selectbox("Tarefa", ["Classificação","Regressão","Previsão (Forecast)"])
    st.subheader("Dados de Treino (via SQL)")
    query_state = code_editor("", lang="sql", height="250px", buttons=[{"name": "Run", "feather": "Play", "hasText": True, "showWithIcon": True, "commands": ["submit"], "alwaysOn": True, "style": {"bottom": "6px", "right": "0.4rem"}}])
    df = pd.DataFrame()
    if query_state['text']:
        try:
            df = sql_query(query_state['text'])
            st.dataframe(df.head(50), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Erro ao executar a query: {e}")
    if not df.empty:
        target = st.selectbox("Coluna alvo", options=df.columns)
        time_col = None
        if mode == "Previsão (Forecast)":
            time_col = st.selectbox("Coluna de tempo", options=df.columns)
            freq = st.text_input("Frequência (ex.: D, W, M)", value="D")
            horizon = st.number_input("Horizonte (nº períodos)", min_value=1, max_value=365, value=30)
        run_ok = st.button("Iniciar AutoML", disabled=(automl is None))
        if automl is None:
            st.warning("Biblioteca 'databricks.automl' não disponível neste ambiente. Execute em um cluster Databricks.")
        if run_ok and automl is not None:
            with st.spinner("Iniciando experimento AutoML..."):
                try:
                    if mode == "Classificação":
                        summary = automl.classify(dataset=df, target_col=target)
                    elif mode == "Regressão":
                        summary = automl.regress(dataset=df, target_col=target)
                    else:
                        summary = automl.forecast(dataset=df, target_col=target, time_col=time_col, frequency=freq, horizon=horizon)
                    st.success("AutoML finalizado (ou iniciado).")
                    try:
                        st.json(summary)  # Best-effort visualização
                    except Exception:
                        st.write(summary)
                except Exception as e:
                    st.error(f"Falha no AutoML: {e}")

def classify():
    if 'etapa_atual' not in st.session_state:
        st.session_state.etapa_atual = 1
    if st.session_state.etapa_atual == 1:
        etapa_1()
    elif st.session_state.etapa_atual == 2:
        etapa_2()

# ----------------------------
# Anomalias em séries temporais
# ----------------------------

def anomaly_detection_page():
    st.title("Detecção de Anomalias")
    st.caption("Forneça uma consulta com tempo e valor; identifique outliers estatísticos e visualize no gráfico.")

    qstate = code_editor("", lang="sql", height="250px", buttons=[{"name":"Run","feather":"Play","hasText":True,"showWithIcon":True,"commands":["submit"],"alwaysOn":True,"style":{"bottom":"6px","right":"0.4rem"}}])
    if not qstate['text']:
        st.info("Escreva sua consulta e clique em Run.")
        return

    # Inferência
    try:
        df_zero = sql_query(f"SELECT * FROM ({qstate['text']}) LIMIT 0")
        cols = list(df_zero.columns)
        time_col = st.selectbox("Coluna de tempo", options=cols, index=0 if cols else None)
        value_col = st.selectbox("Coluna de valor", options=[c for c in cols if c != time_col], index=0 if len(cols) > 1 else None)
    except Exception as e:
        st.error(f"Não foi possível inferir colunas: {e}")
        return

    st.subheader("Parâmetros")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        window = st.number_input("Janela (média móvel)", min_value=3, max_value=365, value=20)
    with c2:
        z_thresh = st.number_input("Z-score limite", min_value=1.0, max_value=10.0, value=3.0)
    with c3:
        limit = st.number_input("Limite de linhas", min_value=100, max_value=200000, value=5000, step=100)

    run = st.button("Detectar anomalias", type="primary")
    if not run:
        return

    with st.spinner("Executando consulta..."):
        df = sql_query(f"SELECT {time_col} AS ds, {value_col} AS y FROM ({qstate['text']}) ORDER BY {time_col} LIMIT {int(limit)}")
    if df.empty:
        st.warning("Sem dados retornados.")
        return

    # Detectar anomalias: z-score sobre resíduo da média móvel
    df['ds'] = pd.to_datetime(df['ds'])
    y_num = pd.to_numeric(df['y'], errors='coerce')
    roll = y_num.rolling(int(window), min_periods=5).mean()
    resid = y_num - roll
    std = resid.rolling(int(window), min_periods=5).std()
    z = (resid / std).abs()
    df['is_anomaly'] = (z > z_thresh)

    # Gráfico
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=y_num, name='Valor', mode='lines', line=dict(color='#2a9d8f')))
    anom = df[df['is_anomaly']]
    if not anom.empty:
        fig.add_trace(go.Scatter(x=anom['ds'], y=pd.to_numeric(anom['y'], errors='coerce'), mode='markers', name='Anomalias', marker=dict(color='#e76f51', size=9)))
    fig.update_layout(title='Série e Anomalias (z-score de resíduos)', xaxis_title='Tempo', yaxis_title=value_col)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resultados")
    st.dataframe(df[['ds','y','is_anomaly']], use_container_width=True, hide_index=True)
    _download_button(df[['ds','y','is_anomaly']], "Baixar (CSV)", "anomalias.csv")

# ----------------------------
# Extração de tópicos / frases-chave
# ----------------------------

def topic_extraction_page():
    st.title("Extração de Tópicos")
    st.caption("Extraia tópicos e frases-chave de um conjunto de textos e visualize sua frequência.")

    qstate = code_editor("", lang="sql", height="250px", buttons=[{"name":"Run","feather":"Play","hasText":True,"showWithIcon":True,"commands":["submit"],"alwaysOn":True,"style":{"bottom":"6px","right":"0.4rem"}}])
    if not qstate['text']:
        st.info("Escreva sua consulta e clique em Run.")
        return

    try:
        df_zero = sql_query(f"SELECT * FROM ({qstate['text']}) LIMIT 0")
        cols = list(df_zero.columns)
        text_col = st.selectbox("Coluna de texto", options=cols, index=0 if cols else None)
    except Exception as e:
        st.error(f"Não foi possível inferir colunas: {e}")
        return

    st.subheader("Parâmetros")
    c1, c2 = st.columns([2,1])
    with c1:
        default_types = ['topic', 'key_phrase']
        types = st_tags(label='Tipos de extração (ex.: topic, key_phrase):', text='Pressione ENTER para adicionar', value=default_types)
    with c2:
        limit = st.number_input("Limite de linhas", min_value=50, max_value=10000, value=1000, step=50)

    run = st.button("Extrair tópicos", type="primary")
    if not run:
        return

    types_clean = [t.strip() for t in types if t and t.strip()]
    if not types_clean:
        st.warning("Informe ao menos um tipo de extração.")
        return

    types_str = "'" + "','".join(types_clean) + "'"
    with st.spinner("Executando ai_extract..."):
        q = f"SELECT {text_col} AS texto, ai_extract({text_col}, ARRAY({types_str})) AS extracao FROM ({qstate['text']}) LIMIT {int(limit)}"
        try:
            df_raw = sql_query(q)
        except Exception as e:
            st.error(f"Falha em ai_extract: {e}")
            return

    # Normalizar
    def _flatten(df):
        rows = []
        for idx, row in df.iterrows():
            data = row.get('extracao')
            if data is None:
                continue
            try:
                if isinstance(data, str):
                    data = json.loads(data)
            except Exception:
                pass
            if isinstance(data, dict):
                for k, vals in data.items():
                    if isinstance(vals, list):
                        for v in vals:
                            rows.append({"linha": idx, "tipo": k, "valor": str(v)})
                    else:
                        rows.append({"linha": idx, "tipo": k, "valor": str(vals)})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['linha','tipo','valor'])

    df_long = _flatten(df_raw)

    st.subheader("Resultados")
    tabs = st.tabs(["Tabela (bruta)", "Normalizada", "Agregados"]) 
    with tabs[0]:
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
        _download_button(df_raw, "Baixar bruto (CSV)", "topicos_bruto.csv")
    with tabs[1]:
        st.dataframe(df_long, use_container_width=True, hide_index=True)
        _download_button(df_long, "Baixar extração (CSV)", "topicos_normalizado.csv")
    with tabs[2]:
        if df_long.empty:
            st.info("Sem dados.")
        else:
            counts = df_long.groupby(["tipo","valor"]).size().reset_index(name="quantidade").sort_values("quantidade", ascending=False)
            fig = px.bar(counts.head(40), x="valor", y="quantidade", color="tipo", title="Top tópicos / frases")
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# PII / Redação (mascaramento)
# ----------------------------

def pii_redaction_page():
    st.title("Mascaramento de PII")
    st.caption("Detecte e mascare informações sensíveis (PII) em textos.")

    qstate = code_editor("", lang="sql", height="250px", buttons=[{"name":"Run","feather":"Play","hasText":True,"showWithIcon":True,"commands":["submit"],"alwaysOn":True,"style":{"bottom":"6px","right":"0.4rem"}}])
    if not qstate['text']:
        st.info("Escreva sua consulta e clique em Run.")
        return

    try:
        df_zero = sql_query(f"SELECT * FROM ({qstate['text']}) LIMIT 0")
        cols = list(df_zero.columns)
        text_col = st.selectbox("Coluna de texto", options=cols, index=0 if cols else None)
    except Exception as e:
        st.error(f"Não foi possível inferir colunas: {e}")
        return

    st.subheader("Parâmetros")
    pii_defaults = ['person','email','phone','credit_card','bank_account','ssn','address']
    types = st_tags(label='Tipos de PII:', text='Pressione ENTER para adicionar', value=pii_defaults)
    mask_token = st.text_input("Token de máscara", value="[REDACTED]")
    limit = st.number_input("Limite de linhas", min_value=50, max_value=10000, value=1000, step=50)

    run = st.button("Mascarar", type="primary")
    if not run:
        return

    t_clean = [t.strip() for t in types if t and t.strip()]
    t_str = "'" + "','".join(t_clean) + "'"
    with st.spinner("Detectando PII..."):
        q = f"SELECT {text_col} AS texto, ai_extract({text_col}, ARRAY({t_str})) AS pii FROM ({qstate['text']}) LIMIT {int(limit)}"
        try:
            df = sql_query(q)
        except Exception as e:
            st.error(f"Falha ao extrair PII: {e}")
            return

    # Aplicar máscara
    def _mask_text(row):
        text = str(row['texto'])
        ents = row.get('pii')
        try:
            if isinstance(ents, str):
                ents = json.loads(ents)
        except Exception:
            pass
        if isinstance(ents, dict):
            for _, vals in ents.items():
                if isinstance(vals, list):
                    for v in vals:
                        try:
                            text = text.replace(str(v), mask_token)
                        except Exception:
                            continue
        return text

    df['mascarado'] = df.apply(_mask_text, axis=1)

    st.subheader("Resultados")
    st.dataframe(df[['texto','mascarado','pii']], use_container_width=True, hide_index=True)
    _download_button(df[['texto','mascarado','pii']], "Baixar (CSV)", "pii_mascarado.csv")

# ----------------------------
# Similaridade / Deduplicação (aprox.)
# ----------------------------

def similarity_page():
    st.title("Similaridade e Deduplicação")
    st.caption("Encontre textos semelhantes ou duplicados (aproximação via Jaccard de tokens).")

    qstate = code_editor("", lang="sql", height="250px", buttons=[{"name":"Run","feather":"Play","hasText":True,"showWithIcon":True,"commands":["submit"],"alwaysOn":True,"style":{"bottom":"6px","right":"0.4rem"}}])
    if not qstate['text']:
        st.info("Escreva sua consulta e clique em Run.")
        return

    try:
        df = sql_query(f"SELECT * FROM ({qstate['text']}) LIMIT 1000")
        cols = list(df.columns)
        text_col = st.selectbox("Coluna de texto", options=cols, index=0 if cols else None)
    except Exception as e:
        st.error(f"Erro ao executar consulta: {e}")
        return

    st.subheader("Parâmetros")
    c1, c2 = st.columns([1,1])
    with c1:
        threshold = st.slider("Limiar de similaridade (Jaccard)", min_value=0.1, max_value=0.9, value=0.7, step=0.05)
    with c2:
        max_pairs = st.number_input("Máx. pares retornados", min_value=20, max_value=5000, value=200, step=20)

    run = st.button("Detectar similares", type="primary")
    if not run:
        return

    # Similaridade aproximada Jaccard sobre tokens simples
    texts = df[text_col].astype(str).tolist()
    tokenized = [set(t.lower().split()) for t in texts]
    pairs = []
    n = len(tokenized)
    for i in range(n):
        for j in range(i+1, n):
            a, b = tokenized[i], tokenized[j]
            if not a or not b:
                continue
            inter = len(a & b)
            union = len(a | b)
            sim = inter / union if union else 0.0
            if sim >= threshold:
                pairs.append({"idx_a": i, "idx_b": j, "sim_jaccard": sim, "texto_a": texts[i][:300], "texto_b": texts[j][:300]})
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    res = pd.DataFrame(pairs)
    if res.empty:
        st.info("Nenhum par semelhante encontrado com o limiar atual.")
        return

    st.subheader("Resultados")
    st.dataframe(res, use_container_width=True, hide_index=True)
    _download_button(res, "Baixar pares (CSV)", "similares.csv")




