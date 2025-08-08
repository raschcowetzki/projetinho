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

# Extração de Entidades

def extract_entities():
    if 'step_ext' not in st.session_state:
        st.session_state.step_ext = 1

    if st.session_state.step_ext == 1:
        st.caption("Forneça uma consulta que retorne uma coluna de texto para extração.")
        query_text, col = _input_query_and_column("Extração de Entidade - Dados iniciais")
        entity_types = st_tags(label='Tipos de entidade (ex.: person, organization, location, date, email, phone, url, money):',
                               text='Pressione ENTER para adicionar mais', value=['person','organization','location'])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Avançar", disabled=st.session_state.button_disabled):
                st.session_state.ext_query = query_text
                st.session_state.ext_col = col
                st.session_state.ext_types = entity_types
                st.session_state.step_ext = 2
                st.rerun()
        with col2:
            if st.button("Limpar"):
                st.session_state.step_ext = 1
                st.rerun()
    else:
        st.title("Extração de Entidade - Resultado")
        types_str = "'"+"','".join([t.strip() for t in st.session_state.ext_types if t.strip()])+"'"
        with st.spinner("Extraindo entidades..."):
            model_query = f"SELECT {st.session_state.ext_col} AS texto, ai_extract({st.session_state.ext_col}, ARRAY({types_str})) AS entidades FROM ({st.session_state.ext_query} limit 1000)"
            df = sql_query(model_query)
        st.dataframe(df, use_container_width=True, hide_index=True)
        # Consolida contagens de entidades
        try:
            long_rows = []
            for _, row in df.iterrows():
                ents = row.get('entidades')
                if isinstance(ents, dict):
                    for k, v in ents.items():
                        if isinstance(v, list):
                            for item in v:
                                long_rows.append({"tipo": k, "valor": str(item)})
                elif isinstance(ents, str):
                    parsed = json.loads(ents)
                    for k, v in parsed.items():
                        for item in v:
                            long_rows.append({"tipo": k, "valor": str(item)})
            if long_rows:
                df_long = pd.DataFrame(long_rows)
                counts = df_long.groupby(["tipo", "valor"]).size().reset_index(name="quantidade")
                top = counts.sort_values("quantidade", ascending=False).head(30)
                fig = px.bar(top, x="valor", y="quantidade", color="tipo", title="Top Entidades Extraídas")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Não foi possível agregar as entidades automaticamente: {e}")
        _download_button(df, "Baixar resultados (CSV)", "entidades.csv")
        if st.button("Voltar"):
            st.session_state.step_ext = 1
            st.rerun()

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
    time_col = None
    target_col = None
    try:
        df_zero = sql_query(f"SELECT * FROM ({query_state['text']}) LIMIT 0")
        cols = list(df_zero.columns)
        time_col = st.selectbox("Coluna de tempo", options=cols, index=0 if cols else None)
        numeric_cols = [c for c in cols if c != time_col]
        target_col = st.selectbox("Coluna alvo (valor)", options=numeric_cols, index=0 if numeric_cols else None)
    except Exception as e:
        st.error(f"Não foi possível inferir as colunas: {e}")
        return

    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        frequency = st.selectbox("Frequência", ["D","W","M"], index=0, help="Periodicidade dos dados")
    with col_b:
        horizon = st.number_input("Horizonte (períodos)", min_value=1, max_value=365, value=30)
    with col_c:
        use_automl = st.toggle("Executar AutoML no Databricks", value=False)

    run = st.button("Executar previsão", type="primary")
    if not run:
        return

    with st.spinner("Consultando dados históricos..."):
        df_hist = sql_query(f"SELECT {time_col} AS ds, {target_col} AS y FROM ({query_state['text']}) ORDER BY {time_col}")

    if df_hist.empty:
        st.warning("A consulta não retornou dados.")
        return

    # Forecast baseline simples (persistência do último valor)
    try:
        df_hist['ds'] = pd.to_datetime(df_hist['ds'])
        last_date = df_hist['ds'].max()
        if frequency == 'D':
            future_index = pd.date_range(last_date, periods=horizon+1, freq='D')[1:]
        elif frequency == 'W':
            future_index = pd.date_range(last_date, periods=horizon+1, freq='W')[1:]
        else:
            future_index = pd.date_range(last_date, periods=horizon+1, freq='MS')[1:]
        last_value = float(pd.to_numeric(df_hist['y'], errors='coerce').dropna().iloc[-1])
        df_fc = pd.DataFrame({"ds": future_index, "yhat": [last_value]*len(future_index)})
    except Exception as e:
        st.error(f"Falha ao construir baseline: {e}")
        return

    # Plot
    hist_plot = df_hist.rename(columns={"y": "valor"})
    fc_plot = df_fc.rename(columns={"yhat": "valor"})
    hist_plot['tipo'] = 'Histórico'
    fc_plot['tipo'] = 'Projeção'
    plot_df = pd.concat([hist_plot, fc_plot], ignore_index=True)
    fig = px.line(plot_df, x='ds', y='valor', color='tipo', title='Histórico e Projeção (baseline)')
    st.plotly_chart(fig, use_container_width=True)

    # Tabela combinada e download
    df_out = df_hist.copy()
    df_out['yhat'] = None
    fc_out = pd.DataFrame({"ds": df_fc['ds'], "y": None, "yhat": df_fc['yhat']})
    results = pd.concat([df_out, fc_out], ignore_index=True)
    st.dataframe(results.tail(horizon+5), use_container_width=True, hide_index=True)
    _download_button(results, "Baixar projeção (CSV)", "projecao.csv")

    # AutoML opcional
    if use_automl:
        try:
            import databricks.automl as automl
        except Exception:
            automl = None
        if automl is None:
            st.warning("AutoML não disponível neste ambiente. Execute em um cluster Databricks.")
        else:
            with st.spinner("Iniciando AutoML Forecast..."):
                try:
                    summary = automl.forecast(dataset=df_hist, target_col='y', time_col='ds', frequency=frequency, horizon=int(horizon))
                    st.success("AutoML iniciado/finalizado. Veja abaixo os detalhes retornados.")
                    try:
                        st.json(summary)
                    except Exception:
                        st.write(summary)
                except Exception as e:
                    st.error(f"Falha ao executar AutoML: {e}")

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




