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

def _fig_to_png_bytes(fig) -> bytes | None:
    try:
        return pio.to_image(fig, format='png', scale=2)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=300)
def _cached_sql_result(query: str) -> pd.DataFrame:
    return sql_query(query)

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
        ai_toggle = st.toggle("Gerar insights (IA)", value=False, help="Resumo textual sobre a distribuição de sentimentos.")
        st.session_state.sent_ai_insights = ai_toggle
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
        if st.session_state.get('sent_ai_insights'):
            try:
                top_sent = counts.sort_values('quantidade', ascending=False).head(3)
                prompt_lines = ["Resuma a distribuição de sentimentos:"]
                for _, row in top_sent.iterrows():
                    prompt_lines.append(f"- {row['sentimento']}: {int(row['quantidade'])}")
                prompt_lines.append("Comente possíveis causas e próximos passos (ex.: filtrar exemplos, olhar segmentos).")
                prompt = "\n".join(prompt_lines).replace("'","''")
                df_ai = sql_query(f"SELECT ai_gen('{prompt}') AS insights")
                st.markdown(df_ai.iloc[0]['insights'] if not df_ai.empty else "")
            except Exception as e:
                st.warning(f"Não foi possível gerar insights: {e}")
        if st.button("Voltar"):
            st.session_state.step_sent = 1
            st.rerun()

def extract_entities():
    st.title("Extração de Entidades")
    st.caption("Extraia e analise entidades de textos usando ai_extract. Forneça uma consulta que retorne uma coluna de texto e selecione os tipos de entidades.")

    query_state = code_editor(
        "",
        lang="sql",
        height="250px",
        buttons=[
            {"name": "Run", "feather": "Play", "hasText": True, "showWithIcon": True, "commands": ["submit"], "alwaysOn": True, "style": {"bottom": "6px", "right": "0.4rem"}}
        ]
    )
    with st.expander("Exemplos de SQL"):
        st.code("""
-- Exemplo simples
SELECT texto
FROM minha_tabela
LIMIT 1000
""", language="sql")
    if not query_state['text']:
        st.info("Escreva sua consulta acima e clique em Run para habilitar os parâmetros.")
        return

    try:
        df_zero = sql_query(f"SELECT * FROM ({query_state['text']}) LIMIT 0")
        cols = list(df_zero.columns)
        text_col = st.selectbox("Coluna de texto", options=cols, index=0 if cols else None, help="Coluna que contém o texto de onde as entidades serão extraídas.")
    except Exception as e:
        st.error(f"Não foi possível inferir as colunas: {e}")
        return

    st.subheader("Parâmetros")
    c1, c2 = st.columns([2,1])
    with c1:
        curated = [
            'person','organization','location','date','time','datetime','email','phone',
            'url','money','quantity','event','product','title','nationality','language'
        ]
        selected = st.multiselect("Tipos de entidade", options=curated, default=['person','organization','location'], help="Tipos padrão de entidades para procurar.")
        extra = st_tags(label='Tipos personalizados:', text='Pressione ENTER para adicionar', value=[])
        entity_types = [t for t in selected] + [t.strip() for t in extra if t and t.strip()]
    with c2:
        limit = st.number_input("Limite de linhas", min_value=50, max_value=20000, value=1000, step=50, help="Quantidade de linhas da sua consulta a processar.")

    with st.expander("Avançado"):
        col_a, col_b = st.columns(2)
        with col_a:
            top_n = st.number_input("Top N (agregados)", min_value=5, max_value=100, value=30, step=5, help="Quantidade máxima de entidades exibidas nos rankings.")
            normalize_case = st.toggle("Normalizar caixa (lowercase)", value=True, help="Converte as entidades para minúsculas para facilitar a agregação.")
        with col_b:
            drop_numeric = st.toggle("Remover números puros", value=True, help="Remove entidades que são apenas números.")
            regex_filter = st.text_input("Regex filtro de valor (opcional)", value="", help="Filtra entidades pelo padrão informado (expressão regular).")

    run = st.button("Extrair entidades", type="primary")
    if not run:
        return

    if not entity_types:
        st.warning("Informe ao menos um tipo de entidade.")
        return

    types_str = "'" + "','".join(entity_types) + "'"
    with st.spinner("Executando ai_extract no warehouse..."):
        q = f"SELECT {text_col} AS texto, ai_extract({text_col}, ARRAY({types_str})) AS entidades FROM ({query_state['text']}) LIMIT {int(limit)}"
        try:
            df_raw = sql_query(q)
        except Exception as e:
            st.error(f"Falha ao executar ai_extract: {e}")
            return

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
                            rows.append({"linha": idx, "tipo": str(ent_type), "valor": '' if v is None else str(v)})
                    else:
                        rows.append({"linha": idx, "tipo": str(ent_type), "valor": '' if ent_values is None else str(ent_values)})
        out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["linha","tipo","valor"])
        if not out.empty:
            if normalize_case:
                out['valor'] = out['valor'].str.lower()
            if drop_numeric:
                out = out[~out['valor'].str.fullmatch(r"\d+(?:[.,]\d+)?", na=False)]
            if regex_filter:
                out = out[out['valor'].str.contains(regex_filter, na=False, regex=True)]
            out = out[out['valor'].str.len() > 0]
        return out

    df_long = _flatten_entities(df_raw)

    st.subheader("Visão geral")
    colm = st.columns(3)
    total_rows = len(df_raw)
    rows_with_entities = df_long['linha'].nunique() if not df_long.empty else 0
    distinct_entities = df_long['valor'].nunique() if not df_long.empty else 0
    colm[0].metric("Linhas avaliadas", f"{total_rows}")
    colm[1].metric("Linhas com entidades", f"{rows_with_entities}")
    colm[2].metric("Entidades distintas", f"{distinct_entities}")

    tab_overview, tab_entities, tab_aggs, tab_raw = st.tabs(["Resumo", "Entidades", "Agregados", "Bruto"]) 

    with tab_overview:
        if df_long.empty:
            st.info("Nenhuma entidade foi detectada.")
        else:
            by_type = df_long.groupby('tipo').size().reset_index(name='quantidade').sort_values('quantidade', ascending=False)
            fig = px.bar(by_type, x='tipo', y='quantidade', title='Distribuição por tipo de entidade')
            st.plotly_chart(fig, use_container_width=True)
            counts_global = df_long.groupby('valor').size().reset_index(name='quantidade').sort_values('quantidade', ascending=False)
            fig2 = px.bar(counts_global.head(int(top_n)), x='valor', y='quantidade', title=f'Top {int(top_n)} entidades (todas)')
            st.plotly_chart(fig2, use_container_width=True)

    with tab_entities:
        if df_long.empty:
            st.info("Sem entidades para exibir.")
        else:
            f1, f2 = st.columns([1,2])
            with f1:
                type_filter = st.multiselect("Filtrar tipos", options=sorted(df_long['tipo'].unique().tolist()), default=None, help="Mostra apenas os tipos selecionados.")
            with f2:
                search = st.text_input("Contém (texto)", value="", help="Mostra apenas entidades que contenham o texto informado.")
            df_show = df_long.copy()
            if type_filter:
                df_show = df_show[df_show['tipo'].isin(type_filter)]
            if search:
                df_show = df_show[df_show['valor'].str.contains(search, case=False, na=False)]
            st.dataframe(df_show, use_container_width=True, hide_index=True)
            _download_button(df_show, "Baixar entidades (CSV)", "entidades_normalizado.csv")

    with tab_aggs:
        if df_long.empty:
            st.info("Sem dados para agregar.")
        else:
            t = st.selectbox("Tipo para detalhar", options=sorted(df_long['tipo'].unique().tolist()))
            subset = df_long[df_long['tipo'] == t]
            counts = subset.groupby('valor').size().reset_index(name='quantidade').sort_values('quantidade', ascending=False)
            fig = px.bar(counts.head(int(top_n)), x='valor', y='quantidade', title=f'Top {int(top_n)} entidades para {t}')
            st.plotly_chart(fig, use_container_width=True)
            _download_button(counts, "Baixar agregados (CSV)", f"agregados_{t}.csv")

    with tab_raw:
        st.dataframe(df_raw, use_container_width=True, hide_index=True)
        _download_button(df_raw, "Baixar bruto (CSV)", "entidades_bruto.csv")

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
    with st.expander("Exemplos de SQL"):
        st.code("""
-- Exemplo simples
SELECT ts AS tempo, valor
FROM minha_tabela
ORDER BY ts
LIMIT 5000
""", language="sql")
    if not query_state['text']:
        st.info("Escreva sua consulta à esquerda e clique em Run.")
        return

    # Inferência de colunas
    try:
        df_zero = sql_query(f"SELECT * FROM ({query_state['text']}) LIMIT 0")
        cols = list(df_zero.columns)
        time_col = st.selectbox("Coluna de tempo", options=cols, index=0 if cols else None, help="Coluna temporal, ordenada, usada como eixo X da série.")
        value_col = st.selectbox("Coluna alvo (valor)", options=[c for c in cols if c != time_col], index=0 if len(cols) > 1 else None, help="Série numérica a ser prevista.")
    except Exception as e:
        st.error(f"Não foi possível inferir as colunas: {e}")
        return

    st.subheader("Parâmetros")
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        horizon_mode = st.selectbox("Modo de horizonte", ["Relativo (N unidades)", "Absoluto (timestamp)"], help="Escolha entre prever até uma data específica (Absoluto) ou por N períodos (Relativo).")
    with col_b:
        frequency = st.selectbox("Frequência", ["D","W","M"], index=0, help="Periodicidade dos dados quando o horizonte é relativo: D=dia, W=semana, M=mês.")
    with col_c:
        show_conf = st.toggle("Exibir intervalo de confiança", value=True, help="Mostra a banda de incerteza da previsão, quando disponível.")

    ai_insights_fc = st.toggle("Gerar insights (IA)", value=False, help="Gera um resumo textual da previsão.")

    if horizon_mode == "Relativo (N unidades)":
        horizon = st.number_input("Horizonte (períodos)", min_value=1, max_value=365, value=30, help="Quantidade de períodos na frente para prever.")
        absolute_horizon_str = None
    else:
        absolute_horizon_str = st.text_input("Horizon (timestamp)", value="", placeholder="YYYY-MM-DD HH:MM:SS", help="Data/hora final da previsão no formato indicado.")
        horizon = None

    run = st.button("Executar previsão", type="primary")
    if not run:
        return

    # Consultar histórico e calcular horizonte e executar previsão
    with st.spinner("Consultando dados históricos..."):
        df_hist = sql_query(f"SELECT {time_col} AS ds, {value_col} AS y FROM ({query_state['text']}) ORDER BY {time_col}")
    if df_hist.empty:
        st.warning("A consulta não retornou dados.")
        return

    try:
        df_hist['ds'] = pd.to_datetime(df_hist['ds'])
        last_date = df_hist['ds'].max()
        if absolute_horizon_str:
            horizon_str = absolute_horizon_str
        else:
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

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.to_datetime(df_hist['ds']), y=pd.to_numeric(df_hist['y'], errors='coerce'), name='Histórico', mode='lines', line=dict(color='#2a9d8f')))
        if show_conf and col_map['yhat_lower'] and col_map['yhat_upper']:
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
        fig.update_layout(title='Histórico e Projeção (ai_forecast)', xaxis_title='Tempo', yaxis_title=value_col)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Falha ao montar o gráfico: {e}")

    st.subheader("Resultados")
    cols_show = [c for c in [col_map['ds'], col_map['yhat_lower'], col_map['yhat'], col_map['yhat_upper']] if c]
    st.dataframe(df_fc[cols_show], use_container_width=True, hide_index=True)
    _download_button(df_fc[cols_show], "Baixar projeção (CSV)", "projecao_ai_forecast.csv")

    if ai_insights_fc:
        try:
            hist_tail = df_hist.tail(10)
            fc_head = df_fc.head(10)[[col_map['ds'], col_map['yhat']]].rename(columns={col_map['ds']:'ds', col_map['yhat']:'yhat'})
            prompt_lines = [
                "Gere um resumo curto sobre a projeção a seguir:",
                f"- Série com {len(df_hist)} pontos; horizonte até {horizon_str}",
                "- Últimos pontos históricos:" 
            ]
            for _, r in hist_tail.iterrows():
                prompt_lines.append(f"  - {r['ds']} : {r['y']}")
            prompt_lines.append("- Primeiros pontos projetados:")
            for _, r in fc_head.iterrows():
                prompt_lines.append(f"  - {r['ds']} : {r['yhat']}")
            if show_conf and col_map['yhat_lower'] and col_map['yhat_upper']:
                prompt_lines.append("- Há uma banda de confiança associada às previsões.")
            prompt_lines.append("Comente tendência, possíveis sazonalidades e cautelas.")
            prompt = "\n".join(prompt_lines).replace("'","''")
            df_ai = sql_query(f"SELECT ai_gen('{prompt}') AS insights")
            insight_text = df_ai.iloc[0]['insights'] if not df_ai.empty else ""
            st.markdown(insight_text)
        except Exception as e:
            st.warning(f"Não foi possível gerar insights: {e}")

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
    with st.expander("Exemplos de SQL"):
        st.code("""
SELECT ts, valor, categoria
FROM minha_tabela
ORDER BY ts
LIMIT 5000
""", language="sql")
    if not qstate['text']:
        st.info("Escreva sua consulta e clique em Run.")
        return

    try:
        df_zero = sql_query(f"SELECT * FROM ({qstate['text']}) LIMIT 0")
        cols = list(df_zero.columns)
        time_col = st.selectbox("Coluna de tempo", options=cols, index=0 if cols else None, help="Coluna temporal usada para ordenação.")
        value_col = st.selectbox("Coluna de valor", options=[c for c in cols if c != time_col], index=0 if len(cols) > 1 else None, help="Série numérica para detecção de anomalias.")
    except Exception as e:
        st.error(f"Não foi possível inferir colunas: {e}")
        return

    with st.expander("Avançado (parâmetros)"):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            method = st.selectbox("Método", ["Z-score (resíduo MM)", "MAD (resíduo MM)"], help="Critério de outlier com base no resíduo da média móvel.")
        with c2:
            window = st.number_input("Janela (MM)", min_value=3, max_value=365, value=20, help="Tamanho da janela da média móvel para suavizar a série.")
        with c3:
            z_thresh = st.number_input("Z/MAD limite", min_value=1.0, max_value=10.0, value=3.0, help="Limiar do escore para marcar anomalias.")
        cA, cB = st.columns([1,1])
        with cA:
            limit = st.number_input("Limite de linhas", min_value=100, max_value=200000, value=5000, step=100, help="Quantidade de linhas da consulta a processar.")
            group_col = st.selectbox("Agrupar por (opcional)", options=["(nenhum)"] + [c for c in cols if c not in (time_col, value_col)], help="Detecta anomalias separadamente por grupo.")
        with cB:
            group_to_plot = st.selectbox("Grupo para visualizar", options=["(auto)"] + ([""] if False else []), help="Grupo a destacar no gráfico quando há agrupamento.")
        ai_insights_anom = st.toggle("Gerar insights (IA)", value=False, help="Resumo textual das anomalias detectadas.")

    run = st.button("Detectar anomalias", type="primary")
    if not run:
        return

    with st.spinner("Executando consulta..."):
        df = sql_query(f"SELECT * FROM ({qstate['text']}) ORDER BY {time_col} LIMIT {int(limit)}")
    if df.empty:
        st.warning("Sem dados retornados.")
        return

    df['ds'] = pd.to_datetime(df[time_col])
    df['y'] = pd.to_numeric(df[value_col], errors='coerce')

    def detect_series(d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values('ds').copy()
        roll = d['y'].rolling(int(window), min_periods=5).mean()
        resid = d['y'] - roll
        if method.startswith("MAD"):
            med = resid.rolling(int(window), min_periods=5).median()
            mad = (resid - med).abs().rolling(int(window), min_periods=5).median()
            score = (resid - med).abs() / (mad.replace(0, pd.NA))
        else:
            std = resid.rolling(int(window), min_periods=5).std()
            score = (resid / std.replace(0, pd.NA)).abs()
        d['is_anomaly'] = (score > z_thresh).fillna(False)
        return d

    if group_col and group_col != "(nenhum)" and group_col in df.columns:
        groups = df[group_col].dropna().unique().tolist()
        if group_to_plot == "(auto)":
            group_to_plot = groups[0] if groups else None
        df = df.groupby(group_col, group_keys=False).apply(detect_series)
        st.subheader("Resultados do grupo selecionado")
        if group_to_plot is not None:
            dplot = df[df[group_col] == group_to_plot]
        else:
            dplot = df
    else:
        df = detect_series(df)
        dplot = df

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dplot['ds'], y=dplot['y'], name='Valor', mode='lines', line=dict(color='#2a9d8f')))
    anom = dplot[dplot['is_anomaly']]
    if not anom.empty:
        fig.add_trace(go.Scatter(x=anom['ds'], y=anom['y'], mode='markers', name='Anomalias', marker=dict(color='#e76f51', size=9)))
    title = 'Série e Anomalias' + (f" - grupo {group_to_plot}" if group_col and group_col != "(nenhum)" and group_to_plot else "")
    fig.update_layout(title=title, xaxis_title='Tempo', yaxis_title=value_col)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Resultados")
    st.dataframe(dplot[[time_col, value_col, 'is_anomaly']], use_container_width=True, hide_index=True)
    _download_button(dplot[[time_col, value_col, 'is_anomaly']], "Baixar (CSV)", "anomalias.csv")

    if group_col and group_col != "(nenhum)" and group_col in df.columns:
        by_group = df.groupby(group_col)['is_anomaly'].sum().reset_index(name='anomalias')
        st.markdown("#### Anomalias por grupo")
        st.dataframe(by_group, use_container_width=True, hide_index=True)
        fig2 = px.bar(by_group.sort_values('anomalias', ascending=False), x=group_col, y='anomalias', title='Contagem de anomalias por grupo')
        st.plotly_chart(fig2, use_container_width=True)

    if ai_insights_anom:
        try:
            total = int(dplot['is_anomaly'].sum())
            prompt_lines = [
                "Resuma a detecção de anomalias:",
                f"- Janela: {int(window)}; Método: {method}; Limite: {z_thresh}",
                f"- Total de anomalias detectadas no recorte: {total}",
            ]
            if group_col and group_col != "(nenhum)" and group_col in df.columns:
                topg = by_group.sort_values('anomalias', ascending=False).head(5)
                prompt_lines.append("- Grupos com mais anomalias:")
                for _, r in topg.iterrows():
                    prompt_lines.append(f"  - {r[group_col]}: {int(r['anomalias'])}")
            prompt_lines.append("Sugira hipóteses de causa e próximos passos (ex.: analisar períodos, segmentações, sazonalidade).")
            prompt = "\n".join(prompt_lines).replace("'","''")
            df_ai = sql_query(f"SELECT ai_gen('{prompt}') AS insights")
            st.markdown(df_ai.iloc[0]['insights'] if not df_ai.empty else "")
        except Exception as e:
            st.warning(f"Não foi possível gerar insights: {e}")

# ----------------------------
# Clusterização automática (KMeans + PCA)
# ----------------------------

def clustering_page():
    st.title("Clusterização")
    st.caption("Agrupe dados automaticamente a partir de uma consulta SQL. Suporta colunas numéricas e visualização com PCA.")

    qstate = code_editor("", lang="sql", height="250px", buttons=[{"name":"Run","feather":"Play","hasText":True,"showWithIcon":True,"commands":["submit"],"alwaysOn":True,"style":{"bottom":"6px","right":"0.4rem"}}])
    with st.expander("Exemplos de SQL"):
        st.code("""
SELECT *
FROM minha_tabela
LIMIT 10000
""", language="sql")
    if not qstate['text']:
        st.info("Escreva sua consulta e clique em Run.")
        return

    st.subheader("Parâmetros")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        k = st.number_input("Número de clusters (k)", min_value=2, max_value=20, value=5, help="Quantidade de grupos a formar.")
    with c2:
        limit = st.number_input("Limite de linhas", min_value=100, max_value=50000, value=5000, step=100, help="Quantidade de linhas da consulta a usar.")
    with c3:
        scale = st.toggle("Padronizar (Z-score)", value=True, help="Padroniza variáveis para média 0 e desvio 1 antes do cluster.")

    with st.expander("Avançado"):
        sel_cols = []
        try:
            df_head = sql_query(f"SELECT * FROM ({qstate['text']}) LIMIT 1")
            num_candidates = df_head.select_dtypes(include=['number']).columns.tolist()
            sel_cols = st.multiselect("Selecionar colunas numéricas específicas", options=num_candidates, default=num_candidates, help="Escolha variáveis numéricas a considerar no cluster.")
        except Exception:
            pass
        elbow = st.toggle("Calcular curva de cotovelo (inércia)", value=False, help="Calcula inércia por k para ajudar a escolher k.")
        elbow_range = st.slider("Faixa k (elbow)", min_value=2, max_value=15, value=(2, 8), help="Intervalo de k para a curva de cotovelo.")
        ai_insights_clu = st.toggle("Gerar insights (IA)", value=False, help="Resumo textual da clusterização.")

    run = st.button("Executar clusterização", type="primary")
    if not run:
        return

    with st.spinner("Executando consulta..."):
        try:
            df = sql_query(f"SELECT * FROM ({qstate['text']}) LIMIT {int(limit)}")
        except Exception as e:
            st.error(f"Erro ao executar consulta: {e}")
            return
    if df.empty:
        st.warning("Sem dados retornados.")
        return

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except ModuleNotFoundError:
        st.error("Dependência ausente: scikit-learn. Instale o pacote 'scikit-learn' no ambiente do app/cluster e recarregue.")
        st.caption("Exemplo: pip install scikit-learn")
        return

    num_df = df.select_dtypes(include=['number']).copy()
    if sel_cols:
        num_df = num_df[sel_cols].copy()
    if num_df.shape[1] < 2:
        st.error("A consulta deve retornar ao menos duas colunas numéricas para clusterização.")
        return

    X = num_df.fillna(num_df.mean(numeric_only=True))
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)
    else:
        X_scaled = X.values

    try:
        kmeans = KMeans(n_clusters=int(k), n_init='auto', random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        centers = kmeans.cluster_centers_
        sil = silhouette_score(X_scaled, labels) if int(k) > 1 and X.shape[0] > int(k) else None
    except Exception as e:
        st.error(f"Falha no KMeans: {e}")
        return

    try:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        vis = pd.DataFrame({'pc1': coords[:,0], 'pc2': coords[:,1], 'cluster': labels.astype(int)})
        fig = px.scatter(vis, x='pc1', y='pc2', color='cluster', title='Clusters (PCA 2D)')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"PCA não pôde ser executado para visualização: {e}")

    out = df.copy()
    out['cluster'] = labels
    st.subheader("Resultados")
    st.dataframe(out, use_container_width=True, hide_index=True)
    _download_button(out, "Baixar dados com cluster (CSV)", "clusters.csv")

    if sil is not None:
        st.caption(f"Silhouette score: {sil:.3f}")

    st.subheader("Centroides (no espaço escalonado)")
    try:
        centers_df = pd.DataFrame(centers, columns=X.columns)
        centers_df['cluster'] = range(int(k))
        st.dataframe(centers_df, use_container_width=True, hide_index=True)
        _download_button(centers_df, "Baixar centroides (CSV)", "centroides.csv")
    except Exception:
        pass

    if elbow:
        ks = list(range(elbow_range[0], elbow_range[1]+1))
        inertias = []
        for kk in ks:
            try:
                km = KMeans(n_clusters=int(kk), n_init='auto', random_state=42)
                km.fit(X_scaled)
                inertias.append(km.inertia_)
            except Exception:
                inertias.append(None)
        df_elbow = pd.DataFrame({"k": ks, "inertia": inertias})
        fig_e = px.line(df_elbow, x='k', y='inertia', markers=True, title='Curva de cotovelo (inércia)')
        st.plotly_chart(fig_e, use_container_width=True)
        _download_button(df_elbow, "Baixar elbow (CSV)", "elbow.csv")

    if ai_insights_clu:
        try:
            dist_counts = out['cluster'].value_counts().to_dict()
            prompt_lines = ["Resuma a clusterização:"]
            prompt_lines.append("- Tamanho dos clusters:")
            for cid, cnt in dist_counts.items():
                prompt_lines.append(f"  - Cluster {cid}: {int(cnt)}")
            if sil is not None:
                prompt_lines.append(f"- Silhouette: {sil:.3f}")
            prompt_lines.append("Comente sobre separação dos grupos, possíveis interpretações e próximos passos (ex.: perfis, variáveis que mais pesam).")
            prompt = "\n".join(prompt_lines).replace("'","''")
            df_ai = sql_query(f"SELECT ai_gen('{prompt}') AS insights")
            st.markdown(df_ai.iloc[0]['insights'] if not df_ai.empty else "")
        except Exception as e:
            st.warning(f"Não foi possível gerar insights: {e}")

# ----------------------------
# Análise Exploratória de Dados (EDA)
# ----------------------------

def eda_page():
    st.title("Análise de Dados (EDA)")
    st.caption("Explore seu conjunto de dados: amostra, estatísticas, distribuições, correlações e insights gerados por IA.")

    qstate = code_editor("", lang="sql", height="250px", buttons=[{"name":"Run","feather":"Play","hasText":True,"showWithIcon":True,"commands":["submit"],"alwaysOn":True,"style":{"bottom":"6px","right":"0.4rem"}}])
    if not qstate['text']:
        st.info("Escreva sua consulta e clique em Run.")
        return

    st.subheader("Parâmetros de análise")
    p1, p2, p3 = st.columns([1,1,1])
    with p1:
        limit = st.number_input("Limite de linhas", min_value=100, max_value=200000, value=5000, step=100, key='eda_limit', help="Quantidade de linhas a amostrar para análise.")
    with p2:
        enable_ai = st.toggle("Gerar insights com IA", value=False, key='eda_ai', help="Gera um resumo automático com base em estatísticas e amostras.")
    with p3:
        bins = st.slider("Bins (histogramas)", min_value=10, max_value=100, value=30, key='eda_bins', help="Número de caixas para os histogramas.")

    if st.button("Executar análise", type="primary"):
        st.session_state._eda_run_token = True

    if not st.session_state.get('_eda_run_token'):
        st.stop()

    with st.spinner("Executando consulta..."):
        try:
            df = _cached_sql_result(f"SELECT * FROM ({qstate['text']}) LIMIT {int(limit)}")
        except Exception as e:
            st.error(f"Erro ao executar consulta: {e}")
            return
    if df.empty:
        st.warning("Sem dados retornados.")
        return

    # Inferir tipos (usados na visão geral) e para sugerir seleção
    inferred_num = df.select_dtypes(include=['number']).columns.tolist()
    inferred_cat = df.select_dtypes(include=['object']).columns.tolist()

    # Visão geral (usa inferência automática)
    st.subheader("Visão Geral")
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.astype(str)
    null_counts = df.isna().sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Linhas", f"{n_rows}")
    m2.metric("Colunas", f"{n_cols}")
    m3.metric("Numéricas (auto)", f"{len(inferred_num)}")
    m4.metric("Categóricas (auto)", f"{len(inferred_cat)}")

    with st.expander("Tipos de dados e nulos"):
        st.write("Tipos de dados")
        st.dataframe(pd.DataFrame({"coluna": dtypes.index, "tipo": dtypes.values}), use_container_width=True, hide_index=True)
        st.write("Nulos por coluna")
        st.dataframe(pd.DataFrame({"coluna": null_counts.index, "nulos": null_counts.values}), use_container_width=True, hide_index=True)

    # Prévia vem antes da seleção de colunas
    st.subheader("Prévia dos dados")
    st.dataframe(df.head(200), use_container_width=True, hide_index=True)
    _download_button(df, "Baixar amostra (CSV)", "amostra.csv")

    # Seleção de colunas (entre Prévia e as Abas)
    st.caption("Seleção de colunas")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Numéricas")
        num_cols = st.multiselect("Colunas numéricas", options=df.columns.tolist(), default=inferred_num, key='eda_num_cols', label_visibility='collapsed', help="Variáveis numéricas a usar nos gráficos e correlação.")
    with c2:
        st.caption("Categóricas")
        cat_cols = st.multiselect("Colunas categóricas", options=df.columns.tolist(), default=inferred_cat, key='eda_cat_cols', label_visibility='collapsed', help="Variáveis categóricas para frequências.")

    # Abas de análise
    tabs = st.tabs(["Numéricas", "Categóricas", "Correlação", "Insights (IA)"])

    with tabs[0]:
        if not num_cols:
            st.info("Selecione ao menos uma coluna numérica.")
        else:
            st.markdown("#### Estatísticas descritivas")
            try:
                desc = df[num_cols].describe().T.reset_index().rename(columns={"index":"coluna"})
                st.dataframe(desc, use_container_width=True, hide_index=True)
                _download_button(desc, "Baixar estatísticas (CSV)", "estatisticas.csv")
            except Exception:
                st.warning("Não foi possível calcular estatísticas descritivas.")

            st.markdown("#### Distribuições")
            for col in num_cols[: min(6, len(num_cols))]:
                try:
                    fig = px.histogram(df, x=col, nbins=st.session_state['eda_bins'], title=f"Histograma - {col}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    continue

    with tabs[1]:
        if not cat_cols:
            st.info("Selecione ao menos uma coluna categórica.")
        else:
            st.markdown("#### Frequências")
            top_k = st.number_input("Top K por coluna", min_value=5, max_value=50, value=20, key='eda_topk', help="Quantidade máxima de categorias para exibir por coluna.")
            for col in cat_cols[: min(6, len(cat_cols))]:
                vc = df[col].astype(str).value_counts().reset_index().head(int(top_k))
                vc.columns = [col, "quantidade"]
                fig = px.bar(vc, x=col, y="quantidade", title=f"Frequências - {col}")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        if len(num_cols) < 2:
            st.info("São necessárias pelo menos duas colunas numéricas para correlação.")
        else:
            st.markdown("#### Matriz de correlação (Pearson)")
            try:
                corr = df[num_cols].corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", origin="lower")
                st.plotly_chart(fig, use_container_width=True)
                _download_button(corr.reset_index().rename(columns={"index":"coluna"}), "Baixar correlação (CSV)", "correlacao.csv")
            except Exception as e:
                st.warning(f"Não foi possível calcular correlação: {e}")

    with tabs[3]:
        if not enable_ai:
            st.info("Habilite 'Gerar insights com IA' nos parâmetros para produzir um resumo automatizado.")
        else:
            st.markdown("#### Insights gerados por IA")
            prompt_lines = []
            prompt_lines.append("Você é um analista de dados. Gere insights concisos e acionáveis sobre o conjunto de dados a seguir.")
            prompt_lines.append(f"- Linhas: {n_rows}; Colunas: {n_cols}")
            if num_cols:
                prompt_lines.append(f"- Colunas numéricas: {', '.join(num_cols[:20])}{'...' if len(num_cols)>20 else ''}")
                try:
                    desc_small = df[num_cols].describe().T[['mean','std','min','max']].round(3)
                    prompt_lines.append("- Estatísticas (amostra):")
                    for cname, row in desc_small.head(8).iterrows():
                        prompt_lines.append(f"  - {cname}: média={row['mean']}, desvio={row['std']}, min={row['min']}, max={row['max']}")
                except Exception:
                    pass
            if cat_cols:
                prompt_lines.append(f"- Colunas categóricas: {', '.join(cat_cols[:20])}{'...' if len(cat_cols)>20 else ''}")
                for c in cat_cols[:3]:
                    vc = df[c].astype(str).value_counts().head(5)
                    top_str = ", ".join([f"{idx}({val})" for idx, val in vc.items()])
                    prompt_lines.append(f"  - {c}: {top_str}")
            prompt_lines.append("Responda com principais tendências, possíveis problemas de qualidade, outliers suspeitos, segmentos relevantes e hipóteses de negócio.")
            prompt = "\n".join(prompt_lines).replace("'", "''")
            try:
                with st.spinner("Consultando modelo de IA..."):
                    df_ai = sql_query(f"SELECT ai_gen('{prompt}') AS insights")
                insight_text = df_ai.iloc[0]['insights'] if not df_ai.empty else ""
                st.markdown(insight_text)
            except Exception as e:
                st.error(f"Falha ao gerar insights com IA: {e}")

def about_page():
    st.title("Sobre o Mini Cientista")
    st.caption("Guia detalhado, dicas e tutoriais para cada funcionalidade.")

    st.markdown("### Visão Geral")
    st.write(
        """
O Mini Cientista é um app Streamlit integrado ao Databricks, pensado para agilizar análises do dia a dia:
- Exploração de dados (EDA)
- Previsões (ai_forecast)
- Análise de sentimento (ai_analyze_sentiment)
- Classificação (ai_classify)
- Extração de tópicos (ai_extract)
- Detecção de anomalias
- Clusterização (KMeans)
- Chat com Genie e integração com AutoML
        """
    )

    with st.expander("Projeção (ai_forecast)"):
        st.markdown("#### O que faz?")
        st.write("Gera previsões de uma série temporal a partir de uma coluna de tempo e uma coluna de valor.")
        st.markdown("#### Como usar")
        st.write("""
1. Escreva um SQL que retorne colunas de tempo e valor. 
2. Selecione as colunas no app. 
3. Escolha o horizonte: 
   - Relativo (N períodos, com D/W/M) 
   - Absoluto (timestamp) 
4. Opcionalmente exiba a banda de confiança.
5. Execute e analise o gráfico e a tabela de resultados.
        """)
        st.markdown("#### Fundamentos (intuitivo)")
        st.write("""
- Série temporal: uma sequência de valores ao longo do tempo (por exemplo, vendas diárias). 
- Horizonte: até quando queremos prever (ex.: próximos 30 dias). 
- Intervalo de confiança: uma faixa que indica incerteza; a linha da previsão é o melhor palpite, e a banda mostra onde é razoável esperar que o valor fique. 
- Sazonalidade e tendência: padrões repetitivos (semanal/mensal) e direção geral (subida/queda) que o modelo tenta capturar.
- Boas práticas: garantir uma coluna temporal ordenada; usar horizontes modestos primeiro; monitorar a qualidade com backtests quando possível.
        """)

    with st.expander("Análise de Sentimento (ai_analyze_sentiment)"):
        st.markdown("#### O que faz?")
        st.write("Classifica opiniões/textos como positivo, negativo, neutro, etc.")
        st.markdown("#### Como usar")
        st.write("""
1. Escreva um SQL com uma coluna de texto. 
2. Selecione a coluna e execute. 
3. Veja a distribuição por sentimento e a tabela com resultados.
        """)
        st.markdown("#### Fundamentos (intuitivo)")
        st.write("""
- Modelos de linguagem reconhecem padrões de palavras/frases associadas a emoções. 
- Textos curtos, ambíguos ou irônicos podem ser difíceis; é normal haver alguma incerteza.
        """)

    with st.expander("Classificação (ai_classify)"):
        st.markdown("#### O que faz?")
        st.write("Classifica textos em categorias definidas por você.")
        st.markdown("#### Como usar")
        st.write("""
1. Escreva um SQL com uma coluna de texto. 
2. Selecione a coluna e defina as categorias (labels). 
3. Execute para obter a classificação e gráficos de distribuição.
        """)
        st.markdown("#### Fundamentos (intuitivo)")
        st.write("""
- Quanto mais claras e distintas forem as categorias, melhor o resultado. 
- Evite categorias muito parecidas; se necessário, comece com poucas e refine.
        """)

    with st.expander("Extração de Tópicos (ai_extract)"):
        st.markdown("#### O que faz?")
        st.write("Extrai tópicos e frases-chave de textos.")
        st.markdown("#### Como usar")
        st.write("""
1. Escreva um SQL com uma coluna de texto. 
2. Selecione a coluna e os tipos (topic, key_phrase). 
3. Opcionalmente ajuste Top N e filtros (regex).
4. Execute e explore a tabela normalizada e os agregados.
        """)
        st.markdown("#### Fundamentos (intuitivo)")
        st.write("""
- Tópicos: temas gerais presentes no texto (ex.: "atendimento", "entrega"). 
- Frases‑chave: expressões relevantes mais específicas (ex.: "atraso na entrega"). 
- O objetivo é resumir conteúdo para facilitar descobertas.
        """)

    with st.expander("Detecção de Anomalias"):
        st.markdown("#### O que faz?")
        st.write("Encontra outliers numa série temporal, com base no resíduo de uma média móvel.")
        st.markdown("#### Como usar")
        st.write("""
1. Escreva um SQL com colunas de tempo e valor (opcionalmente uma de grupo). 
2. Em "Avançado (parâmetros)", defina o método (Z-score/MAD), a janela da MM e o limiar. 
3. Opcionalmente agrupe por categoria para detectar e visualizar por grupo. 
4. Execute para ver série, anomalias e agregados por grupo.
        """)
        st.markdown("#### Fundamentos (intuitivo)")
        st.write("""
- Média móvel (MM): média calculada em uma janela deslizante; ajuda a suavizar ruídos. 
- Resíduo: diferença entre o valor e a média móvel (o quanto "foge" do esperado). 
- Z‑score: resíduo dividido pelo desvio padrão na janela; marca anomalia quando excede um limite (ex.: 3). 
- MAD (Median Absolute Deviation): mede a variabilidade via mediana (mais robusto a outliers). Score MAD ≈ |resíduo − mediana| / MAD. 
- Agrupar por categoria: detecta anomalias separadamente por segmento (ex.: por loja/região), evitando que padrões diferentes se misturem.
        """)

    with st.expander("Clusterização"):
        st.markdown("#### O que faz?")
        st.write("Agrupa observações em k clusters com KMeans e exibe uma projeção 2D por PCA.")
        st.markdown("#### Como usar")
        st.write("""
1. Escreva um SQL que retorne colunas numéricas relevantes. 
2. Ajuste k, limite e se deseja padronizar (recomendado). 
3. Em "Avançado", selecione colunas específicas e/gere curva de cotovelo para sugerir k. 
4. Execute para ver o scatter (PCA), centroides e dados com labels.
        """)
        st.markdown("#### Fundamentos (intuitivo)")
        st.write("""
- KMeans: algoritmo que posiciona k "centroides" (pontos médios) e atribui cada observação ao centro mais próximo; busca minimizar a soma das distâncias ao centro (inércia). 
- Escolha de k: 
  - Elbow (cotovelo): observe a curva da inércia por k; o ponto onde a queda "diminui" costuma ser um bom k. 
  - Silhouette: mede quão bem separadas estão as classes (varia de -1 a 1; mais perto de 1 é melhor). 
- PCA (Análise de Componentes Principais): transforma as variáveis para eixos que capturam a maior variação; útil para visualizar em 2D sem perder muita informação. 
- Escalonamento (padronização): importante para que variáveis em escalas diferentes não dominem a distância.
        """)

    with st.expander("EDA (Análise Exploratória)"):
        st.markdown("#### O que faz?")
        st.write("Oferece visão geral, estatísticas, distribuições, correlação e insights automáticos.")
        st.markdown("#### Como usar")
        st.write("""
1. Escreva um SQL e execute a análise. 
2. Revise a visão geral, amostra, selecione colunas e navegue pelas abas. 
3. Ative "Gerar insights com IA" para um resumo textual com sugestões.
        """)
        st.markdown("#### Fundamentos (intuitivo)")
        st.write("""
- Tipos e nulos: entender que tipo de dado há e onde há ausências. 
- Estatísticas: média, desvio, mínimos/máximos ajudam a ter noção de escala e variação. 
- Histogramas: mostram a distribuição; ajuste o número de "bins" para ver mais/menos detalhe. 
- Frequências (categóricas): top categorias dão uma fotografia rápida do que é mais comum. 
- Correlação (Pearson): indica se duas variáveis variam juntas (positivo) ou em sentidos opostos (negativo); não implica causalidade.
        """)

    with st.expander("Genie Chat"):
        st.markdown("#### O que faz?")
        st.write("Permite conversar com seus dados via um espaço do Genie.")
        st.markdown("#### Como usar")
        st.write("Informe a URL do espaço do Genie (ou configure via variável GENIE_SPACE_URL) e interaja no iframe.")

    with st.expander("AutoML"):
        st.markdown("#### O que faz?")
        st.write("Facilita iniciar experimentos AutoML (classificação, regressão, forecast) quando disponível no ambiente Databricks.")
        st.markdown("#### Como usar")
        st.write("Forneça um SQL que gere o dataset, selecione o alvo (e tempo/frequência/horizonte para forecast) e inicie o AutoML. Em ambientes sem AutoML, uma mensagem informará indisponibilidade.")

    st.markdown("### Glossário rápido")
    st.write("""
- Média móvel (MM): média calculada sobre uma janela deslizante de pontos no tempo. 
- Resíduo: diferença entre o valor observado e o valor esperado (ex.: média móvel). 
- Z-score: valor padronizado que indica quantos desvios padrões o ponto está distante do esperado. 
- MAD (Median Absolute Deviation): medida robusta de variabilidade; menos sensível a outliers que o desvio padrão. 
- KMeans: algoritmo de agrupamento que particiona os dados em k grupos com centróides otimizados. 
- Inércia: soma das distâncias ao centro; usada para avaliar compacidade dos clusters. 
- Silhouette: métrica de separação dos clusters (de -1 a 1). 
- PCA: técnica de redução de dimensionalidade, projeta dados em eixos de maior variância. 
- Correlação (Pearson): grau de relação linear entre duas variáveis (entre -1 e 1). 
- Intervalo de confiança: faixa que expressa incerteza em uma estimativa ou previsão.
    """)

    st.markdown("### Boas práticas gerais")
    st.write("""
- Garanta permissões e Warehouse adequados (Pro/Serverless quando necessário). 
- Valide amostras antes de rodar operações pesadas. 
- Documente seus parâmetros (horizonte, filtros, k) junto dos resultados. 
- Salve/exporte resultados em CSV (ou crie tabelas Delta) para reprodutibilidade.
    """)




