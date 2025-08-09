import streamlit as st
from streamlit_option_menu import option_menu
from functions import *

st.session_state.user_email = st.context.headers.get("X-Forwarded-Email")
user_access_token = st.context.headers.get('X-Forwarded-Access-Token')


# Configurações iniciais
st.set_page_config(
    page_title="Mini Cientista",
    page_icon="https://www.sicredi.com.br/static/home/favicon.ico",
    layout="wide", 
    menu_items={
        "Get help": "https://teams.microsoft.com/l/chat/0/0?users=nicolas_santos@sicredi.com.br,matos_renan@sicredi.com.br",
        "About": "Aplicativo desenvolvido pela Engenharia de Dados do Time de Associação e Contas."
    }
)

# Estilos globais e UX
global_css = '''
<style>
/***** Oculta fullscreen dos gráficos *****/
button[title="View fullscreen"]{visibility: hidden; display: none;}
.st-emotion-cache-1u2dcfn{visibility: hidden; display: none;}
.st-emotion-cache-gi0tri {visibility: hidden; display: none;}

/***** Sidebar *****/
section[data-testid="stSidebar"] > div {background: linear-gradient(180deg, #0b3d0b 0%, #145214 100%);}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p {color: #FFFFFF !important;}

/***** Option Menu tweaks *****/
ul.nav.nav-pills {gap: 4px;}
ul.nav.nav-pills li a {border-radius: 8px !important; padding: 10px 12px !important; color: #f8f9fa !important;}
ul.nav.nav-pills li a:hover {background-color: rgba(255,255,255,0.10) !important;}
ul.nav.nav-pills li a.active {background-color: #3FA110 !important; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.2);} 

/***** Main headings *****/
h1, h2, h3 { letter-spacing: 0.2px; }
</style>
'''
st.markdown(global_css, True)

# Defaults de sessão
if "etapa_atual" not in st.session_state:
    st.session_state.etapa_atual = 1


def on_change(key):
    st.session_state.etapa_atual = int(key)

# Função do Menu do APP  
with st.sidebar:
    st.image("assets/img/logo_gold.png")
    st.markdown("### Mini Cientista")
    st.caption("Seu cientista de dados de bolso no Databricks")
    st.divider()
    st.session_state.selected = option_menu (
        menu_title = "Navegação",
        options = [
            "Projeção",
            "Análise de Sentimento",
            "Classificação",
            "Detecção de Anomalias",
            "Extração de Tópicos",
            "Clusterização",
            "Análise de Dados (EDA)",
            "Genie Chat",
            "AutoML",
            "Sobre"
        ],
        icons=[
            "graph-up",
            "emoji-smile",
            "tags",
            "activity",
            "list-ul",
            "diagram-3",
            "bar-chart",
            "chat",
            "cpu",
            "info-circle"
        ],
        on_change = on_change, key = '1',
        styles={
            "container": {"background-color": "rgba(255, 255, 255, 0)"},
            "icon": {"color": "#f0f0f0", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"font-size": "16px", "background-color": "#3FA110"}}
        )

if st.session_state.selected == "Projeção":
    forecast_projection()
elif st.session_state.selected == "Análise de Sentimento":
    analyze_sentiment()
elif st.session_state.selected == "Classificação":
    classify()
elif st.session_state.selected == "Detecção de Anomalias":
    anomaly_detection_page()
elif st.session_state.selected == "Extração de Tópicos":
    topic_extraction_page()
elif st.session_state.selected == "Clusterização":
    clustering_page()
elif st.session_state.selected == "Análise de Dados (EDA)":
    eda_page()
elif st.session_state.selected == "Genie Chat":
    genie_chat()
elif st.session_state.selected == "AutoML":
    automl_page()
elif st.session_state.selected == "Sobre":
    about_page()
else:
    about_page()
