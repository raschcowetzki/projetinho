# App Camada Gold

## Visão Geral
Este projeto é uma aplicação construída com Streamlit, projetada para gerenciar a criação e alteração de tabelas na **Camada Gold** de um Data Warehouse. Ele inclui funcionalidades que permitem validar e processar queries SQL, garantindo conformidade com padrões de nomenclatura e estruturação de dados.

A aplicação oferece:
- Um menu interativo para criar ou alterar tabelas Gold.
- Validação automática de consultas SQL.
- Validação das regras de tipos de dados e nomenclaturas da Camada Gold.
- Configuração de metadados e esquema de tabelas.

## Estrutura do Projeto

```
/app_gold/
├── app.py                     # Arquivo principal da aplicação
├── functions.py               # Funções auxiliares
├── requirements.txt           # Lista de bibliotecas dependentes
├── app.yaml                   # Configuração para implantação
├── .streamlit/
│   ╰── config.toml            # Configuração do Streamlit
│
╰── assets/                    # Arquivos complementares
    ├── default_values.yaml    # Configuração de valores padrão
    ╰── img/
        ╰── logo_gold.png      # Logo do projeto
```

## Stack utilizada

<div align="center">
	<img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/gitlab.png" alt="GitLab" title="GitLab"/>
	<img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/python.png" alt="Python" title="Python"/>
	<img width="50" src="https://raw.githubusercontent.com/marwin1991/profile-technology-icons/refs/heads/main/icons/databricks.png" alt="Databricks" title="Databricks"/>
</div>

## Instalação e Execução

### 1. Clonar o repositório
```sh
git clone <URL_DO_REPOSITORIO>
cd app_gold
```

### 2. Criar ambiente virtual e instalar dependências
```sh
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Executar a aplicação
```sh
streamlit run app.py
```

## Configuração
- O arquivo `.streamlit/config.toml` pode ser ajustado para modificar parâmetros de execução e customização.
- O `assets/default_values.yaml` contém valores padrão que podem ser personalizados.

## Funcionalidades
1. **Criar Gold**: Permite a criação de uma nova tabela na camada Gold.
   - Seleção de bancos de dados acessíveis ao usuário.
   - Validação de nomes conforme padrões pré-definidos.
   - Verificação de nome de tabela já existente.
   - Interface para escrita e verificação de queries SQL.
   - Validação de prefixos conforme tipos de dados retornados.
   - Identificação e verificação de origens.
   - Análise automática do esquema da tabela, incluindo chave primária e metadados.
   - Validação de Tarefa do Jira relacionada.

2. **Alterar Gold**: Permite a edição de uma tabela já existente na camada Gold.
   - Seleção de bancos de dados acessíveis ao usuário.
   - Validação de nomes conforme padrões pré-definidos.
   - Validação de tabela existente para edição.
   - Interface para escrita e verificação de queries SQL.
   - Validação de prefixos conforme tipos de dados retornados.
   - Identificação e verificação de origens.
   - Análise automática do esquema da tabela, incluindo chave primária e metadados.
   - Validação de Tarefa do Jira relacionada.


