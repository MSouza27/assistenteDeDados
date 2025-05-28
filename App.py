import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from ferramentas import criar_ferramentas
import pdfplumber

# Inicia o app
st.set_page_config(page_title="Assistente de análise de dados com IA", layout="centered")
st.title("🐬 Dolphin Analytics - Assistente de análise de dados com IA")

st.info("""
Este assistente utiliza um agente com Langchain para explorar, analisar e visualizar dados interativamente. Faça upload de um arquivo **CSV, Excel ou PDF** e você poderá:

- 📈 **Gerar relatórios automáticos**
- 🔍 **Fazer perguntas sobre os dados**
- 🤖 **Criar gráficos com linguagem natural**
""")

# Upload do arquivo
st.markdown("### 📁 Faça upload do seu arquivo (CSV, Excel ou PDF)")
arquivo_carregado = st.file_uploader("Selecione um arquivo", type=["csv", "xls", "xlsx", "pdf"], label_visibility="collapsed")


# 🔧 Função atualizada para ler diferentes tipos de arquivos
def carregar_arquivo(arquivo):
    nome = arquivo.name.lower()
    
    try:
        if nome.endswith(".csv"):
            try:
                return pd.read_csv(arquivo)
            except UnicodeDecodeError:
                return pd.read_csv(arquivo, encoding="latin1")
        
        elif nome.endswith((".xls", ".xlsx")):
            return pd.read_excel(arquivo)
        
        elif nome.endswith(".pdf"):
            with pdfplumber.open(arquivo) as pdf:
                texto = ""
                for pagina in pdf.pages:
                    texto += pagina.extract_text() + "\n"
                return pd.DataFrame({"conteudo_pdf": [texto]})
        
        else:
            raise ValueError("Formato de arquivo não suportado.")
    
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {str(e)}")
        return None


if arquivo_carregado:
    df = carregar_arquivo(arquivo_carregado)

    if df is None or df.empty:
        st.error("Não foi possível carregar os dados. Verifique o conteúdo do arquivo.")
    else:
        st.success("Arquivo carregado com sucesso!")
        st.markdown("### Primeiras linhas do DataFrame")
        st.dataframe(df.head())

        colunas = df.columns.tolist()
        exemplo_coluna1 = colunas[0]
        exemplo_coluna2 = colunas[1] if len(colunas) > 1 else colunas[0]

        exemplo_pergunta = f"Qual é a média de {exemplo_coluna1}?"
        exemplo_grafico = f"Crie um gráfico da média de {exemplo_coluna1} por {exemplo_coluna2}."

        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192",
            temperature=0
        )

        tools = criar_ferramentas(df)
        df_head = df.head().to_markdown()

        prompt_react_pt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names", "df_head"],
            template="""
            Você é um assistente que sempre responde em português.

            Você tem acesso a um dataframe pandas chamado `df`.
            Aqui estão as primeiras linhas do DataFrame:
            {df_head}

            Responda às seguintes perguntas da melhor forma possível.

            Para isso, você tem acesso às seguintes ferramentas:
            {tools}

            Use o seguinte formato:

            Question: a pergunta de entrada
            Thought: seu pensamento
            Action: a ação a ser tomada (uma das [{tool_names}])
            Action Input: a entrada da ação
            Observation: o resultado da ação
            ...
            Thought: Agora eu sei a resposta final
            Final Answer: a resposta final

            Question: {input}
            Thought: {agent_scratchpad}
            """
        )

        agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react_pt.partial(
            df_head=df_head,
            tools="\n".join([tool.description for tool in tools]),
            tool_names=", ".join([tool.name for tool in tools])
        ))

        orquestrador = AgentExecutor(agent=agente, tools=tools, verbose=True, handle_parsing_errors=True)

        st.markdown("---")
        st.markdown("### ⚡ Ações rápidas")

        if st.button("📄 Relatório de informações gerais"):
            with st.spinner("Gerando relatório 🧞"):
                resposta = orquestrador.invoke({"input": "Quero um relatório com informações sobre os dados"})
            st.session_state['relatorio_geral'] = resposta["output"]

        if 'relatorio_geral' in st.session_state:
            with st.expander("Resultado: Relatório de informações gerais"):
                st.markdown(st.session_state['relatorio_geral'])
            st.download_button("📅 Baixar relatório", st.session_state['relatorio_geral'], "relatorio_informacoes_gerais.md", "text/markdown")

        if st.button("📄 Relatório de estatísticas descritivas"):
            with st.spinner("Gerando relatório 🧞"):
                resposta = orquestrador.invoke({"input": "Quero um relatório de estatísticas descritivas"})
            st.session_state['relatorio_estatisticas'] = resposta["output"]

        if 'relatorio_estatisticas' in st.session_state:
            with st.expander("Resultado: Relatório de estatísticas descritivas"):
                st.markdown(st.session_state['relatorio_estatisticas'])
            st.download_button("📅 Baixar relatório", st.session_state['relatorio_estatisticas'], "relatorio_estatisticas_descritivas.md", "text/markdown")

        st.markdown("---")
        st.markdown("### 🔎 Perguntas sobre os dados")
        pergunta = st.text_input(f"Faça uma pergunta (ex: '{exemplo_pergunta}')")
        if st.button("Responder"):
            with st.spinner("Analisando os dados 🧞"):
                resposta = orquestrador.invoke({"input": pergunta})
            st.markdown(resposta["output"])

        st.markdown("---")
        st.markdown("### 📈 Criar gráfico com base em uma pergunta")
        pergunta_grafico = st.text_input(f"Digite o que deseja visualizar (ex: '{exemplo_grafico}')")
        if st.button("Gerar gráfico"):
            with st.spinner("Gerando o gráfico 🧞"):
                orquestrador.invoke({"input": pergunta_grafico})

else:
    st.warning("Por favor, carregue um arquivo CSV, Excel ou PDF para continuar.")
