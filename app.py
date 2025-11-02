import streamlit as st
import sqlite3
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from dotenv import load_dotenv

# LangChain関連
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# ===============================
# .envファイルの読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


st.set_page_config(page_title="🧠 Smart Memo Agent", layout="wide")

if not openai_api_key:
    st.error("`.env` に OPENAI_API_KEY が設定されていません。")
    st.stop()

# LLM初期化（APIキーは引数で渡す）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)

# ===============================
# 初期設定
# ===============================
st.set_page_config(page_title="🧠 Smart Memo Agent", layout="wide")

# .env読み込み
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("`.env` に OPENAI_API_KEY が設定されていません。")
    st.stop()

# LLM初期化
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)

# ===============================
# SQLite DB準備
# ===============================
conn = sqlite3.connect("memo.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS memos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                url TEXT,
                category TEXT
            )''')
conn.commit()

# ===============================
# メモ登録フォーム
# ===============================
st.header("📝 メモ・URL 登録")

with st.form("memo_form"):
    memo_text = st.text_area("メモ内容")
    memo_url = st.text_input("URL（任意）")
    submit_btn = st.form_submit_button("登録")

    if submit_btn and memo_text:
        # 自動カテゴリ分類（LLM）
        category_prompt = f"""
        以下のメモを主要カテゴリの1つに分類してください。
        候補: 健康, 仕事, 学習, 人間関係, 投資, 趣味, その他
        メモ: {memo_text}
        出力はカテゴリ名のみ。
        """
        cat = llm.invoke(category_prompt).content.strip()
        c.execute("INSERT INTO memos (content, url, category) VALUES (?, ?, ?)", (memo_text, memo_url, cat))
        conn.commit()
        st.success(f"✅ 登録完了！（カテゴリ: {cat}）")

# ===============================
# DB表示
# ===============================
st.subheader("📂 登録済みメモ")
df = pd.read_sql_query("SELECT * FROM memos", conn)
st.dataframe(df)

# ===============================
# 関係性グラフ可視化
# ===============================
st.subheader("🕸️ メモの関係性グラフ")

if not df.empty:
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row["content"], title=row["category"])
        G.add_edge(row["category"], row["content"])

    net = Network(height="500px", bgcolor="#FFFFFF", directed=False)
    net.from_nx(G)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        net.save_graph(tmp_file.name)
        st.components.v1.html(open(tmp_file.name, 'r', encoding='utf-8').read(), height=520)

# ===============================
# RAG（Retrieval QA）
# ===============================
st.subheader("🤖 Agentに質問")

if not df.empty:
    # ベクトルストア作成
    docs = [Document(page_content=row["content"], metadata={"category": row["category"]}) for _, row in df.iterrows()]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()

    # QA Chain構築
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    # Tool定義
    tools = [
        tool(
            name="KnowledgeBaseQA",
            func=qa_chain.run,
            description="メモDBに基づく質問応答を行うツール。カテゴリや内容の要約、関係性などを答える。"
        )
    ]

    # Prompt定義（ReActスタイル）
    react_prompt = PromptTemplate.from_template("""
    あなたはユーザーのメモデータベースを管理するアシスタントです。
    ユーザーの質問に対して、カテゴリや内容を理解し、必要に応じてKnowledgeBaseQAツールを使って答えてください。
    
    フォーマット:
    Thought: ...
    Action: ...
    Action Input: ...
    Observation: ...
    Final Answer: ...
    
    Human: {input}
    """)

    # Agent作成
    agent = create_react_agent(llm, tools, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 入力欄
    user_query = st.text_input("💬 Agentに質問（例：「健康カテゴリの要約を見せて」）")

    if st.button("送信") and user_query:
        with st.spinner("🤔 考え中..."):
            response = agent_executor.invoke({"input": user_query})
            st.markdown("### 🧩 回答")
            st.success(response["output"])
else:
    st.info("まだメモが登録されていません。")

conn.close()
