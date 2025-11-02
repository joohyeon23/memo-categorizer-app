import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


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
from langchain_classic.tools import Tool
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# .envファイルの読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="🧠 Smart Memo Agent", layout="wide")

if not openai_api_key:
    st.error("`.env` に OPENAI_API_KEY が設定されていません。")
    st.stop()

# LLM初期化（APIキーは引数で渡す）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)

# DB準備
conn = sqlite3.connect("memo.db")
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS memos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT,
    url TEXT,
    category TEXT
)''')
conn.commit()

st.header("📝 メモ・URL 登録")

# 既存カテゴリの読み込み（DBか固定リスト）
default_categories = ["食べ物", "仕事", "AI勉強", "資格", "ダンス", "日記", "お金", "語学勉強", "その他"]

# サイドバーでカテゴリの自由編集
st.sidebar.subheader("カテゴリ管理")
new_category = st.sidebar.text_input("新しいカテゴリを追加")
if st.sidebar.button("カテゴリ追加") and new_category:
    default_categories.append(new_category)
    st.sidebar.success(f"カテゴリ '{new_category}' を追加しました！")

with st.form("memo_form"):
    memo_text = st.text_area("メモ内容")
    memo_url = st.text_input("URL（任意）")
    selected_category = st.selectbox("カテゴリを選択（自動分類も可）", ["自動分類"] + default_categories)
    submit_btn = st.form_submit_button("登録")

    if submit_btn and memo_text:
        if selected_category == "自動分類":
            # カテゴリ分類プロンプト
            category_prompt = f"""
以下のメモを主要カテゴリの1つに分類してください。
候補: {', '.join(default_categories)}
メモ: {memo_text}
出力はカテゴリ名のみ。
"""
            cat = llm.invoke(category_prompt).content.strip()
        else:
            cat = selected_category

        c.execute(
            "INSERT INTO memos (content, url, category) VALUES (?, ?, ?)",
            (memo_text, memo_url, cat)
        )
        conn.commit()
        st.success(f"登録完了！（カテゴリ: {cat}）")

st.subheader("登録済みメモ")
df = pd.read_sql_query("SELECT * FROM memos", conn)
st.dataframe(df)

st.subheader("🗑️ メモの削除")

# DBからメモを取得
df = pd.read_sql_query("SELECT * FROM memos", conn)

# multiselectで削除対象を選択
to_delete = st.multiselect(
    "削除するメモを選択",
    options=df["id"].tolist(),
    format_func=lambda x: df[df["id"] == x]["content"].values[0]
)

if st.button("削除"):
    if to_delete:
        c.executemany("DELETE FROM memos WHERE id=?", [(i,) for i in to_delete])
        conn.commit()
        st.success(f"{len(to_delete)}件のメモを削除しました！")
    else:
        st.warning("削除するメモを選択してください。")

st.subheader("🕸️ メモの関係性グラフ")

if not df.empty:
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row["content"][:40], title=f"Category: {row['category']}")  # 長過ぎる場合カット
        G.add_edge(row["category"], row["content"][:40])

    net = Network(height="500px", bgcolor="#FFFFFF", directed=False)
    net.from_nx(G)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        tmp_file_path = tmp_file.name

    with open(tmp_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=520)

st.subheader("🤖 Agentに質問")

if not df.empty:
    docs = [Document(page_content=row["content"], metadata={"category": row["category"]}) for _, row in df.iterrows()]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    tools = [
        Tool(
            name="KnowledgeBaseQA",
            func=qa_chain.run,
            description="メモDBに基づく質問応答を行うツール。カテゴリや内容の要約、関係性などを答える。"
        )
    ]

    react_prompt = PromptTemplate(
    template="""
あなたはユーザーのメモデータベースを管理するアシスタントです。
ユーザーの質問に答える際は、以下の手順で考えてください：

1. ユーザーの質問をカテゴリや内容に基づいて理解
2. 関連するメモをKnowledgeBaseQAツールで検索
3. サマリーや要点を整理し、分かりやすく出力
4. 必要に応じてアクション（Action）としてツールを呼び出す
5. 最終回答（Final Answer）としてユーザーに伝える

フォーマット:
Thought: 今考えていることや推論
Action: 使うツール名（必要な場合）
Action Input: ツールに渡す入力
Observation: ツールの出力結果
Final Answer: ユーザーへの最終回答

{agent_scratchpad}

ユーザーの質問: {input}
利用可能なツール: {tools}
ツール名一覧: {tool_names}
""",
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
)

    agent = create_react_agent(llm, tools, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    user_query = st.text_input("Agentに質問（例：「健康カテゴリの要約を見せて」）")

    if st.button("送信") and user_query:
        with st.spinner("考え中..."):
            response = agent_executor.invoke({"input": user_query})
            st.markdown("### 回答")
            st.success(response["output"])
else:
    st.info("まだメモが登録されていません。")

conn.close()
