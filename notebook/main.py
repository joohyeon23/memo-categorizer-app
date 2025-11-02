# app.py
import os
import sqlite3
import time
from typing import List

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
import faiss
import numpy as np

# LangChain imports (agent作成用)
from langchain.agents import Tool, create_react_agent, AgentExecutor

# Note: HuggingFacePipeline wrapper for LangChain LLM
from langchain.llms import HuggingFacePipeline

# Embeddings via sentence-transformers (we'll use HF wrapper)
from sentence_transformers import SentenceTransformer

# ---------------------------
# --- 設定・ファイルパス ---
# ---------------------------
DB_PATH = "memos.db"
FAISS_INDEX_PATH = "faiss.index"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 の次元
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"  # 小型のFLAN-T5。重いならさらに小さいものに変更可
SUMMARIZER_MODEL = LLM_MODEL_NAME  # 再利用
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"  # カテゴリ分類用

# ---------------------------
# --- DB ヘルパー ---
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            url TEXT,
            category TEXT,
            summary TEXT,
            created_at REAL
        )
        """
    )
    conn.commit()
    return conn

conn = init_db()

# ---------------------------
# --- Embedding & FAISS ---
# ---------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)

@st.cache_resource
def load_summarizer_pipeline():
    # HF seq2seq pipeline for summarization / text2text
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    # Use LangChain HuggingFacePipeline wrapper later
    summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return summarizer

@st.cache_resource
def load_llm_for_langchain():
    # Prepare a transformers pipeline and wrap with LangChain's HuggingFacePipeline
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    hf_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return HuggingFacePipeline(pipeline=hf_pipe)

def build_faiss_index(embeddings: np.ndarray):
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    return index

def save_faiss(index: faiss.IndexFlatL2):
    faiss.write_index(index, FAISS_INDEX_PATH)

def load_faiss():
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    return None

# ---------------------------
# --- DB 操作関数 ---
# ---------------------------
def insert_memo(text: str, url: str, category: str = None, summary: str = None):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO memos (text, url, category, summary, created_at) VALUES (?, ?, ?, ?, ?)",
        (text, url, category, summary, time.time()),
    )
    conn.commit()
    return cur.lastrowid

def update_memo(id_: int, **kwargs):
    cur = conn.cursor()
    set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
    values = list(kwargs.values())
    values.append(id_)
    cur.execute(f"UPDATE memos SET {set_clause} WHERE id = ?", values)
    conn.commit()

def delete_memo(id_: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM memos WHERE id = ?", (id_,))
    conn.commit()

def list_memos() -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM memos ORDER BY created_at DESC", conn)
    return df

# ---------------------------
# --- RAG ユーティリティ ---
# ---------------------------
@st.cache_resource
def get_emb_model():
    return load_embedding_model()

def build_embeddings_for_all():
    df = list_memos()
    if df.empty:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")
    texts = (df["text"].fillna("") + " " + df["url"].fillna("")).tolist()
    emb_model = get_emb_model()
    embs = emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # ensure float32
    return embs.astype("float32")

def rebuild_faiss_from_db():
    embs = build_embeddings_for_all()
    if embs.shape[0] == 0:
        return None
    index = build_faiss_index(embs)
    save_faiss(index)
    return index

def add_text_to_faiss(text: str, index: faiss.IndexFlatL2):
    emb_model = get_emb_model()
    emb = emb_model.encode([text], convert_to_numpy=True).astype("float32")
    index.add(emb)
    save_faiss(index)

def retrieve_similar(text: str, k: int = 3):
    index = load_faiss()
    if index is None:
        return []
    emb_model = get_emb_model()
    q = emb_model.encode([text], convert_to_numpy=True).astype("float32")
    D, I = index.search(q, k)
    ids = I[0].tolist()
    # map index positions to DB rows (assume ordering same as SELECT by created_at DESC)
    df = list_memos()
    if df.empty:
        return []
    # If FAISS index built from all rows in order, indices match df.index
    # To be safe: clip indices
    docs = []
    for idx in ids:
        if idx < len(df):
            row = df.iloc[idx]
            docs.append({"id": int(row["id"]), "text": row["text"], "url": row["url"], "category": row["category"], "summary": row["summary"]})
    return docs

# ---------------------------
# --- Agent 用ツール ---
# ---------------------------
def memo_search_tool(query: str) -> str:
    # retrieve top-k memos and return readable text
    docs = retrieve_similar(query, k=5)
    if len(docs) == 0:
        return "該当するメモが見つかりませんでした。"
    out = []
    for d in docs:
        s = f"[id:{d['id']}] category:{d.get('category')}\n{d['text']}\nURL: {d.get('url')}\n---"
        out.append(s)
    return "\n".join(out)

def memo_list_by_category_tool(category_q: str) -> str:
    df = list_memos()
    if df.empty:
        return "メモはまだ登録されていません。"
    filtered = df[df["category"] == category_q]
    if filtered.empty:
        return f"カテゴリ '{category_q}' のメモはありません。"
    texts = []
    for _, r in filtered.iterrows():
        texts.append(f"[id:{r['id']}] {r['text']}")
    return "\n".join(texts)

# ---------------------------
# --- Build Agent (create_react_agent + AgentExecutor) ---
# ---------------------------
@st.cache_resource
def build_agent_executor():
    llm = load_llm_for_langchain()
    # Tools: search by semantic similarity, list-by-category
    tools = [
        Tool(name="MemoSearch", func=memo_search_tool, description="メモDBから関連するメモを返す（全文）"),
        Tool(name="ListByCategory", func=memo_list_by_category_tool, description="カテゴリ名を受け取り、そのカテゴリのメモ一覧を返す"),
    ]
    # create_react_agent returns a callable agent; wrap with AgentExecutor
    agent = create_react_agent(llm=llm, tools=tools)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return executor

# ---------------------------
# --- Summarizer using HF pipeline (FLAN-T5) ---
# ---------------------------
@st.cache_resource
def get_summarizer():
    return load_summarizer_pipeline()

def summarize_texts(texts: List[str], max_length: int = 150):
    summarizer = get_summarizer()
    joined = "\n".join(texts)
    # make short prompt to ask summarization
    prompt = "次のメモを要約して短い日本語の要約を1つ作ってください:\n\n" + joined
    out = summarizer(prompt, max_length=max_length, do_sample=False, truncation=True)
    # pipeline returns list of dicts
    if isinstance(out, list) and len(out) > 0:
        return out[0]["generated_text"]
    return ""

# ---------------------------
# --- Graph Visualization ---
# ---------------------------
def build_relation_graph_html(df: pd.DataFrame):
    G = nx.Graph()
    # add category nodes and memo nodes
    for _, r in df.iterrows():
        node_id_memo = f"memo_{int(r['id'])}"
        G.add_node(node_id_memo, label=r["text"][:40], title=r["text"])
        cat = r["category"] if r["category"] else "未分類"
        node_id_cat = f"cat_{cat}"
        if not G.has_node(node_id_cat):
            G.add_node(node_id_cat, label=str(cat), title=str(cat), color="#FFD580")
        G.add_edge(node_id_cat, node_id_memo)
    net = Network(height="600px", width="100%", notebook=False)
    net.from_nx(G)
    # physics for better layout
    net.toggle_physics(True)
    return net.generate_html(notebook=False)

# ---------------------------
# --- Streamlit UI ---
# ---------------------------
st.set_page_config(page_title="RAG Memo Dashboard (local)", layout="wide")
st.title("🧭 RAG + Agent メモダッシュボード（ローカル）")

# Sidebar: 入力
st.sidebar.header("✍️ 新規メモ登録")
memo_text = st.sidebar.text_area("メモ本文（テキスト）", height=120)
memo_url = st.sidebar.text_input("関連 URL（任意）")
# Category selection optional
default_categories = ["仕事", "勉強", "健康", "感情", "生活", "旅行", "その他"]
selected_cat = st.sidebar.selectbox("カテゴリを選択（自動分類したい場合は空欄でOK）", [""] + default_categories)

if st.sidebar.button("登録"):
    text = memo_text.strip()
    if not text:
        st.sidebar.warning("メモ本文を入力してください。")
    else:
        # 自動カテゴリ分類 if not provided
        category = selected_cat if selected_cat else None
        if category is None:
            zero_shot = load_zero_shot()
            res = zero_shot(text, candidate_labels=default_categories)
            category = res["labels"][0]
        # initial summary generate
        summary = summarize_texts([text])[:500]
        new_id = insert_memo(text=text, url=memo_url, category=category, summary=summary)
        # rebuild/add to FAISS
        index = load_faiss()
        if index is None:
            # rebuild whole from DB
            index = rebuild_faiss_from_db()
        else:
            add_text_to_faiss(text, index)
        st.sidebar.success(f"メモを登録しました（id={new_id}、カテゴリ={category}）")

# Main area: show memos table
st.subheader("📚 登録済みメモ")
df = list_memos()
if df.empty:
    st.info("まだメモが登録されていません。サイドバーから追加してください。")
else:
    # display table with edit / delete options via streamlit elements
    # Simple: show dataframe and allow id-based deletion
    st.dataframe(df[["id", "text", "url", "category", "summary", "created_at"]])

    # delete by id
    st.write("----")
    col1, col2 = st.columns([1, 3])
    with col1:
        del_id = st.number_input("削除するメモIDを入力", min_value=0, step=1)
        if st.button("削除"):
            if del_id > 0:
                delete_memo(int(del_id))
                # rebuild faiss
                rebuild_faiss_from_db()
                st.experimental_rerun()
    with col2:
        if st.button("カテゴリ更新（自動再分類）"):
            # re-run zero-shot for all memos without category or to refresh
            zero_shot = load_zero_shot()
            for _, r in df.iterrows():
                text = r["text"]
                res = zero_shot(text, candidate_labels=default_categories)
                update_memo(int(r["id"]), category=res["labels"][0])
            st.success("全メモのカテゴリを更新しました。")
            st.experimental_rerun()

# Graph visualization
st.subheader("🔗 関係性グラフ（カテゴリ - メモ）")
html = build_relation_graph_html(df) if not df.empty else "<p>メモがありません</p>"
components.html(html, height=650, scrolling=True)

# Agent 質問エリア
st.subheader("🤖 Agent に自然言語で質問")
agent_executor = build_agent_executor()
query = st.text_input("例：'健康カテゴリの要約を見せて' / '旅行に関するメモを教えて' etc.")
if st.button("Agent 実行"):
    if not query.strip():
        st.warning("質問文を入力してください。")
    else:
        # run agent; AgentExecutor may have invoke() or run()
        try:
            result = agent_executor.invoke({"input": query})
            # result may be dict-like with "output"
            if isinstance(result, dict) and "output" in result:
                output_text = result["output"]
            else:
                output_text = str(result)
        except Exception:
            try:
                output_text = agent_executor.run(query)
            except Exception as e:
                output_text = f"Agent 実行に失敗しました: {e}"
        st.write("### ✅ Agent の応答")
        # If the agent used MemoSearch tool, it will return memos. Now run summarizer on returned memos if needed.
        st.write(output_text)
        # Post-process: if output contains many memos, ask summarizer
        if "category" in output_text or "id:" in output_text:
            # crude extraction of texts - for demo, summarize top-k similar to query
            docs = retrieve_similar(query, k=5)
            if docs:
                summary = summarize_texts([d["text"] for d in docs])
                st.write("### 🔎 その内容の要約（RAG → Summarize）")
                st.write(summary)

# Footer / tips
st.write("---")
st.caption("⚠️ 注意: このアプリはローカルで動く簡易デモです。大規模データや高頻度アクセスにはチューニングが必要です。")
