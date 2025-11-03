import streamlit as st
import sqlite3
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from dotenv import load_dotenv
import random

# LangChain関連
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
#from langchain_core.tools import tool



import subprocess
import sys
import urllib.request

import tempfile
#import random
import matplotlib.colors as mcolors
# URLメタデータ取得のために追加
import requests
from bs4 import BeautifulSoup
# matplotlib.colorsは使われていないため削除しました

# ----------------------------------------------------
# 0. 初期設定とDB準備
# ----------------------------------------------------
# requirements.txt の依存関係を自動でpip installする関数
#def install_requirements():
#    requirements_path = "requirements.txt"
#    if os.path.exists(requirements_path):
#        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])

# Streamlit デバッグなどで再実行されても無駄な再installを避ける
#if "requirements_installed" not in st.session_state:
#    install_requirements()
#    st.session_state["requirements_installed"] = True

# カテゴリの色マッピング (識別できれば何でも良い)
CATEGORY_COLORS = {
    "食べ物": "#FF7F50",   # Coral
    "qiita記事関連": "#4682B4",     # Steel Blue
    "AI勉強": "#3CB371",   # Medium Sea Green
    "資格": "#FFD700",     # Gold
    "ダンス": "#FF69B4",   # Hot Pink
    "日記メモ": "#A9A9A9",     # Dark Gray
    "お金": "#DAA520",     # Goldenrod
    "語学勉強": "#9370DB", # Medium Purple
    "仕事関連": "#696969",   # Dim Gray
    "自動分類": "#000000"  # Black
}

# .envファイルの読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="🧠 Smart Memo Agent", layout="wide")

if not openai_api_key:
    st.error("`.env` に OPENAI_API_KEY が設定されていません。")
    st.stop()

# LLM初期化（APIキーは引数で渡す）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=openai_api_key)

# 【新しいDB操作関数】データベースの接続・初期化・マイグレーション
def get_db_connection():
    """スレッド問題を回避するため、都度接続オブジェクトを生成"""
    return sqlite3.connect("memo.db", check_same_thread=False)

def initialize_db():
    """データベースの初期テーブルとFTSテーブルを構築する"""
    conn = get_db_connection()
    c = conn.cursor()

    # ------------------------------------
    # 1. memosテーブルの作成とtitleカラムのマイグレーション
    # ------------------------------------
    
    # memosテーブルの定義（titleを含む最新版）
    c.execute('''
    CREATE TABLE IF NOT EXISTS memos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT,
        url TEXT,
        category TEXT,
        title TEXT 
    )''')
    conn.commit()

    # titleカラムの存在チェック（マイグレーションロジック）
    try:
        c.execute("PRAGMA table_info(memos)")
        columns = [col[1] for col in c.fetchall()]
        
        # titleカラムが存在しない場合、追加する
        if 'title' not in columns:
            c.execute("ALTER TABLE memos ADD COLUMN title TEXT")
            conn.commit()
            
            # 既存メモのタイトル初期設定（contentの最初の行をタイトルにする）
            existing_memos_to_migrate = pd.read_sql_query("SELECT id, content FROM memos WHERE title IS NULL", conn)

            if not existing_memos_to_migrate.empty:
                st.info("古いメモデータにタイトルを設定中...")
                for _, row in existing_memos_to_migrate.iterrows():
                    memo_id = row['id']
                    content = row['content']
                    
                    first_line = content.split('\n')[0]
                    initial_title = first_line[:25] + ("..." if len(first_line) > 25 else "")
                    
                    c.execute("UPDATE memos SET title = ? WHERE id = ?", (initial_title, memo_id))
                conn.commit()
                st.info("タイトル設定完了。")

    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise e
    
    # ------------------------------------
    # 2. FTSテーブルの作成と既存データのインポート
    # ------------------------------------
    
    # FTS (Full Text Search) VIRTUAL TABLEの定義
    c.execute('''
    CREATE VIRTUAL TABLE IF NOT EXISTS memos_fts USING fts5(
        id, 
        title, 
        content, 
        tokenize='unicode61' 
    )''')
    conn.commit()
    
    # FTSテーブルへのデータ挿入/更新（FTSテーブルが空の場合に既存データを全てインポート）
    c.execute("SELECT count(*) FROM memos_fts")
    fts_count = c.fetchone()[0]
    
    if fts_count == 0:
        st.info("FTSテーブルに既存メモをインポート中...")
        existing_memos = pd.read_sql_query("SELECT id, content, title FROM memos", conn)
        
        if not existing_memos.empty:
            for _, row in existing_memos.iterrows():
                # titleカラムが必ず存在するように修正済み
                c.execute(
                    "INSERT INTO memos_fts (id, title, content) VALUES (?, ?, ?)",
                    (row['id'], row['title'], row['content'])
                )
            conn.commit()
            st.info("FTSインポート完了。")

    # ------------------------------------
    # 3. memo_relationsテーブルの作成
    # ------------------------------------

    # memo_relationsテーブルの定義
    c.execute('''
    CREATE TABLE IF NOT EXISTS memo_relations (
        memo_id_a INTEGER,
        memo_id_b INTEGER,
        PRIMARY KEY (memo_id_a, memo_id_b),
        FOREIGN KEY (memo_id_a) REFERENCES memos(id) ON DELETE CASCADE,
        FOREIGN KEY (memo_id_b) REFERENCES memos(id) ON DELETE CASCADE
    )''')
    conn.commit()
    conn.close() # 初期化後、すぐに接続を閉じる


# アプリの開始時にDB初期化を一度だけ行う
if 'db_initialized' not in st.session_state:
    try:
        initialize_db()
        st.session_state.db_initialized = True
    except Exception as e:
        st.error(f"データベースの初期化中に致命的なエラーが発生しました: {e}")
        st.stop()


# URLからタイトルを取得する関数
def get_title_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('title')
        if title and title.string:
            return title.string.strip()
        
        og_title = soup.find('meta', property='og:title')
        if og_title and 'content' in og_title.attrs:
            return og_title.attrs['content'].strip()
            
    except Exception:
        return None
    return None

# 既存カテゴリの読み込みを修正
def get_current_categories():
    with get_db_connection() as conn:
        try:
            db_categories = pd.read_sql_query("SELECT DISTINCT category FROM memos", conn)["category"].tolist()
            return db_categories
        except Exception:
            return []

default_categories = list(CATEGORY_COLORS.keys())
for cat in get_current_categories():
    if cat not in default_categories:
        default_categories.append(cat)

# サイドバーでカテゴリの自由編集
st.sidebar.subheader("カテゴリ管理")
new_category = st.sidebar.text_input("新しいカテゴリを追加")
if st.sidebar.button("カテゴリ追加") and new_category and new_category not in default_categories:
    random_color = f'#{random.randint(0, 0xFFFFFF):06x}'
    CATEGORY_COLORS[new_category] = random_color
    
    default_categories.append(new_category)
    st.sidebar.success(f"カテゴリ '{new_category}' を追加しました！")
    st.rerun()

# ----------------------------------------------------
# 1. メモ・URL 登録 (タイトルを明示的に入力)
# ----------------------------------------------------
st.header("📝 メモ・URL 登録")

with st.form("memo_form"):
    # 【修正】タイトルを明示的に入力させる
    memo_title_input = st.text_input("タイトル")
    memo_text = st.text_area("メモ内容")
    memo_url = st.text_input("URL（任意）")
    category_choices = ["自動分類"] + [cat for cat in default_categories if cat != "自動分類"]
    selected_category = st.selectbox("カテゴリを選択（自動分類も可）", category_choices)
    submit_btn = st.form_submit_button("登録")

    if submit_btn and memo_text:
        # ユーザーが入力したタイトルを優先
        memo_title = memo_title_input.strip()
        
        # タイトルが空の場合のフォールバック処理
        if not memo_title:
            if memo_url:
                st.info("URLからタイトルを取得中...")
                memo_title = get_title_from_url(memo_url)
            
            if not memo_title:
                first_line = memo_text.split('\n')[0]
                memo_title = first_line[:25] + ("..." if len(first_line) > 25 else "")
        
        if not memo_title:
             memo_title = "タイトルなし" # 最終フォールバック

        # カテゴリ分類ロジック
        if selected_category == "自動分類":
            category_prompt = f"""
以下のメモを主要カテゴリの1つに分類してください。
候補: {', '.join(default_categories)}
メモ: {memo_title} - {memo_text}
出力はカテゴリ名のみ。候補以外のカテゴリ名は使用しないでください。
"""
            cat = llm.invoke(category_prompt).content.strip()
            if cat not in default_categories:
                cat = "その他"
        else:
            cat = selected_category
        
        # 【修正】DBアクセスをwithブロックでラップ
        with get_db_connection() as conn:
            c = conn.cursor()
            # データベースに挿入 (memosテーブル)
            c.execute(
                "INSERT INTO memos (content, url, category, title) VALUES (?, ?, ?, ?)",
                (memo_text, memo_url, cat, memo_title)
            )
            new_memo_id = c.lastrowid

            # FTSテーブルにも挿入
            c.execute(
                "INSERT INTO memos_fts (id, title, content) VALUES (?, ?, ?)",
                (new_memo_id, memo_title, memo_text)
            )
            
            conn.commit()
            
        st.success(f"登録完了！（タイトル: {memo_title}, カテゴリ: {cat}）")
        st.rerun()

st.subheader("登録済みメモ")
# 【修正】DBアクセスをwithブロックでラップ
with get_db_connection() as conn:
    df = pd.read_sql_query("SELECT * FROM memos", conn)
st.dataframe(df)

# ----------------------------------------------------
# 2. メモデータの編集機能
# ----------------------------------------------------

st.subheader("📝 メモの編集")

# 【修正】DBアクセスをwithブロックでラップ
with get_db_connection() as conn:
    df_edit = pd.read_sql_query("SELECT * FROM memos", conn)

if not df_edit.empty:
    memo_options = {row["id"]: f"ID {row['id']}: {row['title']}" for _, row in df_edit.iterrows()}
    
    selected_memo_id = st.selectbox(
        "編集するメモを選択",
        options=list(memo_options.keys()),
        format_func=lambda x: memo_options[x]
    )

    if selected_memo_id:
        current_memo = df_edit[df_edit["id"] == selected_memo_id].iloc[0]
        
        with st.form("edit_memo_form"):
            edited_title = st.text_input("タイトル", value=current_memo["title"])
            edited_text = st.text_area("メモ内容", value=current_memo["content"])
            edited_url = st.text_input("URL（任意）", value=current_memo["url"] or "")
            
            all_categories = list(set(default_categories + [current_memo["category"]]))
            
            edited_category = st.selectbox(
                "カテゴリを選択", 
                options=all_categories, 
                index=all_categories.index(current_memo["category"])
            )
            
            save_edit_btn = st.form_submit_button("メモを更新")

            if save_edit_btn:
                # 【修正】DBアクセスをwithブロックでラップ (編集が機能しない問題への対応)
                with get_db_connection() as conn:
                    c = conn.cursor()
                    
                    # 1. memosテーブルを更新
                    c.execute(
                        "UPDATE memos SET content=?, url=?, category=?, title=? WHERE id=?",
                        (edited_text, edited_url, edited_category, edited_title, selected_memo_id)
                    )
                    
                    # 2. FTSテーブルを更新 (replace構文を使用)
                    c.execute(
                        "INSERT INTO memos_fts (memos_fts, id, title, content) VALUES ('replace', ?, ?, ?)",
                        (selected_memo_id, edited_title, edited_text)
                    )

                    conn.commit()
                    
                st.success(f"メモID {selected_memo_id} を更新しました！")
                st.rerun()
else:
    st.info("編集可能なメモがありません。")

# ----------------------------------------------------
# 3. メモの削除
# ----------------------------------------------------
st.subheader("🗑️ メモの削除")

# 【修正】DBアクセスをwithブロックでラップ
with get_db_connection() as conn:
    df_delete = pd.read_sql_query("SELECT * FROM memos", conn)

if not df_delete.empty:
    to_delete = st.multiselect(
        "削除するメモを選択",
        options=df_delete["id"].tolist(),
        format_func=lambda x: df_delete[df_delete["id"] == x]["title"].values[0]
    )

    if st.button("削除"):
        if to_delete:
            # 【修正】DBアクセスをwithブロックでラップ
            with get_db_connection() as conn:
                c = conn.cursor()
                c.executemany("DELETE FROM memos WHERE id=?", [(i,) for i in to_delete])
                c.executemany("DELETE FROM memos_fts WHERE id=?", [(i,) for i in to_delete])
                conn.commit()
            
            st.success(f"{len(to_delete)}件のメモを削除しました！")
            st.rerun()
        else:
            st.warning("削除するメモを選択してください。")

# ----------------------------------------------------
# 4. キーワード全文検索
# ----------------------------------------------------
st.header("🔎 キーワード全文検索")

search_query = st.text_input("検索キーワードを入力 (例: ダンスの練習, Pythonコード)")

if search_query:
    # 【修正】DBアクセスをwithブロックでラップ
    with get_db_connection() as conn:
        fts_results = pd.read_sql_query(f"""
            SELECT 
                t1.id, 
                t1.title, 
                t1.content, 
                t1.category,
                t1.url
            FROM memos_fts AS t2
            INNER JOIN memos AS t1 ON t2.id = t1.id
            WHERE t2.memos_fts MATCH ?
            ORDER BY t2.rank
        """, conn, params=(search_query,))

    if not fts_results.empty:
        st.subheader("検索結果")
        
        for _, row in fts_results.iterrows():
            with st.expander(f"**{row['title']}** (ID: {row['id']}, カテゴリ: {row['category']})"):
                st.write(row['content'])
                if row['url']:
                    st.markdown(f"**URL:** [{row['url']}]({row['url']})") 
    else:
        st.info("該当するメモは見つかりませんでした。")

# ----------------------------------------------------
# 5. メモの関連付け機能
# ----------------------------------------------------

st.subheader("🔗 メモの関連付け")
# 【修正】DBアクセスをwithブロックでラップ
with get_db_connection() as conn:
    df_relate = pd.read_sql_query("SELECT * FROM memos", conn)

if len(df_relate) > 1:
    with st.form("relate_memo_form"):
        relate_options = df_relate["id"].tolist()
        format_func = lambda x: f"ID {x}: {df_relate[df_relate['id'] == x]['title'].values[0]}"
        
        memo_a_id = st.selectbox(
            "関連付け元のメモを選択",
            options=relate_options,
            format_func=format_func,
            key='memo_a'
        )
        
        available_memos = [i for i in relate_options if i != memo_a_id]
        
        # 関連付け先のメモの選択肢が空にならないようにチェック
        if available_memos:
            memo_b_id = st.selectbox(
                "関連付け先のメモを選択",
                options=available_memos,
                format_func=format_func,
                key='memo_b'
            )
        else:
            memo_b_id = None # 選択肢がない場合はNoneを設定

        relate_btn = st.form_submit_button("関連付けを登録")
        
        if relate_btn and memo_a_id and memo_b_id:
            id1, id2 = sorted([memo_a_id, memo_b_id])
            
            # 【修正】DBアクセスをwithブロックでラップ
            with get_db_connection() as conn:
                c = conn.cursor()
                try:
                    c.execute(
                        "INSERT INTO memo_relations (memo_id_a, memo_id_b) VALUES (?, ?)",
                        (id1, id2)
                    )
                    conn.commit()
                    st.success(f"メモID {id1} と メモID {id2} の関連付けを登録しました！")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.warning("この関連付けはすでに登録されています。")
                except Exception as e:
                    st.error(f"関連付けエラー: {e}")
else:
    st.info("2件以上のメモがないため、関連付け機能は使えません。")

# ----------------------------------------------------
# 6. メモの関係性グラフ (NameError修正)
# ----------------------------------------------------

st.subheader("🕸️ メモの関係性グラフ")

# 【修正】DBアクセスをwithブロックでラップ
with get_db_connection() as conn:
    df_graph = pd.read_sql_query("SELECT * FROM memos", conn)
    df_relations = pd.read_sql_query("SELECT * FROM memo_relations", conn)

if not df_graph.empty:
    G = nx.Graph()
    
    for cat_name, color in CATEGORY_COLORS.items():
        if cat_name != "自動分類":
            G.add_node(cat_name, title=f"Category: {cat_name}", group=cat_name, color=color, size=30)
        
    memo_id_to_title = {}

    for _, row in df_graph.iterrows():
        memo_id = row["id"]
        memo_title = row["title"] 
        full_content = row["content"]
        category = row["category"]
        
        memo_id_to_title[memo_id] = memo_title
        
        memo_color = CATEGORY_COLORS.get(category, "#696969") 
        
        G.add_node(
            memo_title, 
            title=f"Category: {category}\n\nContent:\n{full_content}", 
            group=category, 
            color=memo_color, 
            size=15, 
            memo_id=memo_id
        )
        
        G.add_edge(category, memo_title, color="#DDDDDD", weight=0.5) 
        
    for _, row in df_relations.iterrows():
        memo_a_id = row["memo_id_a"]
        # 【修正】: memo_id_b が正しく定義されるように修正
        memo_b_id = row["memo_id_b"] 
        
        if memo_a_id in memo_id_to_title and memo_b_id in memo_id_to_title:
            memo_a_label = memo_id_to_title[memo_a_id] 
            memo_b_label = memo_id_to_title[memo_b_id] # ここで memo_id_b を利用
            
            G.add_edge(memo_a_label, memo_b_label, color="#FF0000", weight=1.5, title="関連メモ")

    net = Network(height="500px", bgcolor="#FFFFFF", directed=False)
    net.from_nx(G)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        tmp_file_path = tmp_file.name

    with open(tmp_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=520)

else:
    st.info("まだメモが登録されていません。")

# ----------------------------------------------------
# 7. カテゴリ別サマリー自動生成
# ----------------------------------------------------
st.header("✨ カテゴリ別サマリー生成")

summary_categories = [cat for cat in default_categories if cat != "自動分類"]
selected_summary_cat = st.selectbox(
    "要約するカテゴリを選択", 
    options=summary_categories
)

if st.button("このカテゴリのメモを要約"):
    if not df_graph.empty:
        category_memos = df_graph[df_graph['category'] == selected_summary_cat]
        
        if not category_memos.empty:
            memo_contents = "\n---\n".join([
                f"ID {row['id']} - {row['title']}:\n{row['content']}" 
                for _, row in category_memos.iterrows()
            ])
            
            summary_prompt = f"""
以下のメモの内容を統合し、主要なテーマ、要点、およびそこから導かれる洞察を日本語で分かりやすく要約してください。
メモはカテゴリ「{selected_summary_cat}」に属します。

---
{memo_contents}
---

要約:
"""
            with st.spinner(f"「{selected_summary_cat}」カテゴリのメモをLLMが要約中..."):
                try:
                    summary = llm.invoke(summary_prompt).content
                    st.markdown("### 📝 要約結果")
                    st.success(summary)
                except Exception as e:
                    st.error(f"LLMによる要約中にエラーが発生しました: {e}")
        else:
            st.info(f"「{selected_summary_cat}」カテゴリにはメモがありません。")
    else:
        st.info("要約するメモがまだ登録されていません。")


# ----------------------------------------------------
# 8. Agentに質問
# ----------------------------------------------------

st.subheader("🤖 Agentに質問")

if not df_graph.empty:
    docs = [Document(page_content=row["content"], metadata={"category": row["category"]}) for _, row in df_graph.iterrows()]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # RAGのためにFAISSを作成
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
    st.info("メモを登録するとAgentに質問できます。")
