import streamlit as st
import os
import shutil
from ollama import Client
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. 基础配置 ---
os.environ['NO_PROXY'] = '127.0.0.1,localhost'
DB_PATH = "database"
DATA_PATH = "data/raw"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-r1"

st.set_page_config(page_title="DeepSeek RAG Pro", page_icon="🚀", layout="wide")
os.makedirs(DATA_PATH, exist_ok=True)

# --- 2. 核心函数 ---
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)

@st.cache_resource
def get_vector_db():
    if os.path.exists(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=get_embeddings())
    return None

def process_new_file(uploaded_file):
    save_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == '.pdf': loader = PyPDFLoader(save_path)
    elif ext == '.txt': loader = TextLoader(save_path, encoding='utf-8')
    elif ext == '.md': loader = UnstructuredMarkdownLoader(save_path)
    else: return False, "不支持的格式"

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # 存入数据库
    Chroma.from_documents(documents=chunks, embedding=get_embeddings(), persist_directory=DB_PATH)
    return True, "成功"

def delete_file_from_db(filename):
    """从本地文件夹和向量数据库中同时删除记录"""
    # 1. 从物理路径删除
    file_path = os.path.join(DATA_PATH, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # 2. 从 ChromaDB 中删除 (根据 metadata 中的 source 字段)
    db = get_vector_db()
    if db:
        # 注意：LangChain 的 Chroma 实现通常需要通过 get() 获取 ID 后删除，
        # 或者使用 collection.delete 来根据 metadata 筛选。
        # 这里使用最通用的方法：根据 source 路径匹配删除
        db._collection.delete(where={"source": file_path})
        return True
    return False

# --- 3. 侧边栏：文件管理 ---
with st.sidebar:
    st.header("📂 文档管理")
    
    # --- 上传部分 ---
    st.subheader("新增文档")
    uploaded_files = st.file_uploader("选择 PDF/TXT/MD", accept_multiple_files=True)
    if st.button("🚀 开始同步"):
        if uploaded_files:
            for f in uploaded_files:
                with st.spinner(f"正在处理 {f.name}..."):
                    process_new_file(f)
            st.success("同步完成！")
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")
    
    # --- 删除部分 ---
    st.subheader("已有文档列表")
    current_files = os.listdir(DATA_PATH)
    if not current_files:
        st.info("知识库空空如也")
    else:
        for f in current_files:
            col1, col2 = st.columns([0.7, 0.3])
            col1.text(f"📄 {f}")
            if col2.button("🗑️", key=f):
                if delete_file_from_db(f):
                    st.toast(f"已删除 {f}")
                    st.cache_resource.clear()
                    st.rerun()

# --- 4. 主界面：聊天 (保持不变) ---
st.title("🤖 DeepSeek 本地增强知识库")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("问问你的文档..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        db = get_vector_db()
        if db:
            with st.spinner("检索中..."):
                results = db.similarity_search(prompt, k=3)
                context = "\n\n".join([d.page_content for d in results])
                
                # 记录来源
                sources = list(set([os.path.basename(d.metadata.get('source', '')) for d in results]))
                
                client = Client(host='http://127.0.0.1:11434')
                full_prompt = f"已知信息：\n{context}\n\n问题：{prompt}"
                
                response = client.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': full_prompt}])
                ans = response['message']['content']
                
                final_ans = f"{ans}\n\n> 📚 **参考文档**: {', '.join(sources)}"
                st.markdown(final_ans)
                st.session_state.messages.append({"role": "assistant", "content": final_ans})
        else:
            st.warning("请先在左侧上传文档。")