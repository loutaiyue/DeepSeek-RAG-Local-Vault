import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 配置参数
DATA_PATH = "data/raw"
DB_PATH = "database"

# 定义不同后缀名对应的加载器
LOADERS = {
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".pdf": (PyPDFLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
}

def create_directory_loader(file_ext, loader_cls, loader_kwargs):
    return DirectoryLoader(
        path=DATA_PATH,
        glob=f"**/*{file_ext}",
        loader_cls=loader_cls,
        loader_kwargs=loader_kwargs,
        show_progress=True
    )

def main():
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"❌ 错误：请先在 {DATA_PATH} 放入文件！")
        return

    all_documents = []
    
    # 1. 循环加载不同格式的文件
    for ext, (loader_cls, kwargs) in LOADERS.items():
        print(f"正在扫描 {ext} 文件...")
        loader = create_directory_loader(ext, loader_cls, kwargs)
        docs = loader.load()
        if docs:
            print(f"✅ 成功加载 {len(docs)} 个 {ext} 文件")
            all_documents.extend(docs)

    if not all_documents:
        print("😭 未能识别到任何有效文档，请检查文件格式。")
        return

    # 2. 切分文档（优化切分逻辑）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=100,
        add_start_index=True # 记录在原文档的位置，方便追溯
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"✂️ 文档切分完成，共生成 {len(chunks)} 个数据块。")

    # 3. 向量化入库
    print("🧠 正在生成向量并存入 ChromaDB (这可能需要一点时间)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print(f"🚀 大功告成！本地大脑已就绪，位置：{DB_PATH}")

if __name__ == "__main__":
    main()