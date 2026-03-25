import os
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from ollama import Client

# 告诉系统，访问本地地址时不要走代理
os.environ['NO_PROXY'] = '127.0.0.1,localhost'

# 配置参数（必须和 ingest.py 保持一致）
DB_PATH = "database"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-r1"  # 确保你本地有这个模型

def query_knowledge_base(question):
    # 1. 加载本地向量数据库
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # 2. 检索：寻找最相关的 3 个文档片段
    print(f"\n🔍 正在从数据库中检索相关信息...")
    results = vector_db.similarity_search(question, k=3)
    print(f"DEBUG: 数据库检索到的片段内容是: {results}")
    
    # 3. 提取内容获取页码
    context_parts = []
    sources = []
    
    for i, doc in enumerate(results):
        content = doc.page_content
        # 从元数据中提取文件名和页码
        source_name = os.path.basename(doc.metadata.get('source', '未知文件'))
        page_num = doc.metadata.get('page', '未知页码')
        
        context_parts.append(content)
        sources.append(f"资料 [{i+1}]: {source_name} (第 {page_num} 页)")

    context = "\n\n".join(context_parts)
    
    # 4. 构建 Prompt (RAG 的核心：让 AI 只根据你提供的文档回答)
    prompt = f"""你是一个专业的本地文档助手。请根据【已知信息】简洁、准确地回答【用户问题】。
如果你在信息中找不到答案，请诚实回答“我的知识库中没有相关记录”，不要胡乱编造。

【已知信息】：
{context}

【用户问题】：
{question}
"""

    # 5. 调用本地 DeepSeek
    client = Client(host='http://127.0.0.1:11434')
    print(f"🤖 AI 正在思考中...\n")
    response = client.chat(model=LLM_MODEL, messages=[
        {'role': 'user', 'content': prompt},
    ])
    
    answer= response['message']['content']
    final_output = f"{answer}\n\n📌 答案来源：\n" + "\n".join(set(sources))
    return final_output

if __name__ == "__main__":
    while True:
        user_input = input("请输入你的问题 (输入 quit 退出):")
        if user_input.lower() == 'quit':
            break
        
        if not user_input.strip():
            continue
            
        answer = query_knowledge_base(user_input)
        print("-" * 30)
        print(answer)
        print("-" * 30 + "\n")