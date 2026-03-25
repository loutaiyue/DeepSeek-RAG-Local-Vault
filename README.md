🤖 DeepSeek-RAG-Local-Vault
基于 DeepSeek-R1 的全本地隐私增强知识库解决方案
本项目是一个工业级的 RAG (Retrieval-Augmented Generation) 本地知识库应用。通过集成 DeepSeek-R1 推理模型和 ChromaDB 向量数据库，实现了在完全断网环境下对私有文档（PDF/TXT/MD）的智能检索与深度问答。
🌟 项目亮点 (Key Features)
完全隐私保护：全链路本地化，数据不上传云端，满足极高安全需求。
深度推理能力：核心引擎采用 DeepSeek-R1，利用其思维链（CoT）能力，回答逻辑更严密。
多格式动态管理：
支持 PDF、TXT、Markdown 文档一键上传。
增量入库：无需重启，实时同步新文档。
向量级删除：物理删除文件的同时同步清理向量数据库，确保检索精度。
工业级检索链路：基于 nomic-embed-text 高性能嵌入模型，配合 RecursiveCharacterTextSplitter 优化语义切分。
现代化 UI：使用 Streamlit 构建响应式聊天界面，支持来源追溯（Citations）。
🏗️ 技术架构 (Architecture)
数据层 (Data)：解析 PDF/Text/MD -> 语义切分 (Chunks) -> 向量化 (Embedding)。
存储层 (Storage)：使用 ChromaDB 持久化向量数据，支持高效相似度检索。
检索层 (Retrieval)：基于余弦相似度算法提取 Top-K 相关上下文。
生成层 (Generation)：将上下文喂给 DeepSeek-R1 进行 Context-Aware 推理。
应用层 (App)：Streamlit 驱动的交互式 Web 界面。
🚀 快速开始 (Quick Start)
1. 环境准备
确保已安装 Ollama 并拉取所需模型：
ollama pull deepseek-r1
ollama pull nomic-embed-text
2. 安装依赖
pip install -r requirements.txt
3. 运行应用
由于 Windows 环境变量路径差异，建议使用 Python 模块方式启动：
python -m streamlit run src/app.py
