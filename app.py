import os
import base64
import gc
import tempfile
import uuid
import json
import requests
import time  # 用于跟踪响应时间
import threading  # 用于调度器
import schedule  # 用于定时任务
from typing import List, Dict, Optional
from datetime import datetime
import streamlit as st

from llama_index.readers.file.docs import DocxReader
from llama_index.readers.file.docs import PDFReader
from llama_index.readers.file.docs import HWPReader  # 假设存在MarkdownReader

from bs4 import BeautifulSoup
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer

import tiktoken  # 用于Token计数

# 常量定义
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".doc", ".docx", ".md"]
CACHE_DIR = "cache"
PERSIST_DIR = "storage"
DATA_DIR = "data"  # 数据目录，用于存放域特定的源
MODEL_NAME = "llama3.1"

class KnowledgeSource:
    def __init__(self, source_type: str, content: str, metadata: Dict):
        self.source_type = source_type
        self.content = content
        self.metadata = metadata
        self.timestamp = datetime.now()

class WebScraper:
    @staticmethod
    def scrape_url(url: str) -> Optional[str]:
        """抓取网页内容并保存到临时JSON文件"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除脚本和样式元素
            for script in soup(["script", "style"]):
                script.decompose()
                
            # 提取文本内容
            text = soup.get_text(separator='\n', strip=True)
            
            # 调试输出
            st.write(f"Scraped content length: {len(text)}")
            st.write("First 200 characters of content:", text[:200])
            
            # 提取元数据
            metadata = {
                'title': soup.title.string if soup.title else url,
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
            
            if not text.strip():
                st.warning("Warning: Scraped content is empty")
                return None
            
            # 保存抓取内容到临时JSON文件
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix=".json") as tmp_file:
                json_data = {
                    "source_type": "web",
                    "content": text,
                    "metadata": metadata
                }
                json.dump(json_data, tmp_file)
                temp_file_path = tmp_file.name
                st.write(f"Scraped content saved to temporary JSON file: {temp_file_path}")

            return temp_file_path  # 返回JSON文件路径
        except Exception as e:
            st.error(f"Error scraping URL {url}: {str(e)}")
            return None

def load_from_json_file(json_file_path: str, rag_system) -> bool:
    """从JSON文件加载知识源并添加到知识库"""
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            content = data.get("content", "")
            metadata = data.get("metadata", {})
            
            if not content.strip():
                st.warning("Loaded JSON content is empty")
                return False
            
            # 创建KnowledgeSource对象
            knowledge_source = KnowledgeSource(
                source_type=data.get("source_type", "file"),
                content=content,
                metadata=metadata
            )
            
            # 添加到知识库
            success = rag_system.add_knowledge_source(knowledge_source)
            if success:
                st.success(f"Successfully loaded and added content from {json_file_path}")
                return True
            else:
                st.error(f"Failed to add content from {json_file_path} to knowledge base")
                return False
    except Exception as e:
        st.error(f"Error loading from JSON file {json_file_path}: {str(e)}")
        return False

def process_uploaded_files(uploaded_files, rag_system):
    """处理上传的文件，添加到知识库"""
    if not uploaded_files:
        return
        
    for file in uploaded_files:
        temp_dir = tempfile.mkdtemp()
        temp_path = None
        try:
            temp_path = os.path.join(temp_dir, file.name)
            st.write(f"Processing file: {file.name} in {temp_path}")  # 调试信息
            
            with open(temp_path, 'wb') as tmp_file:
                tmp_file.write(file.getvalue())
            
            knowledge_source = rag_system.document_processor.process_file(temp_path)
            if knowledge_source:
                if knowledge_source.content.strip():
                    success = rag_system.add_knowledge_source(knowledge_source)
                    if success:
                        st.success(f"Successfully processed and added {file.name} to the knowledge base.")
                    else:
                        st.error(f"Failed to add {file.name} to knowledge base")
                else:
                    st.warning(f"No content extracted from {file.name}")
            else:
                st.error(f"Failed to process {file.name}")
                
        except FileNotFoundError as e:
            st.error(f"File not found error while processing {file.name}: {str(e)}")
        except PermissionError as e:
            st.error(f"Permission denied while processing {file.name}: {str(e)}")
        except IOError as e:
            st.error(f"IO error while processing {file.name}: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error processing {file.name}: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
        finally:
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                if temp_dir and os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                st.warning(f"Could not clean up temporary files: {str(e)}")

class DocumentProcessor:
    def __init__(self):
        self.parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )

    def process_file(self, file_path: str) -> Optional[KnowledgeSource]:
        try:
            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext not in SUPPORTED_FILE_TYPES:
                raise ValueError(f"Unsupported file type: {file_ext}")

            if file_ext == '.docx':
                st.write(f"Using DocxReader for file: {file_path}")
                loader = DocxReader()
                docs = loader.load_data(file=file_path)
            elif file_ext == '.pdf':
                st.write(f"Using PDFReader for file: {file_path}")
                loader = PDFReader()
                docs = loader.load_data(file=file_path)
            else:
                st.warning(f"No reader available for file type: {file_ext}")
                return None

            if not docs:
                st.warning("No documents loaded from file")
                return None

            content = docs[0].text
            if not content.strip():
                st.warning("Extracted content is empty")
                return None

            st.write(f"Extracted content length: {len(content)}")
            st.write("First 200 characters:", content[:200])

            metadata = {
                'filename': os.path.basename(file_path),
                'file_type': file_ext,
                'timestamp': datetime.now().isoformat()
            }

            return KnowledgeSource(
                source_type='file',
                content=content,
                metadata=metadata
            )
        except Exception as e:
            st.error(f"Error processing file {file_path}: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None

class EnhancedRAG:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = get_llm(model_name)
        self.embed_model = get_embeddings(model_name)
        self.document_processor = DocumentProcessor()
        self.web_scraper = WebScraper()
        
        # 配置设置
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self._ensure_storage_initialized()
        self._load_domain_specific_sources()  # 加载域特定的源
        self._start_scheduler()  # 启动调度器进行定期更新
        
    def _ensure_storage_initialized(self):
        """确保存储被正确初始化"""
        try:
            os.makedirs(PERSIST_DIR, exist_ok=True)
            
            storage_exists = os.path.exists(PERSIST_DIR) and any(
                os.path.exists(os.path.join(PERSIST_DIR, f))
                for f in ['docstore.json', 'index_store.json', 'vector_store.json']
            )
            
            if not storage_exists:
                self.index = VectorStoreIndex([])
                # Removed adding initial documents during initialization
                # self._add_initial_documents()  # 如果没有存储，则添加初始文档
            else:
                try:
                    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                    self.index = load_index_from_storage(storage_context)
                except Exception:
                    st.warning("Failed to load existing storage, creating new index")
                    self.index = VectorStoreIndex([])
                    
            # 验证索引是否正确初始化
            if not hasattr(self, 'index') or self.index is None:
                self.index = VectorStoreIndex([])
                
            # 强制持久化初始状态
            self.index.storage_context.persist(persist_dir=PERSIST_DIR)
            
        except Exception as e:
            st.error(f"Storage initialization error: {str(e)}")
            self.index = VectorStoreIndex([])
    
    def _add_initial_documents(self):
        """将初始域特定文档添加到索引中，包含去重和更好的错误处理"""
        try:
            # 定义要读取的目录列表
            directories = [
                os.path.join(DATA_DIR, "scientific_literature"),
                os.path.join(DATA_DIR, "technical_blogs")
            ]
            
            all_documents = []
            # 更安全的方式获取现有文档文件名
            existing_filenames = set()
            if hasattr(self, 'index') and self.index and hasattr(self.index, 'docstore'):
                for doc_id, doc in self.index.docstore.docs.items():
                    if doc and hasattr(doc, 'metadata'):
                        filename = doc.metadata.get('filename')
                        if filename:  # 只添加有效的文件名
                            existing_filenames.add(filename)
            
            for dir_path in directories:
                if not os.path.exists(dir_path):
                    st.warning(f"Directory {dir_path} does not exist.")
                    continue
                    
                try:
                    reader = SimpleDirectoryReader(
                        input_dir=dir_path,
                        file_extractor={
                            ".pdf": PDFReader(),
                            ".docx": DocxReader(),
                        },
                        recursive=True  # 递归读取子目录
                    )
                    
                    documents = reader.load_data()
                    
                    # 添加更详细的日志
                    st.info(f"Found {len(documents)} documents in {dir_path}")
                    
                    for doc in documents:
                        if not doc or not hasattr(doc, 'metadata'):
                            st.warning("Found invalid document without metadata")
                            continue
                            
                        filename = doc.metadata.get('filename')
                        if not filename:
                            # Handle missing filename by assigning a default
                            filename = doc.metadata.get('title', f"doc_{uuid.uuid4()}")
                            doc.metadata['filename'] = filename
                            st.warning(f"Document missing filename. Assigned filename: {filename}")
                            
                        if filename not in existing_filenames:
                            st.info(f"Adding new document: {filename}")
                            all_documents.append(doc)
                            existing_filenames.add(filename)  # 更新现有文件集合
                        else:
                            st.info(f"Document '{filename}' already exists in the knowledge base. Skipping.")
                            
                except Exception as e:
                    st.error(f"Error processing directory {dir_path}: {str(e)}")
                    continue
            
            if all_documents:
                # 批量插入文档
                try:
                    for doc in all_documents:
                        self.index.insert(doc)
                        
                    st.success(f"Successfully added {len(all_documents)} new documents to the knowledge base.")
                    
                    # 确保更改被持久化
                    try:
                        self.index.storage_context.persist(persist_dir=PERSIST_DIR)
                        st.success("Successfully persisted the updated knowledge base.")
                    except Exception as e:
                        st.error(f"Error persisting changes: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Error inserting documents: {str(e)}")
            else:
                st.info("No new documents to add to the knowledge base.")
                
        except Exception as e:
            st.error(f"Error in _add_initial_documents: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
    
    def _load_domain_specific_sources(self):
        """加载域特定的源到知识库中"""
        try:
            # 检查索引是否为空，以避免重复添加
            if len(self.index.docstore.docs) == 0:
                self._add_initial_documents()
            else:
                st.info("Knowledge base already initialized with domain-specific sources.")
        except Exception as e:
            st.error(f"Error loading domain-specific sources: {str(e)}")
    
    def _start_scheduler(self):
        """启动一个后台线程，用于调度定期更新任务"""
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def _run_scheduler(self):
        """定义并运行定时任务"""
        schedule.every().day.at("02:00").do(self._update_knowledge_base)
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    
    def _update_knowledge_base(self):
        """使用新的域特定文档更新知识库"""
        try:
            st.info("Starting knowledge base update...")
            self._add_initial_documents()  # 重新添加文档（确保去重）
            st.success("Knowledge base has been updated successfully.")
        except Exception as e:
            st.error(f"Error during knowledge base update: {str(e)}")
    
    def add_knowledge_source(self, source: KnowledgeSource) -> bool:
        """将新的知识源添加到索引中，包含验证"""
        try:
            if not source or not source.content or not source.content.strip():
                st.warning("Empty content source provided")
                return False
                
            # 创建带有内容验证的文档
            doc = Document(
                text=source.content,
                metadata={
                    **source.metadata,
                    'added_at': datetime.now().isoformat()
                }
            )
            
            # 验证文档创建
            if not doc or not doc.text:
                st.warning("Failed to create valid document")
                return False
            
            # 插入并验证
            self.index.insert(doc)
            doc_count = len(self.index.docstore.docs)
            st.info(f"Index now contains {doc_count} documents")
            
            # 持久化更改
            self.index.storage_context.persist(persist_dir=PERSIST_DIR)
            
            # 验证持久化
            if not os.path.exists(os.path.join(PERSIST_DIR, 'docstore.json')):
                st.warning("Failed to persist index")
                return False
                
            return True
                
        except Exception as e:
            st.error(f"Error adding knowledge source: {str(e)}")
            return False
                
    def create_query_engine(self, similarity_threshold: float = 0.7) -> RetrieverQueryEngine:
        """创建一个优化过的查询引擎，使用自定义提示"""
        # 自定义查询提示模板
        query_prompt_tmpl = """Answer the question based on the provided context. If the context doesn't contain relevant information, please indicate that you cannot answer.

Relevant Context:
---------------
{context_str}
---------------

Question: {query_str}

Please provide a detailed answer and reference specific information from the context when possible. For scientific papers, include:
1. Main research objectives
2. Methods used
3. Key findings
4. Important conclusions

Answer:"""
    
        # 创建自定义提示
        query_prompt = PromptTemplate(query_prompt_tmpl)

        # 使用自定义提示创建响应合成器
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=query_prompt,  # 使用自定义查询提示
        )

        # 创建检索器
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,  # 增加检索的文档数量
            filters=None
        )

        # 后处理器
        postprocessor = SimilarityPostprocessor(
            similarity_threshold=similarity_threshold
        )

        # 创建并返回查询引擎，不使用单独的query_prompt参数
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor],
            response_synthesizer=response_synthesizer
        )

    def query(self, query_text: str, context_window: int = 3) -> Dict:
        """增强的查询方法，包含更好的错误处理和性能跟踪"""
        try:
            # 验证索引状态
            if not hasattr(self, 'index') or self.index is None:
                return {
                    'response': "Knowledge base not properly initialized. Please reload documents.",
                    'sources': [],
                    'tokens': 0
                }

            # 检查文档数量
            doc_count = len(self.index.docstore.docs)
            if doc_count == 0:
                return {
                    'response': "Knowledge base is empty. Please add some documents first.",
                    'sources': [],
                    'tokens': 0
                }

            st.info(f"Querying knowledge base containing {doc_count} documents...")

            # 预处理查询
            processed_query = self._preprocess_query(query_text)
            
            # 开始计时
            start_time = time.time()

            # 创建查询引擎
            query_engine = self.create_query_engine()

            # 执行查询
            response = query_engine.query(processed_query)

            # 结束计时
            end_time = time.time()
            response_time = end_time - start_time

            # 存储响应时间
            if 'response_times' not in st.session_state:
                st.session_state.response_times = []
            st.session_state.response_times.append(response_time)

            # 使用tiktoken进行Token计数
            encoding = tiktoken.get_encoding("cl100k_base")  # 根据模型选择合适的编码器
            tokens = len(encoding.encode(response.response)) if response.response else 0

            if 'token_counts' not in st.session_state:
                st.session_state.token_counts = []
            st.session_state.token_counts.append(tokens)

            if not response or not response.response:
                return {
                    'response': "Unable to generate a valid answer. Please try rephrasing the question.",
                    'sources': [],
                    'tokens': tokens
                }

            # 组织源信息
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source = {
                        'content': node.node.text[:300] + "..." if len(node.node.text) > 300 else node.node.text,
                        'metadata': node.node.metadata,
                        'score': node.score
                    }
                    # 添加文档标题
                    if 'title' in node.node.metadata:
                        source['title'] = node.node.metadata['title']
                    elif 'filename' in node.node.metadata:
                        source['title'] = node.node.metadata['filename']
                    sources.append(source)

            return {
                'response': response.response,
                'sources': sources,
                'response_time': response_time,  # 包含响应时间
                'tokens': tokens  # 包含Token计数
            }

        except Exception as e:
            st.error(f"Error occurred during query processing: {str(e)}")
            import traceback
            st.error(f"Error details: {traceback.format_exc()}")
            return {
                'response': "An error occurred while processing the query. Please try again later.",
                'sources': [],
                'tokens': 0
            }

    def _preprocess_query(self, query_text: str) -> str:
        """预处理查询文本"""
        # 处理常见的查询类型
        if query_text.lower().strip() in ['summary', 'summarize', 'summarise']:
            return "Please summarize the main contents of the document, including research objectives, methods, key findings, and conclusions."
        elif query_text.lower().strip() in ['detail', 'details', 'detailed']:
            return "Please provide a detailed content analysis of the document, including research background, methodology, experimental design, results, and discussion."
        
        return query_text

def get_llm(model_name: str) -> Ollama:
    return Ollama(model=model_name, request_timeout=300.0, device='cuda')

def get_embeddings(model_name: str) -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=model_name,
        request_timeout=300.0,
        device='cuda'
    )

class EnhancedChatInterface:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """初始化Streamlit会话状态变量"""
        if "id" not in st.session_state:
            st.session_state.id = str(uuid.uuid4())
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "feedback" not in st.session_state:
            st.session_state.feedback = {}
        if "response_times" not in st.session_state:
            st.session_state.response_times = []  # 初始化响应时间
        if "token_counts" not in st.session_state:
            st.session_state.token_counts = []  # 初始化Token计数
        if "total_queries" not in st.session_state:
            st.session_state.total_queries = 0
        if "positive_feedback" not in st.session_state:
            st.session_state.positive_feedback = 0
        if "negative_feedback" not in st.session_state:
            st.session_state.negative_feedback = 0
        if "feedback_comments" not in st.session_state:
            st.session_state.feedback_comments = []
            
    def render_main_interface(self):
        """渲染主聊天界面"""
        st.title("Enhanced RAG Chatbot")
        
        # 聊天历史
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # 显示助手回答的源信息
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("View Sources"):
                        for source in msg["sources"]:
                            st.markdown(f"**Source:** {source['metadata'].get('title', 'Unknown')}")
                            st.markdown(f"**Relevance:** {source['score']:.2f}")
                            st.markdown(f"**Extract:** {source['content']}")
                
                # 添加反馈按钮
                if msg["role"] == "assistant":
                    cols = st.columns(2)
                    msg_id = msg.get("id", "unknown")
                    
                    if cols[0].button("👍", key=f"thumbs_up_{msg_id}"):
                        st.session_state.feedback[msg_id] = "positive"
                        st.session_state.positive_feedback += 1
                        st.session_state.total_queries += 1
                        st.toast("Thank you for your feedback!")
                        
                    if cols[1].button("👎", key=f"thumbs_down_{msg_id}"):
                        st.session_state.feedback[msg_id] = "negative"
                        feedback = st.text_input(
                            "Please tell us what we could improve:",
                            key=f"feedback_{msg_id}"
                        )
                        if feedback:
                            st.session_state.feedback[msg_id] = {"rating": "negative", "comment": feedback}
                            st.session_state.negative_feedback += 1
                            st.session_state.feedback_comments.append(feedback)
                            st.session_state.total_queries += 1
                            st.toast("Thank you for your feedback!")
                    
    def handle_url_input(self, url_input):
        """处理URL输入，保存内容到JSON并添加到知识库"""
        if not self.rag_system:
            st.error("RAG system is not initialized.")
            return

        # 抓取内容并保存到临时JSON
        json_file_path = self.rag_system.web_scraper.scrape_url(url_input)
        
        if json_file_path:
            # 从JSON文件加载内容到知识库
            success = load_from_json_file(json_file_path, self.rag_system)
            if success:
                st.success(f"Content from {url_input} has been added to the knowledge base.")
            else:
                st.error(f"Failed to add content from {url_input} to the knowledge base.")
        else:
            st.error(f"Failed to fetch content from {url_input}.")
    
    def clear_conversation(self):
        """清除对话和相关会话状态变量"""
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.session_state.total_queries = 0
        st.session_state.positive_feedback = 0
        st.session_state.negative_feedback = 0
        st.session_state.feedback_comments = []
        st.session_state.response_times = []
        st.session_state.token_counts = []
        st.success("Conversation and feedback have been cleared.")
        st.rerun()

    def clear_cache(self):
        """清除缓存和存储目录"""
        try:
            # 删除缓存目录
            if os.path.exists(CACHE_DIR):
                for filename in os.listdir(CACHE_DIR):
                    file_path = os.path.join(CACHE_DIR, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                os.rmdir(CACHE_DIR)
                st.success("Cache directory has been cleared.")
            else:
                st.info("Cache directory does not exist.")
            
            # 删除存储目录
            if os.path.exists(PERSIST_DIR):
                for filename in os.listdir(PERSIST_DIR):
                    file_path = os.path.join(PERSIST_DIR, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                os.rmdir(PERSIST_DIR)
                st.success("Storage directory has been cleared.")
            else:
                st.info("Storage directory does not exist.")
            
            # 清除缓存后重新初始化RAG系统
            self.rag_system._ensure_storage_initialized()
            st.success("RAG system has been reinitialized.")
            
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")
    
    def render_sidebar(self):
        """渲染侧边栏，包含配置选项和统计信息"""
        with st.sidebar:
            st.header("📕 RAG")
            
            # RAG配置
            use_rag = st.checkbox("Enable RAG", value=True)
            
            uploaded_files = None
            if use_rag:
                st.subheader("Knowledge Sources")
                
                # 文件上传
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    type=[ext[1:] for ext in SUPPORTED_FILE_TYPES],
                    accept_multiple_files=True
                )
                
                # URL输入
                url_input = st.text_input("Add Web Page (URL)")
                if url_input and st.button("Fetch Content"):
                    self.handle_url_input(url_input)
                
                # 加载域特定源按钮
                if st.button("Load Domain-Specific Sources"):
                    self.rag_system._add_initial_documents()
                
                # 手动更新知识库按钮
                if st.button("Update Knowledge Base Now"):
                    self.rag_system._update_knowledge_base()
            
            # 显示统计信息
            st.markdown("---")
            st.header("📊 Statistics")
            
            # 生成速度统计
            st.subheader("🔄 Generation Speed")
            if st.session_state.response_times:
                avg_response_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
                recent_times = st.session_state.response_times[-5:]  # 最近5次响应时间
                st.write(f"**Average Response Time:** {avg_response_time:.2f} seconds")
                st.write(f"**Recent Response Times:** {', '.join([f'{t:.2f}s' for t in recent_times])}")
            else:
                st.write("No queries yet.")
            
            # Token生成速度统计
            st.subheader("🔢 Token Generation Speed")
            if st.session_state.response_times and st.session_state.token_counts:
                # 计算每次响应的Token每秒
                tokens_per_second = [tokens / time_sec if time_sec > 0 else 0 for tokens, time_sec in zip(st.session_state.token_counts, st.session_state.response_times)]
                
                # 平均Token生成速度
                avg_tokens_per_sec = sum(tokens_per_second) / len(tokens_per_second)
                
                # 最近的Token生成速度
                recent_tokens_per_sec = tokens_per_second[-5:]
                
                st.write(f"**Average Token Generation Speed:** {avg_tokens_per_sec:.2f} tokens/second")
                st.write(f"**Recent Token Generation Speeds:** {', '.join([f'{t:.2f} t/s' for t in recent_tokens_per_sec])}")
            else:
                st.write("No token data yet.")
            
            # 质量统计
            st.subheader("⭐ Quality")
            st.write(f"**Total Queries:** {st.session_state.total_queries}")
            st.write(f"**👍 Positive Feedback:** {st.session_state.positive_feedback}")
            st.write(f"**👎 Negative Feedback:** {st.session_state.negative_feedback}")
            if st.session_state.total_queries > 0:
                quality_score = (st.session_state.positive_feedback / st.session_state.total_queries) * 100
                st.write(f"**Overall Quality Score:** {quality_score:.2f}%")
            else:
                st.write("**Overall Quality Score:** N/A")
            
            # 可选：显示反馈评论
            if st.session_state.feedback_comments:
                with st.expander("View Feedback Comments"):
                    for idx, comment in enumerate(st.session_state.feedback_comments, 1):
                        st.write(f"{idx}. {comment}")
            
            # 添加分隔线
            st.markdown("---")
            
            # 添加清空对话和清除缓存按钮
            st.header("🛠️ Tools")
            cols = st.columns(2)
            if cols[0].button("Clear ALL"):
                self.clear_conversation()
            if cols[1].button("Clear Cache"):
                self.clear_cache()
                
            return use_rag, uploaded_files

def main():
    # 侧边栏中的模型选择
    with st.sidebar:
        st.header("🤖 Configuration")
        
        # 模型选择
        selected_model = st.selectbox(
            "Choose Model",
            ["llama3.1", "gemma2:9b"],
            index=0
        )
    
    # 使用选择的模型初始化RAG系统
    rag_system = EnhancedRAG(selected_model)
    
    # 使用RAG系统初始化界面
    interface = EnhancedChatInterface(rag_system)
    
    # 渲染侧边栏并获取配置
    use_rag, uploaded_files = interface.render_sidebar()
    
    # 处理上传的文件
    if uploaded_files:
        process_uploaded_files(uploaded_files, rag_system)
                
    # 渲染主界面
    interface.render_main_interface()
    
    # 聊天输入
    if prompt := st.chat_input("Ask a question..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 生成响应
        if use_rag:
            result = rag_system.query(prompt)
            response = result.get('response', "No response generated.")
            sources = result.get('sources', [])
            response_time = result.get('response_time', None)  # 获取响应时间
            tokens = result.get('tokens', 0)  # 获取Token计数
        else:
            start_time = time.time()  # 开始计时
            response = rag_system.llm.complete(prompt).text
            end_time = time.time()  # 结束计时
            response_time = end_time - start_time
            encoding = tiktoken.get_encoding("cl100k_base")  # 选择合适的编码器
            tokens = len(encoding.encode(response)) if response else 0
            sources = []
            
            # 存储响应时间和Token
            if 'response_times' not in st.session_state:
                st.session_state.response_times = []
            st.session_state.response_times.append(response_time)
            
            if 'token_counts' not in st.session_state:
                st.session_state.token_counts = []
            st.session_state.token_counts.append(tokens)
        
        # 添加助手消息和元数据
        msg_id = str(uuid.uuid4())
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
            "id": msg_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # 强制刷新
        try:
            st.rerun()
        except AttributeError:
            try:
                st.rerun()
            except AttributeError:
                # 替代方法
                st.session_state['rerun'] = st.session_state.get('rerun', 0) + 1

if __name__ == "__main__":
    main()
