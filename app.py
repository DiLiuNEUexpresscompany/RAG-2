import os
import base64
import gc
import tempfile
import uuid
import json
import requests
from typing import List, Dict, Optional
from datetime import datetime
import streamlit as st

from llama_index.readers.file.docs import DocxReader

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
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer

# Constants
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".doc", ".docx"]
CACHE_DIR = "cache"
PERSIST_DIR = "storage"
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
        """Scrape the webpage and save the content to a temporary JSON file."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Extract text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Debug output
            st.write(f"Scraped content length: {len(text)}")
            st.write("First 200 characters of content:", text[:200])
            
            # Extract metadata
            metadata = {
                'title': soup.title.string if soup.title else url,
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
            
            if not text.strip():
                st.warning("Warning: Scraped content is empty")
                return None
            
            # Save scraped content to a temporary JSON file
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix=".json") as tmp_file:
                json_data = {
                    "source_type": "web",
                    "content": text,
                    "metadata": metadata
                }
                json.dump(json_data, tmp_file)
                temp_file_path = tmp_file.name
                st.write(f"Scraped content saved to temporary JSON file: {temp_file_path}")
    
            return temp_file_path  # Return JSON file path
        except Exception as e:
            st.error(f"Error scraping URL {url}: {str(e)}")
            return None

def load_from_json_file(json_file_path: str, rag_system) -> bool:
    """Load knowledge source from a JSON file and add it to the knowledge base."""
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            content = data.get("content", "")
            metadata = data.get("metadata", {})
            
            if not content.strip():
                st.warning("Loaded JSON content is empty")
                return False
            
            # 创建 KnowledgeSource 对象
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
    """Process uploaded files with proper error handling"""
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
            else:
                reader = SimpleDirectoryReader(
                    input_files=[file_path]
                )
                docs = reader.load_data()

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
        
        # Configure settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self._ensure_storage_initialized()
        
    def _ensure_storage_initialized(self):
        """Ensure storage is properly initialized"""
        try:
            os.makedirs(PERSIST_DIR, exist_ok=True)
            
            storage_exists = os.path.exists(PERSIST_DIR) and any(
                os.path.exists(os.path.join(PERSIST_DIR, f))
                for f in ['docstore.json', 'index_store.json', 'vector_store.json']
            )
            
            if not storage_exists:
                self.index = VectorStoreIndex([])
            else:
                try:
                    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                    self.index = load_index_from_storage(storage_context)
                except Exception:
                    st.warning("Failed to load existing storage, creating new index")
                    self.index = VectorStoreIndex([])
                    
            # Verify index is properly initialized
            if not hasattr(self, 'index') or self.index is None:
                self.index = VectorStoreIndex([])
                
            # Force persist initial state
            self.index.storage_context.persist(persist_dir=PERSIST_DIR)
            
        except Exception as e:
            st.error(f"Storage initialization error: {str(e)}")
            self.index = VectorStoreIndex([])
    
    def add_knowledge_source(self, source: KnowledgeSource) -> bool:
        """Add a new knowledge source to the index with verification"""
        try:
            if not source or not source.content or not source.content.strip():
                st.warning("Empty content source provided")
                return False
                
            # Create document with content verification
            doc = Document(
                text=source.content,
                metadata={
                    **source.metadata,
                    'added_at': datetime.now().isoformat()
                }
            )
            
            # Verify document creation
            if not doc or not doc.text:
                st.warning("Failed to create valid document")
                return False
            
            # Insert and verify
            self.index.insert(doc)
            doc_count = len(self.index.docstore.docs)
            st.info(f"Index now contains {doc_count} documents")
            
            # Persist changes
            self.index.storage_context.persist(persist_dir=PERSIST_DIR)
            
            # Verify persistence
            if not os.path.exists(os.path.join(PERSIST_DIR, 'docstore.json')):
                st.warning("Failed to persist index")
                return False
                
            return True
            
        except Exception as e:
            st.error(f"Error adding knowledge source: {str(e)}")
            return False
            
    def create_query_engine(self, similarity_threshold: float = 0.7) -> RetrieverQueryEngine:
        """Create an optimized query engine with custom prompts"""
        # Custom query prompt template
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

        # Create custom prompt
        query_prompt = PromptTemplate(query_prompt_tmpl)

        # Create response synthesizer with the custom prompt
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=query_prompt,  # Use the query prompt here
        )

        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5,
            filters=None
        )

        # Post-processor
        postprocessor = SimilarityPostprocessor(
            similarity_threshold=similarity_threshold
        )

        # Create and return query engine without separate query_prompt parameter
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor],
            response_synthesizer=response_synthesizer
        )

    def query(self, query_text: str, context_window: int = 3) -> Dict:
        """Enhanced query method with better error handling"""
        try:
            # Validate index status
            if not hasattr(self, 'index') or self.index is None:
                return {
                    'response': "Knowledge base not properly initialized. Please reload documents.",
                    'sources': []
                }

            # Check document count
            doc_count = len(self.index.docstore.docs)
            if doc_count == 0:
                return {
                    'response': "Knowledge base is empty. Please add some documents first.",
                    'sources': []
                }

            st.info(f"Querying knowledge base containing {doc_count} documents...")

            # Preprocess query
            processed_query = self._preprocess_query(query_text)
            
            # Create query engine
            query_engine = self.create_query_engine()

            # Execute query
            response = query_engine.query(processed_query)

            if not response or not response.response:
                return {
                    'response': "Unable to generate a valid answer. Please try rephrasing the question.",
                    'sources': []
                }

            # Organize source information
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source = {
                        'content': node.node.text[:300] + "..." if len(node.node.text) > 300 else node.node.text,
                        'metadata': node.node.metadata,
                        'score': node.score
                    }
                    # Add document title
                    if 'title' in node.node.metadata:
                        source['title'] = node.node.metadata['title']
                    elif 'filename' in node.node.metadata:
                        source['title'] = node.node.metadata['filename']
                    sources.append(source)

            return {
                'response': response.response,
                'sources': sources
            }

        except Exception as e:
            st.error(f"Error occurred during query processing: {str(e)}")
            import traceback
            st.error(f"Error details: {traceback.format_exc()}")
            return {
                'response': "An error occurred while processing the query. Please try again later.",
                'sources': []
            }

    def _preprocess_query(self, query_text: str) -> str:
        """Preprocess query text"""
        # Handle common query types
        if query_text.lower().strip() in ['summary', 'summarize', 'summarise']:
            return "Please summarize the main contents of the document, including research objectives, methods, key findings, and conclusions."
        elif query_text.lower().strip() in ['detail', 'details', 'detailed']:
            return "Please provide a detailed content analysis of the document, including research background, methodology, experimental design, results, and discussion."
        
        return query_text
        
def get_llm(model_name: str) -> Ollama:
    return Ollama(model=model_name, request_timeout=120.0)

def get_embeddings(model_name: str) -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=model_name,
        request_timeout=120.0
    )

class EnhancedChatInterface:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "id" not in st.session_state:
            st.session_state.id = str(uuid.uuid4())
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "feedback" not in st.session_state:
            st.session_state.feedback = {}   
    def render_main_interface(self):
        """Render main chat interface"""
        st.title("Enhanced RAG Chatbot")
        
        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Show source information for assistant responses
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander("View Sources"):
                        for source in msg["sources"]:
                            st.markdown(f"**Source:** {source['metadata'].get('title', 'Unknown')}")
                            st.markdown(f"**Relevance:** {source['score']:.2f}")
                            st.markdown(f"**Extract:** {source['content']}")
                
                # Add feedback buttons
                if msg["role"] == "assistant":
                    cols = st.columns(2)
                    msg_id = msg.get("id", "unknown")
                    
                    if cols[0].button("👍", key=f"thumbs_up_{msg_id}"):
                        st.session_state.feedback[msg_id] = "positive"
                        st.toast("Thank you for your feedback!")
                        
                    if cols[1].button("👎", key=f"thumbs_down_{msg_id}"):
                        st.session_state.feedback[msg_id] = "negative"
                        feedback = st.text_input(
                            "Please tell us what we could improve:",
                            key=f"feedback_{msg_id}"
                        )
                        if feedback:
                            st.session_state.feedback[msg_id] = {"rating": "negative", "comment": feedback}
                            st.toast("Thank you for your feedback!")
    def handle_url_input(self, url_input):
        """Handle the URL input, save content to JSON, and add it to the knowledge base."""
        if not self.rag_system:
            st.error("RAG system is not initialized.")
            return

        # Scrape content and save to temporary JSON
        json_file_path = self.rag_system.web_scraper.scrape_url(url_input)
        
        if json_file_path:
            # Load content from JSON file to the knowledge base
            success = load_from_json_file(json_file_path, self.rag_system)
            if success:
                st.success(f"Content from {url_input} has been added to the knowledge base.")
            else:
                st.error(f"Failed to add content from {url_input} to the knowledge base.")
        else:
            st.error(f"Failed to fetch content from {url_input}.")
    def render_sidebar(self):
        """Render sidebar with configuration options"""
        with st.sidebar:
            st.header("🤖 Configuration")
            
            # RAG configuration
            use_rag = st.checkbox("Enable RAG", value=True)
            
            uploaded_files = None
            if use_rag:
                st.subheader("Knowledge Sources")
                
                # File upload
                uploaded_files = st.file_uploader(
                    "Upload Documents",
                    type=[ext[1:] for ext in SUPPORTED_FILE_TYPES],
                    accept_multiple_files=True
                )
                
                # URL input
                url_input = st.text_input("Add Web Page (URL)")
                if url_input and st.button("Fetch Content"):
                    self.handle_url_input(url_input)
                        
            return use_rag, uploaded_files

def main():
    # Sidebar for model selection
    with st.sidebar:
        st.header("🤖 Configuration")
        
        # Model selection
        selected_model = st.selectbox(
            "Choose Model",
            ["llama3.1", "gemma2:9b"],
            index=0
        )
    
    # Initialize RAG system with selected model
    rag_system = EnhancedRAG(selected_model)
    
    # Initialize interface with rag_system
    interface = EnhancedChatInterface(rag_system)
    
    # Render sidebar and get configuration
    use_rag, uploaded_files = interface.render_sidebar()
    
    # Process uploaded files
    if uploaded_files:
        process_uploaded_files(uploaded_files, rag_system)
                
    # Render main interface
    interface.render_main_interface()
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        if use_rag:
            result = rag_system.query(prompt)
            response = result['response']
            sources = result['sources']
        else:
            response = rag_system.llm.complete(prompt).text
            sources = []
        
        # Add assistant message with metadata
        msg_id = str(uuid.uuid4())
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
            "id": msg_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Force refresh
        try:
            st.experimental_rerun()
        except AttributeError:
            try:
                st.rerun()
            except AttributeError:
                # Alternative approach
                st.session_state['rerun'] = st.session_state.get('rerun', 0) + 1


if __name__ == "__main__":
    main()