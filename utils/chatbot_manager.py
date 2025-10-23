import requests
import logging
from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from config import Config

logger = logging.getLogger(__name__)

class ChatbotManager:
    def __init__(self):
        self.available_models = {'ollama': [], 'huggingface': []}
        self.current_model = None
        self.current_model_type = 'ollama'
        self.conversation_chains = {}
        
    def check_ollama_status(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                self.available_models['ollama'] = [model['name'] for model in models]
                logger.info(f"✅ Ollama is running. Models: {self.available_models['ollama']}")
                return True
        except Exception as e:
            logger.warning(f"❌ Ollama not available: {str(e)}")
            self.available_models['ollama'] = ["llama2", "mistral"]
        return False
    
    def check_huggingface_status(self):
        """Check if Hugging Face is available"""
        try:
            if Config.HUGGINGFACE_API_KEY:
                self.available_models['huggingface'] = [
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    "google/flan-t5-large",
                    "microsoft/DialoGPT-large",
                    "facebook/blenderbot-400M-distill"
                ]
                logger.info(f"✅ Hugging Face available. Models: {self.available_models['huggingface']}")
                return True
        except Exception as e:
            logger.warning(f"❌ Hugging Face not available: {str(e)}")
            self.available_models['huggingface'] = []
        return False
    
    def get_available_models(self):
        """Get all available models"""
        self.check_ollama_status()
        self.check_huggingface_status()
        return self.available_models
    
    def initialize_llm(self, model_name=None, model_type="ollama"):
        """Initialize the language model"""
        try:
            if model_type == "ollama":
                if not self.available_models['ollama']:
                    raise Exception("No Ollama models available")
                
                model_to_use = model_name or Config.OLLAMA_DEFAULT_MODEL
                if model_to_use not in self.available_models['ollama']:
                    model_to_use = self.available_models['ollama'][0]
                
                llm = Ollama(
                    model=model_to_use,
                    base_url=Config.OLLAMA_BASE_URL,
                    temperature=0.7,
                    top_p=0.9,
                    num_predict=512
                )
                self.current_model = model_to_use
                self.current_model_type = 'ollama'
                return llm
                
            elif model_type == "huggingface":
                if not Config.HUGGINGFACE_API_KEY:
                    raise Exception("Hugging Face API key not configured")
                
                model_to_use = model_name or Config.HUGGINGFACE_DEFAULT_MODEL
                
                llm = HuggingFaceHub(
                    repo_id=model_to_use,
                    model_kwargs={
                        "temperature": 0.7,
                        "max_length": 512,
                        "top_p": 0.9
                    },
                    huggingfacehub_api_token=Config.HUGGINGFACE_API_KEY
                )
                self.current_model = model_to_use
                self.current_model_type = 'huggingface'
                return llm
                
            else:
                raise Exception(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {str(e)}")
            raise e
    
    def get_embeddings(self):
        """Get embeddings model"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {str(e)}")
            raise e
    
    def create_conversation_chain(self, text_data, model_name=None, model_type="ollama"):
        """Create a conversation chain"""
        try:
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            chunks = text_splitter.split_text(text_data)
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # Create vector store
            embeddings = self.get_embeddings()
            vectorstore = FAISS.from_documents(documents, embeddings)
            
            # Initialize LLM
            llm = self.initialize_llm(model_name, model_type)
            
            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create conversation chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                return_source_documents=True,
                output_key="answer"
            )
            
            # Store the chain
            chain_id = f"{model_type}_{model_name or 'default'}"
            self.conversation_chains[chain_id] = {
                'chain': chain,
                'vectorstore': vectorstore,
                'model_name': model_name,
                'model_type': model_type
            }
            
            logger.info(f"✅ Conversation chain created: {chain_id}")
            return chain_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create conversation chain: {str(e)}")
            raise e
    
    def chat(self, chain_id, question):
        """Send a question to the chatbot"""
        try:
            if chain_id not in self.conversation_chains:
                return "Error: Conversation chain not found. Please extract data first."
            
            chain_data = self.conversation_chains[chain_id]
            response = chain_data['chain'].invoke({"question": question})
            
            answer = response.get("answer", "I couldn't generate a response.")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Chat error: {str(e)}")
            return f"Error: {str(e)}"
    
    def clear_conversation(self, chain_id):
        """Clear conversation history"""
        try:
            if chain_id in self.conversation_chains:
                chain_data = self.conversation_chains[chain_id]
                vectorstore = chain_data['vectorstore']
                model_name = chain_data['model_name']
                model_type = chain_data['model_type']
                
                # Create new chain with fresh memory
                llm = self.initialize_llm(model_name, model_type)
                
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
                
                new_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    memory=memory,
                    return_source_documents=True,
                    output_key="answer"
                )
                
                self.conversation_chains[chain_id]['chain'] = new_chain
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to clear conversation: {str(e)}")
            return False

# Global instance
chatbot_manager = ChatbotManager()
