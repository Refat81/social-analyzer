import requests
import logging
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
        self.available_models = {
            'huggingface': [
                "mistralai/Mistral-7B-Instruct-v0.1",
                "google/flan-t5-large", 
                "microsoft/DialoGPT-large",
                "facebook/blenderbot-400M-distill",
                "tiiuae/falcon-7b-instruct"
            ]
        }
        self.current_model = Config.HUGGINGFACE_DEFAULT_MODEL
        self.conversation_chains = {}
        
    def check_huggingface_status(self):
        """Check if Hugging Face is available"""
        try:
            if not Config.HUGGINGFACE_API_KEY:
                logger.error("❌ Hugging Face API key not configured")
                return False
            
            # Test the API key with a simple request
            headers = {"Authorization": f"Bearer {Config.HUGGINGFACE_API_KEY}"}
            response = requests.get(
                "https://huggingface.co/api/models/mistralai/Mistral-7B-Instruct-v0.1",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ Hugging Face API is working correctly")
                return True
            else:
                logger.error(f"❌ Hugging Face API test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Hugging Face connection error: {str(e)}")
            return False
    
    def get_available_models(self):
        """Get all available models"""
        self.check_huggingface_status()
        return self.available_models
    
    def initialize_llm(self, model_name=None):
        """Initialize Hugging Face language model"""
        try:
            if not Config.HUGGINGFACE_API_KEY:
                raise Exception("Hugging Face API key not configured")
            
            model_to_use = model_name or Config.HUGGINGFACE_DEFAULT_MODEL
            
            llm = HuggingFaceHub(
                repo_id=model_to_use,
                model_kwargs={
                    "temperature": 0.7,
                    "max_length": 512,
                    "max_new_tokens": 256,
                    "top_p": 0.9,
                    "do_sample": True
                },
                huggingfacehub_api_token=Config.HUGGINGFACE_API_KEY
            )
            
            self.current_model = model_to_use
            logger.info(f"✅ Hugging Face model initialized: {model_to_use}")
            return llm
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hugging Face model: {str(e)}")
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
    
    def create_conversation_chain(self, text_data, model_name=None):
        """Create a conversation chain with Hugging Face"""
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
            llm = self.initialize_llm(model_name)
            
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
            chain_id = f"hf_{model_name or 'default'}"
            self.conversation_chains[chain_id] = {
                'chain': chain,
                'vectorstore': vectorstore,
                'model_name': model_name or Config.HUGGINGFACE_DEFAULT_MODEL
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
                
                # Create new chain with fresh memory
                llm = self.initialize_llm(model_name)
                
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
                logger.info(f"✅ Conversation cleared for: {chain_id}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to clear conversation: {str(e)}")
            return False

# Global instance
chatbot_manager = ChatbotManager()
