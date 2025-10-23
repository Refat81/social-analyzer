import requests
from bs4 import BeautifulSoup
import re
import logging
from urllib.parse import urlparse
from utils.chatbot_manager import chatbot_manager

logger = logging.getLogger(__name__)

class LinkedInExtractor:
    def __init__(self):
        self.extracted_data = None
        self.conversation_chain_id = None
        
    def extract_data(self, url, data_type="profile", model_name="llama2", model_type="ollama"):
        """Extract data from LinkedIn URL"""
        try:
            logger.info(f"🔗 Extracting LinkedIn {data_type} from: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                return {"error": f"Failed to access page (Status: {response.status_code})"}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Filter meaningful content
            paragraphs = [p.strip() for p in clean_text.split('\n') if len(p.strip()) > 30]
            meaningful_content = [p for p in paragraphs if self._is_meaningful_content(p)]
            
            if not meaningful_content:
                return {"error": "No meaningful content found"}
            
            # Prepare result
            result = {
                "data_type": data_type,
                "url": url,
                "content": meaningful_content[:10],
                "total_blocks": len(meaningful_content),
                "model_used": f"{model_type}: {model_name}",
                "status": "success"
            }
            
            self.extracted_data = result
            
            # Setup chatbot
            chatbot_text = self._prepare_for_chatbot(result)
            self.conversation_chain_id = chatbot_manager.create_conversation_chain(
                chatbot_text, model_name, model_type
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Extraction error: {str(e)}")
            return {"error": f"Extraction failed: {str(e)}"}
    
    def _is_meaningful_content(self, text):
        """Check if content is meaningful"""
        excluded = ['cookie', 'privacy', 'terms', 'sign in', 'login', 'linkedin', '©']
        text_lower = text.lower()
        
        if len(text_lower) < 20:
            return False
            
        if any(pattern in text_lower for pattern in excluded):
            return False
            
        return True
    
    def _prepare_for_chatbot(self, data):
        """Prepare data for chatbot"""
        text = f"LINKEDIN {data['data_type'].upper()} DATA\n\n"
        text += f"URL: {data['url']}\n"
        text += f"Total Content Blocks: {data['total_blocks']}\n"
        text += f"Model: {data['model_used']}\n\n"
        text += "EXTRACTED CONTENT:\n" + "="*50 + "\n"
        
        for i, content in enumerate(data['content'], 1):
            text += f"\nBlock {i}:\n{content}\n" + "-"*40 + "\n"
        
        return text
    
    def chat(self, question):
        """Chat with extracted data"""
        if not self.conversation_chain_id:
            return "Please extract data first."
        
        return chatbot_manager.chat(self.conversation_chain_id, question)
    
    def clear_chat(self):
        """Clear chat history"""
        if self.conversation_chain_id:
            chatbot_manager.clear_conversation(self.conversation_chain_id)
            return "Chat history cleared!"
        return "No active chat session."
