import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import re
from datetime import datetime
from typing import List
from utils.chatbot_manager import chatbot_manager

logger = logging.getLogger(__name__)

class FacebookExtractor:
    def __init__(self):
        self.driver = None
        self.wait = None
        self.is_logged_in = False
        self.extracted_data = None
        self.conversation_chain_id = None
        
    def setup_driver(self):
        """Setup Chrome driver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--start-maximized")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.wait = WebDriverWait(self.driver, 20)
            
            logger.info("✅ Chrome driver setup completed")
            return {"success": True, "message": "Browser ready for login"}
            
        except Exception as e:
            logger.error(f"❌ Driver setup failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def manual_login(self):
        """Open Facebook for manual login"""
        try:
            self.driver.get("https://www.facebook.com")
            time.sleep(3)
            self._handle_cookies()
            
            logger.info("✅ Facebook opened for manual login")
            return {"success": True, "message": "Please login manually in the browser"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def check_login_status(self):
        """Check if user is logged in"""
        try:
            current_url = self.driver.current_url.lower()
            if any(url in current_url for url in ['facebook.com/home', 'facebook.com/?sk']):
                self.is_logged_in = True
                return True
                
            # Check for profile elements
            indicators = ["//a[@aria-label='Profile']", "//div[@aria-label='Account']"]
            for indicator in indicators:
                try:
                    elements = self.driver.find_elements(By.XPATH, indicator)
                    if elements and elements[0].is_displayed():
                        self.is_logged_in = True
                        return True
                except:
                    continue
                    
            return False
            
        except Exception as e:
            logger.error(f"Login check error: {str(e)}")
            return False
    
    def extract_group_data(self, group_url, max_scrolls=5, model_name="llama2", model_type="ollama"):
        """Extract data from Facebook group"""
        try:
            if not self.is_logged_in:
                return {"error": "Please login first", "status": "error"}
            
            logger.info(f"🌐 Accessing group: {group_url}")
            
            self.driver.get(group_url)
            time.sleep(5)
            
            # Extract posts
            posts = self._extract_posts(max_scrolls)
            
            result = {
                "group_url": group_url,
                "posts": posts,
                "total_posts": len(posts),
                "extraction_time": datetime.now().isoformat(),
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
            logger.error(f"❌ Extraction failed: {str(e)}")
            return {"error": str(e), "status": "error"}
    
    def _extract_posts(self, max_scrolls):
        """Extract posts by scrolling"""
        posts = []
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        for i in range(max_scrolls):
            # Extract posts from current view
            current_posts = self._get_posts_from_page()
            for post in current_posts:
                if not self._is_duplicate(post, posts):
                    posts.append(post)
            
            # Scroll down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            
            # Check if reached end
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        return posts
    
    def _get_posts_from_page(self):
        """Get posts from current page"""
        posts = []
        try:
            # Look for post elements
            elements = self.driver.find_elements(By.XPATH, "//div[@role='article']")
            
            for element in elements:
                try:
                    text = element.text.strip()
                    if self._is_valid_post(text):
                        posts.append({
                            "content": text,
                            "timestamp": datetime.now().isoformat()
                        })
                except:
                    continue
                    
        except Exception as e:
            logger.debug(f"Post extraction error: {str(e)}")
            
        return posts
    
    def _is_valid_post(self, text):
        """Check if text is a valid post"""
        if not text or len(text) < 50:
            return False
            
        excluded = ['facebook', 'login', 'sign up', 'menu', 'navigation']
        text_lower = text.lower()
        
        if any(phrase in text_lower for phrase in excluded):
            return False
            
        return len(text.split()) >= 5
    
    def _is_duplicate(self, new_post, existing_posts):
        """Check for duplicate posts"""
        new_content = new_post["content"][:100]
        for existing in existing_posts:
            existing_content = existing["content"][:100]
            if new_content == existing_content:
                return True
        return False
    
    def _handle_cookies(self):
        """Handle cookie consent"""
        try:
            cookie_selectors = [
                "button[data-testid='cookie-policy-manage-dialog-accept-button']",
                "//button[contains(., 'Allow')]",
                "//button[contains(., 'Accept')]"
            ]
            
            for selector in cookie_selectors:
                try:
                    if selector.startswith("//"):
                        elements = self.driver.find_elements(By.XPATH, selector)
                    else:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for element in elements:
                        if element.is_displayed():
                            element.click()
                            time.sleep(2)
                            return
                except:
                    continue
        except:
            pass
    
    def _prepare_for_chatbot(self, data):
        """Prepare data for chatbot"""
        text = f"FACEBOOK GROUP DATA\n\n"
        text += f"Group URL: {data['group_url']}\n"
        text += f"Total Posts: {data['total_posts']}\n"
        text += f"Extraction Time: {data['extraction_time']}\n"
        text += f"Model: {data['model_used']}\n\n"
        text += "EXTRACTED POSTS:\n" + "="*50 + "\n"
        
        for i, post in enumerate(data['posts'], 1):
            text += f"\nPost {i}:\n{post['content']}\n" + "-"*40 + "\n"
        
        return text
    
    def chat(self, question):
        """Chat with extracted data"""
        if not self.conversation_chain_id:
            return "Please extract data first."
        
        return chatbot_manager.chat(self.conversation_chain_id, question)
    
    def close(self):
        """Close browser"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("✅ Browser closed")
            except:
                pass
