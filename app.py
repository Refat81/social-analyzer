from flask import Flask, render_template, request, jsonify, session
import logging
from config import Config
from utils.linkedin_extractor import LinkedInExtractor
from utils.facebook_extractor import FacebookExtractor
from utils.facebook_pro_extractor import FacebookProExtractor
from utils.chatbot_manager import chatbot_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY

# Initialize extractors
linkedin_extractor = LinkedInExtractor()
facebook_extractor = FacebookExtractor()
facebook_pro_extractor = FacebookProExtractor()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/linkedin')
def linkedin_page():
    """LinkedIn analyzer page"""
    return render_template('linkedin.html')

@app.route('/facebook')
def facebook_page():
    """Facebook analyzer page"""
    return render_template('facebook.html')

@app.route('/facebook-pro')
def facebook_pro_page():
    """Facebook Pro analyzer page"""
    return render_template('facebook_pro.html')

# System Status API
@app.route('/api/status')
def system_status():
    """Get system status"""
    try:
        ollama_status = chatbot_manager.check_ollama_status()
        huggingface_status = chatbot_manager.check_huggingface_status()
        
        return jsonify({
            'ollama_available': ollama_status,
            'huggingface_available': huggingface_status,
            'available_models': chatbot_manager.available_models,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Model Management APIs
@app.route('/api/models/available')
def get_available_models():
    """Get all available models"""
    try:
        models = chatbot_manager.get_available_models()
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# LinkedIn APIs
@app.route('/api/linkedin/extract', methods=['POST'])
def extract_linkedin():
    """Extract LinkedIn data"""
    try:
        data = request.json
        url = data.get('url')
        data_type = data.get('data_type', 'profile')
        model_name = data.get('model_name', 'llama2')
        model_type = data.get('model_type', 'ollama')
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        result = linkedin_extractor.extract_data(url, data_type, model_name, model_type)
        
        if result.get('error'):
            return jsonify({'error': result['error']}), 400
        
        session['linkedin_extracted'] = True
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/linkedin/chat', methods=['POST'])
def linkedin_chat():
    """Chat with LinkedIn data"""
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        response = linkedin_extractor.chat(question)
        return jsonify({'answer': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Facebook APIs
@app.route('/api/facebook/login', methods=['POST'])
def facebook_login():
    """Start Facebook login"""
    try:
        # Setup driver
        result = facebook_extractor.setup_driver()
        if not result.get('success'):
            return jsonify({'error': result.get('error')}), 400
        
        # Open Facebook for manual login
        login_result = facebook_extractor.manual_login()
        if not login_result.get('success'):
            return jsonify({'error': login_result.get('error')}), 400
        
        return jsonify({'message': 'Browser opened. Please login manually.'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/facebook/check-login', methods=['POST'])
def check_facebook_login():
    """Check Facebook login status"""
    try:
        if facebook_extractor.check_login_status():
            session['facebook_logged_in'] = True
            return jsonify({'logged_in': True, 'message': 'Successfully logged in!'})
        return jsonify({'logged_in': False, 'message': 'Not logged in yet'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/facebook/extract', methods=['POST'])
def extract_facebook():
    """Extract Facebook group data"""
    try:
        if not session.get('facebook_logged_in'):
            return jsonify({'error': 'Please login to Facebook first'}), 400
        
        data = request.json
        group_url = data.get('group_url')
        max_scrolls = data.get('max_scrolls', 5)
        model_name = data.get('model_name', 'llama2')
        model_type = data.get('model_type', 'ollama')
        
        if not group_url:
            return jsonify({'error': 'Group URL is required'}), 400
        
        result = facebook_extractor.extract_group_data(group_url, max_scrolls, model_name, model_type)
        
        if result.get('status') == 'success':
            session['facebook_extracted'] = True
            return jsonify(result)
        else:
            return jsonify({'error': result.get('error')}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/facebook/chat', methods=['POST'])
def facebook_chat():
    """Chat with Facebook data"""
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        response = facebook_extractor.chat(question)
        return jsonify({'answer': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/facebook/close', methods=['POST'])
def close_facebook():
    """Close Facebook browser"""
    try:
        facebook_extractor.close()
        session['facebook_logged_in'] = False
        session['facebook_extracted'] = False
        return jsonify({'message': 'Browser closed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Facebook Pro APIs (similar to Facebook)
@app.route('/api/facebook-pro/login', methods=['POST'])
def facebook_pro_login():
    """Start Facebook Pro login"""
    try:
        result = facebook_pro_extractor.setup_driver()
        if not result.get('success'):
            return jsonify({'error': result.get('error')}), 400
        
        login_result = facebook_pro_extractor.manual_login()
        if not login_result.get('success'):
            return jsonify({'error': login_result.get('error')}), 400
        
        return jsonify({'message': 'Browser opened. Please login manually.'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/facebook-pro/extract', methods=['POST'])
def extract_facebook_pro():
    """Extract Facebook Pro data"""
    try:
        if not facebook_pro_extractor.is_logged_in:
            return jsonify({'error': 'Please login to Facebook first'}), 400
        
        data = request.json
        group_url = data.get('group_url')
        max_scrolls = data.get('max_scrolls', 5)
        model_name = data.get('model_name', 'llama2')
        model_type = data.get('model_type', 'ollama')
        
        if not group_url:
            return jsonify({'error': 'Group URL is required'}), 400
        
        result = facebook_pro_extractor.extract_group_data(group_url, max_scrolls, model_name, model_type)
        
        if result.get('status') == 'success':
            session['facebook_pro_extracted'] = True
            return jsonify(result)
        else:
            return jsonify({'error': result.get('error')}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/facebook-pro/chat', methods=['POST'])
def facebook_pro_chat():
    """Chat with Facebook Pro data"""
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        response = facebook_pro_extractor.chat(question)
        return jsonify({'answer': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check system status on startup
    chatbot_manager.get_available_models()
    
    # Start the application
    app.run(
        debug=not Config.DEPLOYMENT,
        host='0.0.0.0',
        port=5000
    )
