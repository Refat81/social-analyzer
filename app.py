from flask import Flask, render_template, request, jsonify, session
import logging
from config import Config
from utils.linkedin_extractor import LinkedInExtractor
from utils.chatbot_manager import chatbot_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY

# Initialize extractors
linkedin_extractor = LinkedInExtractor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/linkedin')
def linkedin_page():
    return render_template('linkedin.html')

@app.route('/facebook')
def facebook_page():
    return render_template('facebook.html')

@app.route('/facebook-pro')
def facebook_pro_page():
    return render_template('facebook_pro.html')

@app.route('/api/status')
def system_status():
    try:
        hf_status = chatbot_manager.check_huggingface_status()
        
        return jsonify({
            'huggingface_available': hf_status,
            'available_models': chatbot_manager.available_models,
            'current_model': chatbot_manager.current_model,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/linkedin/extract', methods=['POST'])
def extract_linkedin():
    try:
        data = request.json
        url = data.get('url')
        data_type = data.get('data_type', 'profile')
        model_name = data.get('model_name', Config.HUGGINGFACE_DEFAULT_MODEL)
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Validate Hugging Face
        if not Config.HUGGINGFACE_API_KEY:
            return jsonify({'error': 'Hugging Face API key not configured'}), 400
        
        result = linkedin_extractor.extract_data(url, data_type, model_name, 'huggingface')
        
        if result.get('error'):
            return jsonify({'error': result['error']}), 400
        
        session['linkedin_extracted'] = True
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/linkedin/chat', methods=['POST'])
def linkedin_chat():
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        response = linkedin_extractor.chat(question)
        return jsonify({'answer': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Facebook endpoints (not available in deployment)
@app.route('/api/facebook/login', methods=['POST'])
def facebook_login():
    return jsonify({
        'error': 'Facebook extraction requires local setup',
        'message': 'Run the app locally for Facebook features'
    }), 400

@app.route('/api/facebook/extract', methods=['POST'])
def extract_facebook():
    return jsonify({
        'error': 'Facebook extraction requires local setup', 
        'message': 'Run the app locally for Facebook features'
    }), 400

if __name__ == '__main__':
    # Validate config
    try:
        if Config.DEPLOYMENT:
            Config.validate_config()
        
        # Check Hugging Face
        if chatbot_manager.check_huggingface_status():
            logger.info("✅ Hugging Face API is ready!")
        else:
            logger.error("❌ Hugging Face API not available")
            
    except Exception as e:
        logger.error(f"Configuration error: {e}")
    
    app.run(debug=not Config.DEPLOYMENT, host='0.0.0.0', port=5000)
