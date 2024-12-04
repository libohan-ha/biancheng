from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import openai
from dotenv import load_dotenv
import traceback
import logging
import time
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Debug: Print environment variables
api_key = os.environ.get("SAMBANOVA_API_KEY")
logging.debug(f"API Key: {api_key[:8]}... (length: {len(api_key) if api_key else 0})")

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
try:
    openai.api_key = os.environ.get("SAMBANOVA_API_KEY")
    openai.api_base = "https://api.sambanova.ai/v1"
    logging.debug("OpenAI client initialized successfully")
except Exception as e:
    logging.error(f"Error initializing OpenAI client: {str(e)}")
    raise

def retry_with_backoff(max_retries=3, initial_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for retry in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate limit exceeded" not in str(e).lower() or retry == max_retries - 1:
                        raise
                    logging.warning(f"Rate limit exceeded, retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
            return func(*args, **kwargs)
        return wrapper
    return decorator

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        logging.debug(f"Received message: {user_message}")
        
        if not os.environ.get("SAMBANOVA_API_KEY"):
            raise ValueError("API key not found in environment variables")
        
        @retry_with_backoff(max_retries=3, initial_delay=1)
        def make_api_call():
            return openai.ChatCompletion.create(
                model='Llama-3.2-90B-Vision-Instruct',
                messages=[
                    {"role": "system", "content": """你是一位善于循序渐进、互动教学的编程导师。
                    请按照以下格式回复：
                    1. 概念解释（简明易懂，100字以内）
                    2. 示例学习（提供3个由浅入深的示例）
                    3. 完型填空（提供3个由浅入深的代码填空题，每个题目都应该包含多个需要填写的关键点，用 _____ 表示。）
                    4. 实战练习（提供具体题目和提示）
                    5. 实战答案（提供实战练习的参考答案和详细解释）"""},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                top_p=0.1
            )
        
        response = make_api_call()
        response_data = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        logging.debug("Response generated successfully")
        
        return jsonify({
            "response": response_data
        })
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error occurred: {error_message}")
        
        if "rate limit exceeded" in error_message.lower():
            error_message = "服务器繁忙，请稍后再试"
        elif "API key" in error_message.lower():
            error_message = "API key configuration error. Please check your environment variables."
        
        return jsonify({
            "error": error_message
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
