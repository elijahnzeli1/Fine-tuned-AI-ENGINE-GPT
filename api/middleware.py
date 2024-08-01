from functools import wraps
from flask import request, jsonify, current_app
import time
import logging
from werkzeug.exceptions import HTTPException
import secrets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            logger.warning("Request made without API key")
            return jsonify({"error": "Missing API key"}), 401
        
        if not secrets.compare_digest(api_key, current_app.config['API_KEY']):
            logger.warning(f"Invalid API key used: {api_key[:5]}...")
            return jsonify({"error": "Invalid API key"}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        response = f(*args, **kwargs)
        duration = time.time() - start_time
        
        logger.info(
            f"Request: {request.method} {request.path} | "
            f"Status: {response.status_code} | "
            f"Duration: {duration:.2f}s | "
            f"IP: {request.remote_addr}"
        )
        return response
    return decorated_function

def handle_exceptions(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except HTTPException as e:
            logger.error(f"HTTP exception occurred: {str(e)}")
            return jsonify({"error": str(e)}), e.code
        except Exception as e:
            logger.exception("An unexpected error occurred")
            return jsonify({"error": "An unexpected error occurred"}), 500
    return decorated_function

def rate_limit(limit: int, per: int):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(decorated_function, 'last_called'):
                decorated_function.last_called = {}
            
            current_time = time.time()
            ip_address = request.remote_addr
            
            if ip_address in decorated_function.last_called:
                last_called, call_count = decorated_function.last_called[ip_address]
                if current_time - last_called < per:
                    if call_count >= limit:
                        logger.warning(f"Rate limit exceeded for IP: {ip_address}")
                        return jsonify({"error": "Rate limit exceeded"}), 429
                    decorated_function.last_called[ip_address] = (last_called, call_count + 1)
                else:
                    decorated_function.last_called[ip_address] = (current_time, 1)
            else:
                decorated_function.last_called[ip_address] = (current_time, 1)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_json(*expected_args):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                logger.warning("Request does not contain valid JSON")
                return jsonify({"error": "Request must be JSON"}), 400
            
            data = request.get_json()
            for arg in expected_args:
                if arg not in data:
                    logger.warning(f"Missing required argument in request: {arg}")
                    return jsonify({"error": f"Missing required argument: {arg}"}), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator