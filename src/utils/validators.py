"""
Input Validation Module
Validates handwriting recognition API inputs
"""

import base64
from io import BytesIO
from PIL import Image
from functools import wraps
from flask import request, jsonify

# Configuration constants
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_FORMATS = {'PNG', 'JPG', 'JPEG'}
MIN_RESOLUTION = (50, 50)
MAX_RESOLUTION = (4000, 4000)


def validate_image(f):
    """
    Decorator to validate image input for /predict_handwriting endpoint
    
    Validates:
    - Image parameter exists
    - Base64 encoding is valid
    - File size <= 5MB
    - Image format is PNG, JPG, or JPEG
    - Image resolution is between MIN and MAX
    
    Returns 400 with clear error message if validation fails
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Check if data exists
            if not request.is_json:
                return jsonify({
                    'error': 'Request must be JSON',
                    'code': 'INVALID_FORMAT'
                }), 400
            
            data = request.get_json()
            
            # Check if image parameter exists
            if 'image' not in data:
                return jsonify({
                    'error': 'Missing required parameter: image',
                    'code': 'MISSING_IMAGE'
                }), 400
            
            image_data = data['image']
            
            # Validate base64 format
            try:
                # Extract base64 data (remove 'data:image/png;base64,' prefix if present)
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                return jsonify({
                    'error': 'Invalid base64 encoding',
                    'code': 'INVALID_BASE64'
                }), 400
            
            # Check file size
            file_size = len(image_bytes)
            if file_size == 0:
                return jsonify({
                    'error': 'Image data is empty',
                    'code': 'EMPTY_IMAGE'
                }), 400
            
            if file_size > MAX_IMAGE_SIZE:
                return jsonify({
                    'error': f'Image size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum ({MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB)',
                    'code': 'IMAGE_TOO_LARGE'
                }), 400
            
            # Validate image format and resolution
            try:
                image = Image.open(BytesIO(image_bytes))
                
                # Check format
                img_format = image.format.upper() if image.format else 'UNKNOWN'
                if img_format not in ALLOWED_FORMATS:
                    return jsonify({
                        'error': f'Image format {img_format} not supported. Supported formats: {", ".join(ALLOWED_FORMATS)}',
                        'code': 'UNSUPPORTED_FORMAT'
                    }), 400
                
                # Check resolution
                width, height = image.size
                
                if (width, height) < MIN_RESOLUTION:
                    return jsonify({
                        'error': f'Image resolution {width}x{height} is too small. Minimum: {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]}',
                        'code': 'RESOLUTION_TOO_SMALL'
                    }), 400
                
                if (width, height) > MAX_RESOLUTION:
                    return jsonify({
                        'error': f'Image resolution {width}x{height} is too large. Maximum: {MAX_RESOLUTION[0]}x{MAX_RESOLUTION[1]}',
                        'code': 'RESOLUTION_TOO_LARGE'
                    }), 400
                
            except Exception as e:
                return jsonify({
                    'error': f'Unable to read image: {str(e)}',
                    'code': 'INVALID_IMAGE'
                }), 400
            
            # Validation passed, call original function
            return f(*args, **kwargs)
        
        except Exception as e:
            # Unexpected error in validator
            return jsonify({
                'error': 'Validation error',
                'code': 'VALIDATION_ERROR',
                'details': str(e)
            }), 400
    
    return decorated_function


def validate_parameters(f):
    """
    Decorator to validate optional parameters
    
    Validates:
    - mode: 'single' or 'multi'
    - decode_mode: 'greedy' or 'beam'
    - beam_width: integer between 1-50
    - spellcheck: boolean
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            data = request.get_json() or {}
            
            # Validate mode
            mode = data.get('mode', 'single')
            if mode not in ['single', 'multi']:
                return jsonify({
                    'error': f'Invalid mode: {mode}. Must be "single" or "multi"',
                    'code': 'INVALID_MODE'
                }), 400
            
            # Validate decode_mode
            decode_mode = data.get('decode_mode', 'greedy')
            if decode_mode not in ['greedy', 'beam']:
                return jsonify({
                    'error': f'Invalid decode_mode: {decode_mode}. Must be "greedy" or "beam"',
                    'code': 'INVALID_DECODE_MODE'
                }), 400
            
            # Validate beam_width
            try:
                beam_width = int(data.get('beam_width', 3))
                if beam_width < 1 or beam_width > 50:
                    return jsonify({
                        'error': f'Invalid beam_width: {beam_width}. Must be between 1 and 50',
                        'code': 'INVALID_BEAM_WIDTH'
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    'error': 'beam_width must be an integer',
                    'code': 'INVALID_BEAM_WIDTH'
                }), 400
            
            # Validate spellcheck
            spellcheck = data.get('spellcheck', True)
            if not isinstance(spellcheck, bool):
                return jsonify({
                    'error': 'spellcheck must be boolean (true/false)',
                    'code': 'INVALID_SPELLCHECK'
                }), 400
            
            return f(*args, **kwargs)
        
        except Exception as e:
            return jsonify({
                'error': 'Parameter validation error',
                'code': 'PARAM_VALIDATION_ERROR'
            }), 400
    
    return decorated_function
