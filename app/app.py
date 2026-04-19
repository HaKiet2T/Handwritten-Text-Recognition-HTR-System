import os
import sys
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure root folder is on sys.path so `src` imports work from app/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import handwriting model and utilities
from src.models.handwriting_model import (
    load_handwriting_model,
    decode_sequence,
    idx_to_char,
    SOS_IDX,
    EOS_IDX
)
from src.data.handwriting_preprocessing import (
    preprocess_handwriting_image,
    image_to_base64
)
from src.data.segmentation import segment_text_image, visualize_segmentation
from src.postprocessing.spellcheck import SpellCorrector
from src.postprocessing.enhanced_corrector import get_enhanced_corrector, correct_prediction

# Import validators and logging
from src.config.logging_config import setup_logging, ValidationError, ModelError, PreprocessingError
from src.utils.validators import validate_image, validate_parameters

# Đường dẫn đến mô hình đã huấn luyện
# iam_p3: Larger model (d=384, 6+4 layers) - Better for single chars
model_path = os.path.join(ROOT_DIR, 'weights', 'best_encoder_decoder.pth')

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup logging (before using logger)
logger = setup_logging()

logger.info(f"Loading model on {device}...")
logger.info(f"Model: {model_path}")
model = load_handwriting_model(model_path, device=device)

# Try to load a custom wordlist if present
custom_words = []
custom_wordlist_path = os.path.join(os.path.dirname(__file__), 'data', 'wordlist.txt')
if os.path.isfile(custom_wordlist_path):
    try:
        with open(custom_wordlist_path, 'r', encoding='utf-8') as f:
            custom_words = [l.strip() for l in f.readlines() if l.strip()]
    except Exception as e:
        logger.warning(f"Could not load custom wordlist: {e}")

# Initialize a SpellCorrector (default english, can be extended)
spell_corrector = SpellCorrector(language='en', custom_word_list=custom_words)

# Initialize Enhanced OCR Corrector (Phase 3 improvements)
try:
    enhanced_corrector = get_enhanced_corrector()
    logger.info("Phase 3 Enhanced OCR Corrector loaded successfully!")
except Exception as e:
    logger.warning(f"Could not load enhanced corrector: {e}")
    enhanced_corrector = None
try:
    from src.postprocessing.spellcheck import is_spellchecker_available
    if not is_spellchecker_available():
        logger.warning("SpellChecker backend (pyspellchecker) not available. Spellcheck is disabled until the package is installed.")
except Exception:
    # If import fails for diagnostics, continue silently (we already have fallback)
    pass

app = Flask(__name__)
CORS(app)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)


# Security headers middleware
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response


def predict_multi_word(image_np, decode_mode, beam_width, spellcheck_enabled):
    """
    Predict multi-line or multi-word text by segmenting and processing each word separately
    """
    try:
        logger.info(f"Multi-word mode: Segmenting image...")
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Segment into words
        segments = segment_text_image(gray, method='contours')
        logger.info(f"Found {len(segments)} word segments")
        
        # Debug: print line info
        lines_info = {}
        for _, _, line_idx, word_idx in segments:
            if line_idx not in lines_info:
                lines_info[line_idx] = 0
            lines_info[line_idx] += 1
        logger.info(f"Lines detected: {len(lines_info)}, Words per line: {lines_info}")
        
        if len(segments) == 0:
            return jsonify({'error': 'Không tìm thấy văn bản trong ảnh'}), 400
        
        # Visualize segmentation
        vis_img = visualize_segmentation(gray, segments)
        vis_b64 = image_to_base64(vis_img)
        
        # Process each word
        all_results = []
        reconstructed_text = []
        current_line = 0
        line_words = []
        
        for word_img, bbox, line_idx, word_idx in segments:
            try:
                # Preprocess word
                result = preprocess_handwriting_image(word_img, return_steps=False)
                if isinstance(result, tuple):
                    tensor, _ = result
                else:
                    tensor = result
                tensor = tensor.to(device)
                
                # Predict
                model.eval()
                with torch.no_grad():
                    result = model.generate(
                        tensor, SOS_IDX, EOS_IDX, 
                        max_len=27, 
                        mode=decode_mode, 
                        beam_width=beam_width,
                        verbose=False,
                        return_confidence=True
                    )
                    
                    if isinstance(result, tuple):
                        pred_tokens, confidences = result
                        confidence = confidences[0].item()
                    else:
                        pred_tokens = result
                        confidence = 0.95
                
                # Decode
                pred_text = decode_sequence(pred_tokens[0], idx_to_char)
                
                # Optional spellcheck
                if spellcheck_enabled:
                    try:
                        corrected = spell_corrector.correct_text(pred_text)
                        if corrected and corrected != pred_text:
                            pred_text = corrected
                    except:
                        pass
                
                # Store result
                x, y, w, h = bbox
                all_results.append({
                    'text': pred_text,
                    'confidence': confidence,
                    'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'line': int(line_idx),
                    'word': int(word_idx)
                })
                
                # Reconstruct text line by line
                if line_idx != current_line:
                    if line_words:
                        reconstructed_text.append(' '.join(line_words))
                    line_words = [pred_text]
                    current_line = line_idx
                else:
                    line_words.append(pred_text)
                
                print(f"  L{line_idx}W{word_idx}: '{pred_text}' (conf: {confidence:.2%})")
                
            except Exception as e:
                print(f"⚠️ Error processing word L{line_idx}W{word_idx}: {e}")
                continue
        
        # Add last line
        if line_words:
            reconstructed_text.append(' '.join(line_words))
        
        full_text = '\n'.join(reconstructed_text)
        
        print(f"✅ Multi-word prediction complete!")
        print(f"📝 Full text:\n{full_text}")
        
        return jsonify({
            'mode': 'multi',
            'text': full_text,
            'word_count': len(all_results),
            'line_count': current_line + 1,
            'words': all_results,
            'segmentation_image': vis_b64
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Multi-word error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Lỗi xử lý multi-word: {str(e)}'}), 500


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')


@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'model': 'loaded',
        'device': str(device)
    }), 200


@app.route('/predict_handwriting', methods=['POST'])
@limiter.limit("10 per minute")
@validate_parameters
@validate_image
def predict_handwriting():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Không có ảnh được gửi!'}), 400

        # Decode image
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)

        logger.info(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}, min: {image_np.min()}, max: {image_np.max()}")

        # Validate image
        if image_np.size == 0:
            return jsonify({'error': 'Ảnh rỗng!'}), 400

        if len(image_np.shape) not in [2, 3]:
            return jsonify({'error': f'Định dạng ảnh không hợp lệ: {image_np.shape}'}), 400

        # Get mode: 'single' or 'multi' (multi-line/multi-word)
        mode = data.get('mode', 'single')
        decode_mode = data.get('decode_mode', 'greedy')  # 'greedy' or 'beam'
        spellcheck_enabled = data.get('spellcheck', False)
        
        # Beam width for beam search (default top-3)
        try:
            beam_width = int(data.get('beam_width', 3))
            if beam_width < 1:
                beam_width = 1
            # Cap beam width to something reasonable (e.g., 50)
            beam_width = min(beam_width, 50)
        except Exception:
            beam_width = 3

        # Check if multi-line/multi-word mode
        if mode == 'multi':
            return predict_multi_word(image_np, decode_mode, beam_width, spellcheck_enabled)

        # Single word mode - Preprocess image with steps
        try:
            tensor, steps = preprocess_handwriting_image(image_np, return_steps=True)
            logger.info(f"Preprocessing done. Tensor shape: {tensor.shape}")
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}", exc_info=True)
            return jsonify({'error': f'Lỗi xử lý ảnh: {str(e)}'}), 500

        # Move to device
        tensor = tensor.to(device)

        # Predict
        decode_method = "Beam Search (top-10)" if decode_mode == 'beam' else "Greedy"
        logger.info(f"Running model inference with {decode_method}...")
        model.eval()
        with torch.no_grad():
            result = model.generate(tensor, SOS_IDX, EOS_IDX, max_len=27, mode=decode_mode, beam_width=beam_width, verbose=(decode_mode == 'beam'), return_confidence=True)

            # Unpack result
            if isinstance(result, tuple):
                pred_tokens, confidences = result
                confidence = confidences[0].item()
            else:
                pred_tokens = result
                confidence = 0.95  # Fallback

        # Decode
        pred_text = decode_sequence(pred_tokens[0], idx_to_char)
        logger.info(f"Predicted text: '{pred_text}' (using {decode_method}, confidence: {confidence:.2%})")

        # Optional spellcheck postprocessing (with Phase 3 enhancements)
        corrected_text = pred_text
        if spellcheck_enabled:
            try:
                # Try enhanced corrector first (Phase 3)
                if enhanced_corrector:
                    corrected_text = enhanced_corrector.correct(pred_text)
                    if corrected_text != pred_text:
                        logger.info(f"Phase 3 correction: '{pred_text}' -> '{corrected_text}'")
                else:
                    # Fallback to basic corrector
                    corrected_text = spell_corrector.correct_text(pred_text)
                    if corrected_text and corrected_text != pred_text:
                        logger.info(f"Spell correction: '{pred_text}' -> '{corrected_text}'")
                
                pred_text = corrected_text
            except Exception as e:
                logger.warning(f"Spellcheck error: {e}")

        # Convert steps to array format for frontend visualization
        processing_steps = []
        
        # Define step order and display names
        step_info = {
            '1_original': 'Ảnh gốc (Grayscale)',
            '2_resized': 'Resize về 256×64',
        }
        
        for key in sorted(steps.keys()):
            if key in step_info and isinstance(steps[key], np.ndarray):
                try:
                    img = steps[key]
                    processing_steps.append({
                        'name': step_info[key],
                        'image': image_to_base64(img),
                        'shape': list(img.shape) if hasattr(img, 'shape') else None
                    })
                except Exception as e:
                    logger.warning(f"Could not convert step {key} to base64: {e}")

        return jsonify({
            'mode': 'single',
            'text': pred_text,
            'confidence': confidence,
            'processing_steps': processing_steps
        })

    except Exception as e:
        logger.error(f"Predict error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Error Handlers
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 Not Found: {request.path}")
    return jsonify({'error': 'Endpoint not found', 'code': 'NOT_FOUND'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    logger.warning(f"405 Method Not Allowed: {request.method} {request.path}")
    return jsonify({'error': 'Method not allowed', 'code': 'METHOD_NOT_ALLOWED'}), 405


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 Internal Server Error: {str(error)}", exc_info=True)
    return jsonify({'error': 'Internal server error', 'code': 'INTERNAL_ERROR'}), 500


if __name__ == '__main__':
    # host='0.0.0.0' allows access from other devices on the network
    app.run(host='0.0.0.0', port=5000, debug=True)
