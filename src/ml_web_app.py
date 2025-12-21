#!/usr/bin/env python3
"""
Web Application cho ML Models Fire Detection
Cho phép upload ảnh và test với các mô hình đã train
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import base64
from datetime import datetime
import uuid
import json
from ml_models import MLModelTrainer
from fire_feature_extractor import FireFeatureExtractor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Khởi tạo trainer và load models
# Use path relative to this file so it works regardless of current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
trainer = MLModelTrainer(models_dir=MODELS_DIR)
feature_extractor = FireFeatureExtractor()

# Load models mới nhất nếu có
def load_latest_models():
    """Load models mới nhất từ thư mục trained_models"""
    models_dir = MODELS_DIR
    if not os.path.exists(models_dir):
        print("❌ Không tìm thấy thư mục trained_models")
        return False
    
    # Tìm timestamp mới nhất
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and not f.startswith('scaler')]
    if not model_files:
        print("❌ Không tìm thấy models đã train")
        return False
    
    # Lấy timestamp từ file đầu tiên và đảm bảo format đúng
    first_file = model_files[0]
    # Tìm timestamp đầy đủ (YYYYMMDD_HHMMSS)
    # Format: ModelName_YYYYMMDD_HHMMSS.pkl
    parts = first_file.split('_')
    if len(parts) >= 3:
        # Lấy 2 phần cuối: YYYYMMDD_HHMMSS
        timestamp = f"{parts[-2]}_{parts[-1].replace('.pkl', '')}"
    elif len(parts) >= 2:
        # Fallback: chỉ lấy phần cuối
        timestamp = parts[-1].replace('.pkl', '')
    else:
        print("❌ Không thể parse timestamp từ filename")
        return False
    
    print(f"🔍 Tìm thấy timestamp: {timestamp}")
    print(f"📁 Model files: {model_files[:3]}...")  # Hiển thị 3 files đầu
    
    try:
        trainer.load_models(timestamp)
        print(f"✅ Đã load models từ timestamp: {timestamp}")
        print(f"📊 Models loaded: {list(trainer.trained_models.keys())}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi load models: {e}")
        return False

# Thử load models khi khởi động
print("🚀 Khởi động ML Web Application...")
models_loaded = load_latest_models()
if models_loaded:
    print(f"📊 Models đã load: {list(trainer.trained_models.keys())}")
else:
    print("📊 Models đã load: Không có")

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('ml_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Xử lý upload ảnh và dự đoán"""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được upload'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'})
    
    if file:
        # Tạo tên file unique
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Kiểm tra xem có models nào được load không
            if not trainer.trained_models:
                return jsonify({'error': 'Chưa có models nào được load. Vui lòng load models trước.'})
            
            # Dự đoán với tất cả models
            predictions = trainer.predict_single_image(filepath)
            
            # Chuyển đổi numpy types sang native types
            def convert(obj):
                if isinstance(obj, np.generic):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            
            predictions = convert(predictions)
            
            # Đọc ảnh để encode base64
            with open(filepath, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Tạo response
            response = {
                'success': True,
                'image': encoded_image,
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            # Tìm model tốt nhất
            best_model = max(predictions.items(), key=lambda x: x[1]['confidence'])
            response['best_model'] = {
                'name': best_model[0],
                'prediction': best_model[1]['prediction'],
                'confidence': best_model[1]['confidence']
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Lỗi khi phân tích ảnh: {str(e)}'})

@app.route('/uploaded_file/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    models_loaded = len(trainer.trained_models) > 0
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'num_models': len(trainer.trained_models),
        'model_names': list(trainer.trained_models.keys()) if models_loaded else []
    })

@app.route('/models')
def get_models():
    """Lấy thông tin về các models đã load"""
    models_info = {}
    for name, model in trainer.trained_models.items():
        models_info[name] = {
            'type': type(model).__name__,
            'parameters': str(model.get_params()) if hasattr(model, 'get_params') else 'N/A'
        }
    
    return jsonify({
        'models_loaded': len(trainer.trained_models),
        'models': models_info
    })

@app.route('/load-models', methods=['POST'])
def load_models_endpoint():
    """Load models từ timestamp"""
    data = request.get_json()
    timestamp = data.get('timestamp')
    
    if not timestamp:
        return jsonify({'error': 'Timestamp không được cung cấp'})
    
    try:
        success = load_latest_models() if timestamp == 'latest' else trainer.load_models(timestamp)
        if success:
            return jsonify({
                'success': True,
                'message': f'Đã load models thành công',
                'models': list(trainer.trained_models.keys())
            })
        else:
            return jsonify({'error': 'Không thể load models'})
    except Exception as e:
        return jsonify({'error': f'Lỗi khi load models: {str(e)}'})

@app.route('/train-status')
def train_status():
    """Kiểm tra trạng thái training"""
    models_dir = MODELS_DIR
    if not os.path.exists(models_dir):
        return jsonify({'status': 'no_models', 'message': 'Chưa có models nào được train'})
    
    # Tìm models có sẵn
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and not f.startswith('scaler')]
    if not model_files:
        return jsonify({'status': 'no_models', 'message': 'Không tìm thấy models'})
    
    # Lấy thông tin về models
    timestamps = []
    for f in model_files:
        parts = f.split('_')
        if len(parts) >= 3:
            timestamp = f"{parts[-2]}_{parts[-1].replace('.pkl', '')}"
        else:
            timestamp = parts[-1].replace('.pkl', '')
        timestamps.append(timestamp)
    
    timestamps = list(set(timestamps))
    timestamps.sort(reverse=True)  # Sắp xếp theo thời gian mới nhất
    
    # Kiểm tra xem có models nào đã được load không
    models_loaded = len(trainer.trained_models) > 0
    
    return jsonify({
        'status': 'models_available',
        'available_timestamps': timestamps,
        'latest_timestamp': timestamps[0] if timestamps else None,
        'models_loaded': models_loaded,
        'num_models_loaded': len(trainer.trained_models),
        'model_names': list(trainer.trained_models.keys()) if models_loaded else []
    })

if __name__ == '__main__':
    # Allow cloud platforms (Amplify, etc.) to provide the port via env var
    port = int(os.environ.get('PORT', '8080'))
    # Enable/disable debug via env: DEBUG=true/false
    debug_flag = str(os.environ.get('DEBUG', 'false')).lower() == 'true'
    print("🚀 Khởi động ML Web Application...")
    print("📊 Models đã load:", list(trainer.trained_models.keys()) if trainer.trained_models else "Không có")
    print(f"🌐 Truy cập: http://localhost:{port}")
    app.run(debug=debug_flag, host='0.0.0.0', port=port)