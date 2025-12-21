#!/usr/bin/env python3
"""
Web Application cho hệ thống Vector Database Fire Detection
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import base64
from datetime import datetime
import uuid
import json
from vector_fire_detection import VectorBasedFireClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Khởi tạo classifier
classifier = VectorBasedFireClassifier()

@app.route('/')
def index():
    return render_template('vector_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        try:
            print(f"🔍 Bắt đầu phân tích vector: {filepath}")
            
            # Phân tích ảnh
            result = classifier.classify_new_image(filepath)
            
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
            
            # Chuyển đổi kết quả
            result = convert(result)
            
            # Load ảnh để hiển thị
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Tạo visualization
            viz_path = f"results/vector_analysis_{timestamp}_{unique_id}.png"
            os.makedirs("results", exist_ok=True)
            
            try:
                classifier.visualize_similarities(filepath, viz_path)
                with open(viz_path, 'rb') as viz_file:
                    viz_data = base64.b64encode(viz_file.read()).decode('utf-8')
                viz_available = True
            except Exception as e:
                print(f"❌ Lỗi tạo visualization: {e}")
                viz_data = ""
                viz_available = False
            
            response = {
                'success': True,
                'image_name': filename,
                'image_data': f"data:image/jpeg;base64,{img_data}",
                'visualization_available': viz_available,
                'visualization_data': f"data:image/png;base64,{viz_data}" if viz_available else "",
                'analysis': {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'probability_fire': result['probability_fire'],
                    'probability_no_fire': result['probability_no_fire']
                },
                'features': result['features'],
                'similar_images': [
                    {
                        'filename': os.path.basename(sim['image_path']),
                        'similarity': sim['similarity'],
                        'label': 'FIRE' if sim['label'] == 1 else 'NO FIRE'
                    }
                    for sim in result['similar_images'][:5]
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Error analyzing image: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'classifier_loaded': classifier.vector_database is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/build-database')
def build_database():
    """API để xây dựng vector database"""
    try:
        dataset_path = "../dataset"
        if os.path.exists(dataset_path):
            classifier.build_vector_database(dataset_path)
            return jsonify({
                'success': True,
                'message': 'Vector database built successfully',
                'database_size': classifier.vector_database['total_samples'] if classifier.vector_database else 0
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Dataset not found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("🔥 Vector Database Fire Detection Web App")
    print("=" * 50)
    print("Starting server...")
    print("Access the app at: http://localhost:8084")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=8084) 