from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import base64
from datetime import datetime
import uuid
import json
from detailed_fire_analyzer import DetailedFireAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Khởi tạo analyzer
analyzer = DetailedFireAnalyzer(output_dir="results")

@app.route('/')
def index():
    return render_template('detailed_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        # Tạo tên file unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Lưu file
        file.save(filepath)
        
        try:
            # Phân tích chi tiết
            print(f"🔍 Bắt đầu phân tích: {filepath}")
            report = analyzer.analyze_image_step_by_step(filepath)
            
            # Tạo visualization
            step1 = analyzer._step1_load_and_preprocess(filepath)
            step2 = analyzer._step2_color_analysis(step1)
            step3 = analyzer._step3_fire_region_analysis(step1, step2)
            viz_path = analyzer.visualize_analysis(filepath, step1, step2, step3)
            
            # Đọc ảnh để encode base64
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Đọc visualization để encode base64
            with open(viz_path, 'rb') as viz_file:
                viz_data = base64.b64encode(viz_file.read()).decode('utf-8')
            
            # Chuẩn bị kết quả
            result = {
                'success': True,
                'image_name': filename,
                'image_data': f"data:image/jpeg;base64,{img_data}",
                'visualization_data': f"data:image/png;base64,{viz_data}",
                'analysis': {
                    'classification': report['final_classification']['classification'],
                    'confidence': report['final_classification']['confidence'],
                    'confidence_level': report['final_classification']['confidence_level'],
                    'score': report['final_classification']['score'],
                    'total_conditions': report['final_classification']['total_conditions'],
                    'reasoning': report['final_classification']['reasoning']
                },
                'detailed_results': {
                    'color_analysis': {
                        'red_ratio': report['step_by_step_results']['step2_color_analysis']['color_stats']['red']['percentage'],
                        'orange_ratio': report['step_by_step_results']['step2_color_analysis']['color_stats']['orange']['percentage'],
                        'yellow_ratio': report['step_by_step_results']['step2_color_analysis']['color_stats']['yellow']['percentage'],
                        'total_fire_ratio': f"{report['step_by_step_results']['step2_color_analysis']['total_fire_ratio']*100:.2f}%"
                    },
                    'fire_regions': {
                        'total_regions': report['step_by_step_results']['step3_fire_regions']['total_regions'],
                        'fire_area_ratio': f"{report['step_by_step_results']['step3_fire_regions']['fire_area_ratio']*100:.2f}%",
                        'avg_brightness': f"{report['step_by_step_results']['step3_fire_regions']['brightness_stats']['avg']:.1f}",
                        'max_brightness': f"{report['step_by_step_results']['step3_fire_regions']['brightness_stats']['max']:.1f}",
                        'avg_saturation': f"{report['step_by_step_results']['step3_fire_regions']['saturation_stats']['avg']:.1f}"
                    },
                    'texture_analysis': {
                        'gradient_entropy': f"{report['step_by_step_results']['step4_texture']['texture_stats']['gradient_entropy']:.2f}",
                        'gradient_mean': f"{report['step_by_step_results']['step4_texture']['texture_stats']['gradient_mean']:.2f}",
                        'gradient_std': f"{report['step_by_step_results']['step4_texture']['texture_stats']['gradient_std']:.2f}"
                    },
                    'histogram_analysis': {
                        'fire_hue_ratio': f"{report['step_by_step_results']['step5_histogram']['fire_hue_ratio']*100:.2f}%",
                        'high_saturation_ratio': f"{report['step_by_step_results']['step5_histogram']['saturation_ratio']*100:.2f}%",
                        'high_value_ratio': f"{report['step_by_step_results']['step5_histogram']['value_ratio']*100:.2f}%"
                    }
                },
                'conditions_check': report['final_classification']['conditions'],
                'thresholds': report['thresholds_used'],
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Error analyzing image: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'analyzer_loaded': True,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🔥 Detailed Fire Analysis Web App")
    print("=" * 50)
    print("Starting server...")
    print("Access the app at: http://localhost:8083")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=8083) 