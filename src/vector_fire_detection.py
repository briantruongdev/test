#!/usr/bin/env python3
"""
Hệ thống phát hiện lửa sử dụng Vector Database
So sánh đặc trưng của ảnh mới với database đã lưu
"""

import cv2
import numpy as np
import os
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any
import seaborn as sns

class FireFeatureExtractor:
    """Trích xuất đặc trưng từ ảnh"""
    
    def __init__(self):
        # Định nghĩa các ngưỡng màu lửa
        self.fire_color_ranges = {
            'red_lower': np.array([0, 100, 100]),
            'red_upper': np.array([10, 255, 255]),
            'orange_lower': np.array([10, 100, 100]),
            'orange_upper': np.array([25, 255, 255]),
            'yellow_lower': np.array([25, 100, 100]),
            'yellow_upper': np.array([35, 255, 255])
        }
    
    def preprocess_image(self, image_path: str) -> Dict[str, np.ndarray]:
        """Tiền xử lý ảnh"""
        # Load ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể load ảnh: {image_path}")
        
        # Resize về kích thước cố định
        image = cv2.resize(image, (224, 224))
        
        # Chuyển đổi màu
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return {
            'original': image,
            'rgb': rgb,
            'hsv': hsv,
            'gray': gray
        }
    
    def extract_color_histogram(self, hsv_image: np.ndarray) -> np.ndarray:
        """Trích xuất histogram màu sắc"""
        # Histogram cho từng kênh màu
        h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
        # Chuẩn hóa
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # Kết hợp thành một vector
        color_hist = np.concatenate([h_hist, s_hist, v_hist])
        return color_hist
    
    def extract_fire_color_mask(self, hsv_image: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """Trích xuất mask màu lửa"""
        # Tạo mask cho từng màu
        red_mask = cv2.inRange(hsv_image, self.fire_color_ranges['red_lower'], self.fire_color_ranges['red_upper'])
        orange_mask = cv2.inRange(hsv_image, self.fire_color_ranges['orange_lower'], self.fire_color_ranges['orange_upper'])
        yellow_mask = cv2.inRange(hsv_image, self.fire_color_ranges['yellow_lower'], self.fire_color_ranges['yellow_upper'])
        
        # Kết hợp mask
        fire_mask = cv2.bitwise_or(red_mask, orange_mask)
        fire_mask = cv2.bitwise_or(fire_mask, yellow_mask)
        
        # Tính tỷ lệ
        total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
        red_ratio = np.sum(red_mask > 0) / total_pixels
        orange_ratio = np.sum(orange_mask > 0) / total_pixels
        yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
        total_fire_ratio = np.sum(fire_mask > 0) / total_pixels
        
        fire_features = {
            'red_ratio': red_ratio,
            'orange_ratio': orange_ratio,
            'yellow_ratio': yellow_ratio,
            'total_fire_ratio': total_fire_ratio,
            'fire_pixels': np.sum(fire_mask > 0)
        }
        
        return fire_features, fire_mask
    
    def extract_texture_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Trích xuất đặc trưng texture"""
        # Gradient
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Thống kê gradient
        gradient_mean = np.mean(gradient_magnitude)
        gradient_std = np.std(gradient_magnitude)
        
        # Entropy
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Local Binary Pattern (đơn giản)
        lbp = self._compute_lbp(gray_image)
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
        
        texture_features = {
            'gradient_mean': gradient_mean,
            'gradient_std': gradient_std,
            'entropy': entropy,
            'lbp_histogram': lbp_hist
        }
        
        return texture_features
    
    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """Tính Local Binary Pattern đơn giản"""
        lbp = np.zeros_like(image)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                center = image[i, j]
                code = 0
                # 8 neighbors
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code += 2**k
                lbp[i, j] = code
        return lbp.astype(np.uint8)
    
    def extract_all_features(self, image_path: str) -> Dict[str, Any]:
        """Trích xuất tất cả đặc trưng"""
        # Tiền xử lý
        processed = self.preprocess_image(image_path)
        
        # Trích xuất đặc trưng
        color_hist = self.extract_color_histogram(processed['hsv'])
        fire_features, fire_mask = self.extract_fire_color_mask(processed['hsv'])
        texture_features = self.extract_texture_features(processed['gray'])
        
        # Tạo vector đặc trưng
        feature_vector = {
            'color_histogram': color_hist,
            'fire_features': fire_features,
            'texture_features': texture_features,
            'processed_images': processed,
            'fire_mask': fire_mask
        }
        
        return feature_vector

class VectorBasedFireClassifier:
    """Classifier dựa trên vector database"""
    
    def __init__(self, vector_db_path: str = "vector_database.pkl"):
        self.vector_db_path = vector_db_path
        self.vector_database = None
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_extractor = FireFeatureExtractor()
        
    def create_feature_vector(self, feature_data: Dict[str, Any]) -> np.ndarray:
        """Tạo vector đặc trưng từ dữ liệu đã trích xuất"""
        vectors = []
        
        # 1. Color histogram (692 features: 180 + 256 + 256)
        vectors.append(feature_data['color_histogram'])
        
        # 2. Fire features (5 features)
        fire_feat = list(feature_data['fire_features'].values())
        vectors.append(fire_feat)
        
        # 3. Texture features (3 + 256 features)
        texture_feat = [
            feature_data['texture_features']['gradient_mean'],
            feature_data['texture_features']['gradient_std'],
            feature_data['texture_features']['entropy']
        ]
        vectors.append(texture_feat)
        vectors.append(feature_data['texture_features']['lbp_histogram'])
        
        # Kết hợp tất cả thành một vector
        combined_vector = np.concatenate(vectors)
        return combined_vector
    
    def build_vector_database(self, dataset_path: str, labels_path: str = None):
        """Xây dựng vector database từ dataset"""
        print("🔨 Đang xây dựng vector database...")
        
        # Tìm tất cả ảnh trong dataset
        image_files = []
        labels = []
        
        # Tìm ảnh trong các thư mục con
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    image_files.append(image_path)
                    
                    # Xác định label từ tên file hoặc thư mục
                    if 'fire' in file.lower() or 'train_' in file.lower():
                        labels.append(1)  # Có lửa
                    else:
                        labels.append(0)  # Không có lửa
        
        print(f"📊 Tìm thấy {len(image_files)} ảnh")
        
        # Trích xuất đặc trưng
        all_vectors = []
        all_labels = []
        image_paths = []
        
        for i, (image_path, label) in enumerate(zip(image_files, labels)):
            try:
                print(f"🔍 Đang xử lý {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                
                # Trích xuất đặc trưng
                features = self.feature_extractor.extract_all_features(image_path)
                vector = self.create_feature_vector(features)
                
                all_vectors.append(vector)
                all_labels.append(label)
                image_paths.append(image_path)
                
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {image_path}: {e}")
                continue
        
        # Chuẩn hóa vectors
        all_vectors = np.array(all_vectors)
        scaled_vectors = self.scaler.fit_transform(all_vectors)
        
        # Huấn luyện classifier
        print("🎯 Đang huấn luyện classifier...")
        self.classifier.fit(scaled_vectors, all_labels)
        
        # Đánh giá
        predictions = self.classifier.predict(scaled_vectors)
        accuracy = accuracy_score(all_labels, predictions)
        print(f"✅ Độ chính xác: {accuracy:.3f}")
        
        # Lưu vector database
        vector_database = {
            'all_vectors': scaled_vectors,
            'all_labels': all_labels,
            'image_paths': image_paths,
            'feature_dimension': scaled_vectors.shape[1],
            'total_samples': len(all_labels)
        }
        
        with open(self.vector_db_path, 'wb') as f:
            pickle.dump(vector_database, f)
        
        # Lưu scaler và classifier
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.classifier, 'classifier.pkl')
        
        self.vector_database = vector_database
        print(f"💾 Vector database đã được lưu: {self.vector_db_path}")
        print(f"📊 Kích thước database: {len(all_vectors)} vectors")
        print(f"🔢 Chiều đặc trưng: {scaled_vectors.shape[1]}")
        
        return vector_database
    
    def load_vector_database(self):
        """Load vector database từ file"""
        if os.path.exists(self.vector_db_path):
            with open(self.vector_db_path, 'rb') as f:
                self.vector_database = pickle.load(f)
            
            # Load scaler và classifier
            if os.path.exists('scaler.pkl'):
                self.scaler = joblib.load('scaler.pkl')
            if os.path.exists('classifier.pkl'):
                self.classifier = joblib.load('classifier.pkl')
            
            print(f"✅ Vector database đã được load")
            print(f"📊 Kích thước: {self.vector_database['total_samples']} vectors")
            print(f"🔢 Chiều đặc trưng: {self.vector_database['feature_dimension']}")
            return True
        else:
            print("❌ Không tìm thấy vector database")
            return False
    
    def classify_new_image(self, image_path: str) -> Dict[str, Any]:
        """Phân loại ảnh mới"""
        if self.vector_database is None:
            raise ValueError("Vector database chưa được load")
        
        # Trích xuất đặc trưng
        features = self.feature_extractor.extract_all_features(image_path)
        vector = self.create_feature_vector(features)
        
        # Chuẩn hóa
        scaled_vector = self.scaler.transform([vector])[0]
        
        # Dự đoán
        prediction = self.classifier.predict([scaled_vector])[0]
        probability = self.classifier.predict_proba([scaled_vector])[0]
        
        # Tìm ảnh tương tự
        similarities = self._calculate_similarities(scaled_vector)
        
        result = {
            'image_path': image_path,
            'prediction': 'FIRE' if prediction == 1 else 'NO FIRE',
            'confidence': max(probability),
            'probability_fire': probability[1],
            'probability_no_fire': probability[0],
            'similar_images': similarities,
            'features': {
                'fire_ratio': features['fire_features']['total_fire_ratio'],
                'red_ratio': features['fire_features']['red_ratio'],
                'orange_ratio': features['fire_features']['orange_ratio'],
                'yellow_ratio': features['fire_features']['yellow_ratio'],
                'texture_entropy': features['texture_features']['entropy']
            }
        }
        
        return result
    
    def _calculate_similarities(self, query_vector: np.ndarray) -> List[Dict]:
        """Tính độ tương tự với các ảnh trong database"""
        similarities = []
        
        for i, stored_vector in enumerate(self.vector_database['all_vectors']):
            # Cosine similarity
            similarity = np.dot(query_vector, stored_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(stored_vector)
            )
            
            similarities.append({
                'index': i,
                'similarity': similarity,
                'label': self.vector_database['all_labels'][i],
                'image_path': self.vector_database['image_paths'][i]
            })
        
        # Sắp xếp theo độ tương tự giảm dần
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:5]  # Trả về top 5
    
    def visualize_similarities(self, image_path: str, save_path: str = None):
        """Visualize ảnh gốc và các ảnh tương tự"""
        result = self.classify_new_image(image_path)
        
        # Load ảnh gốc
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Tạo subplot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Ảnh gốc
        axes[0, 0].imshow(original)
        axes[0, 0].set_title(f"Ảnh gốc\n{result['prediction']} ({result['confidence']:.2f})")
        axes[0, 0].axis('off')
        
        # 5 ảnh tương tự nhất
        for i, similar in enumerate(result['similar_images'][:5]):
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            if row < 2:
                similar_img = cv2.imread(similar['image_path'])
                similar_img = cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB)
                similar_img = cv2.resize(similar_img, (224, 224))
                
                label = "FIRE" if similar['label'] == 1 else "NO FIRE"
                axes[row, col].imshow(similar_img)
                axes[row, col].set_title(f"Similar {i+1}\n{label} ({similar['similarity']:.3f})")
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"🖼️ Visualization đã được lưu: {save_path}")
        
        plt.show()
        return result

def main():
    """Hàm chính để test hệ thống"""
    classifier = VectorBasedFireClassifier()
    
    # Kiểm tra xem có vector database không
    if not classifier.load_vector_database():
        print("🔨 Cần xây dựng vector database trước...")
        print("📁 Đang tìm dataset...")
        
        # Tìm dataset
        dataset_path = "../dataset"
        if os.path.exists(dataset_path):
            classifier.build_vector_database(dataset_path)
        else:
            print("❌ Không tìm thấy dataset")
            return
    
    # Test với ảnh mới
    test_images = [
        "../dataset/train/images/train_1.jpg",
        "../dataset/train/images/train_7.jpg",
        "../dataset/train/images/train_100.jpg"
    ]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"\n🔍 Test với: {test_image}")
            result = classifier.classify_new_image(test_image)
            
            print(f"🎯 Kết quả: {result['prediction']}")
            print(f"📊 Độ tin cậy: {result['confidence']:.3f}")
            print(f"🔥 Xác suất có lửa: {result['probability_fire']:.3f}")
            print(f"❌ Xác suất không lửa: {result['probability_no_fire']:.3f}")
            print(f"📈 Tỷ lệ màu lửa: {result['features']['fire_ratio']:.3f}")
            
            # Hiển thị ảnh tương tự nhất
            top_similar = result['similar_images'][0]
            print(f"🖼️ Ảnh tương tự nhất: {os.path.basename(top_similar['image_path'])} (similarity: {top_similar['similarity']:.3f})")

if __name__ == "__main__":
    main() 