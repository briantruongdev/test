#!/usr/bin/env python3
"""
Fire Feature Extractor for ML Training
Trích xuất đặc trưng từ ảnh để training các mô hình ML
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class FireFeatureExtractor:
    """Trích xuất đặc trưng từ ảnh cho fire detection"""
    
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
        
        # Thống kê LBP
        lbp_mean = np.mean(lbp_hist)
        lbp_std = np.std(lbp_hist)
        
        texture_features = {
            'gradient_mean': gradient_mean,
            'gradient_std': gradient_std,
            'entropy': entropy,
            'lbp_mean': lbp_mean,
            'lbp_std': lbp_std,
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
    
    def extract_statistical_features(self, rgb_image: np.ndarray) -> Dict[str, float]:
        """Trích xuất đặc trưng thống kê"""
        # Thống kê từng kênh màu
        r_mean, r_std = np.mean(rgb_image[:,:,0]), np.std(rgb_image[:,:,0])
        g_mean, g_std = np.mean(rgb_image[:,:,1]), np.std(rgb_image[:,:,1])
        b_mean, b_std = np.mean(rgb_image[:,:,2]), np.std(rgb_image[:,:,2])
        
        # Tỷ lệ màu
        total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
        bright_pixels = np.sum(np.mean(rgb_image, axis=2) > 200)
        dark_pixels = np.sum(np.mean(rgb_image, axis=2) < 50)
        
        bright_ratio = bright_pixels / total_pixels
        dark_ratio = dark_pixels / total_pixels
        
        # Skewness và Kurtosis
        r_skew = self._calculate_skewness(rgb_image[:,:,0])
        g_skew = self._calculate_skewness(rgb_image[:,:,1])
        b_skew = self._calculate_skewness(rgb_image[:,:,2])
        
        statistical_features = {
            'r_mean': r_mean, 'r_std': r_std, 'r_skew': r_skew,
            'g_mean': g_mean, 'g_std': g_std, 'g_skew': g_skew,
            'b_mean': b_mean, 'b_std': b_std, 'b_skew': b_skew,
            'bright_ratio': bright_ratio,
            'dark_ratio': dark_ratio
        }
        
        return statistical_features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Tính skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def extract_all_features(self, image_path: str) -> Dict[str, Any]:
        """Trích xuất tất cả đặc trưng"""
        # Tiền xử lý
        processed = self.preprocess_image(image_path)
        
        # Trích xuất đặc trưng
        color_hist = self.extract_color_histogram(processed['hsv'])
        fire_features, fire_mask = self.extract_fire_color_mask(processed['hsv'])
        texture_features = self.extract_texture_features(processed['gray'])
        statistical_features = self.extract_statistical_features(processed['rgb'])
        
        # Tạo vector đặc trưng
        feature_vector = {
            'color_histogram': color_hist,
            'fire_features': fire_features,
            'texture_features': texture_features,
            'statistical_features': statistical_features,
            'processed_images': processed,
            'fire_mask': fire_mask
        }
        
        return feature_vector
    
    def create_feature_vector(self, feature_data: Dict[str, Any]) -> np.ndarray:
        """Tạo vector đặc trưng từ dữ liệu đã trích xuất"""
        vectors = []
        
        # 1. Color histogram (692 features: 180 + 256 + 256)
        vectors.append(feature_data['color_histogram'])
        
        # 2. Fire features (5 features)
        fire_feat = list(feature_data['fire_features'].values())
        vectors.append(fire_feat)
        
        # 3. Texture features (5 features)
        texture_feat = [
            feature_data['texture_features']['gradient_mean'],
            feature_data['texture_features']['gradient_std'],
            feature_data['texture_features']['entropy'],
            feature_data['texture_features']['lbp_mean'],
            feature_data['texture_features']['lbp_std']
        ]
        vectors.append(texture_feat)
        
        # 4. Statistical features (12 features)
        stat_feat = list(feature_data['statistical_features'].values())
        vectors.append(stat_feat)
        
        # Kết hợp tất cả thành một vector
        combined_vector = np.concatenate(vectors)
        return combined_vector

class DatasetLoader:
    """Load và chuẩn bị dataset cho training"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.feature_extractor = FireFeatureExtractor()
        
    def load_dataset(self, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load toàn bộ dataset và trích xuất features"""
        print(f"📁 Loading dataset từ: {self.dataset_path}")
        
        # Tìm tất cả ảnh trong dataset
        image_paths = []
        labels = []
        
        # Load từ train folder
        train_images_dir = os.path.join(self.dataset_path, 'train', 'images')
        
        if os.path.exists(train_images_dir):
            # Kiểm tra cấu trúc thư mục
            train_subdirs = [d for d in os.listdir(train_images_dir) if os.path.isdir(os.path.join(train_images_dir, d))]
            print(f"📁 Train subdirectories: {train_subdirs}")
            
            for subdir in train_subdirs:
                subdir_path = os.path.join(train_images_dir, subdir)
                images = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                print(f"📁 {subdir}: {len(images)} images")
                
                # Xác định label từ tên thư mục
                label = 1 if subdir.lower() == 'fire' else 0
                
                for img_file in images:
                    img_path = os.path.join(subdir_path, img_file)
                    image_paths.append(img_path)
                    labels.append(label)
        
        # Load từ val folder
        val_images_dir = os.path.join(self.dataset_path, 'val', 'images')
        
        if os.path.exists(val_images_dir):
            val_subdirs = [d for d in os.listdir(val_images_dir) if os.path.isdir(os.path.join(val_images_dir, d))]
            print(f"📁 Val subdirectories: {val_subdirs}")
            
            for subdir in val_subdirs:
                subdir_path = os.path.join(val_images_dir, subdir)
                images = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                print(f"📁 {subdir}: {len(images)} images")
                
                label = 1 if subdir.lower() == 'fire' else 0
                
                for img_file in images:
                    img_path = os.path.join(subdir_path, img_file)
                    image_paths.append(img_path)
                    labels.append(label)
        
        # Load từ test folder
        test_images_dir = os.path.join(self.dataset_path, 'test', 'images')
        
        if os.path.exists(test_images_dir):
            test_subdirs = [d for d in os.listdir(test_images_dir) if os.path.isdir(os.path.join(test_images_dir, d))]
            print(f"📁 Test subdirectories: {test_subdirs}")
            
            for subdir in test_subdirs:
                subdir_path = os.path.join(test_images_dir, subdir)
                images = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                print(f"📁 {subdir}: {len(images)} images")
                
                label = 1 if subdir.lower() == 'fire' else 0
                
                for img_file in images:
                    img_path = os.path.join(subdir_path, img_file)
                    image_paths.append(img_path)
                    labels.append(label)
        
        print(f"📊 Tổng cộng: {len(image_paths)} ảnh")
        
        # Giới hạn số lượng mẫu nếu cần
        if max_samples and len(image_paths) > max_samples:
            # Lấy ngẫu nhiên max_samples mẫu, đảm bảo cân bằng
            fire_indices = [i for i, label in enumerate(labels) if label == 1]
            no_fire_indices = [i for i, label in enumerate(labels) if label == 0]
            
            print(f"🔍 Debug: Fire indices: {len(fire_indices)}, No fire indices: {len(no_fire_indices)}")
            
            # Đảm bảo có cả hai class
            if len(fire_indices) == 0 or len(no_fire_indices) == 0:
                print("⚠️ Cảnh báo: Chỉ có một class trong dataset!")
                if len(fire_indices) == 0:
                    # Chỉ có no_fire
                    selected_indices = np.random.choice(no_fire_indices, min(len(no_fire_indices), max_samples), replace=False)
                    image_paths = [image_paths[int(i)] for i in selected_indices]
                    labels = [labels[int(i)] for i in selected_indices]
                    print(f"📊 Giới hạn còn {len(image_paths)} mẫu (Fire: 0, No Fire: {len(image_paths)})")
                else:
                    # Chỉ có fire
                    selected_indices = np.random.choice(fire_indices, min(len(fire_indices), max_samples), replace=False)
                    image_paths = [image_paths[int(i)] for i in selected_indices]
                    labels = [labels[int(i)] for i in selected_indices]
                    print(f"📊 Giới hạn còn {len(image_paths)} mẫu (Fire: {len(image_paths)}, No Fire: 0)")
            else:
                # Có cả hai class, cân bằng
                fire_samples = min(len(fire_indices), max_samples // 2)
                no_fire_samples = min(len(no_fire_indices), max_samples // 2)
                
                # Đảm bảo có ít nhất 1 mẫu từ mỗi class
                if fire_samples == 0:
                    fire_samples = 1
                    no_fire_samples = max_samples - 1
                elif no_fire_samples == 0:
                    no_fire_samples = 1
                    fire_samples = max_samples - 1
                
                # Đảm bảo không vượt quá max_samples
                total_samples = fire_samples + no_fire_samples
                if total_samples > max_samples:
                    if fire_samples > no_fire_samples:
                        fire_samples = max_samples - no_fire_samples
                    else:
                        no_fire_samples = max_samples - fire_samples
                
                selected_fire = np.random.choice(fire_indices, fire_samples, replace=False)
                selected_no_fire = np.random.choice(no_fire_indices, no_fire_samples, replace=False)
                
                selected_indices = np.concatenate([selected_fire, selected_no_fire])
                np.random.shuffle(selected_indices)
                
                image_paths = [image_paths[int(i)] for i in selected_indices]
                labels = [labels[int(i)] for i in selected_indices]
                print(f"📊 Giới hạn còn {len(image_paths)} mẫu (Fire: {fire_samples}, No Fire: {no_fire_samples})")
        
        # Trích xuất features
        print("🔍 Trích xuất features...")
        features = []
        valid_paths = []
        valid_labels = []
        
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            try:
                feature_vector = self.feature_extractor.extract_all_features(img_path)
                vector = self.feature_extractor.create_feature_vector(feature_vector)
                features.append(vector)
                valid_paths.append(img_path)
                valid_labels.append(label)
                
                if (i + 1) % 100 == 0:
                    print(f"📊 Đã xử lý {i + 1}/{len(image_paths)} ảnh")
                    
            except Exception as e:
                print(f"⚠️ Lỗi khi xử lý {img_path}: {e}")
                continue
        
        X = np.array(features)
        y = np.array(valid_labels)
        
        print(f"✅ Dataset đã được load thành công!")
        print(f"📊 Kích thước: {X.shape}")
        print(f"🎯 Số mẫu có lửa: {np.sum(y == 1)}")
        print(f"❌ Số mẫu không lửa: {np.sum(y == 0)}")
        
        return X, y, valid_paths 