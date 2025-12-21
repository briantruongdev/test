import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any
import seaborn as sns

class DetailedFireAnalyzer:
    """
    Hệ thống phân tích lửa chi tiết với từng bước rõ ràng
    """
    
    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Định nghĩa các ngưỡng màu lửa
        self.fire_color_ranges = {
            'red_lower': np.array([0, 100, 100]),
            'red_upper': np.array([10, 255, 255]),
            'orange_lower': np.array([10, 100, 100]),
            'orange_upper': np.array([25, 255, 255]),
            'yellow_lower': np.array([25, 100, 100]),
            'yellow_upper': np.array([35, 255, 255])
        }
        
        # Ngưỡng phân loại
        self.thresholds = {
            'fire_ratio_min': 0.02,  # 2% diện tích tối thiểu
            'fire_brightness_min': 150,  # Độ sáng tối thiểu
            'fire_saturation_min': 100,  # Độ bão hòa tối thiểu
            'texture_entropy_min': 4.0,  # Entropy texture tối thiểu
            'color_histogram_fire_ratio_min': 0.1  # Tỷ lệ màu lửa trong histogram
        }
    
    def analyze_image_step_by_step(self, image_path: str) -> Dict[str, Any]:
        """
        Phân tích ảnh từng bước chi tiết
        """
        print(f"🔥 Bắt đầu phân tích ảnh: {image_path}")
        
        # Bước 1: Load và preprocess ảnh
        step1_result = self._step1_load_and_preprocess(image_path)
        
        # Bước 2: Phân tích màu sắc
        step2_result = self._step2_color_analysis(step1_result)
        
        # Bước 3: Phân tích vùng lửa
        step3_result = self._step3_fire_region_analysis(step1_result, step2_result)
        
        # Bước 4: Phân tích texture
        step4_result = self._step4_texture_analysis(step1_result)
        
        # Bước 5: Phân tích histogram
        step5_result = self._step5_histogram_analysis(step1_result)
        
        # Bước 6: Tổng hợp kết quả
        final_result = self._step6_final_classification(
            step1_result, step2_result, step3_result, step4_result, step5_result
        )
        
        # Bước 7: Tạo báo cáo chi tiết
        report = self._step7_generate_detailed_report(
            image_path, step1_result, step2_result, step3_result, 
            step4_result, step5_result, final_result
        )
        
        return report
    
    def _step1_load_and_preprocess(self, image_path: str) -> Dict[str, Any]:
        """Bước 1: Load và preprocess ảnh"""
        print("  📸 Bước 1: Load và preprocess ảnh...")
        
        # Load ảnh
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể load ảnh: {image_path}")
        
        # Resize về kích thước chuẩn
        image = cv2.resize(image, (224, 224))
        
        # Chuyển đổi màu
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Tính toán thống kê cơ bản
        stats = {
            'image_size': image.shape,
            'mean_brightness': np.mean(gray_image),
            'std_brightness': np.std(gray_image),
            'max_brightness': np.max(gray_image),
            'min_brightness': np.min(gray_image)
        }
        
        result = {
            'original': image,
            'rgb': rgb_image,
            'hsv': hsv_image,
            'gray': gray_image,
            'stats': stats
        }
        
        print(f"    ✅ Ảnh đã được load: {image.shape}")
        print(f"    📊 Độ sáng trung bình: {stats['mean_brightness']:.1f}")
        
        return result
    
    def _step2_color_analysis(self, step1_result: Dict) -> Dict[str, Any]:
        """Bước 2: Phân tích màu sắc"""
        print("  🎨 Bước 2: Phân tích màu sắc...")
        
        hsv_image = step1_result['hsv']
        
        # Tạo mask cho từng màu lửa
        masks = {}
        color_stats = {}
        
        for color_name, (lower, upper) in [
            ('red', (self.fire_color_ranges['red_lower'], self.fire_color_ranges['red_upper'])),
            ('orange', (self.fire_color_ranges['orange_lower'], self.fire_color_ranges['orange_upper'])),
            ('yellow', (self.fire_color_ranges['yellow_lower'], self.fire_color_ranges['yellow_upper']))
        ]:
            mask = cv2.inRange(hsv_image, lower, upper)
            masks[color_name] = mask
            
            # Tính toán thống kê cho màu này
            color_pixels = np.sum(mask > 0)
            color_ratio = color_pixels / (224 * 224)
            
            color_stats[color_name] = {
                'pixel_count': int(color_pixels),
                'ratio': float(color_ratio),
                'percentage': f"{color_ratio * 100:.2f}%"
            }
        
        # Tạo mask tổng hợp cho tất cả màu lửa
        combined_mask = masks['red'] | masks['orange'] | masks['yellow']
        
        result = {
            'masks': masks,
            'combined_mask': combined_mask,
            'color_stats': color_stats,
            'total_fire_pixels': int(np.sum(combined_mask > 0)),
            'total_fire_ratio': float(np.sum(combined_mask > 0) / (224 * 224))
        }
        
        print(f"    🔴 Màu đỏ: {color_stats['red']['percentage']}")
        print(f"    🟠 Màu cam: {color_stats['orange']['percentage']}")
        print(f"    🟡 Màu vàng: {color_stats['yellow']['percentage']}")
        print(f"    🔥 Tổng vùng lửa: {result['total_fire_ratio']*100:.2f}%")
        
        return result
    
    def _step3_fire_region_analysis(self, step1_result: Dict, step2_result: Dict) -> Dict[str, Any]:
        """Bước 3: Phân tích vùng lửa"""
        print("  🔥 Bước 3: Phân tích vùng lửa...")
        
        combined_mask = step2_result['combined_mask']
        hsv_image = step1_result['hsv']
        
        # Tìm contours của vùng lửa
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc contours có kích thước đủ lớn
        min_contour_area = 50  # Tối thiểu 50 pixels
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        
        # Phân tích từng vùng lửa
        fire_regions = []
        total_fire_area = 0
        
        for i, contour in enumerate(valid_contours):
            area = cv2.contourArea(contour)
            total_fire_area += area
            
            # Tạo mask cho vùng này
            region_mask = np.zeros_like(combined_mask)
            cv2.fillPoly(region_mask, [contour], 255)
            
            # Tính toán thống kê cho vùng này
            region_hsv = cv2.bitwise_and(hsv_image, hsv_image, mask=region_mask)
            region_pixels = region_hsv[region_mask > 0]
            
            if len(region_pixels) > 0:
                brightness = np.mean(region_pixels[:, 2])  # V channel
                saturation = np.mean(region_pixels[:, 1])  # S channel
                hue = np.mean(region_pixels[:, 0])  # H channel
            else:
                brightness = saturation = hue = 0
            
            fire_regions.append({
                'id': i,
                'area': int(area),
                'brightness': float(brightness),
                'saturation': float(saturation),
                'hue': float(hue),
                'contour': contour
            })
        
        # Tính toán thống kê tổng hợp
        if fire_regions:
            avg_brightness = np.mean([r['brightness'] for r in fire_regions])
            avg_saturation = np.mean([r['saturation'] for r in fire_regions])
            max_brightness = max([r['brightness'] for r in fire_regions])
        else:
            avg_brightness = avg_saturation = max_brightness = 0
        
        result = {
            'fire_regions': fire_regions,
            'total_regions': len(fire_regions),
            'total_fire_area': int(total_fire_area),
            'avg_brightness': float(avg_brightness),
            'avg_saturation': float(avg_saturation),
            'max_brightness': float(max_brightness),
            'fire_area_ratio': float(total_fire_area / (224 * 224))
        }
        
        print(f"    📊 Số vùng lửa: {len(fire_regions)}")
        print(f"    📏 Tổng diện tích lửa: {result['fire_area_ratio']*100:.2f}%")
        print(f"    💡 Độ sáng trung bình: {avg_brightness:.1f}")
        print(f"    🎨 Độ bão hòa trung bình: {avg_saturation:.1f}")
        
        return result
    
    def _step4_texture_analysis(self, step1_result: Dict) -> Dict[str, Any]:
        """Bước 4: Phân tích texture"""
        print("  🌀 Bước 4: Phân tích texture...")
        
        gray_image = step1_result['gray']
        
        # Tính gradient
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Tính magnitude và direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Tính toán thống kê texture
        texture_stats = {
            'gradient_mean': float(np.mean(magnitude)),
            'gradient_std': float(np.std(magnitude)),
            'gradient_max': float(np.max(magnitude)),
            'direction_mean': float(np.mean(direction)),
            'direction_std': float(np.std(direction))
        }
        
        # Tính entropy của gradient
        hist, _ = np.histogram(magnitude.flatten(), bins=50, range=(0, np.max(magnitude)))
        hist = hist[hist > 0]  # Loại bỏ bins có 0
        if len(hist) > 0:
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob))
        else:
            entropy = 0
        
        texture_stats['gradient_entropy'] = float(entropy)
        
        result = {
            'gradient_magnitude': magnitude,
            'gradient_direction': direction,
            'texture_stats': texture_stats
        }
        
        print(f"    📈 Gradient mean: {texture_stats['gradient_mean']:.2f}")
        print(f"    📊 Gradient std: {texture_stats['gradient_std']:.2f}")
        print(f"    🔍 Gradient entropy: {texture_stats['gradient_entropy']:.2f}")
        
        return result
    
    def _step5_histogram_analysis(self, step1_result: Dict) -> Dict[str, Any]:
        """Bước 5: Phân tích histogram"""
        print("  📊 Bước 5: Phân tích histogram...")
        
        hsv_image = step1_result['hsv']
        
        # Tính histogram cho từng channel
        h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
        # Chuẩn hóa histogram
        h_hist = h_hist.flatten() / np.sum(h_hist)
        s_hist = s_hist.flatten() / np.sum(s_hist)
        v_hist = v_hist.flatten() / np.sum(v_hist)
        
        # Phân tích màu lửa trong histogram
        # Hue ranges: Red (0-10, 170-180), Orange (10-25), Yellow (25-35)
        red_hue_range = np.concatenate([h_hist[0:11], h_hist[170:181]])
        orange_hue_range = h_hist[10:26]
        yellow_hue_range = h_hist[25:36]
        
        fire_hue_ratio = (np.sum(red_hue_range) + np.sum(orange_hue_range) + np.sum(yellow_hue_range)) / 3
        
        # Phân tích saturation và value
        high_sat_ratio = np.sum(s_hist[100:]) / np.sum(s_hist)  # Saturation > 100
        high_val_ratio = np.sum(v_hist[150:]) / np.sum(v_hist)  # Value > 150
        
        result = {
            'h_histogram': h_hist.tolist(),
            's_histogram': s_hist.tolist(),
            'v_histogram': v_hist.tolist(),
            'fire_hue_ratio': float(fire_hue_ratio),
            'high_saturation_ratio': float(high_sat_ratio),
            'high_value_ratio': float(high_val_ratio),
            'histogram_stats': {
                'h_mean': float(np.mean(h_hist)),
                'h_std': float(np.std(h_hist)),
                's_mean': float(np.mean(s_hist)),
                's_std': float(np.std(s_hist)),
                'v_mean': float(np.mean(v_hist)),
                'v_std': float(np.std(v_hist))
            }
        }
        
        print(f"    🎨 Tỷ lệ màu lửa trong histogram: {fire_hue_ratio*100:.2f}%")
        print(f"    📈 Tỷ lệ độ bão hòa cao: {high_sat_ratio*100:.2f}%")
        print(f"    💡 Tỷ lệ độ sáng cao: {high_val_ratio*100:.2f}%")
        
        return result
    
    def _step6_final_classification(self, step1_result: Dict, step2_result: Dict, 
                                  step3_result: Dict, step4_result: Dict, 
                                  step5_result: Dict) -> Dict[str, Any]:
        """Bước 6: Tổng hợp kết quả và phân loại cuối cùng"""
        print("  🎯 Bước 6: Tổng hợp kết quả và phân loại...")
        
        # Thu thập các chỉ số quan trọng
        indicators = {
            'fire_ratio': step2_result['total_fire_ratio'],
            'fire_area_ratio': step3_result['fire_area_ratio'],
            'avg_brightness': step3_result['avg_brightness'],
            'avg_saturation': step3_result['avg_saturation'],
            'max_brightness': step3_result['max_brightness'],
            'texture_entropy': step4_result['texture_stats']['gradient_entropy'],
            'fire_hue_ratio': step5_result['fire_hue_ratio'],
            'high_saturation_ratio': step5_result['high_saturation_ratio'],
            'high_value_ratio': step5_result['high_value_ratio']
        }
        
        # Kiểm tra từng điều kiện
        conditions = {
            'has_fire_colors': indicators['fire_ratio'] > self.thresholds['fire_ratio_min'],
            'has_fire_area': indicators['fire_area_ratio'] > self.thresholds['fire_ratio_min'],
            'has_brightness': indicators['avg_brightness'] > self.thresholds['fire_brightness_min'],
            'has_saturation': indicators['avg_saturation'] > self.thresholds['fire_saturation_min'],
            'has_texture': indicators['texture_entropy'] > self.thresholds['texture_entropy_min'],
            'has_fire_histogram': indicators['fire_hue_ratio'] > self.thresholds['color_histogram_fire_ratio_min']
        }
        
        # Tính điểm tổng hợp
        score = 0
        total_conditions = len(conditions)
        
        for condition_name, condition_met in conditions.items():
            if condition_met:
                score += 1
                print(f"    ✅ {condition_name}: Đạt")
            else:
                print(f"    ❌ {condition_name}: Không đạt")
        
        confidence = score / total_conditions
        
        # Phân loại cuối cùng
        if confidence >= 0.5:  # Ít nhất 50% điều kiện được đáp ứng
            classification = "FIRE"
            confidence_level = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.6 else "LOW"
        else:
            classification = "NO FIRE"
            confidence_level = "HIGH" if confidence <= 0.2 else "MEDIUM" if confidence <= 0.4 else "LOW"
        
        result = {
            'classification': classification,
            'confidence': float(confidence),
            'confidence_level': confidence_level,
            'score': int(score),
            'total_conditions': total_conditions,
            'indicators': indicators,
            'conditions': conditions,
            'reasoning': self._generate_reasoning(conditions, indicators)
        }
        
        print(f"    🎯 Kết quả: {classification}")
        print(f"    📊 Độ tin cậy: {confidence*100:.1f}% ({confidence_level})")
        print(f"    📈 Điểm: {score}/{total_conditions}")
        
        return result
    
    def _generate_reasoning(self, conditions: Dict[str, bool], indicators: Dict[str, float]) -> str:
        """Tạo lý do phân loại"""
        met_conditions = [k for k, v in conditions.items() if v]
        failed_conditions = [k for k, v in conditions.items() if not v]
        
        reasoning = "Lý do phân loại:\n"
        
        if met_conditions:
            reasoning += "✅ Các điều kiện đạt:\n"
            for condition in met_conditions:
                reasoning += f"   - {condition}\n"
        
        if failed_conditions:
            reasoning += "❌ Các điều kiện không đạt:\n"
            for condition in failed_conditions:
                reasoning += f"   - {condition}\n"
        
        reasoning += f"\n📊 Chỉ số quan trọng:\n"
        reasoning += f"   - Tỷ lệ màu lửa: {indicators['fire_ratio']*100:.2f}%\n"
        reasoning += f"   - Diện tích lửa: {indicators['fire_area_ratio']*100:.2f}%\n"
        reasoning += f"   - Độ sáng trung bình: {indicators['avg_brightness']:.1f}\n"
        reasoning += f"   - Độ bão hòa trung bình: {indicators['avg_saturation']:.1f}\n"
        
        return reasoning
    
    def _step7_generate_detailed_report(self, image_path: str, step1_result: Dict, 
                                      step2_result: Dict, step3_result: Dict,
                                      step4_result: Dict, step5_result: Dict,
                                      final_result: Dict) -> Dict[str, Any]:
        """Bước 7: Tạo báo cáo chi tiết"""
        print("  📋 Bước 7: Tạo báo cáo chi tiết...")
        
        # Tạo tên file báo cáo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = os.path.basename(image_path).split('.')[0]
        report_filename = f"report_{image_name}_{timestamp}.json"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Tạo báo cáo
        report = {
            'image_path': image_path,
            'analysis_timestamp': timestamp,
            'final_classification': final_result,
            'step_by_step_results': {
                'step1_preprocessing': {
                    'image_stats': step1_result['stats']
                },
                'step2_color_analysis': {
                    'color_stats': step2_result['color_stats'],
                    'total_fire_ratio': step2_result['total_fire_ratio']
                },
                'step3_fire_regions': {
                    'total_regions': step3_result['total_regions'],
                    'fire_area_ratio': step3_result['fire_area_ratio'],
                    'brightness_stats': {
                        'avg': step3_result['avg_brightness'],
                        'max': step3_result['max_brightness']
                    },
                    'saturation_stats': {
                        'avg': step3_result['avg_saturation']
                    }
                },
                'step4_texture': {
                    'texture_stats': step4_result['texture_stats']
                },
                'step5_histogram': {
                    'fire_hue_ratio': step5_result['fire_hue_ratio'],
                    'saturation_ratio': step5_result['high_saturation_ratio'],
                    'value_ratio': step5_result['high_value_ratio']
                }
            },
            'thresholds_used': self.thresholds
        }
        
        # Chuyển đổi các trường numpy types sang native types
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
        
        report = convert(report)
        
        # Lưu báo cáo
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"    📄 Báo cáo đã được lưu: {report_path}")
        
        return report
    
    def visualize_analysis(self, image_path: str, step1_result: Dict, 
                          step2_result: Dict, step3_result: Dict) -> str:
        """Tạo visualization cho phân tích"""
        print("  🎨 Tạo visualization...")
        
        # Tạo figure với subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Phân tích chi tiết: {os.path.basename(image_path)}', fontsize=16)
        
        # 1. Ảnh gốc
        axes[0, 0].imshow(step1_result['rgb'])
        axes[0, 0].set_title('Ảnh gốc')
        axes[0, 0].axis('off')
        
        # 2. Ảnh HSV
        axes[0, 1].imshow(step1_result['hsv'])
        axes[0, 1].set_title('Ảnh HSV')
        axes[0, 1].axis('off')
        
        # 3. Mask tổng hợp
        axes[0, 2].imshow(step2_result['combined_mask'], cmap='hot')
        axes[0, 2].set_title('Mask màu lửa tổng hợp')
        axes[0, 2].axis('off')
        
        # 4. Mask từng màu
        color_masks = step2_result['masks']
        combined_display = np.zeros((224, 224, 3), dtype=np.uint8)
        combined_display[:, :, 0] = color_masks['red']  # Red channel
        combined_display[:, :, 1] = color_masks['orange']  # Green channel  
        combined_display[:, :, 2] = color_masks['yellow']  # Blue channel
        axes[1, 0].imshow(combined_display)
        axes[1, 0].set_title('Mask màu riêng biệt\n(Đỏ/Cam/Vàng)')
        axes[1, 0].axis('off')
        
        # 5. Contours vùng lửa
        img_with_contours = step1_result['rgb'].copy()
        for region in step3_result['fire_regions']:
            cv2.drawContours(img_with_contours, [region['contour']], -1, (0, 255, 0), 2)
        axes[1, 1].imshow(img_with_contours)
        axes[1, 1].set_title(f'Vùng lửa phát hiện\n({len(step3_result["fire_regions"])} vùng)')
        axes[1, 1].axis('off')
        
        # 6. Thống kê màu sắc
        colors = ['red', 'orange', 'yellow']
        ratios = [step2_result['color_stats'][c]['ratio'] for c in colors]
        axes[1, 2].bar(colors, ratios, color=['red', 'orange', 'yellow'])
        axes[1, 2].set_title('Tỷ lệ màu lửa')
        axes[1, 2].set_ylabel('Tỷ lệ')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Lưu visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = os.path.basename(image_path).split('.')[0]
        viz_filename = f"visualization_{image_name}_{timestamp}.png"
        viz_path = os.path.join(self.output_dir, viz_filename)
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    🖼️ Visualization đã được lưu: {viz_path}")
        return viz_path

def main():
    """Test hệ thống phân tích"""
    analyzer = DetailedFireAnalyzer()
    
    # Test với một số ảnh từ dataset
    test_images = [
        "dataset/train/images/train_1.jpg",
        "dataset/train/images/train_7.jpg", 
        "dataset/train/images/train_155.jpg"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*60}")
            print(f"Phân tích ảnh: {image_path}")
            print(f"{'='*60}")
            
            try:
                # Phân tích chi tiết
                report = analyzer.analyze_image_step_by_step(image_path)
                
                # Tạo visualization
                step1 = analyzer._step1_load_and_preprocess(image_path)
                step2 = analyzer._step2_color_analysis(step1)
                step3 = analyzer._step3_fire_region_analysis(step1, step2)
                viz_path = analyzer.visualize_analysis(image_path, step1, step2, step3)
                
                print(f"\n✅ Hoàn thành phân tích: {image_path}")
                print(f"📊 Kết quả: {report['final_classification']['classification']}")
                print(f"🎯 Độ tin cậy: {report['final_classification']['confidence']*100:.1f}%")
                
            except Exception as e:
                print(f"❌ Lỗi khi phân tích {image_path}: {str(e)}")
        else:
            print(f"❌ Không tìm thấy ảnh: {image_path}")

if __name__ == "__main__":
    main() 