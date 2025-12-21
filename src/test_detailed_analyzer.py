#!/usr/bin/env python3
"""
Script test cho hệ thống phân tích lửa chi tiết
"""

import os
import sys
from detailed_fire_analyzer import DetailedFireAnalyzer

def test_detailed_analyzer():
    """Test hệ thống phân tích chi tiết với các ảnh từ dataset"""
    
    print("🔥 Test hệ thống phân tích lửa chi tiết")
    print("=" * 60)
    
    # Khởi tạo analyzer
    analyzer = DetailedFireAnalyzer(output_dir="results")
    
    # Danh sách ảnh test (cả có lửa và không có lửa)
    test_images = [
        # Ảnh có lửa
        "dataset/train/images/train_1.jpg",
        "dataset/train/images/train_7.jpg", 
        "dataset/train/images/train_155.jpg",
        "dataset/train/images/train_200.jpg",
        "dataset/train/images/train_300.jpg",
        
        # Ảnh không có lửa (nếu có)
        "dataset/train/images/train_500.jpg",
        "dataset/train/images/train_1000.jpg",
    ]
    
    results = []
    
    for i, image_path in enumerate(test_images, 1):
        if not os.path.exists(image_path):
            print(f"❌ Không tìm thấy ảnh: {image_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_images)}: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        try:
            # Phân tích chi tiết
            report = analyzer.analyze_image_step_by_step(image_path)
            
            # Tạo visualization
            step1 = analyzer._step1_load_and_preprocess(image_path)
            step2 = analyzer._step2_color_analysis(step1)
            step3 = analyzer._step3_fire_region_analysis(step1, step2)
            viz_path = analyzer.visualize_analysis(image_path, step1, step2, step3)
            
            # Lưu kết quả
            result = {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'classification': report['final_classification']['classification'],
                'confidence': report['final_classification']['confidence'],
                'score': report['final_classification']['score'],
                'total_conditions': report['final_classification']['total_conditions'],
                'conditions': report['final_classification']['conditions'],
                'visualization_path': viz_path,
                'report_path': report.get('report_path', 'N/A')
            }
            
            results.append(result)
            
            # In kết quả
            print(f"✅ Hoàn thành phân tích: {image_path}")
            print(f"📊 Kết quả: {result['classification']}")
            print(f"🎯 Độ tin cậy: {result['confidence']*100:.1f}%")
            print(f"📈 Điểm: {result['score']}/{result['total_conditions']}")
            
            # In chi tiết các điều kiện
            print("\n📋 Chi tiết các điều kiện:")
            for condition, passed in result['conditions'].items():
                status = "✅ Đạt" if passed else "❌ Không đạt"
                print(f"   {condition}: {status}")
            
            print(f"📄 Báo cáo: {result['report_path']}")
            print(f"🖼️ Visualization: {result['visualization_path']}")
            
        except Exception as e:
            print(f"❌ Lỗi khi phân tích {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Tổng kết
    print(f"\n{'='*60}")
    print("📊 TỔNG KẾT KẾT QUẢ")
    print(f"{'='*60}")
    
    fire_count = sum(1 for r in results if r['classification'] == 'FIRE')
    no_fire_count = sum(1 for r in results if r['classification'] == 'NO FIRE')
    
    print(f"Tổng số ảnh đã phân tích: {len(results)}")
    print(f"Phân loại FIRE: {fire_count}")
    print(f"Phân loại NO FIRE: {no_fire_count}")
    
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"Độ tin cậy trung bình: {avg_confidence*100:.1f}%")
        
        # Phân tích chi tiết từng ảnh
        print(f"\n📋 Chi tiết từng ảnh:")
        for result in results:
            print(f"  {result['image_name']}: {result['classification']} "
                  f"({result['confidence']*100:.1f}%, {result['score']}/{result['total_conditions']})")
    
    print(f"\n📁 Kết quả được lưu trong: results/")
    print("🎨 Visualization được lưu dưới dạng PNG")
    print("📄 Báo cáo chi tiết được lưu dưới dạng JSON")

def analyze_specific_image(image_path):
    """Phân tích một ảnh cụ thể"""
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        return
    
    print(f"🔍 Phân tích ảnh cụ thể: {image_path}")
    print("=" * 60)
    
    analyzer = DetailedFireAnalyzer(output_dir="results")
    
    try:
        # Phân tích chi tiết
        report = analyzer.analyze_image_step_by_step(image_path)
        
        # Tạo visualization
        step1 = analyzer._step1_load_and_preprocess(image_path)
        step2 = analyzer._step2_color_analysis(step1)
        step3 = analyzer._step3_fire_region_analysis(step1, step2)
        viz_path = analyzer.visualize_analysis(image_path, step1, step2, step3)
        
        print(f"✅ Kết quả: {report['final_classification']['classification']}")
        print(f"🎯 Độ tin cậy: {report['final_classification']['confidence']*100:.1f}%")
        print(f"📈 Điểm: {report['final_classification']['score']}/{report['final_classification']['total_conditions']}")
        
        print(f"\n🧠 Lý do phân loại:")
        print(report['final_classification']['reasoning'])
        
        print(f"\n📄 Báo cáo: {report.get('report_path', 'N/A')}")
        print(f"🖼️ Visualization: {viz_path}")
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Nếu có tham số, phân tích ảnh cụ thể
        image_path = sys.argv[1]
        analyze_specific_image(image_path)
    else:
        # Test với danh sách ảnh mặc định
        test_detailed_analyzer() 