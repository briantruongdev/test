#!/usr/bin/env python3
"""
Script test cho hệ thống Vector Database Fire Detection
"""

import os
import sys
from vector_fire_detection import VectorBasedFireClassifier

def test_vector_system():
    """Test hệ thống vector database"""
    print("🔥 Test hệ thống Vector Database Fire Detection")
    print("=" * 60)
    
    # Khởi tạo classifier
    classifier = VectorBasedFireClassifier()
    
    # Kiểm tra vector database
    if not classifier.load_vector_database():
        print("🔨 Cần xây dựng vector database trước...")
        print("📁 Đang tìm dataset...")
        
        dataset_path = "../dataset"
        if os.path.exists(dataset_path):
            print("✅ Tìm thấy dataset, đang xây dựng database...")
            classifier.build_vector_database(dataset_path)
        else:
            print("❌ Không tìm thấy dataset")
            return
    else:
        print("✅ Vector database đã được load")
    
    # Test với các ảnh mẫu
    test_images = [
        "../dataset/train/images/train_1.jpg",
        "../dataset/train/images/train_7.jpg", 
        "../dataset/train/images/train_100.jpg",
        "../dataset/train/images/train_200.jpg",
        "../dataset/train/images/train_500.jpg"
    ]
    
    print(f"\n🧪 Test với {len(test_images)} ảnh mẫu:")
    print("-" * 60)
    
    for i, test_image in enumerate(test_images, 1):
        if os.path.exists(test_image):
            print(f"\n🔍 Test {i}: {os.path.basename(test_image)}")
            print("-" * 40)
            
            try:
                result = classifier.classify_new_image(test_image)
                
                print(f"🎯 Kết quả: {result['prediction']}")
                print(f"📊 Độ tin cậy: {result['confidence']:.3f}")
                print(f"🔥 Xác suất có lửa: {result['probability_fire']:.3f}")
                print(f"❌ Xác suất không lửa: {result['probability_no_fire']:.3f}")
                print(f"📈 Tỷ lệ màu lửa: {result['features']['fire_ratio']:.3f}")
                print(f"🔴 Tỷ lệ màu đỏ: {result['features']['red_ratio']:.3f}")
                print(f"🟠 Tỷ lệ màu cam: {result['features']['orange_ratio']:.3f}")
                print(f"🟡 Tỷ lệ màu vàng: {result['features']['yellow_ratio']:.3f}")
                
                # Hiển thị ảnh tương tự nhất
                top_similar = result['similar_images'][0]
                print(f"🖼️ Ảnh tương tự nhất: {os.path.basename(top_similar['image_path'])}")
                print(f"   Similarity: {top_similar['similarity']:.3f}")
                print(f"   Label: {'FIRE' if top_similar['label'] == 1 else 'NO FIRE'}")
                
            except Exception as e:
                print(f"❌ Lỗi khi phân tích: {e}")
        else:
            print(f"❌ Không tìm thấy ảnh: {test_image}")
    
    print(f"\n✅ Hoàn thành test {len(test_images)} ảnh")

def test_single_image(image_path):
    """Test với một ảnh cụ thể"""
    print(f"🔍 Test với ảnh: {image_path}")
    print("=" * 50)
    
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        return
    
    # Khởi tạo classifier
    classifier = VectorBasedFireClassifier()
    
    if not classifier.load_vector_database():
        print("❌ Vector database chưa được load")
        return
    
    try:
        result = classifier.classify_new_image(image_path)
        
        print(f"🎯 Kết quả: {result['prediction']}")
        print(f"📊 Độ tin cậy: {result['confidence']:.3f}")
        print(f"🔥 Xác suất có lửa: {result['probability_fire']:.3f}")
        print(f"❌ Xác suất không lửa: {result['probability_no_fire']:.3f}")
        print(f"📈 Tỷ lệ màu lửa: {result['features']['fire_ratio']:.3f}")
        
        print(f"\n🖼️ Top 5 ảnh tương tự:")
        for i, similar in enumerate(result['similar_images'][:5], 1):
            label = "FIRE" if similar['label'] == 1 else "NO FIRE"
            print(f"  {i}. {os.path.basename(similar['image_path'])} - {label} ({similar['similarity']:.3f})")
        
        # Tạo visualization
        viz_path = f"results/vector_test_{os.path.basename(image_path)}.png"
        os.makedirs("results", exist_ok=True)
        
        try:
            classifier.visualize_similarities(image_path, viz_path)
            print(f"\n🖼️ Visualization đã được lưu: {viz_path}")
        except Exception as e:
            print(f"❌ Lỗi tạo visualization: {e}")
        
    except Exception as e:
        print(f"❌ Lỗi khi phân tích: {e}")

def build_database():
    """Xây dựng vector database"""
    print("🔨 Xây dựng Vector Database")
    print("=" * 40)
    
    classifier = VectorBasedFireClassifier()
    
    dataset_path = "../dataset"
    if os.path.exists(dataset_path):
        print(f"📁 Tìm thấy dataset: {dataset_path}")
        classifier.build_vector_database(dataset_path)
        print("✅ Vector database đã được xây dựng thành công!")
    else:
        print("❌ Không tìm thấy dataset")

def main():
    """Hàm chính"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            test_vector_system()
        elif command == "single":
            if len(sys.argv) > 2:
                image_path = sys.argv[2]
                test_single_image(image_path)
            else:
                print("❌ Cần cung cấp đường dẫn ảnh")
                print("Usage: python test_vector_system.py single <image_path>")
        elif command == "build":
            build_database()
        else:
            print("❌ Lệnh không hợp lệ")
            print("Usage:")
            print("  python test_vector_system.py test          # Test với nhiều ảnh")
            print("  python test_vector_system.py single <path> # Test với ảnh cụ thể")
            print("  python test_vector_system.py build         # Xây dựng database")
    else:
        # Chế độ tương tác
        print("🔥 Vector Database Fire Detection Test")
        print("=" * 50)
        
        while True:
            print("\n📋 Menu:")
            print("1. 🧪 Test với nhiều ảnh")
            print("2. 🔍 Test với ảnh cụ thể")
            print("3. 🔨 Xây dựng database")
            print("4. ❌ Thoát")
            
            choice = input("\n👉 Chọn (1-4): ").strip()
            
            if choice == "1":
                test_vector_system()
            elif choice == "2":
                image_path = input("📸 Nhập đường dẫn ảnh: ").strip()
                if image_path:
                    test_single_image(image_path)
            elif choice == "3":
                build_database()
            elif choice == "4":
                print("👋 Tạm biệt!")
                break
            else:
                print("❌ Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main() 