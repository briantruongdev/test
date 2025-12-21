#!/usr/bin/env python3
"""
Script tổng hợp để chạy cả hai hệ thống phát hiện lửa
1. Detailed Analysis System (Port 8083)
2. Vector Database System (Port 8084)
"""

import os
import sys
import argparse
import subprocess
import time
import requests
from pathlib import Path

def check_port(port):
    """Kiểm tra port có đang được sử dụng không"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_detailed_system():
    """Khởi động hệ thống phân tích chi tiết"""
    print("🔥 Khởi động Detailed Analysis System...")
    print("=" * 50)
    
    if check_port(8083):
        print("✅ Detailed Analysis System đã đang chạy tại: http://localhost:8083")
        return True
    
    try:
        print("🚀 Đang khởi động Detailed Analysis System...")
        subprocess.Popen([sys.executable, "detailed_web_app.py"], 
                        cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # Chờ khởi động
        print("⏳ Đang chờ khởi động...")
        for i in range(10):
            time.sleep(1)
            if check_port(8083):
                print("✅ Detailed Analysis System đã khởi động thành công!")
                print("🌐 Truy cập: http://localhost:8083")
                return True
            print(f"   Đang chờ... ({i+1}/10)")
        
        print("❌ Không thể khởi động Detailed Analysis System")
        return False
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False

def start_vector_system():
    """Khởi động hệ thống vector database"""
    print("🔥 Khởi động Vector Database System...")
    print("=" * 50)
    
    if check_port(8084):
        print("✅ Vector Database System đã đang chạy tại: http://localhost:8084")
        return True
    
    try:
        print("🚀 Đang khởi động Vector Database System...")
        subprocess.Popen([sys.executable, "vector_web_app.py"], 
                        cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # Chờ khởi động
        print("⏳ Đang chờ khởi động...")
        for i in range(10):
            time.sleep(1)
            if check_port(8084):
                print("✅ Vector Database System đã khởi động thành công!")
                print("🌐 Truy cập: http://localhost:8084")
                return True
            print(f"   Đang chờ... ({i+1}/10)")
        
        print("❌ Không thể khởi động Vector Database System")
        return False
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False

def test_systems():
    """Test cả hai hệ thống"""
    print("🧪 Test cả hai hệ thống")
    print("=" * 50)
    
    test_image = "../dataset/train/images/train_7.jpg"
    if not os.path.exists(test_image):
        print(f"❌ Không tìm thấy ảnh test: {test_image}")
        return
    
    print(f"🔍 Test với ảnh: {test_image}")
    
    # Test Detailed Analysis System
    print("\n📊 Test Detailed Analysis System:")
    print("-" * 30)
    try:
        from test_detailed_analyzer import analyze_specific_image
        analyze_specific_image(test_image)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    
    # Test Vector Database System
    print("\n📊 Test Vector Database System:")
    print("-" * 30)
    try:
        from test_vector_system import test_single_image
        test_single_image(test_image)
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def show_status():
    """Hiển thị trạng thái các hệ thống"""
    print("📊 Trạng thái các hệ thống")
    print("=" * 50)
    
    detailed_status = "✅ Đang chạy" if check_port(8083) else "❌ Không chạy"
    vector_status = "✅ Đang chạy" if check_port(8084) else "❌ Không chạy"
    
    print(f"🔥 Detailed Analysis System (Port 8083): {detailed_status}")
    print(f"🔥 Vector Database System (Port 8084): {vector_status}")
    
    if check_port(8083):
        print("   🌐 URL: http://localhost:8083")
    if check_port(8084):
        print("   🌐 URL: http://localhost:8084")

def show_comparison():
    """So sánh hai hệ thống"""
    print("📋 So sánh hai hệ thống")
    print("=" * 50)
    
    comparison = {
        "Detailed Analysis System": {
            "Phương pháp": "Rule-based với 6 điều kiện",
            "Ưu điểm": [
                "Giải thích rõ ràng từng bước",
                "Visualization chi tiết",
                "Không cần training data",
                "Dễ tùy chỉnh ngưỡng"
            ],
            "Nhược điểm": [
                "Có thể thiếu chính xác",
                "Cần điều chỉnh thủ công"
            ],
            "Port": "8083"
        },
        "Vector Database System": {
            "Phương pháp": "Machine Learning với vector similarity",
            "Ưu điểm": [
                "Độ chính xác cao hơn",
                "Học từ dữ liệu thực tế",
                "Tìm ảnh tương tự",
                "Tự động cải thiện"
            ],
            "Nhược điểm": [
                "Cần training data",
                "Khó giải thích",
                "Phức tạp hơn"
            ],
            "Port": "8084"
        }
    }
    
    for system, info in comparison.items():
        print(f"\n🔥 {system}:")
        print(f"   📊 Phương pháp: {info['Phương pháp']}")
        print(f"   🌐 Port: {info['Port']}")
        
        print(f"   ✅ Ưu điểm:")
        for advantage in info['Ưu điểm']:
            print(f"      • {advantage}")
        
        print(f"   ❌ Nhược điểm:")
        for disadvantage in info['Nhược điểm']:
            print(f"      • {disadvantage}")

def main():
    parser = argparse.ArgumentParser(description="Hệ thống phát hiện lửa tổng hợp")
    parser.add_argument("--detailed", action="store_true", help="Khởi động Detailed Analysis System")
    parser.add_argument("--vector", action="store_true", help="Khởi động Vector Database System")
    parser.add_argument("--both", action="store_true", help="Khởi động cả hai hệ thống")
    parser.add_argument("--test", action="store_true", help="Test cả hai hệ thống")
    parser.add_argument("--status", action="store_true", help="Hiển thị trạng thái")
    parser.add_argument("--compare", action="store_true", help="So sánh hai hệ thống")
    
    args = parser.parse_args()
    
    print("🔥 Hệ thống phát hiện lửa tổng hợp")
    print("=" * 60)
    
    if args.detailed:
        start_detailed_system()
    elif args.vector:
        start_vector_system()
    elif args.both:
        start_detailed_system()
        time.sleep(2)
        start_vector_system()
    elif args.test:
        test_systems()
    elif args.status:
        show_status()
    elif args.compare:
        show_comparison()
    else:
        # Chế độ tương tác
        while True:
            print("\n📋 Menu chính:")
            print("1. 🔥 Khởi động Detailed Analysis System (Port 8083)")
            print("2. 🔥 Khởi động Vector Database System (Port 8084)")
            print("3. 🔥 Khởi động cả hai hệ thống")
            print("4. 🧪 Test cả hai hệ thống")
            print("5. 📊 Hiển thị trạng thái")
            print("6. 📋 So sánh hai hệ thống")
            print("7. ❌ Thoát")
            
            choice = input("\n👉 Chọn (1-7): ").strip()
            
            if choice == "1":
                start_detailed_system()
            elif choice == "2":
                start_vector_system()
            elif choice == "3":
                start_detailed_system()
                time.sleep(2)
                start_vector_system()
            elif choice == "4":
                test_systems()
            elif choice == "5":
                show_status()
            elif choice == "6":
                show_comparison()
            elif choice == "7":
                print("👋 Tạm biệt!")
                break
            else:
                print("❌ Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main() 