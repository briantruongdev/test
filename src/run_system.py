#!/usr/bin/env python3
"""
Script tổng hợp để chạy hệ thống phân tích lửa chi tiết
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def run_web_app():
    """Khởi động web application"""
    print("🔥 Khởi động Web Application...")
    print("=" * 50)
    
    try:
        # Kiểm tra xem web app đã chạy chưa
        import requests
        try:
            response = requests.get("http://localhost:8083/health", timeout=2)
            if response.status_code == 200:
                print("✅ Web app đã đang chạy tại: http://localhost:8083")
                return
        except:
            pass
        
        # Khởi động web app
        print("🚀 Đang khởi động web app...")
        subprocess.Popen([sys.executable, "detailed_web_app.py"], 
                        cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # Chờ web app khởi động
        print("⏳ Đang chờ web app khởi động...")
        time.sleep(5)
        
        # Kiểm tra lại
        try:
            response = requests.get("http://localhost:8083/health", timeout=5)
            if response.status_code == 200:
                print("✅ Web app đã khởi động thành công!")
                print("🌐 Truy cập: http://localhost:8083")
                print("📱 Upload ảnh để phân tích chi tiết từng bước")
            else:
                print("❌ Web app khởi động thất bại")
        except Exception as e:
            print(f"❌ Lỗi khi kiểm tra web app: {e}")
            
    except ImportError:
        print("❌ Cần cài đặt requests: pip install requests")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def test_single_image(image_path):
    """Test với một ảnh cụ thể"""
    print(f"🔍 Test với ảnh: {image_path}")
    print("=" * 50)
    
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        return
    
    try:
        from test_detailed_analyzer import analyze_specific_image
        analyze_specific_image(image_path)
    except Exception as e:
        print(f"❌ Lỗi khi phân tích: {e}")

def test_multiple_images():
    """Test với nhiều ảnh"""
    print("🧪 Test với nhiều ảnh")
    print("=" * 50)
    
    try:
        from test_detailed_analyzer import test_detailed_analyzer
        test_detailed_analyzer()
    except Exception as e:
        print(f"❌ Lỗi khi test: {e}")

def show_results():
    """Hiển thị kết quả đã lưu"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("❌ Không có kết quả nào")
        return
    
    print("📊 Kết quả đã lưu:")
    print("=" * 50)
    
    # Liệt kê các file JSON
    json_files = list(Path(results_dir).glob("*.json"))
    if json_files:
        print(f"📄 Báo cáo JSON: {len(json_files)} file")
        for file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            print(f"   - {file.name}")
    
    # Liệt kê các file PNG
    png_files = list(Path(results_dir).glob("*.png"))
    if png_files:
        print(f"🖼️ Visualization: {len(png_files)} file")
        for file in sorted(png_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            print(f"   - {file.name}")

def main():
    parser = argparse.ArgumentParser(description="Hệ thống phân tích lửa chi tiết")
    parser.add_argument("--web", action="store_true", help="Khởi động web application")
    parser.add_argument("--test", type=str, help="Test với ảnh cụ thể")
    parser.add_argument("--test-all", action="store_true", help="Test với tất cả ảnh mẫu")
    parser.add_argument("--results", action="store_true", help="Hiển thị kết quả đã lưu")
    
    args = parser.parse_args()
    
    print("🔥 Hệ thống phân tích lửa chi tiết từng bước")
    print("=" * 60)
    
    if args.web:
        run_web_app()
    elif args.test:
        test_single_image(args.test)
    elif args.test_all:
        test_multiple_images()
    elif args.results:
        show_results()
    else:
        # Chế độ tương tác
        while True:
            print("\n📋 Menu:")
            print("1. 🌐 Khởi động Web Application")
            print("2. 🔍 Test với ảnh cụ thể")
            print("3. 🧪 Test với tất cả ảnh mẫu")
            print("4. 📊 Xem kết quả đã lưu")
            print("5. ❌ Thoát")
            
            choice = input("\n👉 Chọn (1-5): ").strip()
            
            if choice == "1":
                run_web_app()
            elif choice == "2":
                image_path = input("📸 Nhập đường dẫn ảnh: ").strip()
                if image_path:
                    test_single_image(image_path)
            elif choice == "3":
                test_multiple_images()
            elif choice == "4":
                show_results()
            elif choice == "5":
                print("👋 Tạm biệt!")
                break
            else:
                print("❌ Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main() 