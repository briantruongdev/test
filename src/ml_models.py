#!/usr/bin/env python3
"""
Machine Learning Models for Fire Detection
Training và so sánh các mô hình: KNN, SVM, Decision Tree, Logistic Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

class MLModelTrainer:
    """Trainer cho các mô hình ML"""
    
    def __init__(self, models_dir: str = "trained_models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Định nghĩa các mô hình
        self.models = {
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42)
        }
        
        # Hyperparameter grids cho GridSearch - Enhanced for better performance
        self.param_grids = {
            'KNN': {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'p': [1, 2, 3]  # For minkowski metric
            },
            'SVM': {
                'C': [0.01, 0.1, 1, 10, 100, 1000],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'degree': [2, 3, 4]  # For poly kernel
            },
            'Decision Tree': {
                'max_depth': [3, 5, 7, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy'],
                'max_features': ['sqrt', 'log2', None]
            },
                                 'Logistic Regression': {
                         'C': [0.01, 0.1, 1, 10, 100, 1000],
                         'penalty': ['l1', 'l2'],
                         'solver': ['liblinear', 'saga']
                     },
            'Random Forest': {
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
        }
        
        self.trained_models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
        """Chuẩn bị dữ liệu training và testing"""
        print("📊 Chuẩn bị dữ liệu...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"📈 Training set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        print(f"🎯 Training - Fire: {np.sum(y_train == 1)}, No Fire: {np.sum(y_train == 0)}")
        print(f"🎯 Test - Fire: {np.sum(y_test == 1)}, No Fire: {np.sum(y_test == 0)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                   use_grid_search: bool = True) -> Dict[str, Any]:
        """Train một mô hình cụ thể"""
        print(f"\n🔥 Training {model_name}...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} không được hỗ trợ")
        
        model = self.models[model_name]
        
        if use_grid_search and model_name in self.param_grids:
            print(f"🔍 Grid Search cho {model_name}...")
            grid_search = GridSearchCV(
                model, self.param_grids[model_name], 
                cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print(f"✅ Best parameters: {best_params}")
            print(f"📊 Best CV score: {best_score:.4f}")
            
        else:
            print(f"🚀 Training {model_name} với default parameters...")
            best_model = model
            best_model.fit(X_train, y_train)
            best_params = {}
            best_score = cross_val_score(best_model, X_train, y_train, cv=5).mean()
            print(f"📊 CV score: {best_score:.4f}")
        
        # Lưu model
        self.trained_models[model_name] = best_model
        
        return {
            'model': best_model,
            'best_params': best_params,
            'cv_score': best_score
        }
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Đánh giá một mô hình"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} chưa được train")
        
        model = self.trained_models[model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\n📊 Kết quả {model_name}:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"   ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, use_grid_search: bool = True):
        """Train tất cả các mô hình"""
        print("🔥 Bắt đầu training tất cả các mô hình...")
        print("=" * 60)
        
        # Chuẩn bị dữ liệu
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Train từng mô hình
        for model_name in self.models.keys():
            try:
                # Train model
                train_result = self.train_model(model_name, X_train, y_train, use_grid_search)
                
                # Evaluate model
                eval_result = self.evaluate_model(model_name, X_test, y_test)
                
                # Lưu kết quả
                self.results[model_name] = {
                    'train_result': train_result,
                    'eval_result': eval_result
                }
                
            except Exception as e:
                print(f"❌ Lỗi khi train {model_name}: {e}")
                continue
        
        print(f"\n✅ Hoàn thành training {len(self.results)} mô hình!")
        
        return X_test, y_test
    
    def compare_models(self) -> pd.DataFrame:
        """So sánh tất cả các mô hình"""
        if not self.results:
            raise ValueError("Chưa có mô hình nào được train")
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            eval_result = result['eval_result']
            train_result = result['train_result']
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': eval_result['accuracy'],
                'Precision': eval_result['precision'],
                'Recall': eval_result['recall'],
                'F1-Score': eval_result['f1_score'],
                'ROC AUC': eval_result['roc_auc'],
                'CV Score': train_result['cv_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print("\n📊 So sánh các mô hình:")
        print("=" * 80)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return comparison_df
    
    def plot_results(self, save_path: str = None):
        """Vẽ biểu đồ so sánh kết quả"""
        if not self.results:
            raise ValueError("Chưa có mô hình nào được train")
        
        # Chuẩn bị dữ liệu
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Tạo subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🔥 So sánh hiệu suất các mô hình ML', fontsize=16, fontweight='bold')
        
        # Bar plot cho từng metric
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            values = [self.results[model]['eval_result'][metric] for model in models]
            
            bars = axes[row, col].bar(models, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'])
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_ylim(0, 1)
            
            # Thêm giá trị trên bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                  f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Xoay labels nếu cần
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Biểu đồ đã được lưu: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, save_path: str = None):
        """Vẽ confusion matrix cho tất cả mô hình"""
        if not self.results:
            raise ValueError("Chưa có mô hình nào được train")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🔥 Confusion Matrices', fontsize=16, fontweight='bold')
        
        for i, (model_name, result) in enumerate(self.results.items()):
            row, col = i // 3, i % 3
            
            cm = result['eval_result']['confusion_matrix']
            
            # Vẽ confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Fire', 'Fire'],
                       yticklabels=['No Fire', 'Fire'],
                       ax=axes[row, col])
            
            axes[row, col].set_title(f'{model_name}\nAccuracy: {result["eval_result"]["accuracy"]:.3f}')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Ẩn subplot thừa nếu có
        if n_models < 6:
            for i in range(n_models, 6):
                row, col = i // 3, i % 3
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrices đã được lưu: {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None):
        """Vẽ ROC curves cho tất cả mô hình"""
        if not self.results:
            raise ValueError("Chưa có mô hình nào được train")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            model = self.trained_models[model_name]
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('🔥 ROC Curves Comparison', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 ROC curves đã được lưu: {save_path}")
        
        plt.show()
    
    def save_models(self):
        """Lưu tất cả các mô hình đã train"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.trained_models.items():
            model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.pkl")
            joblib.dump(model, model_path)
            print(f"💾 {model_name} đã được lưu: {model_path}")
        
        # Lưu scaler
        scaler_path = os.path.join(self.models_dir, f"scaler_{timestamp}.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"💾 Scaler đã được lưu: {scaler_path}")
        
        # Lưu kết quả
        results_path = os.path.join(self.models_dir, f"results_{timestamp}.json")
        import json
        # Chuyển đổi numpy arrays thành lists để JSON serializable
        results_to_save = {}
        for model_name, result in self.results.items():
            results_to_save[model_name] = {
                'train_result': {
                    'cv_score': result['train_result']['cv_score'],
                    'best_params': result['train_result']['best_params']
                },
                'eval_result': {
                    'accuracy': result['eval_result']['accuracy'],
                    'precision': result['eval_result']['precision'],
                    'recall': result['eval_result']['recall'],
                    'f1_score': result['eval_result']['f1_score'],
                    'roc_auc': result['eval_result']['roc_auc'],
                    'confusion_matrix': result['eval_result']['confusion_matrix'].tolist()
                }
            }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"💾 Kết quả đã được lưu: {results_path}")
    
    def load_models(self, timestamp: str):
        """Load các mô hình đã train"""
        print(f"🔍 Đang load models với timestamp: {timestamp}")
        
        models_loaded = 0
        for model_name in self.models.keys():
            model_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}.pkl")
            print(f"🔍 Kiểm tra: {model_path}")
            if os.path.exists(model_path):
                try:
                    self.trained_models[model_name] = joblib.load(model_path)
                    print(f"✅ {model_name} đã được load: {model_path}")
                    models_loaded += 1
                except Exception as e:
                    print(f"❌ Lỗi khi load {model_name}: {e}")
            else:
                print(f"❌ Không tìm thấy: {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, f"scaler_{timestamp}.pkl")
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Scaler đã được load: {scaler_path}")
            except Exception as e:
                print(f"❌ Lỗi khi load scaler: {e}")
        else:
            print(f"❌ Không tìm thấy scaler: {scaler_path}")
        
        # Load results
        results_path = os.path.join(self.models_dir, f"results_{timestamp}.json")
        if os.path.exists(results_path):
            try:
                import json
                with open(results_path, 'r') as f:
                    results_data = json.load(f)
                
                # Chuyển đổi lại thành numpy arrays
                for model_name, result in results_data.items():
                    result['eval_result']['confusion_matrix'] = np.array(
                        result['eval_result']['confusion_matrix']
                    )
                    self.results[model_name] = result
                
                print(f"✅ Kết quả đã được load: {results_path}")
            except Exception as e:
                print(f"❌ Lỗi khi load results: {e}")
        else:
            print(f"❌ Không tìm thấy results: {results_path}")
        
        print(f"📊 Tổng cộng đã load {models_loaded} models")
        return models_loaded > 0
    
    def predict_single_image(self, image_path: str, model_name: str = None) -> Dict[str, Any]:
        """Dự đoán cho một ảnh"""
        if not self.trained_models:
            raise ValueError("Chưa có mô hình nào được load")
        
        # Trích xuất đặc trưng
        from fire_feature_extractor import FireFeatureExtractor
        feature_extractor = FireFeatureExtractor()
        
        features = feature_extractor.extract_all_features(image_path)
        vector = feature_extractor.create_feature_vector(features)
        
        # Scale features
        vector_scaled = self.scaler.transform([vector])
        
        # Dự đoán với tất cả mô hình
        predictions = {}
        
        if model_name:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} không tồn tại")
            
            model = self.trained_models[model_name]
            pred = model.predict(vector_scaled)[0]
            proba = model.predict_proba(vector_scaled)[0]
            
            predictions[model_name] = {
                'prediction': 'FIRE' if pred == 1 else 'NO FIRE',
                'confidence': max(proba),
                'probability_fire': proba[1],
                'probability_no_fire': proba[0]
            }
        else:
            for name, model in self.trained_models.items():
                pred = model.predict(vector_scaled)[0]
                proba = model.predict_proba(vector_scaled)[0]
                
                predictions[name] = {
                    'prediction': 'FIRE' if pred == 1 else 'NO FIRE',
                    'confidence': max(proba),
                    'probability_fire': proba[1],
                    'probability_no_fire': proba[0]
                }
        
        return predictions 