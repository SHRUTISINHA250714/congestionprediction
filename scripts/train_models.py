# """
# ============================================================================
# Final Year Project: ML-Based Predictive Congestion Control Using NS2
# File: train_models.py
# Description: Train multiple ML models for congestion prediction with FIXED params
# ============================================================================
# """

# import pandas as pd
# import numpy as np
# import os
# import sys
# import pickle
# import warnings
# warnings.filterwarnings('ignore')

# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
#                             f1_score, confusion_matrix, roc_auc_score)

# try:
#     from xgboost import XGBClassifier
#     XGBOOST_AVAILABLE = True
# except ImportError:
#     XGBOOST_AVAILABLE = False
#     print("[WARNING] XGBoost not installed. Install with: pip install xgboost")

# class ModelTrainer:
#     def __init__(self, dataset_path):
#         self.dataset_path = dataset_path
#         self.models = {}
#         self.results = {}
#         self.scaler = StandardScaler()
#         self.feature_names = None
        
#     def load_data(self):
#         print("[INFO] Loading dataset...")
#         try:
#             df = pd.read_csv(self.dataset_path)
#         except FileNotFoundError:
#             print(f"[ERROR] Dataset not found: {self.dataset_path}")
#             sys.exit(1)
            
#         print(f"[SUCCESS] Loaded dataset: {df.shape}")
        
#         drop_cols = ['time_window', 'congestion']
#         X = df.drop(columns=[col for col in drop_cols if col in df.columns])
#         y = df['congestion']
#         self.feature_names = X.columns.tolist()
        
#         # Split dataset
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )
        
#         # Scale
#         self.X_train = self.scaler.fit_transform(self.X_train)
#         self.X_test = self.scaler.transform(self.X_test)
        
#     def train_all(self):
#         # 1. Logistic Regression
#         lr = LogisticRegression(penalty='l2', C=1.0, random_state=42)
#         lr.fit(self.X_train, self.y_train)
#         self.models['Logistic Regression'] = lr
#         self._evaluate_model('Logistic Regression', lr)

#         # 2. Random Forest
#         rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
#         rf.fit(self.X_train, self.y_train)
#         self.models['Random Forest'] = rf
#         self._evaluate_model('Random Forest', rf)

#         # 3. SVM
#         svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
#         svm.fit(self.X_train, self.y_train)
#         self.models['SVM'] = svm
#         self._evaluate_model('SVM', svm)

#         # 4. XGBoost
#         if XGBOOST_AVAILABLE:
#             xgb = XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
#             xgb.fit(self.X_train, self.y_train)
#             self.models['XGBoost'] = xgb
#             self._evaluate_model('XGBoost', xgb)
            
#     def _evaluate_model(self, name, model):
#         y_pred = model.predict(self.X_test)
#         try:
#             y_pred_proba = model.predict_proba(self.X_test)[:, 1]
#             auc = roc_auc_score(self.y_test, y_pred_proba)
#         except:
#             auc = 0.0
            
#         accuracy = accuracy_score(self.y_test, y_pred)
#         precision = precision_score(self.y_test, y_pred, zero_division=0)
#         recall = recall_score(self.y_test, y_pred, zero_division=0)
#         f1 = f1_score(self.y_test, y_pred, zero_division=0)
#         cm = confusion_matrix(self.y_test, y_pred)
        
#         cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        
#         self.results[name] = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1,
#             'roc_auc': auc,
#             'cv_mean': cv_scores.mean(),
#             'cv_std': cv_scores.std(),
#             'confusion_matrix': cm
#         }
#         print(f"[SUCCESS] {name} Evaluated -> AUC: {auc:.4f}, Acc: {accuracy:.4f}")

#     def save_comparison(self, output_dir):
#         comparison = []
#         for name, metrics in self.results.items():
#             comparison.append({
#                 'Model': name,
#                 'Accuracy': metrics['accuracy'],
#                 'Precision': metrics['precision'],
#                 'Recall': metrics['recall'],
#                 'F1_Score': metrics['f1_score'],
#                 'ROC_AUC': metrics['roc_auc'],
#                 'CV_Mean': metrics['cv_mean'],
#                 'CV_Std': metrics['cv_std']
#             })
            
#         df = pd.DataFrame(comparison)
#         os.makedirs(output_dir, exist_ok=True)
#         df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
#         return df
        
#     def save_models(self, output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#         with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
#             pickle.dump(self.scaler, f)
            
#         for name, model in self.models.items():
#             filename = f"{name.lower().replace(' ', '_')}.pkl"
#             with open(os.path.join(output_dir, filename), 'wb') as f:
#                 pickle.dump(model, f)
                
#         with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
#             pickle.dump(self.feature_names, f)
            
#     def get_best_model(self):
#         # Select by ROC-AUC
#         best_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
#         return best_name, self.results[best_name]

# def main():
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     dataset_path = os.path.join(project_root, 'dataset', 'congestion_dataset.csv')
#     models_dir = os.path.join(project_root, 'models')
#     results_dir = os.path.join(project_root, 'results')
    
#     trainer = ModelTrainer(dataset_path)
#     trainer.load_data()
#     trainer.train_all()
#     trainer.save_comparison(results_dir)
#     trainer.save_models(models_dir)
    
#     best_name, metrics = trainer.get_best_model()
#     print(f"\nBest Model Selected: {best_name} (ROC-AUC: {metrics['roc_auc']:.4f})")

# if __name__ == "__main__":
#     main()

"""
============================================================================
Final Year Project: ML-Based Predictive Congestion Control Using NS2
File: train_models.py
Description: Train multiple ML models for congestion prediction (SVM Removed)
============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Install with: pip install xgboost")


class ModelTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        print("[INFO] Loading dataset...")
        try:
            df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            print(f"[ERROR] Dataset not found: {self.dataset_path}")
            sys.exit(1)
            
        print(f"[SUCCESS] Loaded dataset: {df.shape}")
        
        drop_cols = ['time_window', 'congestion']
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df['congestion']
        self.feature_names = X.columns.tolist()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def train_all(self):

        # 1️⃣ Logistic Regression
        lr = LogisticRegression(penalty='l2', C=1.0, random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        self._evaluate_model('Logistic Regression', lr)

        # 2️⃣ Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        self._evaluate_model('Random Forest', rf)

        # 3️⃣ XGBoost (Optional)
        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(
                learning_rate=0.1,
                max_depth=6,
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            xgb.fit(self.X_train, self.y_train)
            self.models['XGBoost'] = xgb
            self._evaluate_model('XGBoost', xgb)

    def _evaluate_model(self, name, model):
        y_pred = model.predict(self.X_test)

        try:
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, y_pred_proba)
        except:
            auc = 0.0
            
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred)
        
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        
        self.results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': cm
        }

        print(f"[SUCCESS] {name} Evaluated -> AUC: {auc:.4f}, Acc: {accuracy:.4f}")

    def save_comparison(self, output_dir):
        comparison = []
        for name, metrics in self.results.items():
            comparison.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'ROC_AUC': metrics['roc_auc'],
                'CV_Mean': metrics['cv_mean'],
                'CV_Std': metrics['cv_std']
            })
            
        df = pd.DataFrame(comparison)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
        return df
        
    def save_models(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        for name, model in self.models.items():
            filename = f"{name.lower().replace(' ', '_')}.pkl"
            with open(os.path.join(output_dir, filename), 'wb') as f:
                pickle.dump(model, f)
                
        with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(self.feature_names, f)
            
    def get_best_model(self):
        best_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        return best_name, self.results[best_name]


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, 'dataset', 'congestion_dataset.csv')
    models_dir = os.path.join(project_root, 'models')
    results_dir = os.path.join(project_root, 'results')
    
    trainer = ModelTrainer(dataset_path)
    trainer.load_data()
    trainer.train_all()
    trainer.save_comparison(results_dir)
    trainer.save_models(models_dir)
    
    best_name, metrics = trainer.get_best_model()
    print(f"\nBest Model Selected: {best_name} (ROC-AUC: {metrics['roc_auc']:.4f})")


if __name__ == "__main__":
    main()