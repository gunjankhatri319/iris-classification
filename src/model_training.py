"""
Model Training Module
Trains multiple classification models for Iris flower classification
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np


class IrisModelTrainer:
    """Trainer class for multiple classification models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.cv_results = {}
    
    def train_all_models(self, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
        """
        Train all five classification models.
        
        Args:
            X_train: Unscaled training features
            X_test: Unscaled testing features
            X_train_scaled: Scaled training features
            X_test_scaled: Scaled testing features
            y_train: Training target
            y_test: Testing target
        """
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        # Logistic Regression
        print("\n1. Training Logistic Regression...")
        model_lr = LogisticRegression(max_iter=200, random_state=42)
        model_lr.fit(X_train_scaled, y_train)
        accuracy_lr = model_lr.score(X_test_scaled, y_test)
        self.models['Logistic Regression'] = model_lr
        self.results['Logistic Regression'] = accuracy_lr
        print(f"   ✓ Accuracy: {accuracy_lr:.4f}")
        
        # Decision Tree
        print("\n2. Training Decision Tree...")
        model_dt = DecisionTreeClassifier(random_state=42)
        model_dt.fit(X_train, y_train)
        accuracy_dt = model_dt.score(X_test, y_test)
        self.models['Decision Tree'] = model_dt
        self.results['Decision Tree'] = accuracy_dt
        print(f"   ✓ Accuracy: {accuracy_dt:.4f}")
        
        # Random Forest
        print("\n3. Training Random Forest...")
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_rf.fit(X_train, y_train)
        accuracy_rf = model_rf.score(X_test, y_test)
        self.models['Random Forest'] = model_rf
        self.results['Random Forest'] = accuracy_rf
        print(f"   ✓ Accuracy: {accuracy_rf:.4f}")
        
        # Support Vector Machine
        print("\n4. Training Support Vector Machine...")
        model_svm = SVC(kernel='rbf', random_state=42, probability=True)
        model_svm.fit(X_train_scaled, y_train)
        accuracy_svm = model_svm.score(X_test_scaled, y_test)
        self.models['SVM'] = model_svm
        self.results['SVM'] = accuracy_svm
        print(f"   ✓ Accuracy: {accuracy_svm:.4f}")
        
        # K-Nearest Neighbors
        print("\n5. Training K-Nearest Neighbors...")
        model_knn = KNeighborsClassifier(n_neighbors=3)
        model_knn.fit(X_train_scaled, y_train)
        accuracy_knn = model_knn.score(X_test_scaled, y_test)
        self.models['KNN'] = model_knn
        self.results['KNN'] = accuracy_knn
        print(f"   ✓ Accuracy: {accuracy_knn:.4f}")
    
    def get_best_model(self):
        """
        Get the model with highest accuracy.
        
        Returns:
            tuple: (model_name, model_object)
        """
        best_model_name = max(self.results, key=self.results.get)
        best_model = self.models[best_model_name]
        return best_model_name, best_model
    
    def cross_validate_models(self, X_train, X_train_scaled, y_train, cv=5):
        """
        Perform k-fold cross-validation for all models.
        
        Args:
            X_train: Unscaled training features
            X_train_scaled: Scaled training features
            y_train: Training target
            cv (int): Number of folds
        """
        print("\n" + "=" * 60)
        print(f"K-FOLD CROSS-VALIDATION (k={cv})")
        print("=" * 60)
        
        for model_name, model in self.models.items():
            # Handle scaled vs unscaled data
            if model_name in ['Logistic Regression', 'SVM', 'KNN']:
                X_train_current = X_train_scaled
            else:
                X_train_current = X_train
            
            scores = cross_val_score(model, X_train_current, y_train, cv=cv)
            self.cv_results[model_name] = scores
            
            print(f"\n{model_name}:")
            print(f"  Fold Scores: {np.round(scores, 4)}")
            print(f"  Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    def hyperparameter_tuning(self, X_train, X_train_scaled, y_train):
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            X_train: Unscaled training features
            X_train_scaled: Scaled training features
            y_train: Training target
        """
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING (GridSearchCV)")
        print("=" * 60)
        
        # Random Forest tuning
        print("\nTuning Random Forest...")
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        grid_search_rf = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid_rf,
            cv=5,
            n_jobs=-1,
            verbose=0
        )
        grid_search_rf.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search_rf.best_params_}")
        print(f"Best cross-validation score: {grid_search_rf.best_score_:.4f}")
        self.models['Random Forest (Tuned)'] = grid_search_rf.best_estimator_
        
        # KNN tuning
        print("\nTuning K-Nearest Neighbors...")
        param_grid_knn = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
        
        grid_search_knn = GridSearchCV(
            KNeighborsClassifier(),
            param_grid_knn,
            cv=5,
            n_jobs=-1,
            verbose=0
        )
        grid_search_knn.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {grid_search_knn.best_params_}")
        print(f"Best cross-validation score: {grid_search_knn.best_score_:.4f}")
        self.models['KNN (Tuned)'] = grid_search_knn.best_estimator_
    
    def get_results_summary(self):
        """
        Get summary of all model results.
        
        Returns:
            dict: Dictionary with model names and accuracies
        """
        return self.results