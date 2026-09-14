"""
Evaluation Module
Evaluates trained models and creates visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluator:
    """Evaluator class for model performance analysis"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_models(self, models, X_test, X_test_scaled, y_test):
        """
        Comprehensive evaluation of all models.
        
        Args:
            models (dict): Dictionary of trained models
            X_test: Unscaled test features
            X_test_scaled: Scaled test features
            y_test: Test target
        """
        print("\n" + "=" * 60)
        print("DETAILED MODEL EVALUATION")
        print("=" * 60)
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            print(f"{'='*60}")
            
            # Handle scaled vs unscaled data
            if any(keyword in model_name for keyword in ['Logistic', 'SVM', 'KNN']):
                X_test_current = X_test_scaled
            else:
                X_test_current = X_test
            
            # Predictions
            y_pred = model.predict(X_test_current)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            # Classification Report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                  target_names=['Setosa', 'Versicolor', 'Virginica'],
                  zero_division=0))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)
            
            # Visualize Confusion Matrix
            self._plot_confusion_matrix(cm, model_name)
            
            self.evaluation_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'confusion_matrix': cm
            }
    
    def _plot_confusion_matrix(self, cm, model_name):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            model_name (str): Name of the model
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                    yticklabels=['Setosa', 'Versicolor', 'Virginica'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(f'visualizations/confusion_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()}.png', 
                   dpi=100, bbox_inches='tight')
        plt.close()
    
    def feature_importance_analysis(self, models, feature_names):
        """
        Analyze feature importance for tree-based models.
        
        Args:
            models (dict): Dictionary of trained models
            feature_names (list): List of feature names
        """
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        # Decision Tree feature importance
        if 'Decision Tree' in models:
            dt_importance = models['Decision Tree'].feature_importances_
            print("\nDecision Tree Feature Importance:")
            for feature, importance in zip(feature_names, dt_importance):
                print(f"  {feature}: {importance:.4f}")
        
        # Random Forest feature importance
        if 'Random Forest' in models:
            rf_importance = models['Random Forest'].feature_importances_
            print("\nRandom Forest Feature Importance:")
            for feature, importance in zip(feature_names, rf_importance):
                print(f"  {feature}: {importance:.4f}")
            
            self._plot_feature_importance(rf_importance, feature_names, 'Random Forest')
        
        # Random Forest (Tuned) feature importance
        if 'Random Forest (Tuned)' in models:
            rf_tuned_importance = models['Random Forest (Tuned)'].feature_importances_
            print("\nRandom Forest (Tuned) Feature Importance:")
            for feature, importance in zip(feature_names, rf_tuned_importance):
                print(f"  {feature}: {importance:.4f}")
            
            self._plot_feature_importance(rf_tuned_importance, feature_names, 'Random Forest (Tuned)')
    
    def _plot_feature_importance(self, importances, feature_names, model_name):
        """
        Plot feature importance.
        
        Args:
            importances: Array of feature importances
            feature_names (list): List of feature names
            model_name (str): Name of the model
        """
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances, color='steelblue')
        plt.xlabel('Importance')
        plt.title(f'{model_name} - Feature Importance')
        plt.tight_layout()
        plt.savefig(f'visualizations/feature_importance_{model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()}.png', 
                   dpi=100, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, results):
        """
        Plot accuracy comparison of all models.
        
        Args:
            results (dict): Dictionary with model names and accuracies
        """
        models = list(results.keys())
        accuracies = list(results.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, accuracies, color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{accuracy:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.ylim(0.9, 1.0)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("✓ Model comparison plot saved")
    
    def get_evaluation_summary(self):
        """
        Get summary of evaluation results.
        
        Returns:
            dict: Dictionary with evaluation results
        """
        return self.evaluation_results
    
    def print_summary(self, results):
        """
        Print summary of all models.
        
        Args:
            results (dict): Dictionary with model names and accuracies
        """
        print("\n" + "=" * 60)
        print("PROJECT SUMMARY")
        print("=" * 60)
        print("\nModel Accuracy Comparison:")
        for model_name in sorted(results, key=results.get, reverse=True):
            print(f"  {model_name}: {results[model_name]:.4f}")