"""
Main Execution Script
Orchestrates the entire Iris Flower Classification pipeline
"""

# IMPORTANT: Set matplotlib backend BEFORE importing other libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging

import os
import sys
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing import load_and_explore_data, visualize_data, prepare_and_split_data, get_feature_names
from model_training import IrisModelTrainer
from evaluation import ModelEvaluator


def create_directories():
    """Create necessary directories for output files."""
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("✓ Output directories created")


def make_predictions(best_model, scaler, feature_names, needs_scaling=True):
    """
    Make predictions on new samples.
    
    Args:
        best_model: Trained model object
        scaler: StandardScaler object for feature scaling
        feature_names (list): List of feature names
        needs_scaling (bool): Whether model requires scaled features
    """
    print("\n" + "=" * 60)
    print("PREDICTIONS ON NEW SAMPLES")
    print("=" * 60)
    
    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    
    # Example samples
    samples = [
        [5.1, 3.5, 1.4, 0.2],  # Likely Setosa
        [6.0, 2.7, 5.1, 1.6],  # Likely Versicolor
        [7.7, 2.8, 6.7, 2.0]   # Likely Virginica
    ]
    
    sample_names = ['Sample 1 (Likely Setosa)', 'Sample 2 (Likely Versicolor)', 'Sample 3 (Likely Virginica)']
    
    for sample, name in zip(samples, sample_names):
        print(f"\n{name}")
        print(f"Features: {dict(zip(feature_names, sample))}")
        
        sample_array = np.array(sample).reshape(1, -1)
        
        # Apply scaling if needed
        if needs_scaling:
            sample_scaled = scaler.transform(sample_array)
            prediction = best_model.predict(sample_scaled)
            if hasattr(best_model, 'predict_proba'):
                probabilities = best_model.predict_proba(sample_scaled)
            else:
                probabilities = None
        else:
            prediction = best_model.predict(sample_array)
            if hasattr(best_model, 'predict_proba'):
                probabilities = best_model.predict_proba(sample_array)
            else:
                probabilities = None
        
        print(f"Predicted Species: {species_map[prediction[0]]}")
        
        if probabilities is not None:
            print("Prediction Probabilities:")
            for i, species in enumerate(['Setosa', 'Versicolor', 'Virginica']):
                print(f"  {species}: {probabilities[0][i]:.2%}")


def main():
    """Main execution flow."""
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "IRIS FLOWER CLASSIFICATION PROJECT" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Create output directories
    create_directories()
    
    # Step 1: Load and explore data
    print("\n[1/8] Loading and exploring data...")
    iris, df = load_and_explore_data()
    
    # Step 2: Visualize data
    print("\n[2/8] Creating visualizations...")
    visualize_data(df, iris)
    
    # Step 3: Prepare and split data
    print("\n[3/8] Preparing and splitting data...")
    feature_names = get_feature_names()
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = \
        prepare_and_split_data(df, feature_names)
    
    # Step 4: Train models
    print("\n[4/8] Training models...")
    trainer = IrisModelTrainer()
    trainer.train_all_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 5: Evaluate models
    print("\n[5/8] Evaluating models...")
    evaluator = ModelEvaluator()
    evaluator.evaluate_models(trainer.models, X_test, X_test_scaled, y_test)
    
    # Step 6: Feature importance analysis
    print("\n[6/8] Analyzing feature importance...")
    evaluator.feature_importance_analysis(trainer.models, feature_names)
    
    # Step 7: Cross-validation
    print("\n[7/8] Performing cross-validation...")
    trainer.cross_validate_models(X_train, X_train_scaled, y_train, cv=5)
    
    # Step 8: Hyperparameter tuning
    print("\n[8/8] Tuning hyperparameters...")
    trainer.hyperparameter_tuning(X_train, X_train_scaled, y_train)
    
    # Get best model and make predictions
    best_model_name, best_model = trainer.get_best_model()
    print(f"\n✓ Best Model: {best_model_name} (Accuracy: {trainer.results[best_model_name]:.4f})")
    
    # Determine if model needs scaling
    needs_scaling = any(keyword in best_model_name for keyword in ['Logistic', 'SVM', 'KNN'])
    make_predictions(best_model, scaler, feature_names, needs_scaling)
    
    # Plot model comparison
    print("\nCreating model comparison chart...")
    evaluator.plot_model_comparison(trainer.results)
    
    # Final summary
    evaluator.print_summary(trainer.results)
    
    print("\n✓ All visualizations saved to 'visualizations/' directory")
    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()