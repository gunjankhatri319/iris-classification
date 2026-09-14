"""
Data Processing Module
Handles data loading, exploration, and preprocessing for Iris classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_explore_data():
    """
    Loads Iris dataset into a DataFrame and displays basic summary info.
    
    Returns:
        tuple: (iris dataset object, pandas DataFrame)
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({
        0: 'Setosa', 
        1: 'Versicolor', 
        2: 'Virginica'
    })
    
    print("=" * 60)
    print("IRIS FLOWER CLASSIFICATION - DATA EXPLORATION")
    print("=" * 60)
    
    print("\n--- Head ---")
    print(df.head())
    
    print("\n--- Info ---")
    print(df.info())
    
    print("\n--- Description ---")
    print(df.describe())
    
    print("\n--- Target Counts ---")
    print(df['species'].value_counts())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    return iris, df


def visualize_data(df, iris):
    """
    Create comprehensive data visualizations.
    
    Args:
        df (DataFrame): Iris dataset as DataFrame
        iris: Iris dataset object from sklearn
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Pairplot
    print("\nCreating pairplot...")
    sns.pairplot(df, hue='species', diag_kind='hist', palette='Set2')
    plt.suptitle('Iris Features Pairplot', y=1.00)
    plt.savefig('visualizations/01_pairplot.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Pairplot saved")
    
    # Correlation heatmap
    print("Creating correlation heatmap...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/02_correlation_heatmap.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Correlation heatmap saved")
    
    # Distribution of features
    print("Creating feature distributions...")
    df[iris.feature_names].hist(figsize=(12, 8), bins=15)
    plt.suptitle('Distribution of Iris Features')
    plt.tight_layout()
    plt.savefig('visualizations/03_distributions.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Distribution plots saved")
    
    # Box plot by species
    print("Creating box plots by species...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, col in enumerate(iris.feature_names):
        ax = axes[idx // 2, idx % 2]
        df.boxplot(column=col, by='species', ax=ax)
        ax.set_title(col)
    plt.suptitle('Feature Distribution by Species')
    plt.tight_layout()
    plt.savefig('visualizations/04_boxplots.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Box plots saved")


def prepare_and_split_data(df, feature_names, test_size=0.2, random_state=42):
    """
    Splits and standardizes the feature data.
    
    Args:
        df (DataFrame): Iris dataset as DataFrame
        feature_names (list): List of feature column names
        test_size (float): Proportion of test set (default: 0.2)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    X = df[feature_names]
    y = df['target']

    # 80-20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"Training set scaled - Mean: {X_train_scaled.mean():.4f}, Std: {X_train_scaled.std():.4f}")
    print(f"Class distribution in training set:")
    for i, species in enumerate(['Setosa', 'Versicolor', 'Virginica']):
        print(f"  {species}: {sum(y_train == i)}")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_feature_names():
    """
    Get feature names from Iris dataset.
    
    Returns:
        list: List of feature names
    """
    iris = load_iris()
    return iris.feature_names