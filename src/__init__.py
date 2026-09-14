"""
Iris Flower Classification Package

A complete machine learning project for classifying iris flowers into three species
using multiple classification algorithms.

Author: ML Project
Version: 1.0.0
"""

# Import main classes for easier access
from .data_processing import (
    load_and_explore_data,
    visualize_data,
    prepare_and_split_data,
    get_feature_names
)

from .model_training import IrisModelTrainer

from .evaluation import ModelEvaluator

# Package metadata
__version__ = "1.0.0"
__author__ = "ML Project"
__all__ = [
    'load_and_explore_data',
    'visualize_data',
    'prepare_and_split_data',
    'get_feature_names',
    'IrisModelTrainer',
    'ModelEvaluator'
]

__description__ = "Iris Flower Classification using Multiple ML Algorithms"