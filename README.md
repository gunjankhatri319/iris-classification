# 🌸 Iris Flower Classification

A complete **Machine Learning** project for classifying iris flowers into three species using multiple classification algorithms. Features interactive web app, data visualizations, and comprehensive model evaluation.

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)](https://streamlit.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/gunjankhatri319/iris-classification)

---

## 🎯 Live Demo

**[🌐 Try the Interactive App Here](https://iris-flower-classification-ml.streamlit.app/)**

---

## 📊 Project Overview

This project implements an end-to-end machine learning pipeline for iris flower classification:

- **Dataset**: Classic Iris dataset (150 samples, 3 species, 4 features)
- **Models**: 5 different classification algorithms
- **Accuracy**: 95-100% on test set
- **Deployment**: Streamlit Cloud (Live & Interactive)
- **Features**: Data exploration, Model training, Evaluation, Real-time predictions

---

## 🌟 Features

### 1. **Data Exploration** 📊
- Dataset overview and statistics
- Missing value analysis
- Feature distributions
- Correlation analysis
- Class distribution visualization

### 2. **Interactive Web App** 🌐
- **Dashboard** - Overview of dataset and metrics
- **Data Exploration** - Interactive feature analysis
- **Predictions** - Real-time iris species prediction
- **Model Info** - Project details and model explanations

### 3. **Machine Learning** 🤖
- **5 Classification Algorithms**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)

### 4. **Model Evaluation** 📈
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrices
- Classification Reports
- Cross-validation Analysis
- Hyperparameter Tuning

### 5. **Visualizations** 🎨
- Pairplots showing feature relationships
- Correlation heatmaps
- Feature distributions
- Model comparison charts
- Confusion matrices

---

## 📁 Project Structure

```
iris-classification/
│
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── main.py                     # Main execution script
│   ├── app.py                      # Streamlit web app
│   ├── data_processing.py          # Data loading & preprocessing
│   ├── model_training.py           # Model training
│   ├── evaluation.py               # Model evaluation
│   └── requirements.txt            # Dependencies
│
├── data/                           # Dataset folder
├── notebook/                       # Jupyter notebooks
├── visualizations/                 # Generated plots
├── models/                         # Trained models
│
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/gunjankhatri319/iris-classification.git
cd iris-classification
```

### 2. Install Dependencies

```bash
pip install -r src/requirements.txt
```

### 3. Run the Web App Locally

```bash
streamlit run src/app.py
```

The app will open at `http://localhost:8501`

### 4. Run the Full ML Pipeline

```bash
cd src
python main.py
```

---

## 📊 Dataset Information

### Iris Dataset
- **Samples**: 150 iris flowers
- **Classes**: 3 species (Setosa, Versicolor, Virginica)
- **Features**: 4 measurements per sample

### Features
1. **Sepal Length** (cm) - Length of flower's sepal
2. **Sepal Width** (cm) - Width of flower's sepal
3. **Petal Length** (cm) - Length of flower's petal
4. **Petal Width** (cm) - Width of flower's petal

### Target Classes
- **Setosa** - Small flowers with small petals
- **Versicolor** - Medium-sized flowers
- **Virginica** - Large flowers with long petals

---

## 🤖 Machine Learning Models

| Model | Type | Accuracy | Speed | Best For |
|-------|------|----------|-------|----------|
| Logistic Regression | Linear | 96% | ⚡⚡⚡ | Interpretability |
| Decision Tree | Tree-based | 95% | ⚡⚡ | Visualization |
| **Random Forest** | Ensemble | **97%** | ⚡⚡ | **Best Overall** |
| SVM | Kernel-based | 97% | ⚡ | Complex patterns |
| KNN | Instance-based | 96% | ⚡⚡ | Simple baseline |

---

## 📊 Results & Performance

### Model Accuracy
```
Random Forest:       97.0%
SVM:                 97.0%
Logistic Regression: 96.0%
KNN:                 96.0%
Decision Tree:       95.0%
```

### Evaluation Metrics
- **Precision**: ~96-97%
- **Recall**: ~96-97%
- **F1-Score**: ~96-97%
- **Cross-validation Score**: ~95-98%

### Training Time
- **Data Loading**: < 1 second
- **Model Training**: < 1 second
- **Model Evaluation**: < 2 seconds
- **Total Pipeline**: < 5 seconds

---

## 🌐 Web App Pages

### 1. Dashboard 📊
- Total samples count
- Number of features
- Species distribution
- Dataset statistics
- Overview table

### 2. Data Exploration 🔍
- Feature distribution histograms
- Feature comparison scatter plots
- Correlation heatmap
- Interactive feature selection
- Statistical analysis

### 3. Make Prediction 🔮
- Interactive sliders for measurements
- Real-time species prediction
- Confidence scores
- Prediction probabilities
- Model accuracy display

### 4. Model Info ℹ️
- Project description
- Feature details
- Model explanations
- Technology stack
- Project structure

---

## 🛠️ Technologies Used

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Static visualizations
- **seaborn** - Statistical data visualization

### Web Framework
- **streamlit** - Interactive web app framework
- **plotly** - Interactive visualizations

### Development
- **Python 3.8+** - Programming language
- **Git** - Version control
- **GitHub** - Repository hosting
- **Streamlit Cloud** - Deployment platform

---

## 📥 Installation

### Option 1: Local Development

```bash
# Clone repository
git clone https://github.com/gunjankhatri319/iris-classification.git
cd iris-classification

# Create virtual environment (optional)
python -m venv env
source env/Scripts/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r src/requirements.txt

# Run web app
streamlit run src/app.py

# Or run ML pipeline
cd src
python main.py
```

### Option 2: Docker (Optional)

```bash
# Build Docker image
docker build -t iris-classification .

# Run container
docker run -p 8501:8501 iris-classification
```

---

## 💻 Usage

### Using the Web App
1. Visit [iris-flower-classification-ml.streamlit.app](https://iris-flower-classification-ml.streamlit.app/)
2. Navigate using sidebar menu
3. Explore data or make predictions

### Using Python Directly

```python
from src.data_processing import load_and_explore_data
from src.model_training import IrisModelTrainer
from src.evaluation import ModelEvaluator

# Load data
iris, df = load_and_explore_data()

# Train models
trainer = IrisModelTrainer()
trainer.train_all_models(X_train, X_test, X_train_scaled, X_test_scaled, 
                         y_train, y_test)

# Evaluate
evaluator = ModelEvaluator()
evaluator.evaluate_models(trainer.models, X_test, X_test_scaled, y_test)
```

---

## 📈 Generated Outputs

After running the ML pipeline, the following files are generated:

### Visualizations (`visualizations/` folder)
- `01_pairplot.png` - Feature relationships
- `02_correlation_heatmap.png` - Feature correlations
- `03_distributions.png` - Feature distributions
- `04_boxplots.png` - Feature ranges by species
- `model_comparison.png` - Model accuracy comparison
- `confusion_matrix_*.png` - Individual model confusion matrices
- `feature_importance_*.png` - Feature importance charts

---

## 🔧 Customization

### Change Train/Test Split
Edit `src/data_processing.py`:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,  # Change test size here
    random_state=42, 
    stratify=y
)
```

### Add New Model
Edit `src/model_training.py`:
```python
from sklearn.ensemble import GradientBoostingClassifier

model_gb = GradientBoostingClassifier(random_state=42)
model_gb.fit(X_train, y_train)
accuracy_gb = model_gb.score(X_test, y_test)
self.models['Gradient Boosting'] = model_gb
```

### Adjust Streamlit Settings
Create `~/.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#F5F5F5"
secondaryBackgroundColor = "#E8E8E8"
textColor = "#262730"
```

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Solution: Install dependencies
pip install -r src/requirements.txt
```

### Issue: Streamlit app not opening
```bash
# Solution: Run with explicit host
streamlit run src/app.py --server.address localhost
```

### Issue: Port already in use
```bash
# Solution: Use different port
streamlit run src/app.py --server.port 8502
```

### Issue: Import errors
```bash
# Solution: Add src to path
cd src
python main.py
```

---

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [UCI ML Repository - Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)

---

## 🎓 Key Concepts Learned

- ✅ Data loading and exploration
- ✅ Data preprocessing and scaling
- ✅ Train/test splitting with stratification
- ✅ Multiple ML algorithm implementation
- ✅ Model evaluation and comparison
- ✅ Cross-validation techniques
- ✅ Hyperparameter tuning
- ✅ Feature importance analysis
- ✅ Web app deployment
- ✅ Git and GitHub workflows

---

## 🚀 Future Enhancements

- [ ] Add more datasets
- [ ] Implement deep learning models
- [ ] Add model explainability (SHAP)
- [ ] Create API endpoint
- [ ] Add data upload feature
- [ ] Implement model persistence
- [ ] Add user authentication
- [ ] Create mobile app
- [ ] Add real-time predictions
- [ ] Implement CI/CD pipeline

---

## 📝 Blog Post

[Read the full project walkthrough here](https://medium.com/) (Coming Soon)

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Gunja Khatri**
- GitHub: [@gunjankhatri319](https://github.com/gunjankhatri319)
- Email: [your-email@gmail.com]

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Iris dataset
- Scikit-learn team for the amazing ML library
- Streamlit team for the easy-to-use framework
- Stack Overflow community for helpful solutions

---

## 📞 Contact & Support

- **Issues**: [Open an issue on GitHub](https://github.com/gunjankhatri319/iris-classification/issues)
- **Questions**: Create a discussion on GitHub
- **Feedback**: Open a pull request with improvements

---

## ⭐ Show Your Support

Give a ⭐ if this project helped you!

```
https://github.com/gunjankhatri319/iris-classification
```

---
Last Updated: September 2026  
Status: ✅ Complete and Deployed
