"""
Streamlit Web App for Iris Flower Classification
Interactive dashboard to visualize data and make predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# Title
st.title("🌸 Iris Flower Classification Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    return iris, df

iris, df = load_data()

# Sidebar
st.sidebar.header("📊 Options")
page = st.sidebar.radio("Select Page", ["Dashboard", "Data Exploration", "Make Prediction", "Model Info"])

# =====================================================
# PAGE 1: DASHBOARD
# =====================================================
if page == "Dashboard":
    st.header("📈 Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Number of Features", 4)
    with col3:
        st.metric("Species Count", 3)
    
    st.markdown("---")
    
    # Data overview
    st.subheader("Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Class distribution
    st.subheader("Class Distribution")
    fig = px.bar(df['species'].value_counts(), 
                 labels={'index': 'Species', 'value': 'Count'},
                 color=df['species'].value_counts().index,
                 color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE 2: DATA EXPLORATION
# =====================================================
elif page == "Data Exploration":
    st.header("🔍 Data Exploration")
    
    # Feature selection
    feature1 = st.selectbox("Select Feature 1", iris.feature_names, index=0)
    feature2 = st.selectbox("Select Feature 2", iris.feature_names, index=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{feature1} Distribution")
        fig = px.histogram(df, x=feature1, color='species', 
                          nbins=20, 
                          color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(f"{feature2} Distribution")
        fig = px.histogram(df, x=feature2, color='species',
                          nbins=20,
                          color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot
    st.subheader(f"Relationship: {feature1} vs {feature2}")
    fig = px.scatter(df, x=feature1, y=feature2, color='species',
                     size_max=100,
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Correlation Matrix")
    fig = px.imshow(df[iris.feature_names].corr(),
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    labels=dict(color="Correlation"))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE 3: MAKE PREDICTION
# =====================================================
elif page == "Make Prediction":
    st.header("🔮 Make a Prediction")
    
    st.markdown("Enter iris flower measurements to predict the species:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.5, 0.1)
    
    with col2:
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)
    
    st.markdown("---")
    
    # Train model
    @st.cache_resource
    def train_model():
        X = df[iris.feature_names]
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        
        return model, scaler, accuracy
    
    model, scaler, accuracy = train_model()
    
    # Make prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = species_map[prediction[0]]
    
    # Display result
    st.subheader("🎯 Prediction Result")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Predicted Species", predicted_species)
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    with col2:
        st.subheader("Prediction Confidence")
        fig = go.Figure(data=[
            go.Bar(
                x=['Setosa', 'Versicolor', 'Virginica'],
                y=probabilities[0],
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
        ])
        fig.update_layout(
            yaxis_title="Probability",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PAGE 4: MODEL INFO
# =====================================================
elif page == "Model Info":
    st.header("ℹ️ Model Information")
    
    st.subheader("About This Project")
    st.write("""
    This is an **Iris Flower Classification** machine learning project that:
    
    - 📊 **Loads** the classic Iris dataset (150 samples, 3 species)
    - 🔄 **Trains** multiple classification algorithms
    - 📈 **Evaluates** model performance with detailed metrics
    - 🎯 **Predicts** iris species from flower measurements
    
    **Dataset Info:**
    - **Samples:** 150 iris flowers
    - **Features:** 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
    - **Classes:** 3 (Setosa, Versicolor, Virginica)
    """)
    
    st.markdown("---")
    
    st.subheader("📚 Features")
    for feature in iris.feature_names:
        st.write(f"- **{feature}**")
    
    st.markdown("---")
    
    st.subheader("🎓 Machine Learning Models")
    models_info = {
        "Logistic Regression": "Simple linear model, fast and interpretable",
        "Decision Tree": "Tree-based model, easy to visualize",
        "Random Forest": "Ensemble of trees, high accuracy",
        "Support Vector Machine": "Kernel-based model, great for complex patterns",
        "K-Nearest Neighbors": "Instance-based model, simple but effective"
    }
    
    for model_name, description in models_info.items():
        st.write(f"- **{model_name}:** {description}")
    
    st.markdown("---")
    
    st.subheader("🔧 Project Structure")
    st.code("""
iris_classification/
├── src/
│   ├── main.py
│   ├── app.py                    (This file!)
│   ├── data_processing.py
│   ├── model_training.py
│   └── evaluation.py
├── visualizations/               (Generated plots)
├── models/                       (Trained models)
└── README.md
    """)

# Footer
st.markdown("---")
st.markdown("💡 Built with Streamlit | Machine Learning | Python")