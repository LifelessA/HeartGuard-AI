import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64
from datetime import datetime
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="HeartGuard AI - Cardiac Health Analyzer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light/dark theme
st.markdown("""
<style>
    /* Light Theme (Default) */
    .light-theme {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2c3e50;
    }
    
    .light-theme .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        color: #2c3e50;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.8);
    }
    
    .light-theme .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        color: #2c3e50;
    }
    
    /* Dark Theme */
    .dark-theme {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 50%, #2c3e50 100%);
        color: #ecf0f1;
    }
    
    .dark-theme .main .block-container {
        background: rgba(44, 62, 80, 0.95);
        color: #ecf0f1;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .dark-theme .sidebar .sidebar-content {
        background: rgba(44, 62, 80, 0.95);
        color: #ecf0f1;
    }
    
    /* Common Styles */
    .main-header {
        font-size: 4rem;
        background: linear-gradient(135deg, #e74c3c, #3498db, #9b59b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .light-theme .main-header {
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .dark-theme .main-header {
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
    }
    
    .sub-header {
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .light-theme .sub-header {
        background: rgba(255, 255, 255, 0.9);
        color: #2c3e50;
    }
    
    .dark-theme .sub-header {
        background: rgba(52, 73, 94, 0.9);
        color: #ecf0f1;
    }
    
    .metric-card {
        padding: 1.5rem;
        border-radius: 20px;
        margin: 0.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 2px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    .light-theme .metric-card {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.9), rgba(155, 89, 182, 0.9));
        color: white;
    }
    
    .dark-theme .metric-card {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.9), rgba(230, 126, 34, 0.9));
        color: white;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .prediction-card {
        padding: 2.5rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border: 3px solid rgba(255,255,255,0.4);
        backdrop-filter: blur(15px);
        animation: pulse 2s infinite;
    }
    
    .light-theme .prediction-card {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.95), rgba(192, 57, 43, 0.95));
    }
    
    .dark-theme .prediction-card {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.95), rgba(192, 57, 43, 0.95));
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 15px 35px rgba(231, 76, 60, 0.4); }
        50% { transform: scale(1.02); box-shadow: 0 20px 40px rgba(231, 76, 60, 0.6); }
        100% { transform: scale(1); box-shadow: 0 15px 35px rgba(231, 76, 60, 0.4); }
    }
    
    .safe-card {
        padding: 2.5rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border: 3px solid rgba(255,255,255,0.4);
        backdrop-filter: blur(15px);
        animation: float 3s ease-in-out infinite;
    }
    
    .light-theme .safe-card {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.95), rgba(39, 174, 96, 0.95));
    }
    
    .dark-theme .safe-card {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.95), rgba(39, 174, 96, 0.95));
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .training-status {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
        border: 2px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
    }
    
    .light-theme .training-status {
        background: linear-gradient(135deg, rgba(241, 196, 15, 0.9), rgba(243, 156, 18, 0.9));
        color: #2c3e50;
    }
    
    .dark-theme .training-status {
        background: linear-gradient(135deg, rgba(241, 196, 15, 0.9), rgba(243, 156, 18, 0.9));
        color: #2c3e50;
    }
    
    .training-success {
        animation: success-glow 2s infinite;
    }
    
    @keyframes success-glow {
        0%, 100% { box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3); }
        50% { box-shadow: 0 5px 25px rgba(46, 204, 113, 0.6); }
    }
    
    /* Button styling */
    .stButton>button {
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .light-theme .stButton>button {
        background: linear-gradient(135deg, #3498db, #9b59b6);
        color: white;
    }
    
    .dark-theme .stButton>button {
        background: linear-gradient(135deg, #e74c3c, #e67e22);
        color: white;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .light-theme .stButton>button:hover {
        background: linear-gradient(135deg, #9b59b6, #3498db);
    }
    
    .dark-theme .stButton>button:hover {
        background: linear-gradient(135deg, #e67e22, #e74c3c);
    }
    
    /* Form elements */
    .light-theme .stSlider>div>div>div {
        background: linear-gradient(90deg, #3498db, #9b59b6);
    }
    
    .dark-theme .stSlider>div>div>div {
        background: linear-gradient(90deg, #e74c3c, #e67e22);
    }
    
    .light-theme .stSelectbox>div>div,
    .light-theme .stNumberInput>div>div>input,
    .light-theme .stRadio>div {
        background: rgba(255,255,255,0.9);
        color: #2c3e50;
        border-radius: 10px;
    }
    
    .dark-theme .stSelectbox>div>div,
    .dark-theme .stNumberInput>div>div>input,
    .dark-theme .stRadio>div {
        background: rgba(52, 73, 94, 0.9);
        color: #ecf0f1;
        border-radius: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .light-theme .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.8);
        color: #2c3e50;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: bold;
    }
    
    .dark-theme .stTabs [data-baseweb="tab"] {
        background: rgba(52, 73, 94, 0.8);
        color: #ecf0f1;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: bold;
    }
    
    .light-theme .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db, #9b59b6) !important;
        color: white !important;
    }
    
    .dark-theme .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e74c3c, #e67e22) !important;
        color: white !important;
    }
    
    /* Theme toggle button */
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                           'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        self.model_path = 'heart_disease_model.joblib'
        self.scaler_path = 'heart_disease_scaler.joblib'
        self.is_trained = False
        
    def save_model(self):
        """Save trained model and scaler"""
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            return True
        return False
    
    def load_model(self):
        """Load pre-trained model and scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                return True
            return False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
        
    def load_and_preprocess_data(self, data):
        """Load and preprocess the heart disease data"""
        X = data[self.feature_names]
        y = data['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_model(self, data, n_neighbors=5):
        """Train KNN model using Euclidean distance"""
        # Check if model already exists
        if self.load_model():
            st.markdown(f'<div class="training-status training-info">🎯 Using pre-trained model (K={n_neighbors})</div>', unsafe_allow_html=True)
            return self.get_model_performance(data)
        
        # Show training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🚀 Loading and preprocessing data...")
        X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = self.load_and_preprocess_data(data)
        progress_bar.progress(25)
        
        status_text.text("🤖 Training KNN model with Euclidean distance...")
        # Create and train KNN model
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric='euclidean',
            weights='distance'
        )
        
        self.model.fit(X_train_scaled, y_train)
        progress_bar.progress(75)
        
        status_text.text("📊 Evaluating model performance...")
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        progress_bar.progress(100)
        
        # Save the trained model
        if self.save_model():
            status_text.text("✅ Model trained and saved successfully!")
        else:
            status_text.text("⚠️ Model trained but could not be saved!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_importance': self.get_feature_importance(X_train, y_train),
            'model_trained': True
        }
    
    def get_model_performance(self, data):
        """Get model performance metrics without retraining"""
        X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = self.load_and_preprocess_data(data)
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_importance': self.get_feature_importance(X_train, y_train),
            'model_trained': False  # Model was loaded, not trained
        }
    
    def get_feature_importance(self, X_train, y_train):
        """Calculate feature importance using permutation importance approximation"""
        try:
            from sklearn.inspection import permutation_importance
            base_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
            base_model.fit(X_train, y_train)
            
            result = permutation_importance(
                base_model, X_train, y_train, n_repeats=10, random_state=42
            )
            
            return dict(zip(self.feature_names, result.importances_mean))
        except:
            # Return default importance if calculation fails
            return {feature: 1.0 for feature in self.feature_names}
    
    def predict_single(self, features):
        """Predict heart disease for a single patient"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'probability_0': float(probability[0]),
            'probability_1': float(probability[1]),
            'risk_level': 'High' if prediction == 1 else 'Low',
            'confidence': float(max(probability))
        }

def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from weight and height"""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return bmi, category

def create_medical_report(patient_data, prediction_result, bmi_info):
    """Generate comprehensive medical report in JSON format"""
    
    report = {
        "patient_info": {
            "age": patient_data['age'],
            "sex": "Male" if patient_data['sex'] == 1 else "Female",
            "bmi": round(bmi_info[0], 2),
            "bmi_category": bmi_info[1],
            "timestamp": datetime.now().isoformat()
        },
        "vital_signs": {
            "resting_blood_pressure": patient_data['trestbps'],
            "cholesterol": patient_data['chol'],
            "max_heart_rate": patient_data['thalach'],
            "blood_sugar": "High" if patient_data['fbs'] == 1 else "Normal"
        },
        "prediction_results": prediction_result,
        "risk_factors": {
            "chest_pain_type": patient_data['cp'],
            "exercise_induced_angina": "Yes" if patient_data['exang'] == 1 else "No",
            "st_depression": patient_data['oldpeak'],
            "number_of_vessels": patient_data['ca']
        },
        "recommendations": generate_recommendations(prediction_result, bmi_info, patient_data)
    }
    
    return json.dumps(report, indent=2)

def generate_recommendations(prediction_result, bmi_info, patient_data):
    """Generate personalized recommendations"""
    recommendations = []
    
    if prediction_result['risk_level'] == 'High':
        recommendations.extend([
            "Consult a cardiologist immediately",
            "Consider stress testing and ECG",
            "Monitor blood pressure regularly",
            "Maintain low-sodium diet",
            "Regular cardiovascular checkups"
        ])
    else:
        recommendations.extend([
            "Continue regular health checkups",
            "Maintain healthy lifestyle",
            "Regular cardiovascular exercise",
            "Balanced diet recommended"
        ])
    
    if bmi_info[1] in ["Overweight", "Obese"]:
        recommendations.extend([
            "Weight management program recommended",
            "Consult nutritionist for diet plan",
            "Regular physical activity advised"
        ])
    
    if patient_data['chol'] > 200:
        recommendations.append("Cholesterol management needed - consider dietary changes")
    
    if patient_data['trestbps'] > 140:
        recommendations.append("Blood pressure monitoring crucial - reduce salt intake")
    
    if patient_data['thalach'] < 120:
        recommendations.append("Consider cardiac stress testing")
    
    return recommendations

def create_visualizations(data, prediction_result, patient_data, bmi_info, feature_importance, theme):
    """Create comprehensive visualizations"""
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Risk Analysis", "📊 Feature Importance", "⚖️ BMI Analysis", "👥 Patient Comparison"])
    
    with tab1:
        # Risk probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prediction_result['probability_1'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Heart Disease Risk Score", 'font': {'size': 24}},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#00b894'},
                    {'range': [30, 70], 'color': '#fdcb6e'},
                    {'range': [70, 100], 'color': '#e17055'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        if theme == "dark":
            fig.update_layout(
                height=400,
                font={'color': "white", 'family': "Arial"},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
        else:
            fig.update_layout(
                height=400,
                font={'color': "darkblue", 'family': "Arial"},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Feature importance
        if feature_importance:
            fig = px.bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h',
                title="🔍 Feature Importance in Heart Disease Prediction",
                labels={'x': 'Importance Score', 'y': 'Medical Features'},
                color=list(feature_importance.values()),
                color_continuous_scale='viridis'
            )
            
            if theme == "dark":
                fig.update_layout(
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(color='white'),
                    yaxis=dict(color='white')
                )
            else:
                fig.update_layout(
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50')
                )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # BMI analysis with healthy ranges
        bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
        bmi_ranges = [18.5, 25, 30, 40]
        current_bmi = bmi_info[0]
        
        fig = go.Figure()
        
        # Add BMI ranges
        colors = ['#ff9999', '#99ff99', '#ffff99', '#ff9999']
        for i in range(len(bmi_ranges)-1):
            fig.add_trace(go.Bar(
                x=[bmi_categories[i]],
                y=[bmi_ranges[i+1] - bmi_ranges[i]],
                name=bmi_categories[i],
                marker_color=colors[i],
                opacity=0.7
            ))
        
        # Add patient's BMI
        fig.add_trace(go.Scatter(
            x=[bmi_info[1]],
            y=[current_bmi],
            mode='markers+text',
            marker=dict(size=25, color='red', symbol='star', line=dict(width=2, color='darkred')),
            text=[f'Your BMI: {current_bmi:.1f}'],
            textposition='top center',
            name='Your BMI'
        ))
        
        if theme == "dark":
            fig.update_layout(
                title="⚖️ BMI Analysis with Healthy Ranges",
                xaxis_title="BMI Categories",
                yaxis_title="BMI Value",
                barmode='stack',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(color='white'),
                yaxis=dict(color='white')
            )
        else:
            fig.update_layout(
                title="⚖️ BMI Analysis with Healthy Ranges",
                xaxis_title="BMI Categories",
                yaxis_title="BMI Value",
                barmode='stack',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50')
            )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Compare with dataset statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Age comparison
            fig_age = go.Figure()
            fig_age.add_trace(go.Box(y=data['age'], name='Dataset Age', boxpoints='outliers',
                                   marker_color='#4ecdc4', line_color='#2c3e50'))
            fig_age.add_trace(go.Scatter(x=['Dataset Age'], y=[patient_data['age']], 
                                     mode='markers', name='Patient Age',
                                     marker=dict(size=20, color='red', symbol='diamond',
                                               line=dict(width=2, color='darkred'))))
            
            if theme == "dark":
                fig_age.update_layout(
                    title="📅 Age Comparison", 
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(color='white'),
                    yaxis=dict(color='white')
                )
            else:
                fig_age.update_layout(
                    title="📅 Age Comparison", 
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50')
                )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Cholesterol comparison
            fig_chol = go.Figure()
            fig_chol.add_trace(go.Box(y=data['chol'], name='Dataset Cholesterol', boxpoints='outliers',
                                    marker_color='#ff6b6b', line_color='#2c3e50'))
            fig_chol.add_trace(go.Scatter(x=['Dataset Cholesterol'], y=[patient_data['chol']], 
                                       mode='markers', name='Patient Cholesterol',
                                       marker=dict(size=20, color='blue', symbol='diamond',
                                                 line=dict(width=2, color='darkblue'))))
            
            if theme == "dark":
                fig_chol.update_layout(
                    title="🩸 Cholesterol Comparison", 
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(color='white'),
                    yaxis=dict(color='white')
                )
            else:
                fig_chol.update_layout(
                    title="🩸 Cholesterol Comparison", 
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50')
                )
            st.plotly_chart(fig_chol, use_container_width=True)

def initialize_app():
    """Initialize the app with model training"""
    
    # Load data
    @st.cache_data
    def load_data():
        data = pd.read_csv('UCI.csv')
        return data
    
    data = load_data()
    
    # Initialize and train model
    predictor = HeartDiseasePredictor()
    
    # Try to load existing model first
    if not predictor.load_model():
        # Model doesn't exist, train new one
        st.markdown('<div class="training-status training-info">🚀 Training Heart Disease Detection Model...</div>', unsafe_allow_html=True)
        
        # Default parameters for initial training
        n_neighbors = 5
        results = predictor.train_model(data, n_neighbors)
        
        if results.get('model_trained', False):
            st.markdown(f'<div class="training-status training-success">✅ Model trained successfully! Accuracy: {results["accuracy"]:.2%}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="training-status training-success">✅ Model loaded from cache! Accuracy: {results["accuracy"]:.2%}</div>', unsafe_allow_html=True)
    else:
        # Model loaded successfully
        results = predictor.get_model_performance(data)
        st.markdown(f'<div class="training-status training-success">🎯 Pre-trained model loaded! Accuracy: {results["accuracy"]:.2%}</div>', unsafe_allow_html=True)
    
    return predictor, data, results

def main():
    # Theme toggle
    if 'theme' not in st.session_state:
        st.session_state.theme = "light"
    
    # Theme toggle button in sidebar
    with st.sidebar:
        st.markdown("### 🎨 Theme Settings")
        if st.button("🌙 Switch to Dark Mode" if st.session_state.theme == "light" else "☀️ Switch to Light Mode"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    
    # Apply theme
    theme_class = "dark-theme" if st.session_state.theme == "dark" else "light-theme"
    st.markdown(f'<div class="{theme_class}">', unsafe_allow_html=True)
    
    # Header with enhanced styling
    st.markdown('<div class="main-header">❤️ HeartGuard AI</div>', unsafe_allow_html=True)
    
    if st.session_state.theme == "light":
        st.markdown('<div style="text-align: center; font-size: 1.4rem; color: #2c3e50; margin-bottom: 2rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">Advanced Cardiac Health Analysis using Machine Learning</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: center; font-size: 1.4rem; color: #ecf0f1; margin-bottom: 2rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Advanced Cardiac Health Analysis using Machine Learning</div>', unsafe_allow_html=True)
    
    # Initialize app and train/load model
    predictor, data, training_results = initialize_app()
    
    # Model configuration in sidebar with better styling
    if st.session_state.theme == "light":
        st.sidebar.markdown("""
        <div style='background: rgba(255,255,255,0.9); 
                    padding: 1.5rem; 
                    border-radius: 15px; 
                    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                    border: 2px solid rgba(255,255,255,0.3);'>
            <h2 style='color: #2c3e50; text-align: center; margin-bottom: 1rem;'>🔧 Model Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style='background: rgba(52, 73, 94, 0.9); 
                    padding: 1.5rem; 
                    border-radius: 15px; 
                    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                    border: 2px solid rgba(255,255,255,0.1);'>
            <h2 style='color: #ecf0f1; text-align: center; margin-bottom: 1rem;'>🔧 Model Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.theme == "light":
        st.sidebar.markdown(f"""
        <div style='background: rgba(255,255,255,0.9); 
                    padding: 1rem; 
                    border-radius: 10px; 
                    margin: 1rem 0;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50; margin: 0;'>Current Accuracy</h3>
            <h2 style='color: #3498db; margin: 0;'>{training_results['accuracy']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"""
        <div style='background: rgba(52, 73, 94, 0.9); 
                    padding: 1rem; 
                    border-radius: 10px; 
                    margin: 1rem 0;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
            <h3 style='color: #ecf0f1; margin: 0;'>Current Accuracy</h3>
            <h2 style='color: #e74c3c; margin: 0;'>{training_results['accuracy']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Retrain option
    if st.sidebar.button("🔄 Retrain Model", use_container_width=True):
        with st.spinner("Retraining model with current parameters..."):
            new_results = predictor.train_model(data, n_neighbors=5)
            st.sidebar.success(f"Model retrained! New Accuracy: {new_results['accuracy']:.2%}")
            training_results = new_results
    
    # Main content - Patient input form
    st.markdown('<div class="sub-header">👤 Patient Health Assessment</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.theme == "light":
            st.markdown("""
            <div style='background: rgba(255,255,255,0.9); 
                        padding: 1.5rem; 
                        border-radius: 15px; 
                        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                        margin-bottom: 1rem;'>
                <h3 style='color: #2c3e50;'>Personal Information</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(52, 73, 94, 0.9); 
                        padding: 1.5rem; 
                        border-radius: 15px; 
                        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                        margin-bottom: 1rem;'>
                <h3 style='color: #ecf0f1;'>Personal Information</h3>
            </div>
            """, unsafe_allow_html=True)
        
        age = st.slider("Age", 20, 100, 50)
        weight = st.number_input("Weight (kg)", 40.0, 200.0, 70.0)
        height = st.number_input("Height (cm)", 140.0, 220.0, 170.0)
        sex = st.radio("Sex", ["Male", "Female"])
        
        # Calculate BMI
        bmi, bmi_category = calculate_bmi(weight, height)
        if st.session_state.theme == "light":
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(52, 152, 219, 0.9), rgba(155, 89, 182, 0.9));
                        padding: 1rem; 
                        border-radius: 15px; 
                        color: white;
                        text-align: center;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.2);'>
                <h4 style='margin: 0;'>BMI</h4>
                <h2 style='margin: 0;'>{bmi:.1f}</h2>
                <p style='margin: 0; font-size: 1.1rem;'>{bmi_category}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(231, 76, 60, 0.9), rgba(230, 126, 34, 0.9));
                        padding: 1rem; 
                        border-radius: 15px; 
                        color: white;
                        text-align: center;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.3);'>
                <h4 style='margin: 0;'>BMI</h4>
                <h2 style='margin: 0;'>{bmi:.1f}</h2>
                <p style='margin: 0; font-size: 1.1rem;'>{bmi_category}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.theme == "light":
            st.markdown("""
            <div style='background: rgba(255,255,255,0.9); 
                        padding: 1.5rem; 
                        border-radius: 15px; 
                        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                        margin-bottom: 1rem;'>
                <h3 style='color: #2c3e50;'>Medical Parameters</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(52, 73, 94, 0.9); 
                        padding: 1.5rem; 
                        border-radius: 15px; 
                        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                        margin-bottom: 1rem;'>
                <h3 style='color: #ecf0f1;'>Medical Parameters</h3>
            </div>
            """, unsafe_allow_html=True)
        
        cp = st.selectbox("Chest Pain Type", 
                         [("0: Typical Angina", 0), 
                          ("1: Atypical Angina", 1),
                          ("2: Non-anginal Pain", 2),
                          ("3: Asymptomatic", 3)], format_func=lambda x: x[0])[1]
        
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG", 
                              [("0: Normal", 0),
                               ("1: ST-T Wave Abnormality", 1),
                               ("2: Left Ventricular Hypertrophy", 2)], format_func=lambda x: x[0])[1]
    
    col3, col4 = st.columns(2)
    
    with col3:
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression induced by exercise", 0.0, 6.0, 1.0, 0.1)
    
    with col4:
        slope = st.selectbox("Slope of Peak Exercise ST Segment",
                           [("0: Upsloping", 0),
                            ("1: Flat", 1),
                            ("2: Downsloping", 2)], format_func=lambda x: x[0])[1]
        ca = st.slider("Number of Major Vessels colored by Fluoroscopy", 0, 3, 0)
        thal = st.selectbox("Thalassemia",
                          [("1: Normal", 1),
                           ("2: Fixed Defect", 2),
                           ("3: Reversible Defect", 3)], format_func=lambda x: x[0])[1]
    
    # Prepare patient data
    patient_data = {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == "Yes" else 0,
        'restecg': restecg,
        'thalach': thalach,
        'exang': 1 if exang == "Yes" else 0,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # Prediction button with enhanced styling
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Analyze Heart Disease Risk", use_container_width=True):
        try:
            # Make prediction
            prediction_result = predictor.predict_single(list(patient_data.values()))
            bmi_info = (bmi, bmi_category)
            
            # Display results
            if prediction_result['prediction'] == 1:
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>🚨 HIGH RISK DETECTED</h2>
                    <h3>Probability of Heart Disease: {prediction_result['probability_1']:.1%}</h3>
                    <p>Confidence: {prediction_result['confidence']:.1%} | Immediate medical consultation recommended</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-card">
                    <h2>✅ LOW RISK</h2>
                    <h3>Probability of Heart Disease: {prediction_result['probability_1']:.1%}</h3>
                    <p>Confidence: {prediction_result['confidence']:.1%} | Continue maintaining healthy lifestyle</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Create visualizations
            create_visualizations(data, prediction_result, patient_data, bmi_info, training_results['feature_importance'], st.session_state.theme)
            
            # Generate and display report
            st.markdown('<div class="sub-header">📊 Comprehensive Medical Report</div>', unsafe_allow_html=True)
            report_json = create_medical_report(patient_data, prediction_result, bmi_info)
            
            col5, col6 = st.columns(2)
            with col5:
                st.download_button(
                    label="📥 Download JSON Report",
                    data=report_json,
                    file_name=f"heart_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col6:
                # Create a simple text report for download
                text_report = f"""
HEARTGUARD AI - CARDIAC HEALTH REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT SUMMARY:
- Age: {patient_data['age']}
- Sex: {'Male' if patient_data['sex'] == 1 else 'Female'}
- BMI: {bmi_info[0]:.1f} ({bmi_info[1]})

VITAL SIGNS:
- Resting BP: {patient_data['trestbps']} mm Hg
- Cholesterol: {patient_data['chol']} mg/dl
- Max Heart Rate: {patient_data['thalach']} bpm
- Blood Sugar: {'High' if patient_data['fbs'] == 1 else 'Normal'}

RISK ASSESSMENT:
- Heart Disease Risk: {prediction_result['probability_1']:.1%}
- Risk Level: {prediction_result['risk_level']}
- Confidence: {prediction_result['confidence']:.1%}

RECOMMENDATIONS:
{chr(10).join(generate_recommendations(prediction_result, bmi_info, patient_data))}
"""
                st.download_button(
                    label="📄 Download Text Report",
                    data=text_report,
                    file_name=f"heart_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Show JSON report
            with st.expander("🔍 View Detailed JSON Report"):
                st.json(json.loads(report_json))
                
        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")
    
    # Enhanced footer
    st.markdown("---")
    if st.session_state.theme == "light":
        st.markdown(
            """
            <div style='text-align: center; color: #2c3e50; padding: 2rem; 
                        background: rgba(255,255,255,0.8); 
                        border-radius: 15px;
                        backdrop-filter: blur(10px);
                        border: 1px solid rgba(255,255,255,0.5);'>
                <h3 style='color: #2c3e50; margin-bottom: 1rem;'>❤️ HeartGuard AI</h3>
                <p style='margin: 0; font-size: 1.1rem;'>
                    Powered by Machine Learning & Euclidean Distance KNN<br>
                    <small>For educational and medical screening purposes only</small>
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style='text-align: center; color: #ecf0f1; padding: 2rem; 
                        background: rgba(52, 73, 94, 0.8); 
                        border-radius: 15px;
                        backdrop-filter: blur(10px);
                        border: 1px solid rgba(255,255,255,0.1);'>
                <h3 style='color: #ecf0f1; margin-bottom: 1rem;'>❤️ HeartGuard AI</h3>
                <p style='margin: 0; font-size: 1.1rem;'>
                    Powered by Machine Learning & Euclidean Distance KNN<br>
                    <small>For educational and medical screening purposes only</small>
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Close theme div
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

