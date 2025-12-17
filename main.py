import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import joblib
import os
import warnings
import db_manager as db

warnings.filterwarnings('ignore')

# --- CONFIGURATION & STYLES ---
class AppConfig:
    @staticmethod
    def set_page_config():
        st.set_page_config(
            page_title="HeartGuard AI - Cardiac Health Analyzer",
            page_icon="❤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    @staticmethod
    def apply_custom_css():
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
            /* Common Styles */
            .main-header {
                font-size: 3rem;
                background: linear-gradient(135deg, #e74c3c, #3498db, #9b59b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: 900;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            .sub-header {
                font-size: 1.5rem;
                margin-bottom: 1.5rem;
                font-weight: 700;
                text-align: center;
                padding: 1rem;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .light-theme .sub-header { background: rgba(255, 255, 255, 0.9); color: #2c3e50; }
            .dark-theme .sub-header { background: rgba(52, 73, 94, 0.9); color: #ecf0f1; }
            
            .prediction-card, .safe-card, .warning-card {
                padding: 2rem;
                border-radius: 25px;
                color: white;
                text-align: center;
                margin: 1rem 0;
                box-shadow: 0 15px 35px rgba(0,0,0,0.2);
                border: 3px solid rgba(255,255,255,0.4);
                backdrop-filter: blur(15px);
            }
            .prediction-card { background: linear-gradient(135deg, rgba(231, 76, 60, 0.95), rgba(192, 57, 43, 0.95)); animation: pulse 2s infinite; }
            .warning-card { background: linear-gradient(135deg, #e67e22, #d35400); animation: pulse 2s infinite; }
            .safe-card { background: linear-gradient(135deg, rgba(46, 204, 113, 0.95), rgba(39, 174, 96, 0.95)); animation: float 3s ease-in-out infinite; }
            
            @keyframes pulse {
                0% { transform: scale(1); box-shadow: 0 15px 35px rgba(231, 76, 60, 0.4); }
                50% { transform: scale(1.02); box-shadow: 0 20px 40px rgba(231, 76, 60, 0.6); }
                100% { transform: scale(1); box-shadow: 0 15px 35px rgba(231, 76, 60, 0.4); }
            }
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-5px); }
                100% { transform: translateY(0px); }
            }
            div[data-baseweb="input"] {
                border-radius: 10px;
                background-color: rgba(255,255,255,0.1);
            }
        </style>
        """, unsafe_allow_html=True)

# --- CONSTANTS & MAPPINGS ---
CP_MAP = {
    0: "Typical Angina (0)",
    1: "Atypical Angina (1)",
    2: "Non-anginal Pain (2)",
    3: "Asymptomatic (3)"
}
RESTECG_MAP = {
    0: "Normal (0)",
    1: "ST-T Wave Abnormality (1)",
    2: "Left Ventricular Hypertrophy (2)"
}
SLOPE_MAP = {
    0: "Upsloping (0)",
    1: "Flat (1)",
    2: "Downsloping (2)"
}
THAL_MAP = {
    1: "Normal (1)",
    2: "Fixed Defect (2)",
    3: "Reversable Defect (3)"
}

# --- BUSINESS LOGIC ---
class PatientProfile:
    def __init__(self, age, sex, weight_kg, height_cm, medical_data):
        self.age = age
        self.sex = sex
        self.weight_kg = weight_kg
        self.height_cm = height_cm
        self.medical_data = medical_data 

    @property
    def sex_numeric(self):
        return 1 if self.sex == "Male" else 0

    def calculate_bmi(self):
        # BMI Logic
        if self.height_cm <= 0: return 0, "Invalid"
        height_m = self.height_cm / 100
        bmi = self.weight_kg / (height_m ** 2)
        category = "Normal"

        if self.age < 19:
            if self.age < 5:
                if bmi < 14: category = "Underweight"
                elif bmi < 18: category = "Healthy Weight (Child)"
                elif bmi < 19: category = "Overweight (Child)"
                else: category = "Obese (Child)"
            elif self.age < 10:
                if bmi < 14.5: category = "Underweight"
                elif bmi < 20: category = "Healthy Weight (Child)"
                elif bmi < 23: category = "Overweight (Child)"
                else: category = "Obese (Child)"
            else:
                if bmi < 16: category = "Underweight"
                elif bmi < 24: category = "Healthy Weight (Teen)"
                elif bmi < 28: category = "Overweight (Teen)"
                else: category = "Obese (Teen)"
        elif self.age >= 65:
            if bmi < 23: category = "Underweight (Elderly)"
            elif bmi < 29: category = "Normal (Elderly)"
            elif bmi < 32: category = "Overweight (Elderly)"
            else: category = "Obese (Elderly)"
        else:
            if bmi < 18.5: category = "Underweight"
            elif bmi < 25: category = "Normal"
            elif bmi < 30: category = "Overweight"
            else: category = "Obese"
            
        return bmi, category

    def to_feature_list(self):
        return [
            self.age,
            self.sex_numeric,
            self.medical_data['cp'],
            self.medical_data['trestbps'],
            self.medical_data['chol'],
            self.medical_data['fbs'],
            self.medical_data['restecg'],
            self.medical_data['thalach'],
            self.medical_data['exang'],
            self.medical_data['oldpeak'],
            self.medical_data['slope'],
            self.medical_data['ca'],
            self.medical_data['thal']
        ]

    def to_dict(self):
        data = {
            'age': self.age,
            'sex': self.sex_numeric,
            **self.medical_data
        }
        return data


class HeartDiseasePredictor:
    def __init__(self, model_path='heart_disease_rf_model.joblib', scaler_path='heart_disease_scaler.joblib'):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                           'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    def load_data(self, filepath='UCI.csv'):
        return pd.read_csv(filepath)

    def train(self, data):
        X = data[self.feature_names]
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scaling is less critical for RF but good for consistency
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Using Random Forest
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'status': 'Trained'
        }

    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                return True
            except:
                return False
        return False

    def predict(self, patient_profile: PatientProfile):
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        # 1. Standard ML Prediction
        features = patient_profile.to_feature_list()
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        probs = self.model.predict_proba(features_scaled)[0]
        
        # 2. Medical Rule Safeguards (Override Rules)
        is_override = False
        override_reason = ""
        
        chol = patient_profile.medical_data['chol']
        bp = patient_profile.medical_data['trestbps']
        oldpeak = patient_profile.medical_data['oldpeak']
        thalach = patient_profile.medical_data['thalach']
        
        # Rule: Cholesterol > 350
        if chol > 350:
            prediction = 1
            is_override = True
            override_reason += "Critical Cholesterol (>350). "
            probs[1] = max(probs[1], 0.98) 
            
        # Rule: BP > 180
        if bp > 180:
            prediction = 1
            is_override = True
            override_reason += "Critical BP (>180). "
            probs[1] = max(probs[1], 0.98)
            
        # Rule: Oldpeak > 4 (Severe ST Depression)
        if oldpeak > 4.0:
            prediction = 1
            is_override = True
            override_reason += "Severe ST Depression (>4.0). "
            probs[1] = max(probs[1], 0.95)
            
        # Rule: Max HR > 220 
        if thalach > 220:
             prediction = 1
             is_override = True
             override_reason += "Critical Heart Rate (>220). "
             probs[1] = max(probs[1], 0.95)

        return {
            'prediction': int(prediction),
            'probability_1': float(probs[1]),
            'risk_level': 'High' if prediction == 1 else 'Low',
            'confidence': float(max(probs)),
            'is_override': is_override,
            'override_reason': override_reason
        }


# --- VISUALIZATION ---
class Visualizer:
    def __init__(self, theme="light"):
        self.theme = theme
        self.colors = {"text": "white" if theme == "dark" else "#2c3e50"}

    def plot_risk_gauge(self, risk_probability):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Heart Disease Risk Score", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': '#00b894'},
                    {'range': [30, 70], 'color': '#fdcb6e'},
                    {'range': [70, 100], 'color': '#e17055'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': self.colors['text']})
        return fig

    def plot_feature_importance(self, feature_importance):
        if not feature_importance: return None
        fig = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            title="🔍 Feature Importance",
            labels={'x': 'Importance', 'y': 'Feature'},
            color=list(feature_importance.values()),
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            showlegend=False, 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': self.colors['text']},
            xaxis={'color': self.colors['text']},
            yaxis={'color': self.colors['text']}
        )
        return fig

    def plot_bmi_analysis(self, current_bmi, age):
        if age < 19:
            bmi_categories = ['Underweight', 'Healthy (Child)', 'Overweight', 'Obese']
            if age < 5: range_vals = [0, 14, 18, 19, 30]
            elif age < 10: range_vals = [0, 14.5, 20, 23, 35]
            else: range_vals = [0, 16, 24, 28, 40]
        elif age >= 65:
            bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
            range_vals = [0, 23, 29, 32, 45]
        else:
            bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
            range_vals = [0, 18.5, 25, 30, 45]

        fig = go.Figure()
        colors = ['#ff9999', '#99ff99', '#ffff99', '#ff9999']

        for i in range(4):
            fig.add_trace(go.Bar(
                x=["BMI Scale"], 
                y=[range_vals[i+1] - range_vals[i]], 
                name=bmi_categories[i], 
                marker_color=colors[i]
            ))

        fig.add_trace(go.Scatter(
            x=["BMI Scale"], y=[current_bmi], 
            mode='markers+text', 
            marker=dict(size=25, color='red', symbol='star'),
            text=[f'Your BMI: {current_bmi:.1f}'], textposition='middle right', name='You'
        ))

        title = f"⚖️ BMI Analysis ({'Child' if age < 19 else ('Elderly' if age >= 65 else 'Standard')})"
        fig.update_layout(
            title=title, barmode='stack', 
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': self.colors['text']},
            yaxis_title="BMI Value"
        )
        return fig

    def plot_comparison(self, dataset, patient_val, feature_name, color):
        fig = go.Figure()
        fig.add_trace(go.Box(y=dataset[feature_name], name=f'Dataset {feature_name}', marker_color=color))
        fig.add_trace(go.Scatter(
            x=[f'Dataset {feature_name}'], y=[patient_val], mode='markers', 
            marker=dict(size=20, color='red', symbol='diamond'), name='You'
        ))
        fig.update_layout(
            title=f"{feature_name.capitalize()} Comparison",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': self.colors['text']}
        )
        return fig


# --- REPORTING ---
class ReportGenerator:
    @staticmethod
    def generate_recommendations(prediction_result, bmi_info, patient_data):
        recs = []
        if prediction_result['risk_level'] == 'High':
            recs.extend(["Consult cardiologist immediately", "Schedule Stress testing & ECG", "Strict BP Monitoring"])
            if prediction_result.get('is_override'):
                recs.insert(0, f"**CRITICAL WARNING**: {prediction_result['override_reason']}")
        else:
            recs.extend(["Regular annual checkups", "Maintain Healthy lifestyle", "30 mins Cardio exercise daily"])
            
        if "Overweight" in bmi_info[1] or "Obese" in bmi_info[1]:
            recs.append("Consult nutritionist for weight management")
            
        if patient_data['chol'] > 200: recs.append("Limit saturated fats (Cholesterol High)")
        if patient_data['trestbps'] > 130: recs.append("Reduce salt intake (BP Elevated)")
        
        return recs

    @staticmethod
    def create_json(patient_profile: PatientProfile, prediction_result, bmi_info):
        data = patient_profile.to_dict()
        recs = ReportGenerator.generate_recommendations(prediction_result, bmi_info, data)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "patient_info": {
                "age": data['age'], "sex": patient_profile.sex,
                "bmi": round(bmi_info[0], 2), "bmi_category": bmi_info[1]
            },
            "risk_assessment": prediction_result,
            "recommendations": recs
        }
        return json.dumps(report, indent=2)

    @staticmethod
    def create_text(patient_profile: PatientProfile, prediction_result, bmi_info):
        data = patient_profile.to_dict()
        recs = ReportGenerator.generate_recommendations(prediction_result, bmi_info, data)
        return f"""
HEARTGUARD AI REPORT
Date: {datetime.now()}
Patient: {data['age']} yrs, {patient_profile.sex}
BMI: {bmi_info[0]:.1f} ({bmi_info[1]})

Risk: {prediction_result['risk_level']} ({prediction_result['probability_1']:.1%})
{f"WARN: {prediction_result['override_reason']}" if prediction_result.get('is_override') else ""}

RECOMMENDATIONS:
- """ + "\n- ".join(recs)


# --- MAIN APPLICATION ---
class HeartGuardApp:
    def __init__(self):
        self.config = AppConfig()
        self.predictor = HeartDiseasePredictor()
        
        if 'user' not in st.session_state: st.session_state.user = None
        if 'theme' not in st.session_state: st.session_state.theme = "light"
        db.init_db()

    def run(self):
        self.config.set_page_config()
        self.config.apply_custom_css()
        self.render_header()
        
        if st.session_state.user is None:
            self.render_auth_page()
        else:
            self.render_main_app()

    def render_auth_page(self):
        st.markdown("<h3 style='text-align: center;'>Welcome to HeartGuard AI</h3>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            _, col2, _ = st.columns([2, 2, 2])
            with col2:
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Login", use_container_width=True)
                    if submitted:
                        user = db.verify_user(username, password)
                        if user:
                            st.session_state.user = user
                            st.success(f"Welcome back, {user['full_name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
        
        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("Username")
                new_pass = st.text_input("Password", type="password")
                full_name = st.text_input("Full Name")
                col1, col2 = st.columns(2)
                age = col1.number_input("Age", 1, 120, 30)
                sex = col2.selectbox("Sex", ["Male", "Female"])
                weight = col1.number_input("Weight (kg)", 10.0, 300.0, 70.0)
                height = col2.number_input("Height (cm)", 50.0, 250.0, 170.0)
                
                submitted = st.form_submit_button("Register")
                if submitted:
                    if new_user and new_pass and full_name:
                        success, msg = db.create_user(new_user, new_pass, full_name, age, sex, weight, height)
                        if success: st.success("Account created! Please login.")
                        else: st.error(f"Error: {msg}")
                    else:
                        st.warning("Please fill all fields")

    def render_main_app(self):
        self.render_sidebar()
        data = self.predictor.load_data()
        training_results = self._ensure_model_trained(data)
        
        tab_predict, tab_history, tab_exp = st.tabs(["🏥 New Analysis", "📜 History", "📈 Health Trends"])
        
        with tab_predict:
            patient_profile = self.render_input_form()
            self.render_prediction_section(patient_profile, data, training_results)
            
        with tab_history:
            self.render_history_section()
            
        with tab_exp:
            self.render_trends_section() 
        
        self.render_footer()

    def _ensure_model_trained(self, data):
        if not self.predictor.load_model():
            with st.spinner("Training model..."):
                return self.predictor.train(data)
        else:
             return self.predictor.train(data) 

    def render_sidebar(self):
        with st.sidebar:
            st.write(f"Logged in as: **{st.session_state.user['full_name']}**")
            if st.button("🚪 Logout"):
                st.session_state.user = None
                st.rerun()
            st.markdown("---")
            st.markdown("### 🎨 Settings")
            if st.button("🌙 Toggle Theme" if st.session_state.theme == "light" else "☀️ Toggle Theme"):
                st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
                st.rerun()
            st.markdown("---")
            st.info("AI Model: Random Forest (v2.1)\nAccuracy: ~90%")

    def render_header(self):
        theme_class = "dark-theme" if st.session_state.theme == "dark" else "light-theme"
        st.markdown(f'<div class="{theme_class}">', unsafe_allow_html=True)
        st.markdown('<div class="main-header">❤️ HeartGuard AI</div>', unsafe_allow_html=True)

    def render_input_form(self):
        st.markdown('<div class="sub-header">👤 Patient Data</div>', unsafe_allow_html=True)
        user = st.session_state.user
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 2, 100, user['age'] if user else 50)
            weight = st.number_input("Weight (kg)", 5.0, 200.0, user['weight_kg'] if user else 70.0)
            height = st.number_input("Height (cm)", 50.0, 220.0, user['height_cm'] if user else 170.0)
            sex_idx = 0 if user and user['sex'] == "Male" else 1
            sex = st.radio("Sex", ["Male", "Female"], index=sex_idx)
            
            # Real-time BMI
            temp_profile = PatientProfile(age, sex, weight, height, {})
            bmi, bmi_cat = temp_profile.calculate_bmi()
            self._display_bmi_widget(bmi, bmi_cat, age)

        with col2:
            cp_label = st.selectbox("Chest Pain Type", list(CP_MAP.values()))
            cp = [k for k, v in CP_MAP.items() if v == cp_label][0]
            
            trestbps = st.slider("Resting BP (mm Hg)", 90, 250, 120, help="High BP (>180) is critical.")
            chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200, help=">240 is high. >350 is critical.")
            fbs = 1 if st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"]) == "Yes" else 0
            
            restecg_label = st.selectbox("Resting ECG Result", list(RESTECG_MAP.values()))
            restecg = [k for k, v in RESTECG_MAP.items() if v == restecg_label][0]

        col3, col4 = st.columns(2)
        with col3:
            thalach = st.slider("Max Heart Rate", 60, 220, 150)
            exang = 1 if st.radio("Exercise Induced Angina", ["No", "Yes"]) == "Yes" else 0
            oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
        with col4:
            slope_label = st.selectbox("Slope of Peak Exercise ST", list(SLOPE_MAP.values()))
            slope = [k for k, v in SLOPE_MAP.items() if v == slope_label][0]
            
            ca = st.slider("Major Vessels Colored by Flourosopy (0-3)", 0, 3, 0)
            
            thal_label = st.selectbox("Thalassemia", list(THAL_MAP.values()))
            thal = [k for k, v in THAL_MAP.items() if v == thal_label][0]

        medical_data = {
            'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
            'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        
        return PatientProfile(age, sex, weight, height, medical_data)

    def _display_bmi_widget(self, bmi, bmi_cat, age):
        bg_color = "linear-gradient(135deg, #3498db, #9b59b6)" if st.session_state.theme == "light" else "linear-gradient(135deg, #e74c3c, #e67e22)"
        
        if age < 19:
             st.markdown(f"""
                <div style='background: {bg_color}; padding: 1rem; border-radius: 15px; color: white; text-align: center;'>
                    <h4>Child Growth Status</h4>
                    <h2>{bmi:.1f}</h2>
                    <p style='font-size: 1.2rem; font-weight: bold; color: #ffeaa7;'>{bmi_cat}</p>
                    <small>BMI {bmi:.1f} is {bmi_cat.split('(')[0]} for age {age}</small>
                </div>
            """, unsafe_allow_html=True)
        else:
             st.markdown(f"""
                <div style='background: {bg_color}; padding: 1rem; border-radius: 15px; color: white; text-align: center;'>
                    <h4>BMI</h4>
                    <h2>{bmi:.1f}</h2>
                    <p>{bmi_cat}</p>
                </div>
            """, unsafe_allow_html=True)

    def render_prediction_section(self, patient: PatientProfile, data, training_results):
        if st.button("🔍 Analyze Risk", use_container_width=True):
            try:
                result = self.predictor.predict(patient)
                bmi_info = patient.calculate_bmi()
                
                # Save to DB
                if st.session_state.user:
                    db.add_medical_record(st.session_state.user['id'], patient.to_dict(), result)
                    st.toast("Result saved to history!", icon="💾")
                
                # Display Card
                if result.get('is_override'):
                     st.markdown(f"""<div class="warning-card">
                        <h2>⚠️ CRITICAL ALERT</h2>
                        <h3>{result['override_reason']}</h3>
                        <p>High Risk Assumed regardless of other factors.</p>
                     </div>""", unsafe_allow_html=True)
                elif result['prediction'] == 1:
                     st.markdown(f"""<div class="prediction-card"><h2>🚨 HIGH RISK ({result['probability_1']:.1%})</h2></div>""", unsafe_allow_html=True)
                else:
                     st.markdown(f"""<div class="safe-card"><h2>✅ LOW RISK ({result['probability_1']:.1%})</h2></div>""", unsafe_allow_html=True)

                # Visualizations
                viz = Visualizer(st.session_state.theme)
                t1, t2, t3, t4 = st.tabs(["🎯 Risk", "📊 Features", "⚖️ BMI", "👥 Compare"])
                
                with t1: st.plotly_chart(viz.plot_risk_gauge(result['probability_1']), width='stretch')
                with t2: st.plotly_chart(viz.plot_feature_importance(training_results['feature_importance']), width='stretch')
                with t3: st.plotly_chart(viz.plot_bmi_analysis(bmi_info[0], patient.age), width='stretch')
                with t4: 
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(viz.plot_comparison(data, patient.age, 'age', '#4ecdc4'), width='stretch')
                    with c2: st.plotly_chart(viz.plot_comparison(data, patient.medical_data['chol'], 'chol', '#ff6b6b'), width='stretch')

                # Report Download
                json_rep = ReportGenerator.create_json(patient, result, bmi_info)
                txt_rep = ReportGenerator.create_text(patient, result, bmi_info)
                
                d1, d2 = st.columns(2)
                d1.download_button("📥 JSON Report", json_rep, "report.json", "application/json")
                d2.download_button("📄 Text Report", txt_rep, "report.txt", "text/plain")

            except Exception as e:
                st.error(f"Error: {e}")

    def render_history_section(self):
        st.subheader("📜 Medical History")
        user = st.session_state.user
        if not user:
            st.warning("Please login to view history.")
            return
            
        history = db.get_user_history(user['id'])
        
        if not history:
            st.info("No medical records found.")
            return
            
        for record in history:
            with st.expander(f"{record['timestamp']} - Risk: {record['risk_level']} ({float(record['probability']):.1%})"):
                md = record['medical_data']
                c1, c2, c3 = st.columns(3)
                c1.write(f"**BP:** {md.get('trestbps')} | **Chol:** {md.get('chol')}")
                c2.write(f"**Recall:** {record['risk_level']}")
                if 'Critical' in str(record.get('medical_data')):
                    st.error("Includes Critical Vitals Override")
    
    def render_trends_section(self):
        st.subheader("📈 Health Trends (Vitals over Time)")
        user = st.session_state.user
        if not user:
            st.warning("Please login to see your health trends.")
            return

        history = db.get_user_history(user['id'])
        if not history or len(history) < 2:
            st.info("Not enough data to generate trends. Please run more analyses to build your history!")
            return

        # Flatten data for dataframe
        data_list = []
        for r in history:
            row = {'timestamp': r['timestamp']}
            # medical_data is a dict like {'chol': 123, ...}
            row.update(r['medical_data']) 
            data_list.append(row)
        
        df = pd.DataFrame(data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Selector
        metrics = {
            'Cholesterol': 'chol',
            'Resting Blood Pressure': 'trestbps', 
            'Max Heart Rate': 'thalach',
            'ST Depression (Oldpeak)': 'oldpeak'
        }
        
        col1, col2 = st.columns([1, 3])
        with col1:
            metric_name = st.selectbox("Select Vital Sign", list(metrics.keys()))
            col_name = metrics[metric_name]
            
            # Show summary stats
            current_val = df.iloc[-1][col_name]
            prev_val = df.iloc[-2][col_name]
            delta = current_val - prev_val
            st.metric(f"Current {metric_name}", round(current_val, 2), round(delta, 2))

        with col2:
            fig = px.line(df, x='timestamp', y=col_name, markers=True, title=f" {metric_name} History")
            fig.update_traces(line_color='#e74c3c', line_width=4)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'gray'},
                xaxis_title="Date",
                yaxis_title=metric_name,
                hovermode="x unified"
            )
            st.plotly_chart(fig, width='stretch')

    def render_footer(self):
        st.markdown('</div>', unsafe_allow_html=True) # Close theme div
        st.markdown("---")
        st.markdown("<p style='text-align: center'>Powered by HeartGuard AI OOP v2.2</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    app = HeartGuardApp()
    app.run()
