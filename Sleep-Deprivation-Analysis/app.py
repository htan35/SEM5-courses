import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path
import io
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import base64

st.set_page_config(page_title="Sleep Health Predictor", layout="wide", initial_sidebar_state="expanded")

MODEL_PATH = Path("models/sleep_prediction_model.pkl")

# ------------------
# Enhanced Styling
# ------------------
st.markdown(
    """
    <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    :root {
        --primary: #6ee7b7;
        --primary-dark: #06d6a0;
        --dark-bg: #0f1724;
        --dark-card: #0b1220;
        --text-light: #e0e0e0;
        --text-muted: #9ca3af;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-orange: #f59e0b;
        --accent-red: #ef4444;
    }
    
    body, .main { background-color: var(--dark-bg); color: var(--text-light); }
    
    .app-header {
        background: linear-gradient(135deg, #06121a 0%, #0f1f27 50%, #042f2e 100%);
        padding: 32px 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        border: 1px solid rgba(110, 231, 183, 0.1);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
    }
    
    .logo {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #6ee7b7, #06d6a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
    }
    
    .sub {
        color: rgba(230, 230, 230, 0.8);
        margin-top: 8px;
        font-size: 15px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .card {
        background: linear-gradient(135deg, #0b1220, #0d1424);
        padding: 28px;
        border-radius: 14px;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(110, 231, 183, 0.08);
        margin-bottom: 24px;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: rgba(110, 231, 183, 0.15);
        box-shadow: 0 16px 64px rgba(0, 0, 0, 0.4);
    }
    
    .card h2, .card h3 {
        color: #6ee7b7;
        margin-bottom: 16px;
        font-weight: 700;
    }
    
    .small { font-size: 13px; color: var(--text-muted); }
    
    .metric-badge {
        background: linear-gradient(135deg, rgba(110, 231, 183, 0.1), rgba(6, 214, 160, 0.05));
        padding: 14px 18px;
        border-radius: 10px;
        border: 1px solid rgba(110, 231, 183, 0.15);
        margin: 8px 0;
    }
    
    .recommend {
        background: linear-gradient(90deg, rgba(6, 214, 160, 0.08), rgba(59, 130, 246, 0.05));
        padding: 16px;
        border-radius: 10px;
        margin: 12px 0;
        border-left: 4px solid #06d6a0;
        color: var(--text-light);
        font-size: 14px;
    }
    
    .status-optimal { color: var(--accent-green); font-weight: 700; }
    .status-below { color: var(--accent-orange); font-weight: 700; }
    .status-above { color: var(--accent-red); font-weight: 700; }
    
    .footer {
        color: var(--text-muted);
        font-size: 13px;
        margin-top: 32px;
        text-align: center;
        padding-top: 20px;
        border-top: 1px solid rgba(110, 231, 183, 0.1);
    }
    
    [data-testid="stMetricValue"] { font-size: 28px !important; color: #6ee7b7 !important; }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
    
    .stButton > button {
        background: linear-gradient(135deg, #06d6a0, #06b87f);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(6, 214, 160, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(6, 214, 160, 0.3);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.2) !important;
        width: 100% !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(59, 130, 246, 0.3) !important;
    }
    
    .success-badge {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        margin: 8px 0;
    }
    
    .warning-badge {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #f59e0b;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        margin: 8px 0;
    }
    
    .divider { border: 1px solid rgba(110, 231, 183, 0.1); margin: 20px 0; }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    "<div class='app-header'><div class='logo'>🛌 Sleep Health Predictor</div><div class='sub'>AI-Powered Personalized Sleep Duration Analysis • Powered by Machine Learning + Domain Expertise</div></div>",
    unsafe_allow_html=True,
)

st.write("")

# PDF Report Generation

def generate_pdf_report(user_data, predicted, recommendations):
    """Generate a professional PDF report"""
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page 1: Title & Summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('#0f1724')
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        title_text = "🛌 Sleep Health Analysis Report"
        ax.text(0.5, 0.90, title_text, ha='center', va='top', fontsize=28, 
                fontweight='bold', color='#6ee7b7', transform=ax.transAxes)
        
        # Date
        date_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}"
        ax.text(0.5, 0.84, date_text, ha='center', va='top', fontsize=12, 
                color='#9ca3af', transform=ax.transAxes)
        
        # User Info
        user_info = f"{user_data['Gender']}, Age {user_data['Age']} | {user_data['Occupation']}"
        ax.text(0.5, 0.78, user_info, ha='center', va='top', fontsize=13, 
                color='#e0e0e0', transform=ax.transAxes)
        
        # Divider
        ax.plot([0.1, 0.9], [0.74, 0.74], color='#6ee7b7', linewidth=2, transform=ax.transAxes, alpha=0.3)
        
        # Main Result Box
        if predicted < 7:
            status = "BELOW RECOMMENDED"
            color = '#f59e0b'
        elif predicted <= 9:
            status = "OPTIMAL"
            color = '#10b981'
        else:
            status = "ABOVE RECOMMENDED"
            color = '#ef4444'
        
        # Result Box
        ax.add_patch(plt.Rectangle((0.15, 0.55), 0.7, 0.15, transform=ax.transAxes, 
                                   facecolor=color, alpha=0.1, edgecolor=color, linewidth=2))
        ax.text(0.5, 0.67, f"Optimal Sleep Duration", ha='center', va='center', 
                fontsize=16, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.60, f"{predicted} hours", ha='center', va='center', 
                fontsize=32, fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.56, status, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=color, transform=ax.transAxes)
        
        # Health Metrics Summary
        ax.text(0.5, 0.48, "Your Health Profile", ha='center', va='top', 
                fontsize=14, fontweight='bold', color='#6ee7b7', transform=ax.transAxes)
        
        metrics_text = f"""
Physical Activity: {user_data['Physical Activity Level']} min/day
Stress Level: {user_data['Stress Level']}/10
Daily Steps: {user_data['Daily Steps']:,}
Heart Rate: {user_data['Heart Rate']} bpm
BMI Category: {user_data['BMI Category']}
Blood Pressure: {user_data['Blood Pressure']}
Sleep Disorder: {user_data['Sleep Disorder'] if user_data['Sleep Disorder'] else 'None'}
        """
        
        ax.text(0.15, 0.42, metrics_text, ha='left', va='top', fontsize=11, 
                color='#e0e0e0', transform=ax.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#0b1220', alpha=0.5, edgecolor='#6ee7b7', linewidth=1))
        
        # Footer
        ax.text(0.5, 0.05, "Sleep Health Predictor | Powered by ML + Domain Expertise", 
                ha='center', va='bottom', fontsize=9, color='#9ca3af', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='#0f1724')
        plt.close()
        
        # Page 2: Recommendations
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('#0f1724')
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, "Personalized Sleep Recommendations", ha='center', va='top', 
                fontsize=20, fontweight='bold', color='#6ee7b7', transform=ax.transAxes)
        
        y_pos = 0.88
        for i, rec in enumerate(recommendations, 1):
            wrapped_text = f"{i}. {rec}"
            ax.text(0.1, y_pos, wrapped_text, ha='left', va='top', fontsize=11, 
                   color='#e0e0e0', transform=ax.transAxes, wrap=True,
                   bbox=dict(boxstyle='round', facecolor='#0b1220', alpha=0.3, 
                            edgecolor='#06d6a0', linewidth=1, pad=0.8))
            y_pos -= 0.14
        
        # General Tips
        ax.text(0.5, 0.15, "General Sleep Hygiene Tips", ha='center', va='top', 
                fontsize=12, fontweight='bold', color='#6ee7b7', transform=ax.transAxes)
        
        tips = "• Maintain consistent sleep schedule\n• Keep bedroom dark, cool, and quiet\n• Avoid screens 1 hour before bed\n• Regular exercise (not before bedtime)"
        ax.text(0.1, 0.10, tips, ha='left', va='top', fontsize=10, color='#e0e0e0', 
               transform=ax.transAxes, family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight', facecolor='#0f1724')
        plt.close()
        
        # Page 3: Insights Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.patch.set_facecolor('#0f1724')
        
        # Sleep Duration Chart
        ax1.barh(['Predicted', 'Min Recommended', 'Max Recommended'], 
                [predicted, 7, 9], color=['#6ee7b7', '#3b82f6', '#3b82f6'], 
                edgecolor='white', linewidth=1.5)
        ax1.set_xlim(0, 10)
        ax1.set_xlabel('Hours', color='#e0e0e0', fontsize=11)
        ax1.set_title('Sleep Duration Comparison', color='#6ee7b7', fontsize=13, fontweight='bold')
        ax1.tick_params(colors='#e0e0e0')
        ax1.spines['bottom'].set_color('#6ee7b7')
        ax1.spines['left'].set_color('#6ee7b7')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_facecolor('#0b1220')
        ax1.grid(axis='x', alpha=0.2, color='#6ee7b7')
        
        # Health Factors
        factors = {
            'Activity': min(user_data['Physical Activity Level'] / 120, 1.0),
            'Stress Mgmt': 1 - (user_data['Stress Level'] / 10),
            'Daily Steps': min(user_data['Daily Steps'] / 10000, 1.0),
            'Heart Rate': 1 - min(abs(user_data['Heart Rate'] - 70) / 30, 1.0),
        }
        
        colors_factors = ['#10b981', '#f59e0b', '#3b82f6', '#6ee7b7']
        ax2.barh(list(factors.keys()), list(factors.values()), color=colors_factors, 
                edgecolor='white', linewidth=1.5)
        ax2.set_xlim(0, 1.1)
        ax2.set_xlabel('Score', color='#e0e0e0', fontsize=11)
        ax2.set_title('Health Factors Profile', color='#6ee7b7', fontsize=13, fontweight='bold')
        ax2.tick_params(colors='#e0e0e0')
        ax2.spines['bottom'].set_color('#6ee7b7')
        ax2.spines['left'].set_color('#6ee7b7')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_facecolor('#0b1220')
        ax2.grid(axis='x', alpha=0.2, color='#6ee7b7')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', facecolor='#0f1724')
        plt.close()
    
    buffer.seek(0)
    return buffer

# Helper Functions

def load_model_components():
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, 'rb') as f:
                components = pickle.load(f)
            return components
        except Exception as e:
            return None
    return None

def predict_sleep_duration_advanced(user, pipeline, my_cols, use_hybrid=True):
    """Advanced sleep duration prediction using ML model with domain knowledge"""
    user_df = pd.DataFrame([user])
    
    missing_features = [col for col in my_cols if col not in user_df.columns]
    for feature in missing_features:
        if feature == 'Sleep Duration':
            user_df[feature] = 7.0
        elif 'Centroid' in feature:
            user_df[feature] = 0.5
    
    try:
        ml_prediction = pipeline.predict(user_df[my_cols])[0]
    except:
        return 7.0
    
    if not use_hybrid:
        return round(ml_prediction, 1)
    
    # Age-based adjustment
    age_factor = 0.5 if user['Age'] < 18 else (0.4 if user['Age'] < 25 else (0.2 if user['Age'] < 35 else (0.0 if user['Age'] < 50 else (-0.1 if user['Age'] < 65 else -0.2))))
    
    activity_level = user['Physical Activity Level']
    occupation = user['Occupation']
    physical_jobs = ['Nurse', 'Doctor', 'Teacher', 'Sales Representative']
    job_bonus = 0.1 if occupation in physical_jobs else 0.0
    
    if activity_level >= 75:
        activity_factor = 0.5 + job_bonus
    elif activity_level >= 50:
        activity_factor = 0.4 + job_bonus
    elif activity_level >= 30:
        activity_factor = 0.2 + job_bonus
    elif activity_level >= 15:
        activity_factor = 0.0
    else:
        activity_factor = -0.2
    
    stress_level = user['Stress Level']
    stress_factor = 0.5 if stress_level >= 8 else ((stress_level - 5) / 10) * 0.6
    
    bmi_factor = 0 if user['BMI Category'] == 'Normal' else (0.2 if user['BMI Category'] == 'Overweight' else 0.5)
    
    disorder = user.get('Sleep Disorder')
    disorder_factor = 0.8 if disorder == 'Sleep Apnea' else (0.6 if disorder == 'Insomnia' else 0.0)
    
    hr = user['Heart Rate']
    hr_factor = 0.0 if 60 <= hr <= 70 else (0.1 if 55 <= hr <= 75 else 0.3)
    
    steps = user['Daily Steps']
    steps_factor = 0.3 if steps >= 12000 else (0.2 if steps >= 10000 else (0.1 if steps >= 7500 else (0.0 if steps >= 4000 else -0.1)))
    
    gender_factor = 0.3 if (user['Gender'] == 'Female' and user['Age'] > 50) else (0.2 if user['Gender'] == 'Female' else 0.0)
    
    high_stress_jobs = ['Sales Representative', 'Doctor', 'Lawyer']
    shift_work_jobs = ['Nurse', 'Doctor']
    mental_jobs = ['Software Engineer', 'Accountant', 'Scientist']
    
    if occupation in high_stress_jobs:
        occupation_factor = 0.2
    elif occupation in shift_work_jobs:
        occupation_factor = 0.3
    elif occupation in mental_jobs:
        occupation_factor = 0.15
    else:
        occupation_factor = 0.0
    
    bp_factor = 0.0
    if 'Blood Pressure' in user:
        try:
            sys, dias = map(int, user['Blood Pressure'].split('/'))
            bp_factor = 0.3 if (sys >= 160 or dias >= 100) else (0.2 if (sys >= 140 or dias >= 90) else (0.1 if (sys >= 130 or dias >= 80) else 0.0))
        except:
            pass
    
    domain_adjustment = sum([age_factor, activity_factor, stress_factor, bmi_factor, disorder_factor, hr_factor, steps_factor, gender_factor, occupation_factor, bp_factor])
    
    if activity_level >= 50 or steps >= 10000:
        hybrid = (0.6 * ml_prediction) + (0.4 * (ml_prediction + domain_adjustment))
    else:
        hybrid = (0.7 * ml_prediction) + (0.3 * (ml_prediction + domain_adjustment))
    
    return round(max(6.0, min(10.0, hybrid)), 1)

def predict_sleep_duration_fallback(user):
    """Fallback heuristic-based prediction"""
    base = 7.0
    
    activity = user.get('Physical Activity Level', 30)
    base += 0.3 if activity >= 60 else (-0.2 if activity < 30 else 0)
    
    stress = user.get('Stress Level', 5)
    base += 0.4 if stress >= 8 else (-0.2 if stress <= 3 else 0)
    
    steps = user.get('Daily Steps', 5000)
    base += 0.2 if steps >= 10000 else (-0.1 if steps < 5000 else 0)
    
    if user.get('BMI Category') == 'Obese':
        base += 0.3
    
    disorder = user.get('Sleep Disorder')
    base += 0.5 if disorder == 'Sleep Apnea' else (0.3 if disorder == 'Insomnia' else 0)
    
    return round(max(6.0, min(10.0, base)), 1)

def get_personalized_recommendations(user, predicted):
    """Generate personalized recommendations"""
    recommendations = []
    
    if user['Physical Activity Level'] < 15:
        recommendations.append("Increase physical activity to at least 30 minutes daily")
    elif user['Physical Activity Level'] < 30:
        recommendations.append("Gradually increase activity to 30-45 minutes daily")
    elif user['Physical Activity Level'] >= 75:
        recommendations.append("Maintain activity, but avoid workouts 2-3 hours before bed")
    
    if user['Stress Level'] >= 8:
        recommendations.append("Prioritize stress reduction with meditation or breathing exercises")
    elif user['Stress Level'] >= 6:
        recommendations.append("Try 10-minute mindfulness practice before bed")
    
    if user['Daily Steps'] < 5000:
        recommendations.append("Increase daily steps to 7,500+ for better sleep quality")
    elif user['Daily Steps'] < 7500:
        recommendations.append("Aim to increase steps to 7,500-10,000 per day")
    
    if user['Heart Rate'] > 80:
        recommendations.append("Cardiovascular exercise can help lower your resting heart rate")
    
    if user['BMI Category'] == 'Overweight':
        recommendations.append("Focus on balanced nutrition and consistent physical activity")
    elif user['BMI Category'] == 'Obese':
        recommendations.append("Consider medical evaluation for sleep apnea if you snore")
    
    disorder = user.get('Sleep Disorder')
    if disorder == 'Sleep Apnea':
        recommendations.append("Try sleeping on your side rather than your back")
    elif disorder == 'Insomnia':
        recommendations.append("Establish a consistent bedtime routine every day")
    
    if user['Age'] < 30:
        recommendations.append("Young adults need consistent sleep even on weekends")
    elif user['Age'] >= 50:
        recommendations.append("Focus on sleep quality over quantity as you age")
    
    return recommendations[:5]

# Load components
components = load_model_components()

# Main Layout
left_col, right_col = st.columns([1.1, 1.3], gap='large')

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👤 Your Profile")
    
    with st.form(key='profile_form'):
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        occupation = st.selectbox("Occupation", [
            "Doctor", "Engineer", "Lawyer", "Teacher", "Nurse", 
            "Accountant", "Sales Representative", "Software Engineer", 
            "Manager", "Office Worker", "Student", "Other"
        ], index=9)
        
        st.write("**Health Metrics**")
        physical_activity = st.slider("Physical Activity (min/day)", 0, 120, 30)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        daily_steps = st.number_input("Daily Steps", 0, 50000, 5000, step=500)
        heart_rate = st.slider("Resting Heart Rate (bpm)", 40, 120, 70)
        
        bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
        
        col_bp1, col_bp2 = st.columns(2)
        with col_bp1:
            systolic = st.number_input("Systolic BP", 80, 200, 120)
        with col_bp2:
            diastolic = st.number_input("Diastolic BP", 40, 120, 80)
        
        blood_pressure = f"{int(systolic)}/{int(diastolic)}"
        sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Insomnia", "Sleep Apnea"])
        sleep_disorder = None if sleep_disorder == "None" else sleep_disorder
        
        submit = st.form_submit_button("🔮 Predict Sleep Duration", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ️ Model Status")
    if components is None:
        st.markdown("<div class='warning-badge'>⚠️ No trained model detected. Using fallback heuristic algorithm for predictions.</div>", unsafe_allow_html=True)
        st.caption("💡 Tip: Place `models/sleep_prediction_model.pkl` to enable ML-powered predictions.")
    else:
        st.markdown("<div class='success-badge'>✓ Trained model loaded successfully</div>", unsafe_allow_html=True)
        st.caption("🚀 Using scikit-learn pipeline with domain expertise for predictions")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Results")
    
    if submit:
        user = {
            'Person ID': 999,
            'Gender': gender,
            'Age': int(age),
            'Occupation': occupation,
            'Sleep Duration': None,
            'Quality of Sleep': 5,
            'Physical Activity Level': int(physical_activity),
            'Stress Level': int(stress_level),
            'BMI Category': bmi_category,
            'Blood Pressure': blood_pressure,
            'Heart Rate': int(heart_rate),
            'Daily Steps': int(daily_steps),
            'Sleep Disorder': sleep_disorder
        }
        
        predicted = None
        source = None
        
        if components is not None and 'pipeline' in components and 'feature_columns' in components:
            try:
                pipeline = components['pipeline']
                my_cols = components['feature_columns']
                predicted = predict_sleep_duration_advanced(user, pipeline, my_cols)
                source = 'Trained ML Model'
            except:
                predicted = predict_sleep_duration_fallback(user)
                source = 'Fallback (model error)'
        else:
            predicted = predict_sleep_duration_fallback(user)
            source = 'Heuristic Fallback'
        
        # Status
        if predicted < 7:
            status = "BELOW RECOMMENDED"
        elif predicted <= 9:
            status = "OPTIMAL"
        else:
            status = "ABOVE RECOMMENDED"
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Sleep Duration", f"{predicted}h")
        m2.metric("Status", status)
        m3.metric("Source", source)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        recs = get_personalized_recommendations(user, predicted)
        for i, rec in enumerate(recs, 1):
            st.markdown(f"<div class='recommend'>**{i}. {rec}**</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Visualization
        st.subheader("📈 Health Factors")
        factors_data = {
            'Activity': min(physical_activity / 120, 1.0),
            'Stress Mgmt': 1 - (stress_level / 10),
            'Daily Steps': min(daily_steps / 10000, 1.0),
            'Heart Rate': 1 - min(abs(heart_rate - 70) / 30, 1.0),
            'BMI OK': 0.9 if bmi_category == 'Normal' else (0.5 if bmi_category == 'Overweight' else 0.3)
        }
        
        fig = go.Figure(data=[
            go.Bar(
                y=list(factors_data.keys()),
                x=list(factors_data.values()),
                orientation='h',
                marker_color=['#6ee7b7', '#06d6a0', '#00d084', '#00b870', '#009b5c'],
                text=[f"{v:.0%}" for v in factors_data.values()],
                textposition='auto'
            )
        ])
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_range=[0, 1],
            template='plotly_dark',
            paper_bgcolor='#0b1220',
            plot_bgcolor='#0b1220'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Download Options
        st.subheader("📥 Export Your Results")
        
        col_csv, col_pdf = st.columns(2)
        
        with col_csv:
            csv_data = pd.DataFrame([user]).to_csv(index=False)
            st.download_button(
                label="📄 Download CSV Profile",
                data=csv_data,
                file_name=f"sleep_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_pdf:
            pdf_buffer = generate_pdf_report(user, predicted, recs)
            pdf_data = pdf_buffer.getvalue()
            st.download_button(
                label="📊 Download PDF Report",
                data=pdf_data,
                file_name=f"Sleep_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Expandable profile
        with st.expander("📋 Full Profile Details"):
            st.json(user)
    
    else:
        st.markdown(
            "<div class='small'>Fill your profile on the left and click <b>Predict Sleep Duration</b> to get a personalized sleep recommendation.</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>🛌 Sleep Health Predictor | Advanced AI Analysis • Built with ❤️ using Streamlit<br>Privacy: Your data is never stored or shared. Analysis happens locally on your device.</div>",
    unsafe_allow_html=True
)
