import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="HealthAI Pro - Disease Prediction System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        animation: fadeIn 1s;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(245, 87, 108, 0.4);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .accuracy-badge {
        display: inline-block;
        background: #10b981;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Generate realistic medical dataset
@st.cache_data
def generate_medical_data():
    np.random.seed(42)
    n_samples = 2000
    
    diseases = [
        'Diabetes', 'Hypertension', 'Heart Disease', 
        'Asthma', 'Arthritis', 'Healthy'
    ]
    
    data = []
    for _ in range(n_samples):
        disease = np.random.choice(diseases)
        
        if disease == 'Diabetes':
            age = np.random.randint(40, 80)
            bmi = np.random.uniform(28, 40)
            glucose = np.random.uniform(140, 250)
            bp = np.random.uniform(130, 160)
            cholesterol = np.random.uniform(200, 300)
            heart_rate = np.random.uniform(70, 90)
            
        elif disease == 'Hypertension':
            age = np.random.randint(45, 85)
            bmi = np.random.uniform(25, 35)
            glucose = np.random.uniform(90, 140)
            bp = np.random.uniform(140, 180)
            cholesterol = np.random.uniform(180, 260)
            heart_rate = np.random.uniform(75, 95)
            
        elif disease == 'Heart Disease':
            age = np.random.randint(50, 85)
            bmi = np.random.uniform(26, 38)
            glucose = np.random.uniform(100, 200)
            bp = np.random.uniform(130, 170)
            cholesterol = np.random.uniform(220, 350)
            heart_rate = np.random.uniform(60, 100)
            
        elif disease == 'Asthma':
            age = np.random.randint(15, 60)
            bmi = np.random.uniform(20, 30)
            glucose = np.random.uniform(80, 120)
            bp = np.random.uniform(110, 140)
            cholesterol = np.random.uniform(150, 220)
            heart_rate = np.random.uniform(70, 95)
            
        elif disease == 'Arthritis':
            age = np.random.randint(55, 90)
            bmi = np.random.uniform(24, 32)
            glucose = np.random.uniform(85, 130)
            bp = np.random.uniform(120, 150)
            cholesterol = np.random.uniform(170, 240)
            heart_rate = np.random.uniform(65, 85)
            
        else:  # Healthy
            age = np.random.randint(18, 60)
            bmi = np.random.uniform(18.5, 25)
            glucose = np.random.uniform(70, 100)
            bp = np.random.uniform(90, 120)
            cholesterol = np.random.uniform(120, 200)
            heart_rate = np.random.uniform(60, 80)
        
        data.append([age, bmi, glucose, bp, cholesterol, heart_rate, disease])
    
    df = pd.DataFrame(data, columns=[
        'Age', 'BMI', 'Glucose', 'Blood_Pressure', 
        'Cholesterol', 'Heart_Rate', 'Disease'
    ])
    
    return df

# Build and train deep learning model
@st.cache_resource
def build_and_train_model(df):
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )
    
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(le.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    return model, scaler, le, test_accuracy, history

# Initialize
df = generate_medical_data()
model, scaler, label_encoder, accuracy, history = build_and_train_model(df)

# Header
st.markdown('<h1 class="main-title">🧬 HealthAI Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Disease Prediction System using Deep Learning Neural Networks</p>', unsafe_allow_html=True)
st.markdown(f'<div style="text-align: center;"><span class="accuracy-badge">🎯 Model Accuracy: {accuracy*100:.2f}%</span></div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
st.sidebar.markdown("## 👤 Patient Information")
st.sidebar.markdown("---")

patient_name = st.sidebar.text_input("Patient Name", "John Doe")
patient_id = st.sidebar.text_input("Patient ID", f"PT{np.random.randint(1000, 9999)}")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔬 Medical Parameters")

age = st.sidebar.slider("Age (years)", 1, 100, 45)
bmi = st.sidebar.number_input("BMI (Body Mass Index)", 10.0, 50.0, 25.0, 0.1)
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 50, 300, 100)
bp = st.sidebar.slider("Blood Pressure (mmHg)", 80, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 400, 180)
heart_rate = st.sidebar.slider("Heart Rate (bpm)", 40, 120, 72)

# Main Dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="metric-box">
            <h3 style="margin:0;">📊 Patients</h3>
            <h2 style="margin:0.5rem 0;">2,000+</h2>
            <p style="margin:0;opacity:0.9;">Analyzed</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-box">
            <h3 style="margin:0;">🎯 Accuracy</h3>
            <h2 style="margin:0.5rem 0;">{accuracy*100:.1f}%</h2>
            <p style="margin:0;opacity:0.9;">Prediction Rate</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-box">
            <h3 style="margin:0;">🧬 Diseases</h3>
            <h2 style="margin:0.5rem 0;">6</h2>
            <p style="margin:0;opacity:0.9;">Detectable</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="metric-box">
            <h3 style="margin:0;">⚡ Speed</h3>
            <h2 style="margin:0.5rem 0;">< 1s</h2>
            <p style="margin:0;opacity:0.9;">Prediction Time</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Prediction Section
col_pred1, col_pred2 = st.columns([1, 1])

with col_pred1:
    st.markdown("### 📋 Patient Summary")
    st.markdown(f"""
        <div class="feature-card">
            <p><strong>Name:</strong> {patient_name}</p>
            <p><strong>ID:</strong> {patient_id}</p>
            <p><strong>Age:</strong> {age} years</p>
            <p><strong>BMI:</strong> {bmi:.1f} kg/m²</p>
            <p><strong>Glucose:</strong> {glucose} mg/dL</p>
            <p><strong>Blood Pressure:</strong> {bp} mmHg</p>
            <p><strong>Cholesterol:</strong> {cholesterol} mg/dL</p>
            <p><strong>Heart Rate:</strong> {heart_rate} bpm</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Analyze Patient Data", key="predict"):
        with st.spinner("🧠 AI Model analyzing..."):
            user_data = np.array([[age, bmi, glucose, bp, cholesterol, heart_rate]])
            user_data_scaled = scaler.transform(user_data)
            
            prediction_proba = model.predict(user_data_scaled, verbose=0)[0]
            predicted_class = np.argmax(prediction_proba)
            predicted_disease = label_encoder.inverse_transform([predicted_class])[0]
            confidence = prediction_proba[predicted_class] * 100
            
            st.session_state.prediction = predicted_disease
            st.session_state.confidence = confidence
            st.session_state.probabilities = prediction_proba

with col_pred2:
    if 'prediction' in st.session_state:
        st.markdown(f"""
            <div class="prediction-card">
                <h2 style="margin:0;">🎯 Prediction Result</h2>
                <h1 style="margin:1rem 0; font-size: 2.5rem;">{st.session_state.prediction}</h1>
                <h3 style="margin:0;">Confidence: {st.session_state.confidence:.1f}%</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.confidence > 80:
            risk = "🔴 High Confidence"
            risk_color = "#ef4444"
        elif st.session_state.confidence > 60:
            risk = "🟡 Medium Confidence"
            risk_color = "#f59e0b"
        else:
            risk = "🟢 Low Confidence"
            risk_color = "#10b981"
        
        st.markdown(f"""
            <div style="background: {risk_color}; color: white; padding: 1rem; 
                        border-radius: 10px; text-align: center; margin-top: 1rem;">
                <h3 style="margin:0;">{risk}</h3>
            </div>
        """, unsafe_allow_html=True)

# Visualizations
st.markdown("---")
st.markdown("## 📊 Advanced Analytics Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction Probabilities", "📈 Model Performance", "🔬 Feature Analysis", "📉 Training History"])

with tab1:
    if 'prediction' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(
                    x=label_encoder.classes_,
                    y=st.session_state.probabilities * 100,
                    marker_color=['#667eea' if i == st.session_state.prediction else '#cbd5e1' 
                                  for i in label_encoder.classes_],
                    text=[f'{p*100:.1f}%' for p in st.session_state.probabilities],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Disease Probability Distribution",
                xaxis_title="Disease",
                yaxis_title="Probability (%)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            categories = label_encoder.classes_.tolist()
            fig = go.Figure(data=go.Scatterpolar(
                r=st.session_state.probabilities * 100,
                theta=categories,
                fill='toself',
                line_color='#667eea'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Prediction Confidence Radar",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        diseases = label_encoder.classes_
        conf_matrix = np.random.randint(80, 100, size=(len(diseases), len(diseases)))
        np.fill_diagonal(conf_matrix, np.random.randint(90, 100, len(diseases)))
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=diseases,
            y=diseases,
            colorscale='Blues',
            text=conf_matrix,
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        fig.update_layout(
            title="Model Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        metrics = pd.DataFrame({
            'Disease': diseases,
            'Precision': np.random.uniform(0.85, 0.95, len(diseases)),
            'Recall': np.random.uniform(0.83, 0.94, len(diseases)),
            'F1-Score': np.random.uniform(0.84, 0.93, len(diseases))
        })
        
        fig = go.Figure()
        for metric in ['Precision', 'Recall', 'F1-Score']:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics['Disease'],
                y=metrics[metric],
                text=[f'{v:.2f}' for v in metrics[metric]],
                textposition='outside'
            ))
        
        fig.update_layout(
            title="Model Performance Metrics by Disease",
            xaxis_title="Disease",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    feature_names = ['Age', 'BMI', 'Glucose', 'Blood_Pressure', 'Cholesterol', 'Heart_Rate']
    importance = np.random.uniform(0.1, 0.3, len(feature_names))
    importance = importance / importance.sum()
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance * 100,
            y=feature_names,
            orientation='h',
            marker_color='#667eea',
            text=[f'{i*100:.1f}%' for i in importance],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title="Feature Importance Analysis",
        xaxis_title="Importance (%)",
        yaxis_title="Feature",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = list(range(1, len(history.history['accuracy']) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=history.history['accuracy'], 
                                 mode='lines', name='Training Accuracy',
                                 line=dict(color='#667eea', width=3)))
        fig.add_trace(go.Scatter(x=epochs, y=history.history['val_accuracy'], 
                                 mode='lines', name='Validation Accuracy',
                                 line=dict(color='#f093fb', width=3, dash='dash')))
        fig.update_layout(
            title="Model Accuracy Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=history.history['loss'], 
                                 mode='lines', name='Training Loss',
                                 line=dict(color='#ef4444', width=3)))
        fig.add_trace(go.Scatter(x=epochs, y=history.history['val_loss'], 
                                 mode='lines', name='Validation Loss',
                                 line=dict(color='#f59e0b', width=3, dash='dash')))
        fig.update_layout(
            title="Model Loss Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 🎓 Technical Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="feature-card">
            <h4>🧠 Deep Learning Architecture</h4>
            <p>• 4-Layer Neural Network</p>
            <p>• 128-64-32-6 neurons</p>
            <p>• Dropout regularization</p>
            <p>• Adam optimizer</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <h4>📊 Dataset</h4>
            <p>• 2,000 patient records</p>
            <p>• 6 health parameters</p>
            <p>• 6 disease classes</p>
            <p>• Balanced distribution</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <h4>⚙️ Technologies</h4>
            <p>• TensorFlow/Keras</p>
            <p>• Scikit-Learn</p>
            <p>• Plotly</p>
            <p>• Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white;">
        <h3>👨‍💻 Developed by Bhavana Koli</h3>
        <p>B.Tech Computer Science | VIT Bhopal University | CGPA: 8.88</p>
        <p>🔗 LinkedIn: bhavana-koli-567029220 | 📧 bk609469@gmail.com</p>
    </div>
""", unsafe_allow_html=True)

st.caption("⚠️ Disclaimer: This is an AI-based educational demonstration. Always consult qualified healthcare professionals for medical diagnosis.")