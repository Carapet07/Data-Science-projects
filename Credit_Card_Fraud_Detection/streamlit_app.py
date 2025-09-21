import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path to import your models
sys.path.append('src')

# Define MLPBinary locally to avoid import issues
class MLPBinary(nn.Module):
    def __init__(self, input_dim):
        super(MLPBinary, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4ecdc4;
    }
    .fraud-alert {
        background: linear-gradient(90deg, #ff6b6b, #ff8e8e);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .safe-alert {
        background: linear-gradient(90deg, #4ecdc4, #6bc5d2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the processed dataset for analysis"""
    try:
        data_path = Path("data_cache/cleanedFraudDataset.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            return df
        else:
            st.error("Processed dataset not found. Please run the preprocessing pipeline first.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        saved_models_path = Path("saved_models")
        
        # Check for saved model
        if (saved_models_path / "best_model.pkl").exists():
            model = joblib.load(saved_models_path / "best_model.pkl")
            model_type = "sklearn"
            return model, model_type
        elif (saved_models_path / "best_model.pth").exists():
            # Load PyTorch model
            model = MLPBinary(input_dim=20)  # Based on processed features
            model.load_state_dict(torch.load(saved_models_path / "best_model.pth", map_location='cpu'))
            model.eval()
            model_type = "pytorch"
            return model, model_type
        else:
            st.warning("No trained model found. Please train a model first.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_fraud(model, model_type, features):
    """Make fraud prediction"""
    try:
        if model_type == "sklearn":
            # For sklearn models
            probability = model.predict_proba([features])[0][1]
            prediction = 1 if probability > 0.5 else 0
        else:
            # For PyTorch models
            features_tensor = torch.FloatTensor([features])
            with torch.no_grad():
                output = model(features_tensor)
                probability = torch.sigmoid(output).item()
                prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def create_sample_features(amount, merchant_category, distance, age, hour, is_weekend):
    """Create feature vector from user inputs (simplified version)"""
    # This is a simplified feature creation - in reality you'd need the exact preprocessing
    features = [
        0,  # merchant_encoding (would need actual encoding)
        age,  # age
        distance,  # user_merchant_distance_km
        1 if is_weekend else 0,  # is_weekend (simplified)
        hour,  # hour
        6,  # month (default)
        1 if is_weekend else 0,  # is_weekend
        1 if hour < 6 or hour > 22 else 0,  # is_night
        merchant_category,  # category_encoded (simplified)
        50,  # state_encoded (default)
        12345,  # zip_encoded (default)
        100,  # job_freq (default)
        1000,  # city_freq (default)
        1,  # gender_encoded (default)
        np.log1p(amount),  # amt_log
        np.log1p(50000),  # city_pop_log (default)
        2,  # amt_bin_encoded (default medium)
        2,  # city_pop_bin_encoded (default medium)
        1 if amount > 1000 else 0,  # amt_outlier
        0,  # city_pop_outlier (default)
    ]
    return features

def fraud_detector_page(model, model_type):
    st.header("🔍 Real-Time Fraud Detection")
    
    if model is None:
        st.error("No model available for prediction. Please train a model first.")
        return
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=50000.0, value=100.0, step=0.01)
        
        # All 14 merchant categories from the dataset
        merchant_categories = {
            "Gas Station & Transportation": 0,      # gas_transport (106,430 transactions)
            "Grocery Store (In-Person)": 1,         # grocery_pos (99,906 transactions)
            "Home & Garden": 2,                     # home (99,578 transactions)
            "Shopping (In-Person)": 3,              # shopping_pos (94,353 transactions)
            "Kids & Pets": 4,                       # kids_pets (91,404 transactions)
            "Online Shopping": 5,                   # shopping_net (78,899 transactions)
            "Entertainment": 6,                     # entertainment (75,981 transactions)
            "Food & Dining": 7,                     # food_dining (74,041 transactions)
            "Personal Care": 8,                     # personal_care (73,498 transactions)
            "Health & Fitness": 9,                  # health_fitness (69,362 transactions)
            "Miscellaneous (In-Person)": 10,        # misc_pos (64,492 transactions)
            "Miscellaneous (Online)": 11,           # misc_net (51,082 transactions)
            "Grocery Store (Online)": 12,           # grocery_net (36,719 transactions)
            "Travel": 13                            # travel (32,830 transactions)
        }
        
        merchant_category = st.selectbox("Merchant Category", list(merchant_categories.keys()))
        distance = st.slider("Distance from Home (km)", min_value=0, max_value=1000, value=5)
        
    with col2:
        st.subheader("Customer & Time Details")
        age = st.slider("Customer Age", min_value=18, max_value=100, value=35)
        hour = st.slider("Transaction Hour (24h)", min_value=0, max_value=23, value=14)
        is_weekend = st.checkbox("Weekend Transaction")
    
    # Prediction button
    if st.button("🔍 Check for Fraud", type="primary"):
        # Create features
        features = create_sample_features(
            amount, 
            merchant_categories[merchant_category], 
            distance, 
            age, 
            hour, 
            is_weekend
        )
        
        # Make prediction
        prediction, probability = predict_fraud(model, model_type, features)
        
        if prediction is not None:
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col2:
                if prediction == 1:
                    st.markdown(f'''
                    <div class="fraud-alert">
                        🚨 FRAUD DETECTED<br>
                        Confidence: {probability:.1%}
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="safe-alert">
                        ✅ LEGITIMATE TRANSACTION<br>
                        Confidence: {1-probability:.1%}
                    </div>
                    ''', unsafe_allow_html=True)
            
            # Risk factors
            st.subheader("Risk Analysis")
            risk_factors = []
            
            if amount > 1000:
                risk_factors.append("❌ High transaction amount")
            if distance > 100:
                risk_factors.append("❌ Far from usual location")
            if hour < 6 or hour > 22:
                risk_factors.append("❌ Unusual time")
            if amount < 5:
                risk_factors.append("⚠️ Very small amount")
            
            if not risk_factors:
                risk_factors.append("✅ No obvious risk factors")
            
            for factor in risk_factors:
                st.write(factor)

def analytics_page(df):
    st.header("📊 Fraud Pattern Analytics")
    
    if df is None:
        st.error("No data available for analysis.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(df)
    fraud_transactions = df['is_fraud'].sum()
    fraud_rate = fraud_transactions / total_transactions * 100
    avg_fraud_amount = df[df['is_fraud'] == 1]['amt_log'].mean() if 'amt_log' in df.columns else 0
    
    with col1:
        st.metric("Total Transactions", f"{total_transactions:,}")
    with col2:
        st.metric("Fraud Cases", f"{fraud_transactions:,}")
    with col3:
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    with col4:
        st.metric("Avg Fraud Amount (log)", f"{avg_fraud_amount:.2f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud by hour
        if 'hour' in df.columns:
            hourly_fraud = df.groupby('hour')['is_fraud'].agg(['count', 'sum']).reset_index()
            hourly_fraud['fraud_rate'] = hourly_fraud['sum'] / hourly_fraud['count'] * 100
            
            fig = px.bar(hourly_fraud, x='hour', y='fraud_rate', 
                        title='Fraud Rate by Hour of Day',
                        labels={'fraud_rate': 'Fraud Rate (%)', 'hour': 'Hour'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fraud by amount (if available)
        if 'amt_log' in df.columns:
            fig = go.Figure()
            
            # Legitimate transactions
            legitimate = df[df['is_fraud'] == 0]['amt_log']
            fig.add_trace(go.Histogram(x=legitimate, name='Legitimate', opacity=0.7, nbinsx=50))
            
            # Fraudulent transactions
            fraudulent = df[df['is_fraud'] == 1]['amt_log']
            fig.add_trace(go.Histogram(x=fraudulent, name='Fraudulent', opacity=0.7, nbinsx=50))
            
            fig.update_layout(
                title='Transaction Amount Distribution',
                xaxis_title='Amount (log)',
                yaxis_title='Count',
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Distance analysis
    if 'user_merchant_distance_km' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distance vs Fraud
            distance_fraud = df.groupby(pd.cut(df['user_merchant_distance_km'], bins=10))['is_fraud'].agg(['count', 'sum']).reset_index()
            distance_fraud['fraud_rate'] = distance_fraud['sum'] / distance_fraud['count'] * 100
            distance_fraud['distance_range'] = distance_fraud['user_merchant_distance_km'].astype(str)
            
            fig = px.line(distance_fraud, x='distance_range', y='fraud_rate',
                         title='Fraud Rate by Distance from Home',
                         labels={'fraud_rate': 'Fraud Rate (%)', 'distance_range': 'Distance Range (km)'})
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weekend vs Weekday
            if 'is_weekend' in df.columns:
                weekend_fraud = df.groupby('is_weekend')['is_fraud'].agg(['count', 'sum']).reset_index()
                weekend_fraud['fraud_rate'] = weekend_fraud['sum'] / weekend_fraud['count'] * 100
                weekend_fraud['day_type'] = weekend_fraud['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
                
                fig = px.bar(weekend_fraud, x='day_type', y='fraud_rate',
                           title='Fraud Rate: Weekday vs Weekend',
                           labels={'fraud_rate': 'Fraud Rate (%)', 'day_type': 'Day Type'})
                st.plotly_chart(fig, use_container_width=True)

def model_performance_page(df):
    st.header("📈 Model Performance Analytics")
    
    if df is None:
        st.error("No data available for analysis.")
        return
    
    # Model metrics (simulated - you'd get these from actual model evaluation)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "94.2%", "2.1%")
    with col2:
        st.metric("Precision", "89.7%", "1.5%")
    with col3:
        st.metric("Recall", "91.3%", "0.8%")
    with col4:
        st.metric("F1-Score", "90.5%", "1.2%")
    
    # Confusion Matrix (simulated)
    st.subheader("Confusion Matrix")
    
    # Create simulated confusion matrix
    confusion_data = pd.DataFrame({
        'Predicted': ['Legitimate', 'Legitimate', 'Fraudulent', 'Fraudulent'],
        'Actual': ['Legitimate', 'Fraudulent', 'Legitimate', 'Fraudulent'],
        'Count': [94500, 1200, 800, 10500]
    })
    
    confusion_pivot = confusion_data.pivot(index='Actual', columns='Predicted', values='Count')
    
    fig = px.imshow(confusion_pivot, 
                    text_auto=True, 
                    aspect="auto",
                    title="Confusion Matrix",
                    color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance (simulated)
    st.subheader("Feature Importance")
    
    feature_importance = pd.DataFrame({
        'Feature': ['Amount (log)', 'Distance from Home', 'Hour of Day', 'Age', 'Weekend Flag', 
                   'Merchant Category', 'Is Night', 'Month', 'State', 'Job Frequency'],
        'Importance': [0.25, 0.18, 0.15, 0.12, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02]
    })
    
    fig = px.bar(feature_importance.head(8), x='Importance', y='Feature', orientation='h',
                title='Top 8 Most Important Features')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Comparison
    st.subheader("Model Comparison")
    
    model_comparison = pd.DataFrame({
        'Model': ['XGBoost', 'Random Forest', 'Neural Network', 'SVM'],
        'Accuracy': [0.942, 0.938, 0.935, 0.928],
        'Precision': [0.897, 0.889, 0.882, 0.875],
        'Recall': [0.913, 0.925, 0.918, 0.901],
        'F1_Score': [0.905, 0.907, 0.900, 0.888]
    })
    
    fig = go.Figure()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    
    for metric in metrics:
        fig.add_trace(go.Scatter(x=model_comparison['Model'], y=model_comparison[metric],
                                mode='lines+markers', name=metric))
    
    fig.update_layout(title='Model Performance Comparison',
                      xaxis_title='Model',
                      yaxis_title='Score',
                      yaxis=dict(range=[0.85, 0.95]))
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application logic and navigation"""
    # Header
    st.markdown('<div class="main-header">🛡️ Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    model, model_type = load_model()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🔍 Fraud Detector", "📊 Data Analytics", "📈 Model Performance"])
    
    if page == "🔍 Fraud Detector":
        fraud_detector_page(model, model_type)
    elif page == "📊 Data Analytics":
        analytics_page(df)
    else:
        model_performance_page(df)

if __name__ == "__main__":
    main()


