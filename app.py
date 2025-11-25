import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Passenger Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">✈️ Passenger Satisfaction Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predicting passenger satisfaction using Machine Learning")

# Create sample data if no dataset is uploaded
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Customer Type': np.random.choice(['Loyal Customer', 'disloyal Customer'], n_samples),
        'Age': np.random.randint(7, 85, n_samples),
        'Type of Travel': np.random.choice(['Personal Travel', 'Business travel'], n_samples),
        'Class': np.random.choice(['Eco', 'Eco Plus', 'Business'], n_samples),
        'Flight Distance': np.random.randint(31, 4983, n_samples),
        'Inflight wifi service': np.random.randint(0, 6, n_samples),
        'Departure/Arrival time convenient': np.random.randint(0, 6, n_samples),
        'Ease of Online booking': np.random.randint(0, 6, n_samples),
        'Gate location': np.random.randint(0, 6, n_samples),
        'Food and drink': np.random.randint(0, 6, n_samples),
        'Online boarding': np.random.randint(0, 6, n_samples),
        'Seat comfort': np.random.randint(0, 6, n_samples),
        'Inflight entertainment': np.random.randint(0, 6, n_samples),
        'On-board service': np.random.randint(0, 6, n_samples),
        'Leg room service': np.random.randint(0, 6, n_samples),
        'Baggage handling': np.random.randint(1, 6, n_samples),
        'Checkin service': np.random.randint(0, 6, n_samples),
        'Inflight service': np.random.randint(0, 6, n_samples),
        'Cleanliness': np.random.randint(0, 6, n_samples),
        'Arrival Delay in Minutes': np.random.exponential(scale=15, size=n_samples),
        'satisfaction': np.random.choice(['satisfied', 'neutral or dissatisfied'], n_samples)
    }
    
    return pd.DataFrame(data)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page", 
    ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Predictions", "📈 Model Comparison"]
)

if page == "🏠 Home":
    st.markdown("## Welcome to the Passenger Satisfaction Predictor!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Objective</h3>
        <p>Predict passenger satisfaction based on flight experience factors using Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🔬 Models Used</h3>
        <p>Logistic Regression & Decision Tree for comprehensive analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Features</h3>
        <p>20+ features including service ratings, delays, and demographics</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File upload
    st.markdown("### 📁 Upload Your Dataset (Optional)")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.success("✅ Data uploaded successfully!")
        st.dataframe(df.head())
    else:
        st.info("Using sample dataset for demonstration")
        df = create_sample_data()
        st.session_state['data'] = df
        st.dataframe(df.head())

elif page == "📊 Data Analysis":
    st.markdown('<h2 class="sub-header">📊 Data Analysis & Exploration</h2>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state:
        df = create_sample_data()
        st.session_state['data'] = df
    else:
        df = st.session_state['data']
    
    # Data preprocessing
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    
    # Handle missing values
    if 'Arrival Delay in Minutes' in df.columns:
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())
    
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns)-1)
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        satisfaction_rate = (df['satisfaction'] == 'satisfied').mean() * 100 if 'satisfaction' in df.columns else 50
        st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
    
    # Satisfaction distribution
    if 'satisfaction' in df.columns:
        st.markdown("### Satisfaction Distribution")
        fig = px.pie(df, names='satisfaction', title="Distribution of Passenger Satisfaction")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.markdown("### Feature Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        selected_feature = st.selectbox("Select a feature to analyze:", numeric_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=selected_feature, title=f"Distribution of {selected_feature}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'satisfaction' in df.columns:
                fig = px.box(df, x='satisfaction', y=selected_feature, 
                           title=f"{selected_feature} by Satisfaction")
                st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Training":
    st.markdown('<h2 class="sub-header">🤖 Model Training & Evaluation</h2>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state:
        df = create_sample_data()
        st.session_state['data'] = df
    else:
        df = st.session_state['data']
    
    # Data preprocessing
    df_processed = df.copy()
    
    # Drop unnecessary columns
    if 'Unnamed: 0' in df_processed.columns:
        df_processed = df_processed.drop(['Unnamed: 0'], axis=1)
    if 'id' in df_processed.columns:
        df_processed = df_processed.drop(['id'], axis=1)
    
    # Handle missing values
    if 'Arrival Delay in Minutes' in df_processed.columns:
        df_processed['Arrival Delay in Minutes'] = df_processed['Arrival Delay in Minutes'].fillna(
            df_processed['Arrival Delay in Minutes'].median()
        )
    
    # Encode target variable
    if 'satisfaction' in df_processed.columns:
        df_processed['satisfaction'] = df_processed['satisfaction'].map({
            'satisfied': 1,
            'neutral or dissatisfied': 0
        })
    
    # Encode categorical features
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Drop highly correlated features if they exist
    if 'Departure Delay in Minutes' in df_processed.columns:
        df_processed = df_processed.drop('Departure Delay in Minutes', axis=1)
    
    if st.button("🚀 Train Models"):
        with st.spinner("Training models..."):
            # Prepare data
            X = df_processed.drop('satisfaction', axis=1)
            y = df_processed['satisfaction']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Logistic Regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            lr_model = LogisticRegression(max_iter=2000, random_state=42)
            lr_model.fit(X_train_scaled, y_train)
            
            # Decision Tree
            dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
            dt_model.fit(X_train, y_train)
            
            # Predictions
            lr_pred = lr_model.predict(X_test_scaled)
            dt_pred = dt_model.predict(X_test)
            
            # Save models
            st.session_state['lr_model'] = lr_model
            st.session_state['dt_model'] = dt_model
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = X.columns.tolist()
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Logistic Regression Results")
                lr_accuracy = accuracy_score(y_test, lr_pred)
                st.metric("Accuracy", f"{lr_accuracy:.3f}")
                
                # Confusion Matrix
                cm_lr = confusion_matrix(y_test, lr_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Dissatisfied', 'Satisfied'],
                           yticklabels=['Dissatisfied', 'Satisfied'], ax=ax)
                ax.set_title('Logistic Regression Confusion Matrix')
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 🌳 Decision Tree Results")
                dt_accuracy = accuracy_score(y_test, dt_pred)
                st.metric("Accuracy", f"{dt_accuracy:.3f}")
                
                # Confusion Matrix
                cm_dt = confusion_matrix(y_test, dt_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens',
                           xticklabels=['Dissatisfied', 'Satisfied'],
                           yticklabels=['Dissatisfied', 'Satisfied'], ax=ax)
                ax.set_title('Decision Tree Confusion Matrix')
                st.pyplot(fig)
            
            # Feature importance
            st.markdown("### 🎯 Feature Importance (Decision Tree)")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': dt_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        orientation='h', title="Top 10 Most Important Features")
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Models trained successfully!")

elif page == "🔮 Predictions":
    st.markdown('<h2 class="sub-header">🔮 Make Predictions</h2>', unsafe_allow_html=True)
    
    if 'lr_model' not in st.session_state:
        st.warning("⚠️ Please train the models first in the 'Model Training' section.")
    else:
        st.markdown("### Enter Passenger Details:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
            age = st.slider("Age", 7, 85, 35)
            travel_type = st.selectbox("Type of Travel", ["Personal Travel", "Business travel"])
            
        with col2:
            class_type = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
            flight_distance = st.slider("Flight Distance", 31, 4983, 1000)
            wifi_service = st.slider("Inflight WiFi Service (0-5)", 0, 5, 3)
            time_convenient = st.slider("Departure/Arrival Time Convenient (0-5)", 0, 5, 3)
            
        with col3:
            online_booking = st.slider("Ease of Online Booking (0-5)", 0, 5, 3)
            gate_location = st.slider("Gate Location (0-5)", 0, 5, 3)
            food_drink = st.slider("Food and Drink (0-5)", 0, 5, 3)
            online_boarding = st.slider("Online Boarding (0-5)", 0, 5, 3)
        
        # More features
        st.markdown("### Additional Service Ratings:")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            seat_comfort = st.slider("Seat Comfort (0-5)", 0, 5, 3)
            entertainment = st.slider("Inflight Entertainment (0-5)", 0, 5, 3)
            onboard_service = st.slider("On-board Service (0-5)", 0, 5, 3)
            
        with col5:
            leg_room = st.slider("Leg Room Service (0-5)", 0, 5, 3)
            baggage_handling = st.slider("Baggage Handling (1-5)", 1, 5, 3)
            checkin_service = st.slider("Check-in Service (0-5)", 0, 5, 3)
            
        with col6:
            inflight_service = st.slider("Inflight Service (0-5)", 0, 5, 3)
            cleanliness = st.slider("Cleanliness (0-5)", 0, 5, 3)
            arrival_delay = st.slider("Arrival Delay (minutes)", 0, 300, 0)
        
        if st.button("🔮 Predict Satisfaction"):
            # Encode categorical inputs
            gender_encoded = 1 if gender == "Male" else 0
            customer_encoded = 1 if customer_type == "Loyal Customer" else 0
            travel_encoded = 1 if travel_type == "Business travel" else 0
            class_encoded = {"Eco": 0, "Eco Plus": 1, "Business": 2}[class_type]
            
            # Create input array
            input_data = np.array([[
                gender_encoded, customer_encoded, age, travel_encoded, class_encoded,
                flight_distance, wifi_service, time_convenient, online_booking,
                gate_location, food_drink, online_boarding, seat_comfort,
                entertainment, onboard_service, leg_room, baggage_handling,
                checkin_service, inflight_service, cleanliness, arrival_delay
            ]])
            
            # Make predictions
            lr_model = st.session_state['lr_model']
            dt_model = st.session_state['dt_model']
            scaler = st.session_state['scaler']
            
            # Scale input for logistic regression
            input_scaled = scaler.transform(input_data)
            
            lr_pred = lr_model.predict(input_scaled)[0]
            lr_prob = lr_model.predict_proba(input_scaled)[0]
            
            dt_pred = dt_model.predict(input_data)[0]
            dt_prob = dt_model.predict_proba(input_data)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Logistic Regression Prediction")
                if lr_pred == 1:
                    st.success(f"✅ **SATISFIED** (Confidence: {lr_prob[1]:.2%})")
                else:
                    st.error(f"❌ **DISSATISFIED** (Confidence: {lr_prob[0]:.2%})")
            
            with col2:
                st.markdown("### 🌳 Decision Tree Prediction")
                if dt_pred == 1:
                    st.success(f"✅ **SATISFIED** (Confidence: {dt_prob[1]:.2%})")
                else:
                    st.error(f"❌ **DISSATISFIED** (Confidence: {dt_prob[0]:.2%})")

elif page == "📈 Model Comparison":
    st.markdown('<h2 class="sub-header">📈 Model Comparison & Insights</h2>', unsafe_allow_html=True)
    
    if 'lr_model' not in st.session_state:
        st.warning("⚠️ Please train the models first in the 'Model Training' section.")
    else:
        # Get model performance
        lr_model = st.session_state['lr_model']
        dt_model = st.session_state['dt_model']
        scaler = st.session_state['scaler']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        X_test_scaled = scaler.transform(X_test)
        
        lr_pred = lr_model.predict(X_test_scaled)
        dt_pred = dt_model.predict(X_test)
        
        lr_accuracy = accuracy_score(y_test, lr_pred)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        
        # Model comparison
        st.markdown("### 🏆 Model Performance Comparison")
        
        comparison_data = pd.DataFrame({
            'Model': ['Logistic Regression', 'Decision Tree'],
            'Accuracy': [lr_accuracy, dt_accuracy],
            'Type': ['Linear', 'Non-linear']
        })
        
        fig = px.bar(comparison_data, x='Model', y='Accuracy', 
                    title="Model Accuracy Comparison", 
                    color='Model',
                    color_discrete_map={
                        'Logistic Regression': '#1f77b4',
                        'Decision Tree': '#2ca02c'
                    })
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.markdown("### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📊 Logistic Regression
            - **Strengths:**
              - Highly interpretable coefficients
              - Provides probability estimates
              - Less prone to overfitting
              - Good baseline model
            
            - **Best for:**
              - Understanding feature impact
              - Regulatory environments
              - Simple deployment
            """)
        
        with col2:
            st.markdown("""
            #### 🌳 Decision Tree
            - **Strengths:**
              - Captures non-linear relationships
              - Handles feature interactions automatically
              - No scaling required
              - Visual decision rules
            
            - **Best for:**
              - Complex pattern recognition
              - Mixed data types
              - Feature selection insights
            """)
        
        st.markdown("---")
        st.markdown("### 🎯 Business Recommendations")
        st.info("""
        **Key Factors for Passenger Satisfaction:**
        1. **Online Boarding Experience** - Most critical factor
        2. **Inflight WiFi Service** - Essential for modern travelers
        3. **Check-in Service** - First impression matters
        4. **Seat Comfort & Leg Room** - Physical comfort drives satisfaction
        5. **On-time Performance** - Minimize delays
        
        **Actionable Insights:**
        - Invest in digital boarding solutions
        - Upgrade WiFi infrastructure
        - Train ground staff for better check-in experience
        - Consider seat upgrades for frequent flyers
        - Implement proactive delay management
        """)

# Footer
st.markdown("---")
st.markdown("""

""", unsafe_allow_html=True)