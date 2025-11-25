# ✈️ Passenger Satisfaction Predictor

A comprehensive machine learning application built with Streamlit that predicts passenger satisfaction using Logistic Regression and Decision Tree models.

## 🚀 Live Demo

**Deploy this app on Streamlit Cloud to get your 15% grade boost!**

## 📋 Features

- **Interactive Data Analysis**: Explore passenger satisfaction datasets with visualizations
- **Model Training**: Train both Logistic Regression and Decision Tree models
- **Real-time Predictions**: Make predictions for new passengers
- **Model Comparison**: Compare performance between different algorithms
- **Business Insights**: Get actionable recommendations for improving passenger satisfaction

## 🛠️ Installation

### Local Setup

1. Clone this repository:
```bash
git clone <your-github-repo-url>
cd Streamlit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud

### Step 1: Push to GitHub
1. Create a new repository on GitHub
2. Push your code:
```bash
git init
git add .
git commit -m "Initial commit - Passenger Satisfaction Predictor"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and the `app.py` file
5. Click "Deploy!"

Your app will be live in minutes! 🎉

## 📊 Models Used

### Logistic Regression
- **Accuracy**: ~92.6%
- **Strengths**: Interpretable, provides probability estimates
- **Best for**: Understanding feature impact and coefficients

### Decision Tree
- **Accuracy**: ~94.2%
- **Strengths**: Captures non-linear relationships, handles interactions
- **Best for**: Complex pattern recognition and feature importance

## 🎯 Key Features Analyzed

- **Demographics**: Age, Gender, Customer Type
- **Travel Details**: Flight Distance, Class, Type of Travel
- **Service Ratings**: WiFi, Boarding, Food, Seat Comfort, etc.
- **Operational**: Arrival Delays, Gate Location, Check-in Service

## 📈 Business Insights

### Top Satisfaction Drivers:
1. **Online Boarding Experience** - Most critical factor
2. **Inflight WiFi Service** - Essential for modern travelers  
3. **Check-in Service** - First impression matters
4. **Seat Comfort & Leg Room** - Physical comfort drives satisfaction
5. **On-time Performance** - Minimize delays

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Deployment**: Streamlit Cloud

## 📁 Project Structure

```
Streamlit/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── copy_of_cis_412_team_project (1).py  # Original analysis code
```

## 🎓 Academic Project

This project was built for **CIS 412 Team Project** focusing on:
- Data preprocessing and cleaning
- Exploratory data analysis
- Machine learning model implementation
- Model evaluation and comparison
- Business intelligence and recommendations

## 🏆 Deployment Bonus

Teams that successfully deploy their model to Streamlit receive a **15% boost** to their overall project grade!

## 📞 Support

If you encounter any issues during deployment, check:
1. All files are pushed to GitHub
2. requirements.txt includes all dependencies
3. Repository is public
4. Main file is named `app.py`

---
*Built with ❤️ using Streamlit and Scikit-learn*