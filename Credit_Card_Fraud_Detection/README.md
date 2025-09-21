# Credit Card Fraud Detection

**Dataset:** https://huggingface.co/datasets/dazzle-nu/CIS435-CreditCardFraudDetection

This project focuses on predicting credit card fraud based on transaction features such as Age, Distance from home, Merchant category, Day of Week, and transaction amount. The system includes a web-based demo where users can input synthetic transaction data to test whether the trained model classifies it as fraudulent or legitimate.

## Project Structure

The `src/` folder contains two main modules:
- `data_preprocessing.py` - Handles data loading, cleaning, and feature engineering
- `model_building.py` - Implements multiple ML models including custom MLP, XGBoost, SVM, and Random Forest

These modules are orchestrated through `main.py`. Running `python src/main.py` initiates the complete pipeline:
1. Data preprocessing (if not already done)
2. Training multiple classification models
3. Model evaluation and saving the best performer to `saved_models/` folder

⚠️ **Note:** The full training process can take considerable time.

## Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
## Quick Start

### Option 1: Quick Demo (Recommended for Testing)

For rapid testing and demo purposes:

1. **Train a demo model:**
   ```bash
   python demo_model.py
   ```

2. **Launch the Streamlit web application:**
   ```bash
   streamlit run streamlit_app.py
   ```

### Option 2: Full Model Training & Evaluation

If you want to train all provided models (MLP, XGBoost, Random Forest, SVM) and automatically select the best performer:
All of the provided above will give much better results than the custom MLP model

```bash
python src/main.py
```

```bash
   streamlit run streamlit_app.py
```

⚠️ **Note:** This will:
- Train multiple classification models
- Evaluate and compare their performance  
- Save the best model to `saved_models/` folder
- Take considerably longer than the demo

## Web Application Features

The app will open in your browser at `http://localhost:8501`

The web app features:
- **Fraud Detector:** Real-time fraud prediction with user inputs
- **Data Analytics:** Fraud pattern visualization and insights  
- **Model Performance:** Metrics, confusion matrix, and feature importance

## Future Enhancements

- Command line arguments (argparse) for flexible configuration
- YAML/JSON configuration files for model parameters
- Structured logging instead of print statements
- Comprehensive error handling and validation
- Unit tests for core functions
- Docker containerization for easy deployment
- Model versioning and experiment tracking (MLflow/Weights & Biases)
- REST API development (FastAPI)
- Cloud deployment with monitoring and health checks
