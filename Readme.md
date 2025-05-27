# 🌊 Flood Prediction using IndoFloods Dataset

This project focuses on **flood risk prediction** using real-world flood event data from Indonesia (IndoFloods dataset). It leverages machine learning models like XGBoost, Random Forest, MLP, and SVR to predict flood risks based on various hydrological and geographical indicators.

## 📌 Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Multiple ML model training and evaluation
- Model persistence for reuse
- Interactive flood risk visualization
- Streamlit web interface

## ⚙️ How It Works

1. **Data Preparation**: Raw data is processed and saved to `data/processed/`
2. **EDA**: Analysis performed in `notebooks/01_EDA.ipynb`
3. **Modeling**: Models trained in `notebooks/02_Modeling.ipynb` and saved to `models/`
4. **Web App**: Streamlit interface at `streamlit_app/app.py`

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Santhosh071/flood_prediction.git
cd flood_prediction

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
cd streamlit_app
streamlit run app.py


## Project Structure

flood_prediction/
├── data/
│   ├── raw/                
│   ├── processed/          
│   └── outputs/            
├── models/                 
├── notebooks/              
│   ├── 01_EDA.ipynb        
│   └── 02_Modeling.ipynb   
├── src/                    
├── streamlit_app/          
│   └── app.py              
├── utils/                  
├── requirements.txt        
└── README.md               