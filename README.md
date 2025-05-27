# 🌊 Flood Prediction using IndoFloods Dataset

This project focuses on **flood risk prediction** using real-world flood event data from India's Indo Valley (IndoFloods dataset). It leverages machine learning models like XGBoost, Random Forest, MLP, and SVR to predict flood risks based on various hydrological and geographical indicators.

## 📌 Features

- Data preprocessing and cleaning for Indo Valley flood data
- Exploratory Data Analysis (EDA) specific to Indian geography
- Multiple ML model training and evaluation
- Model persistence for reuse
- Interactive flood risk visualization for Indian regions
- Streamlit web interface

## ⚙️ How It Works

1. **Data Preparation**: Raw Indo Valley data is processed and saved to `data/processed/`
2. **EDA**: Analysis performed in `notebooks/01_EDA.ipynb` focusing on Indian flood patterns
3. **Modeling**: Models trained in `notebooks/02_Modeling.ipynb` and saved to `models/`
4. **Web App**: Streamlit interface at `streamlit_app/app.py` with India-focused visualizations

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
