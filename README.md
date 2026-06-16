<<<<<<< HEAD
# Nassau Candy Factory Optimization Dashboard

## Setup
```bash
cd nassau_candy
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

The app will open at http://localhost:8501

## Project Structure
```
nassau_candy/
├── app.py              ← Streamlit dashboard (5 tabs)
├── data_prep.py        ← Data loading & feature engineering
├── models_train.py     ← ML training, simulation & recommendations
├── requirements.txt
├── data/
│   └── Nassau_Candy_Distributor.csv
└── models/             ← Pre-trained model (auto-generated on first run)
    ├── best_model.pkl
    └── encoders.pkl
```

## Dashboard Tabs
1. **Factory Simulator** – See predicted lead times for all 5 factories
2. **What-If Analysis** – Compare current vs proposed factory assignment
3. **Recommendations** – Top-N ranked reassignment suggestions
4. **EDA & Insights** – Charts, heatmaps, trends
5. **Model Evaluation** – RMSE/MAE/R² comparison, feature importance
=======
---
title: Nasa Candy App
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---

# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).
>>>>>>> 2ace27931e0d7313e79f1c838ab3e056f52ebeed
