# Machine Learning-driven Predictive Maintenance of Turbofan Engines                                
 Written on 07/22/2025                                                                     


## Problem Definition
Predictive maintenance is a critical task in aerospace engineering. The global commercial fleet counts more than 27,000 aircraft, nearly all powered by turbofan engines. Even with high reliability, in-flight shutdowns occur about once every 375,000 flight hours, which translates into multiple incidents each year across the global fleet
. The goal of this project is to **predict the Remaining Useful Life (RUL) of turbofan engines using sensor data**, enabling early detection of degradation and reducing unexpected failures. This project was completed with a team of 4 other graduate students as part of CS7641 - Machine Learning at Georgia Tech.

We use the **NASA C-MAPSS dataset** for simulation of engine degradation under various operating conditions and fault modes.  
Dataset link: [NASA C-MAPSS Dataset](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/ff5v-kuh6)

---

## Methodology
1. **Exploratory Data Analysis (EDA)**: Understand trends, sensor behaviors, and degradation patterns.  
2. **Baseline Modeling**: Define evaluation metrics (RMSE) and compare with simple models.  
3. **Feature Engineering and Processing**: Normalize sensor readings, apply rolling statistics to reduce noise, and split train/test sets.  
4. **Anomaly Detection (CUSUM + Autoencoder)**: A two-step anomaly detection pipeline was used to identify the onset of abnormal engine degradation.  
   - **CUSUM** detects small but persistent shifts in sensor signals (temperature, pressure, and fuel flow ratios) by monitoring cumulative deviations from expected values.  
   - **Autoencoders** are trained only on healthy engine cycles to reconstruct normal behavior. Their reconstruction errors highlight deviations. Applying CUSUM to these errors pinpoints the first significant departure from healthy operation.  
   This process clips each engine’s time series into pre-anomaly and post-anomaly phases, allowing models to focus training on the degradation period and improving RUL prediction accuracy:contentReference[oaicite:0]{index=0}.  
5. **Model Training**: Train regression models (e.g., Random Forest, LSTM, Gradient Boosting, TCN) on the processed and anomaly-clipped datasets.  
6. **Evaluation**: Compare models using RMSE, MAE, MAPE and a custom score to quantify predictive accuracy.  

---

## Results
The framework achieved **RMSE in the range of 16.24 cycles** on the dataset FD001, significantly improving over the baseline model. This demonstrates that ML-driven approaches can provide accurate and robust RUL predictions.

---

## Environment Setup
```bash
conda env create -f ml_project.yml
conda activate my_ml_env

 


