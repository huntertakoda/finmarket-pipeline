# finmarket-pipeline

# Advanced Financial Analytics and Prediction System

## 1. Summary  
A full-stack financial analytics pipeline combining **Golang**, **Rust**, **SQL**, and **Python** for ingesting, processing, analyzing, and predicting market trends with machine learning.

## 2. High-Value Summary  
- **Multi-Technology Integration**: Efficient **Go**-based data ingestion, high-performance **Rust** processing, and advanced **SQL** analytics.  
- **Predictive Modeling**: Comprehensive **Python ML pipeline** with models like Stacked Ensembles, Random Forest, and Neural Networks.  
- **High-Value Insights**: Rolling correlations, volatility analysis, moving averages, and **visualized predictions** for financial forecasting.

---

## Project Layout

### 1. **Data Ingestion**  
**Technology:** Go (Golang)  
- Extracts and ingests financial data efficiently.

### 2. **Core Processing**  
**Technology:** Rust  
- High-performance data cleaning and transformation.

### 3. **Data Analytics**  
**Technology:** SQL  
- Advanced analytics with window functions, CTEs, and volatility calculations.  

### 4. **Machine Learning & Insights**  
**Technology:** Python  
- Predictive models:  
  - Linear Regression, SVR, Random Forest, Stacked Ensembles, and Neural Networks.  
- Visualizations:  
  - Actual vs Predicted Prices  
  - Moving Averages, Candlestick Charts, and Rolling Correlations.

---

## Key Screenshots & Visualizations  

![Candlestick Chart](plots/daily-candlestick-chart-with-vol.png)  
![Moving Averages](plots/close-price_moving-avg.png)  
![Model Predictions](plots/actual-vs-predicted-stacked.png)  
![Rolling Correlations](plots/30-day-rolling-correlation_close-price-vs-volume.png)  

---

## Future Enhancements  
- Real-time prediction API deployment.  
- Pipeline automation with Apache Airflow.  
- Enhanced models for live data analysis.
