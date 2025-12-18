# Jakarta-Air-Pollution-Predictor
_Predicting Jakarta’s PM2.5 air pollution up to 7 days ahead using machine learning and weather forecasts._

## Background
Jakarta frequently experiences PM2.5 levels exceeding WHO guidelines, posing serious health risks. Despite this, accessible short-term air quality forecasting tools remain limited. This project applies machine learning to anticipate pollution spikes before they occur.

## Objective / Goal:
Build a machine-learning model to predict Jakarta’s PM2.5 levels up to 7 days ahead using only upcoming weather forecasts, enabling early warnings and better planning.

## Tools & Technologies
- Google Colab
- Python 3.12.12
- pandas 2.2.2
- NumPy 2.0.2
- scikit-learn 1.6.1
- Matplotlib 3.10.0
- Seaborn 0.13.2
- XGBoost 3.1.2
- LightGBM 4.6.0
- TensorFlow / Keras 2.19.0
- Meteostat API
- AQICN API
- Open-Meteo API

## Dataset Description:
This project combines multiple data sources to build a rich, time-series dataset for training and forecasting PM2.5 levels in Jakarta.

**Data Sources**
- AQICN – Historical PM2.5 measurements for Jakarta.
- Meteostat – Historical weather observations (temperature, wind speed, pressure, precipitation, etc.).
- Open-Meteo – 7-day weather forecasts used for future PM2.5 prediction.

**Dataset Coverage**
- Location: Jakarta, Indonesia.
- Time Range: 2020–2025 (2020 is later excluded due to missing values).
- Frequency: Daily data.

| Feature      | Description                                      |
|--------------|--------------------------------------------------|
| pm25         | Historical PM2.5 concentration (target variable) |
| temp_avg     | Daily average temperature                        |
| temp_min     | Daily minimum temperature                        |
| temp_max     | Daily maximum temperature                        |
| wind_speed   | Average daily wind speed                         |
| pres         | Atmospheric pressure                             |
| precip       | Total daily precipitation                        |
| day_of_week  | Numerical day of week (0–6)                      |
| day_of_month | Day of month (1–31)                              |
| month        | Month (1–12)                                     |
| year         | Observation year                                 |

## Methodology:
**Data cleaning & preprocessing**
- Drop null columns – values are missing completely.
- Dropped rows from January to December 2020 – precipitation values are missing throughout the entire year of 2020.
- Explored distributions of features to identify outliers – precipitation is extremely skewed (log transformed later).
- Linear interpolation (time-based) for gaps ≤5 days – preserve short-term trends without altering the natural patterns of the data. 
- Dropped rows in the gap between December 2021 and January 2022 – massive gap of missing values, better to split the data there.
- Dropped remaining rows with missing values – to ensure clean, fully observed samples.
- All preprocessing steps were applied chronologically to prevent information leakage between training and testing sets.

**Exploratory Data Analysis (visualizations, trends)**
- Correlation plot
- Scatter plots against PM2.5
- Line graph of PM2.5 against time
- These analyses guided feature engineering decisions.

**Feature engineering**
- Feature interactions (temp_avg x precip, temp_min x wind_speed, etc).
- Cyclical features (month_sin, month_cos, etc) – to preserve the cyclical pattern of time features (day_of_week: 7 is close to 0).
- Additional features such as daily temperature range, precipitation lag and rolling features constructed using shifted values to avoid target leakage, etc.
- Log transform precipitation – as mentioned earlier, precipitation is highly skewed.
- Scale values using StandardScaler – does not rescale the data into a tiny range as MinMaxScaler does.

**Model training, tuning, & evaluation (regression, random forest, etc.)**
- Models used:
  1. Linear Regression (as baseline model)
  2. Random Forest Regressor
  3. LGBM Regressor
  4. XGBoost Regressor
  5. Feedforward Neural Network (MLP)
- Random Forest, LGBM, XGBoost, and Neural Net parameters tuned using a mix of RandomSearchCV and GridSearchCV.

## Project Workflow Diagram:
<img width="1161" height="231" alt="Jakarta Pollution Predictor Workflow drawio (2)" src="https://github.com/user-attachments/assets/6ba73453-ceed-4a38-9b07-b4e1b0775a72" />

## Results:
| Model             |  RMSE |   MSE   |  MAE  |   R²  |
|-------------------|:-----:|:-------:|:-----:|:-----:|
| Linear Regression | 43.85 | 1922.71 | 38.18 | -2.36 |
| Random Forest     | 24.68 | 609.19  | 19.02 | -0.07 |
| LGBM              | 22.73 | 516.77  | 17.09 | 0.10  |
| XGBoost           | 22.64 | 512.73  | 17.36 | 0.10  |
| Neural Net        | 28.88 | 834.14  | 23.59 | -0.46 |

Tree-based models (Random Forest, LightGBM, and XGBoost) significantly outperformed linear regression, indicating strong non-linear relationships between meteorological features and PM2.5 levels. XGBoost achieved the best results (lowest RMSE), while linear regression struggled to capture complex interactions, resulting in negative R² values. The neural network did not outperform tree-based models, likely due to the limited dataset size and the noisy nature of air pollution data.

## Conclusions & Key Takeaways: 
- Tree-based models significantly outperform linear regression for PM2.5 prediction.
- Weather variables exhibit strong non-linear relationships with air pollution.
- XGBoost achieved the best overall performance on this dataset.
- Neural networks were less effective, likely due to limited data and high noise.

## Future Work
- Include traffic, industrial activity, or holidays as additional features.
- Expand the range of the system to support multiple cities and spatial forecasting.
- Deploy the model as a light web application or API for public access.
