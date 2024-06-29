# traffic-congestion-prediction
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Data cleaning and feature engineering
    # Add relevant features like traffic volume, weather conditions, etc.
    # Handle missing values
    
    # Feature scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data
