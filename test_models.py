#!/usr/bin/env python3
"""Test script to check if models can be loaded properly"""

import joblib
import os

def test_model_loading():
    """Test loading all model files"""
    models_dir = "analysis2/models/models"
    
    model_files = [
        "timeseries_model.joblib",
        "regression_model.joblib", 
        "hotspot_model.joblib",
        "regression_models.joblib"
    ]
    
    print("Testing model loading...")
    print("=" * 50)
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        try:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                print(f"✅ {model_file}: Loaded successfully")
                print(f"   Type: {type(model)}")
            else:
                print(f"❌ {model_file}: File not found")
        except Exception as e:
            print(f"❌ {model_file}: Error loading - {str(e)}")
    
    print("=" * 50)
    print("Model loading test completed")

if __name__ == "__main__":
    test_model_loading()




