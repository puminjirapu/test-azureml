import os
import numpy as np
import pandas as pd
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    """
    Load the model when the batch inference service starts.
    This method is called once when the deployment starts.
    """
    global model
    
    # Get the path to the model file
    model_path = os.path.join(os.environ.get("AZUREML_MODEL_DIR", ""), "model.pkl")
    
    try:
        # Load the model
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def run(mini_batch):
    """
    Perform batch inference on the input data.
    
    Args:
        mini_batch (list or pandas.DataFrame): Input data for batch inference
    
    Returns:
        pandas.DataFrame: Predictions with input data
    """
    try:
        # Handle different input types
        if isinstance(mini_batch, list):
            # If input is a list of file paths, read files
            dataframes = []
            for file_path in mini_batch:
                # Detect file type and read accordingly
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_path}")
                dataframes.append(df)
            
            # Concatenate if multiple files
            input_data = pd.concat(dataframes) if len(dataframes) > 1 else dataframes[0]
        elif isinstance(mini_batch, pd.DataFrame):
            # If already a DataFrame, use as-is
            input_data = mini_batch
        else:
            raise ValueError("Input must be a list of file paths or a pandas DataFrame")
        
        # Prepare input data (remove target column if present)
        X = input_data.copy()
        if 'target' in X.columns:
            X = X.drop('target', axis=1)
        
        # Perform prediction
        predictions = model.predict(X)
        
        # Create output DataFrame with predictions
        output = pd.DataFrame({
            'prediction': predictions
        })
        
        # Optionally add input data columns
        output = pd.concat([input_data, output], axis=1)
        
        logger.info(f"Batch prediction completed. Processed {len(output)} records.")
        return output
    
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise
