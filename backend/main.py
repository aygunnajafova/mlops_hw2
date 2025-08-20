from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import logging

# Keeping some logging for debubbing errors.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Azercell HW2 API",
    description="Creating predictions from FASTAPI and serialized models.",
    version="1.0.0"
)


class HousingFeatures(BaseModel):
    features: List[float]
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 2:
            raise ValueError('Housing model requires exactly 2 features: area and room')
        return v

class RamenFeatures(BaseModel):
    features: List[float]
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 10:
            raise ValueError('Ramen model requires exactly 10 features')
        return v

class PredictionResponse(BaseModel):
    prediction: Union[float, int, str]
    model_name: str
    confidence: Optional[float] = None

class ErrorResponse(BaseModel):
    error: str
    detail: str

# This is where I am storing my loaded models.
models = {}
scalers = {}

def load_models():
    """Load trained model weights"""
    try:
        # Try to find models directory - works for both local and Docker
        models_dir = None
        current_dir = Path.cwd()
        logger.info(f"Current working directory: {current_dir}")
        
        # Check if models/ exists in current directory
        models_path_current = Path("models")
        logger.info(f"Checking current directory for models: {models_path_current.absolute()}")
        logger.info(f"models/ exists in current dir: {models_path_current.exists()}")
        if models_path_current.exists():
            logger.info(f"Contents of current models/ directory: {list(models_path_current.iterdir())}")
        
        # Check if ../models exists
        models_path_parent = Path("../models")
        logger.info(f"Checking parent directory for models: {models_path_parent.absolute()}")
        logger.info(f"../models exists: {models_path_parent.exists()}")
        if models_path_parent.exists():
            logger.info(f"Contents of ../models directory: {list(models_path_parent.iterdir())}")
        
        # Try to find models directory - works for both local and Docker
        if models_path_current.exists():
            models_dir = models_path_current  # Docker: /app/models
            logger.info("Using models directory: models/ (current directory)")
        elif models_path_parent.exists():
            models_dir = models_path_parent  # Local: ../models
            logger.info("Using models directory: ../models/ (parent directory)")
        else:
            logger.error("Neither models/ nor ../models/ directories found!")
            logger.error(f"Current directory contents: {list(current_dir.iterdir())}")
            if current_dir.parent.exists():
                logger.error(f"Parent directory contents: {list(current_dir.parent.iterdir())}")
            raise FileNotFoundError("Models directory not found. Tried 'models/' and '../models/'")
        
        logger.info(f"Selected models directory: {models_dir.absolute()}")
        
        # Load Housing models
        housing_path = models_dir / "housing_price_model.pkl"
        housing_scaler_path = models_dir / "housing_price_scaler.pkl"
        
        logger.info(f"Looking for housing model at: {housing_path}")
        logger.info(f"Housing model exists: {housing_path.exists()}")
        
        if housing_path.exists():
            models["housing"] = joblib.load(housing_path)
            logger.info("✅ Successfully loaded Housing Price model")
        else:
            logger.warning("❌ Housing Price model not found")
        
        if housing_scaler_path.exists():
            scalers["housing"] = joblib.load(housing_scaler_path)
            logger.info("✅ Successfully loaded Housing Price scaler")
        else:
            logger.warning("❌ Housing Price scaler not found")
        
        # Load Ramen models
        ramen_path = models_dir / "ramen_regression_model.pkl"
        ramen_scaler_path = models_dir / "ramen_regression_scaler.pkl"
        
        logger.info(f"Looking for ramen model at: {ramen_path}")
        logger.info(f"Ramen model exists: {ramen_path.exists()}")
        
        if ramen_path.exists():
            models["ramen"] = joblib.load(ramen_path)
            logger.info("✅ Successfully loaded Ramen Regression model")
        else:
            logger.warning("❌ Ramen Regression model not found")
        
        if ramen_scaler_path.exists():
            scalers["ramen"] = joblib.load(ramen_scaler_path)
            logger.info("✅ Successfully loaded Ramen Regression scaler")
        else:
            logger.warning("❌ Ramen Regression scaler not found")
        
        logger.info(f"=== Model loading complete. Loaded {len(models)} models ===")
        logger.info(f"Available models: {list(models.keys())}")
        logger.info(f"Available scalers: {list(scalers.keys())}")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """This is initial startup event, we will first call this function and then every other requests will be processed."""
    load_models()

@app.get("/status")
async def root():
    """To get status of the service."""
    return {"message": "My Azercell HW2 API is running.", "status": "running"}


@app.get("/models")
async def list_models():
    """Display all available models"""
    return {"available_models": list(models.keys())}

@app.post("/predict/housing", response_model=PredictionResponse)
async def predict_housing(features: HousingFeatures):
    """Make predictions using Housing Price model"""
    try:
        if "housing" not in models:
            raise HTTPException(status_code=500, detail="Housing model not available")
        
        # Check if we have exactly 2 features (area, room)
        if len(features.features) != 2:
            raise HTTPException(status_code=400, detail="Housing model expects exactly 2 features: area and room")
        
        X = np.array(features.features).reshape(1, -1)
        
        if "housing" in scalers:
            X = scalers["housing"].transform(X)
        
        # Make prediction
        prediction = models["housing"].predict(X)[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            model_name="Housing Price",
            confidence=None
        )
        
    except Exception as e:
        logger.error(f"Error in Housing prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/ramen", response_model=PredictionResponse)
async def predict_ramen(features: RamenFeatures):
    """Make predictions using Ramen Regression model"""
    try:
        if "ramen" not in models:
            raise HTTPException(status_code=500, detail="Ramen model not available")
        
        # Check if we have exactly 10 features
        if len(features.features) != 10:
            raise HTTPException(status_code=400, detail="Ramen model expects exactly 10 features")
        
        # The ramen model expects 200 features, so we'll pad with average values
        required_features = 200
        
        # Create a feature array with the 10 provided features + average values for the rest
        if len(features.features) < required_features:
            # Calculate average of provided features for padding
            avg_value = sum(features.features) / len(features.features)
            
            # Pad with average values
            padded_features = list(features.features) + [avg_value] * (required_features - len(features.features))
            X = np.array(padded_features).reshape(1, -1)
        else:
            X = np.array(features.features).reshape(1, -1)
        
        # Make prediction
        prediction = models["ramen"].predict(X)[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            model_name="Ramen Rating",
            confidence=None
        )
        
    except Exception as e:
        logger.error(f"Error in Ramen prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
