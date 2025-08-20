import streamlit as st
import requests
import plotly.graph_objects as go
from typing import List, Dict, Any
import os

# Page configuration
st.set_page_config(
    page_title="Azercell HW2 Prediction Service.",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL for backend service.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def check_backend_health():
    """Health check for our backend service."""
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException:
        return False, {"error": "Backend service is down."}

def get_available_models():
    """Get list of hosted models in the backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json()["available_models"]
        return []
    except requests.exceptions.RequestException:
        return []

def run_inference(endpoint: str, features: List[float]) -> Dict[str, Any]:
    """Make inference requests."""
    try:
        payload = {"features": features}
        response = requests.post(f"{BACKEND_URL}/{endpoint}", json=payload, timeout=10)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.json().get("detail", "Unknown error")}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}

def create_feature_inputs(num_features: int, feature_names: List[str] = None) -> List[float]:
    """Feature engineering."""
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(num_features)]
    
    features = []
    cols = st.columns(min(3, num_features))
    
    for i, name in enumerate(feature_names):
        col_idx = i % 3
        with cols[col_idx]:
            feature = st.number_input(
                name,
                value=0.0,
                step=0.01,
                format="%.4f",
                key=f"feature_{i}"
            )
            features.append(feature)
    
    return features

def display_prediction_results(results: Dict[str, Any], model_type: str):
    """Display prediction results."""
    if not results["success"]:
        st.error(f"Prediction failed: {results['error']}")
        return
    
    data = results["data"]
    
    # Handle single prediction
    st.success("Prediction completed successfully!")
    
    # Create gauge chart for single prediction
    if isinstance(data["prediction"], (int, float)):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=data["prediction"],
            title={'text': f"{data['model_name']} Prediction"},
            gauge={'axis': {'range': [None, data["prediction"] * 1.2]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, data["prediction"]], 'color': "lightgray"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': data["prediction"]}}
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"**Model**: {data['model_name']}")
    st.info(f"**Prediction**: {data['prediction']}")

def main():
    st.title("ML Ops Prediction Interface")
    st.markdown("---")
    
    st.sidebar.title("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ["Housing Price", "Ramen Rating"]
    )
    
    st.markdown("---")
    
    if model_choice == "Housing Price":
        st.header("Housing Price Prediction Model")
        st.markdown("""
        This model predicts housing prices based on area and number of rooms.
        
        **Feature ranges:**
        - Area: 30-320 sq meters (typical Amsterdam housing)
        - Rooms: 1-7 rooms
        """)
        
        features = create_feature_inputs(2, [
            "Area (sq meters)", "Number of Rooms"
        ])
        
        if st.button("Predict Housing Price", type="primary"):
            with st.spinner("Making prediction..."):
                results = run_inference("predict/housing", features)
                display_prediction_results(results, "housing")
    
    elif model_choice == "Ramen Rating":
        st.header("Ramen Rating Prediction Model")
        st.markdown("""
        This model predicts ramen ratings based on 10 key features.
        The remaining 190 features will be automatically filled with average values.
        
        **Key Features (enter values 0-10):**
        - Brand Score, Variety Score, Style Score
        - Country Score, Packaging Score, Price Score
        - Additional features 7-10
        
        **Note:** The model internally uses 200 features total.
        """)
        
        features = create_feature_inputs(10, [
            "Brand Score", "Variety Score", "Style Score",
            "Country Score", "Packaging Score", "Price Score",
            "Feature 7", "Feature 8", "Feature 9", "Feature 10"
        ])
        
        if st.button("Predict Ramen Rating", type="primary"):
            with st.spinner("Making prediction..."):
                results = run_inference("predict/ramen", features)
                display_prediction_results(results, "ramen")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        ML Ops Prediction Interface | Built with Streamlit and FastAPI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
