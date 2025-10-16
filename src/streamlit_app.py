import streamlit as st
import requests
from datetime import datetime, timezone
import json
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Logistics AI Assistant",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Initialize session state for conversation memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "eta_predictions" not in st.session_state:
    st.session_state.eta_predictions = []

# Helper function to display weather with visualization
def display_weather_info(weather_data: Dict[str, Any]):
    """Display weather information with nice formatting and icons"""
    st.markdown("### 🌤️ Weather Conditions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp = weather_data.get("temperature", "N/A")
        if temp != "N/A":
            st.metric("🌡️ Temperature", f"{temp}°C")
        else:
            st.metric("🌡️ Temperature", "N/A")
    
    with col2:
        rainfall = weather_data.get("rainfall_since_9am", 0)
        if rainfall is None:
            rainfall = 0
        rain_icon = "🌧️" if rainfall > 0 else "☀️"
        st.metric(f"{rain_icon} Rainfall", f"{rainfall} mm")
    
    with col3:
        wind_speed = weather_data.get("wind_speed_kmh", "N/A")
        if wind_speed != "N/A":
            st.metric("💨 Wind Speed", f"{wind_speed} km/h")
        else:
            st.metric("💨 Wind Speed", "N/A")
    
    with col4:
        location = weather_data.get("location", "N/A")
        st.markdown(f"**📍 Location**")
        st.markdown(f"{location}")

# Sidebar configuration
with st.sidebar:
    st.title("Logistics Assistant")
    st.markdown("---")
    
    # Route selection
    st.subheader("Route Configuration")
    route = st.selectbox(
        "Select Route",
        ["None", "Hume_Highway", "Inland_Route"],
        help="Select a route for ETA predictions"
    )
    
    # Advanced settings
    st.subheader("Advanced Settings")
    k_value = st.slider("RAG Context Chunks (k)", min_value=1, max_value=10, value=5)
    
    # API health check
    st.markdown("---")
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("API Connected")
        else:
            st.error("API Error")
    except Exception as e:
        st.error(f"API Unavailable")
    
    # Clear conversation button
    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.eta_predictions = []
        st.rerun()

# Main chat interface
st.title("Logistics AI Assistant")
st.markdown("Ask me anything about logistics, routes, weather, traffic, or get ETA predictions!")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display ETA data if available
        if "eta_data" in message and message["eta_data"]:
            with st.expander("📊 ETA Prediction Details"):
                eta = message["eta_data"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Time", f"{eta.get('prediction_hours', 0):.2f} hours")
                with col2:
                    st.metric("Minutes", f"{eta.get('prediction_minutes', 0):.2f} min")
                with col3:
                    st.metric("Route", eta.get('route', 'N/A'))
        
        # Display weather if available
        if "included" in message and message.get("included", {}).get("weather"):
            with st.expander("🌤️ Weather Information"):
                display_weather_info(message["included"]["weather"])

# Chat input
if prompt := st.chat_input("Ask me about logistics, routes, or ETA predictions..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare request based on route selection
                request_data = {
                    "question": prompt,
                    "k": k_value,
                }
                
                # Add route if selected
                if route != "None":
                    request_data["route"] = route
                    request_data["ts_utc"] = datetime.now(timezone.utc).isoformat()
                
                # Call the agent/compose endpoint
                response = requests.post(
                    f"{API_BASE_URL}/agent/compose",
                    json=request_data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No response received")
                    included = data.get("included", {})
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Display ETA if available
                    if included.get("eta"):
                        eta = included["eta"]
                        with st.expander("📊 ETA Prediction Details"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Predicted Time", f"{eta.get('prediction_hours', 0):.2f} hours")
                            with col2:
                                st.metric("Minutes", f"{eta.get('prediction_minutes', 0):.2f} min")
                            with col3:
                                st.metric("Route", eta.get('route', 'N/A'))
                    
                    # Display weather if available
                    if included.get("weather"):
                        with st.expander("🌤️ Weather Information"):
                            display_weather_info(included["weather"])
                    
                    # Save to conversation history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "eta_data": included.get("eta"),
                        "included": included
                    })
                    
                    # Save ETA prediction if available
                    if included.get("eta"):
                        st.session_state.eta_predictions.append({
                            "timestamp": datetime.now().isoformat(),
                            "question": prompt,
                            "eta": included["eta"]
                        })
                    
                else:
                    error_msg = f"Error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    
            except requests.exceptions.Timeout:
                error_msg = "⏱️ Request timed out. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
