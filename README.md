#  Logistic Trip Planner Agent

A sophisticated AI-powered logistics and transportation planning system that combines Machine Learning, RAG (Retrieval-Augmented Generation), and real-time data integration for intelligent route planning and ETA prediction.

##  Live Demo

**API Documentation:** [https://logistic-agent-1077932806589.australia-southeast2.run.app/docs](https://logistic-agent-1077932806589.australia-southeast2.run.app/docs)

##  Features

- **ML-Powered ETA Prediction** - XGBoost model with weather & traffic features
- **RAG System** - Intelligent document retrieval and question answering
- **Real-time Weather Integration** - BoM data feeds both ML model and RAG context
- **Live Traffic Data** - TomTom data enhances both predictions and responses
- **Hybrid AI Agent** - Combines ML predictions with RAG context intelligently
- **Docker Ready** - Production-ready containerization
- **Cloud Deployed** - Live on Google Cloud Run

##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Logistic Agent Planner                       │
│                        (FastAPI App)                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
         ┌──────────▼─────┐     │     ┌─────▼──────────┐
         │   ML Model     │     │     │   RAG System   │
         │  (XGBoost)     │     │     │  (ChromaDB)    │
         │                │     │     │                │
         │ • Predicts     │     │     │ • Retrieves    │
         │   travel time  │     │     │   documents    │
         │ • Uses weather │     │     │ • Provides     │
         │   & traffic    │     │     │   context      │
         └────────────────┘     │     └────────────────┘
                    │           │           │
                    └───────────┼───────────┘
                                │
                    ┌───────────▼───────────┐
                    │    Hybrid Agent       │
                    │   (OpenAI GPT-4o)     │
                    │                       │
                    │ • Combines ML         │
                    │   predictions with    │
                    │   RAG context         │
                    │ • Generates smart     │
                    │   responses           │
                    └───────────────────────┘
                                │
                                ▼
                         External APIs       
                                │                 
                    ┌───────────|──────────┐
                    │           │          │          
                ┌────▼────┐ ┌───▼───┐ ┌────▼────┐    
                │  BoM    │ │TomTom │ │ OpenAI  │    
                │Weather  │ │Traffic│ │   LLM   │    
                └─────────┘ └───────┘ └─────────┘    
```

**How it works:**
1. **Weather & Traffic APIs** provide real-time data
2. **ML Model** uses this data to predict travel times
3. **RAG System** retrieves relevant documents for context
4. **Hybrid Agent** combines both to give intelligent answers

##  Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- API Keys:
  - OpenAI API Key
  - TomTom API Key

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Logistic_Agent_Planner
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   OPENAI_API_KEY=your_openai_key_here
   TOMTOM_API_KEY=your_tomtom_key_here
   ```

4. **Run the application**
   ```bash
   uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

```bash
# Build the image
docker build -t logistic-agent .

# Run with environment variables
docker run --env-file .env -p 8000:8000 logistic-agent
```

##  API Endpoints

### 1. Health Check
```bash
curl -X GET "https://logistic-agent-1077932806589.australia-southeast2.run.app/health"
```

**Response:**
```json
{
  "ok": true,
  "model_loaded": true,
  "preprocessor_loaded": true,
  "features_count": 25,
  "artifact_dir": "/app/src/model_folder_v1"
}
```

### 2. Travel Time Prediction
```bash
curl -X POST "https://logistic-agent-1077932806589.australia-southeast2.run.app/predict_travel_time" \
  -H "Content-Type: application/json" \
  -d '{
    "route": "Hume_Highway"
  }'
```

**Response:**
```json
{
  "route": "Hume_Highway",
  "ts_utc": "2024-01-15T10:30:00Z",
  "ts_local": "2024-01-15 21:30:00 AEDT",
  "features_used": {
    "route": "Hume_Highway",
    "day_of_week": 1,
    "hour_of_day": 21,
    "wx_temp_c": 18.5,
    "wx_rain_mm": 0.0,
    "wx_wind_kmh": 15.2,
    "leg_incident_count": 2,
    "leg_incident_delay_sum_sec": 180.0
  },
  "prediction_seconds": 6840,
  "prediction_minutes": 114.0,
  "prediction_hours": 1.9,
  "sources": {
    "bom": {
      "location": "Melbourne",
      "temperature": 18.5,
      "rainfall_since_9am": 0.0,
      "wind_speed_kmh": 15.2
    },
    "tomtom_summary": {
      "bbox": "144.5,-38.0,152.0,-33.0",
      "count": 2
    }
  }
}
```

### 3. RAG Query
```bash
curl -X POST "https://logistic-agent-1077932806589.australia-southeast2.run.app/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main freight routes in the Hume region?",
    "k": 5
  }'
```

**Response:**
```json
{
  "answer": "Based on the retrieved documents, the main freight routes in the Hume region include the Hume Highway (M31) which serves as the primary north-south corridor connecting Melbourne to Sydney, and the Inland Route which provides an alternative path through regional Victoria and New South Wales.",
  "used_k": 5
}
```

### 4. Hybrid Agent (Recommended)
```bash
curl -X POST "https://logistic-agent-1077932806589.australia-southeast2.run.app/agent/compose" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the current travel time from Melbourne to Sydney via Hume Highway and let me know about the resting points on the way?",
    "k": 5,
    "route": "Hume_Highway"
  }'
```

**Response:**
```json
{
  "answer": "The current travel time from Melbourne to Sydney via the Hume Highway is approximately 10 hours and 46 minutes, based on the provided ETA. Along the way, there are several resting points that can be utilized for breaks. Notable rest areas include:
  1. **Wodonga** - A key location with facilities for drivers, especially near livestock saleyards. 
  2. **Wangaratta** - Another significant stop with amenities for truck drivers.
  3. **Shepparton** - Offers rest facilities and is a major hub in the region.
  4. **Yea** - Provides resting options for travelers.
  5. **Euroa** - A convenient stop along the route.
  6. **Yarrawonga** - Also has facilities for resting.
  There are a total of 35 designated truck parking areas along the A and B road network in the region, which include full-service centers and roadside laybys. These rest areas are essential for ensuring driver safety and compliance with fatigue management regulations.",
  "used_k": 5,
  "included": {
    "eta": {
      "route": "Hume_Highway",
      "prediction_seconds": 6840,
      "prediction_minutes": 114.0,
      "prediction_hours": 1.9,
      "ts_local": "2024-01-15 21:30:00 AEDT"
    },
    "weather": {
      "location": "Melbourne",
      "temperature": 18.5,
      "rainfall_since_9am": 0.0,
      "wind_speed_kmh": 15.2
    },
    "traffic": {
      "bbox": "144.5,-38.0,152.0,-33.0",
      "count": 2
    }
  }
}
```

## Supported Routes

### Available Routes:
- **`Hume_Highway`** - Primary route from Melbourne to Sydney via Albury and Goulburn
- **`Inland_Route`** - Alternative route through Shepparton and Dubbo

> **Note:** Due to limited information on the Inland Route, the RAG system has been primarily trained on Hume Highway data. The ML model works for both routes, but RAG context is more comprehensive for Hume Highway.

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
TOMTOM_API_KEY=your_tomtom_api_key
ARTIFACT_DIR=src/model_folder_v1  # Optional, defaults to this path
```

### Model Configuration
The system uses:
- **ML Model**: XGBoost for ETA prediction
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: GPT-4o-mini for RAG responses
- **Vector DB**: ChromaDB for document storage

## Data Sources

### Real-time Data Integration:
1. **Bureau of Meteorology (BoM)** - Weather conditions
   - **ML Model**: Temperature, rainfall, wind speed as prediction features
   - **RAG System**: Weather context for intelligent responses
2. **TomTom Traffic API** - Live traffic incidents
   - **ML Model**: Incident count, delay duration as prediction features
   - **RAG System**: Traffic context for route planning advice
3. **OSRM** - Base travel time calculations

### RAG Knowledge Base:
- Hume Region Planning documents
- Freight transport policies
- Road infrastructure reports
- Regional development plans

## Project Structure

```
Logistic_Agent_Planner/
├── src/
│   ├── app.py              # FastAPI application
│   ├── tools.py            # External API integrations
│   ├── rag_service.py      # RAG system implementation
│   └── model_folder_v1/    # ML model artifacts
├── data/                   # Training data and documents
├── notebooks/              # Jupyter notebooks for analysis
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Deployment

### Google Cloud Run
The application is currently deployed on Google Cloud Run at:
[https://logistic-agent-1077932806589.australia-southeast2.run.app/docs](https://logistic-agent-1077932806589.australia-southeast2.run.app/docs)




