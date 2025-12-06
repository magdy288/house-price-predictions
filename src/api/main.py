from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas import HousePredictionRequest, PredictionResponse
from inference import predict_price, batch_predict
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize FastAPI app with metadata
app = FastAPI(
    title='House Price Prediction',
    description=(
        'An PI for predicting house orices based on various features.'
        'This application is part of MLops Bootcamp '
    ),
    version='1.0.0',
    contact={
        'name': 'magdy288',
        'email': 'magdy288m@gmail.com'
    },
    license_info={
        'name':'Apache 2.0',
        'url':'https://www.apache.org/licenses/LICENSE-2.0.html'
    }
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Initialize and insroment Prometheus metrics
Instrumentator().instrument(app).expose(app) # Add this 

# Health check endpoint
@app.get('/health', response_model=dict)
async def health_check():
    return {'status': 'healthy', 'model_loaded': True}


# Prediction endpoint
@app.post('/predict', response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    return predict_price(request)

# Batch prediction endpoint
@app.post('/batch-predict', response_model=list)
async def batch_predict_endpoint(requests: list[HousePredictionRequest]):
    return batch_predict(requests)