"""
FastAPI Application for MLOps Pipeline

This application provides FASTAPI endpoints for:
- Health checks
- Model training pipeline
- Target variable forecast
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import forecasting, health, training

# Create FastAPI app
app = FastAPI(
    title="Forecasting API",
    description="MLOps Pipeline API for training and forecasting.",
    version="v1",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.health_router, prefix="/api/v1", tags=["Health"])
app.include_router(training.training_router, prefix="/api/v1", tags=["Training"])
app.include_router(
    forecasting.prediction_router, prefix="/api/v1", tags=["Forecasting"]
)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
