# ==============================================================================
# 2. Forecasting Router Tests (POST /api/v1/forecast)
# ==============================================================================

"""
Forecasting Router

Provides endpoints for target variable forecasting
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from scripts.forecast_pipeline import ForecastingPipeline
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Global router
forcast_router = APIRouter()


# Endpoint to perform prediction
@forcast_router.post("/inventory_forecast")
async def inventory_forecast_pipeline(request: Request):
    try:
        data = await request.json()

        # Check if data is a list with config and input
        if isinstance(data, list) and len(data) >= 2:
            config_data = data[0]
            input_data = data[1]
        else:
            # Assume data is a single dictionary with all required fields
            config_data = {
                "CLIENT_NUMBER": data.get("CLIENT_NUMBER"),
                "SOURCE_NAME": data.get("SOURCE_NAME"),
                "TARGET_VARIABLE": data.get("TARGET_VARIABLE"),
                "ITEMSKU": data.get("ITEMSKU"),
                "MODEL_PIPELINE": data.get("MODEL_PIPELINE"),
            }
            input_data = data

        if not config_data or not input_data:
            return JSONResponse(
                content={"message": "config_data or input_data is missing"},
                status_code=400,
            )

        # Initialize prediction pipeline
        pipeline = ForecastingPipeline(config_data=config_data)

        # Extract forecast parameters from input_data
        periods = input_data.get("forecast_periods", 30)
        exog_features = input_data.get("exog_features", None)

        # Convert exog_features dict to DataFrame if provided
        exog_df = None
        if exog_features:
            # Create a DataFrame with the exog features repeated for all periods
            exog_df = pd.DataFrame([exog_features] * periods)

        # Call forecast method with correct parameters
        forecast_df = pipeline.forecast(
            periods=periods, exog_features_df=exog_df, date_index_col="date"
        )

        # Reset index to convert date index to a column
        forecast_df = forecast_df.reset_index()

        # Merge with inventory data BEFORE converting dates to strings
        # This ensures both dataframes have datetime types for the merge
        forecast_df = pipeline.final_inventory_consumption_data(forecast_df=forecast_df)

        # Now convert datetime columns to ISO format strings for JSON serialization
        for col in forecast_df.columns:
            if pd.api.types.is_datetime64_any_dtype(forecast_df[col]):
                forecast_df[col] = forecast_df[col].dt.strftime("%Y-%m-%d")

        # Final Response of the API
        forecast_dict = forecast_df.to_dict(orient="records")

        return JSONResponse(
            content={"message": "Forecast successful", "forecast": forecast_dict},
            status_code=200,
        )

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return JSONResponse(
            content={"message": f"API Error: {str(e)}", "traceback": error_trace},
            status_code=500,
        )


# Endpoint to perform prediction
@forcast_router.post("/revenue_forecast")
async def revenue_forecast_pipeline(request: Request):
    try:
        data = await request.json()

        # Check if data is a list with config and input
        if isinstance(data, list) and len(data) >= 2:
            config_data = data[0]
            input_data = data[1]
        else:
            # Assume data is a single dictionary with all required fields
            config_data = {
                "CLIENT_NUMBER": data.get("CLIENT_NUMBER"),
                "SOURCE_NAME": data.get("SOURCE_NAME"),
                "TARGET_VARIABLE": data.get("TARGET_VARIABLE"),
                "ITEMSKU": data.get("ITEMSKU"),
                "MODEL_PIPELINE": data.get("MODEL_PIPELINE"),
            }
            input_data = data

        if not config_data or not input_data:
            return JSONResponse(
                content={"message": "config_data or input_data is missing"},
                status_code=400,
            )

        # Initialize prediction pipeline
        pipeline = ForecastingPipeline(config_data=config_data)

        # Extract forecast parameters from input_data
        periods = input_data.get("forecast_periods", 30)
        exog_features = input_data.get("exog_features", None)

        # Convert exog_features dict to DataFrame if provided
        exog_df = None
        if exog_features:
            # Create a DataFrame with the exog features repeated for all periods
            exog_df = pd.DataFrame([exog_features] * periods)

        # Call forecast method with correct parameters
        forecast_df = pipeline.forecast(
            periods=periods, exog_features_df=exog_df, date_index_col="date"
        )

        # Now convert datetime columns to ISO format strings for JSON serialization
        for col in forecast_df.columns:
            if pd.api.types.is_datetime64_any_dtype(forecast_df[col]):
                forecast_df[col] = forecast_df[col].dt.strftime("%Y-%m-%d")

        # Final Response of the API
        forecast_dict = forecast_df.to_dict(orient="records")

        return JSONResponse(
            content={"message": "Forecast successful", "forecast": forecast_dict},
            status_code=200,
        )

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return JSONResponse(
            content={"message": f"API Error: {str(e)}", "traceback": error_trace},
            status_code=500,
        )
