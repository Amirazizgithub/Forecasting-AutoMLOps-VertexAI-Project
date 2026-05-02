# ==============================================================================
# 3. Training Router Tests (POST /api/v1/train)
# ==============================================================================

"""
Training Router

Provides endpoints for triggering and monitoring ML pipeline training
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from scripts.run_pipeline import Compile_Run_Pipeline
from pipeline.model_pipeline import _forecasting_pipeline_
import warnings

warnings.filterwarnings("ignore")

# Global router
training_router = APIRouter()


# Endpoint to compile & execute vertexai pipeline
@training_router.post("/inventory_train")
async def compile_run_pipeline_for_inventory(request: Request):
    try:
        data = await request.json()
        client_number = data.get("CLIENT_NUMBER")
        source_name = data.get("SOURCE_NAME")
        target_variable = data.get("TARGET_VARIABLE")

        if not client_number or not source_name or not target_variable:
            return JSONResponse(
                content={
                    "message": "client_number or source_name or target_variable is missing"
                },
                status_code=400,
            )
        data["ROUTES_SUFFIX"] = "inventory_forecast"
        # Initialize and run the pipeline compilation and execution
        runner = Compile_Run_Pipeline(data=data)

        # Pass the definition object to the compile method
        runner.compile_pipeline(pipeline_func=_forecasting_pipeline_)
        runner.run_pipeline()

        return JSONResponse(
            content={
                "message": "Training successfully: Vertex AI pipeline's compilation & execution done."
            },
            status_code=200,
        )

    except Exception as e:
        return JSONResponse(
            content={"message": f"API Error: {str(e)}"}, status_code=500
        )


# Endpoint to compile & execute vertexai pipeline
@training_router.post("/revenue_train")
async def compile_run_pipeline_for_revenue(request: Request):
    try:
        data = await request.json()
        client_number = data.get("CLIENT_NUMBER")
        source_name = data.get("SOURCE_NAME")
        target_variable = data.get("TARGET_VARIABLE")

        if not client_number or not source_name or not target_variable:
            return JSONResponse(
                content={
                    "message": "client_number or source_name or target_variable is missing"
                },
                status_code=400,
            )

        data["ROUTES_SUFFIX"] = "revenue_forecast"

        # Initialize and run the pipeline compilation and execution
        runner = Compile_Run_Pipeline(data=data)

        # Pass the definition object to the compile method
        runner.compile_pipeline(pipeline_func=_forecasting_pipeline_)
        runner.run_pipeline()

        return JSONResponse(
            content={
                "message": "Training successfully: Vertex AI pipeline's compilation & execution done."
            },
            status_code=200,
        )

    except Exception as e:
        return JSONResponse(
            content={"message": f"API Error: {str(e)}"}, status_code=500
        )
