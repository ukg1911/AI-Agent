import os
import uvicorn
import uuid
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field, field_validator
from masumi.config import Config
from masumi.payment import Payment, Amount
# We keep this import if you have the file, otherwise it might error if file is missing
try:
    from crew_definition import ResearchCrew
except ImportError:
    pass 
from logging_config import setup_logging

# Configure logging
logger = setup_logging()

# Load environment variables
load_dotenv(override=True)

# Retrieve API Keys and URLs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PAYMENT_SERVICE_URL = os.getenv("PAYMENT_SERVICE_URL")
PAYMENT_API_KEY = os.getenv("PAYMENT_API_KEY")
NETWORK = os.getenv("NETWORK")

logger.info("Starting application with configuration:")
logger.info(f"PAYMENT_SERVICE_URL: {PAYMENT_SERVICE_URL}")

# Initialize FastAPI
app = FastAPI(
    title="API following the Masumi API Standard",
    description="API for running Agentic Services tasks with Masumi payment integration",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────────────────────
# Temporary in-memory job store
# ─────────────────────────────────────────────────────────────────────────────
jobs = {}
payment_instances = {}

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Masumi Payment Config
# ─────────────────────────────────────────────────────────────────────────────
config = Config(
    payment_service_url=PAYMENT_SERVICE_URL,
    payment_api_key=PAYMENT_API_KEY
)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────
class AdOfferInput(BaseModel):
    ad_category: str
    bid_amount: str
    ad_content_url: str

class StartJobRequest(BaseModel):
    identifier_from_purchaser: str
    input_data: AdOfferInput
    
    class Config:
        json_schema_extra = {
            "example": {
                "identifier_from_purchaser": "marketer_123",
                "input_data": {
                    "ad_category": "Technology",
                    "bid_amount": "0.5",
                    "ad_content_url": "https://ipfs.io/ipfs/..."
                }
            }
        }

class ProvideInputRequest(BaseModel):
    job_id: str

# ─────────────────────────────────────────────────────────────────────────────
# Task Execution (Gatekeeper Logic)
# ─────────────────────────────────────────────────────────────────────────────
# Mock user preferences
USER_PREFERENCES = {
    "interested_categories": ["Technology", "Gaming", "DeFi"],
    "min_bid": 0.5 
}

async def execute_crew_task(input_data: dict) -> str:
    """ 
    Acts as the 'Gatekeeper'. 
    """
    logger.info(f"Evaluating Ad Offer: {input_data}")
    
    # 1. Check Category
    incoming_category = input_data.get("ad_category")
    
    if not incoming_category:
        return "ERROR: No category provided."

    if incoming_category not in USER_PREFERENCES["interested_categories"]:
        return f"REJECTED: User is not interested in {incoming_category}."

    return f"ACCEPTED: Ad for {incoming_category} displayed. User credited."

# ─────────────────────────────────────────────────────────────────────────────
# 1) Start Job (MIP-003: /start_job)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/start_job")
async def start_job(data: StartJobRequest):
    """ Initiates a job and creates a payment request """
    print(f"Received data: {data}")
    
    try:
        job_id = str(uuid.uuid4())
        agent_identifier = os.getenv("AGENT_IDENTIFIER")
        
        # --- FIX START ---
        # accessing fields directly from Pydantic model, not dictionary key ["text"]
        input_desc = f"Ad Offer: {data.input_data.ad_category}"
        logger.info(f"Received job request: '{input_desc}'")
        logger.info(f"Starting job {job_id} with agent {agent_identifier}")

        # Convert Pydantic model to dict for storage and Masumi SDK
        input_data_dict = data.input_data.model_dump()
        # --- FIX END ---

        # Define payment amounts
        payment_amount = os.getenv("PAYMENT_AMOUNT", "10000000")  # Default 10 ADA
        payment_unit = os.getenv("PAYMENT_UNIT", "lovelace")

        amounts = [Amount(amount=payment_amount, unit=payment_unit)]
        logger.info(f"Using payment amount: {payment_amount} {payment_unit}")
        
        # Create a payment request using Masumi
        payment = Payment(
            agent_identifier=agent_identifier,
            amounts=amounts,
            config=config,
            identifier_from_purchaser=data.identifier_from_purchaser,
            input_data=input_data_dict, # Pass the dict here
            network=NETWORK
        )
        
        logger.info("Creating payment request...")
        payment_request = await payment.create_payment_request()
        blockchain_identifier = payment_request["data"]["blockchainIdentifier"]
        payment.payment_ids.add(blockchain_identifier)
        logger.info(f"Created payment request with blockchain identifier: {blockchain_identifier}")

        # Store job info (Awaiting payment)
        jobs[job_id] = {
            "status": "awaiting_payment",
            "payment_status": "pending",
            "blockchain_identifier": blockchain_identifier,
            "input_data": input_data_dict, # Store as dict so execute_crew_task can read it
            "result": None,
            "identifier_from_purchaser": data.identifier_from_purchaser
        }

        async def payment_callback(blockchain_identifier: str):
            await handle_payment_status(job_id, blockchain_identifier)

        # Start monitoring the payment status
        payment_instances[job_id] = payment
        logger.info(f"Starting payment status monitoring for job {job_id}")
        await payment.start_status_monitoring(payment_callback)

        # Return the response
        return {
            "status": "success",
            "job_id": job_id,
            "blockchainIdentifier": blockchain_identifier,
            "submitResultTime": payment_request["data"]["submitResultTime"],
            "unlockTime": payment_request["data"]["unlockTime"],
            "externalDisputeUnlockTime": payment_request["data"]["externalDisputeUnlockTime"],
            "agentIdentifier": agent_identifier,
            "sellerVKey": os.getenv("SELLER_VKEY"),
            "identifierFromPurchaser": data.identifier_from_purchaser,
            "amounts": amounts,
            "input_hash": payment.input_hash,
            "payByTime": payment_request["data"]["payByTime"],
        }
    except KeyError as e:
        logger.error(f"Missing required field in request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Missing field: {str(e)}")
    except Exception as e:
        logger.error(f"Error in start_job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 2) Process Payment and Execute AI Task
# ─────────────────────────────────────────────────────────────────────────────
async def handle_payment_status(job_id: str, payment_id: str) -> None:
    """ Executes Task after payment confirmation """
    try:
        logger.info(f"Payment {payment_id} completed for job {job_id}, executing task...")
        
        # Update job status to running
        jobs[job_id]["status"] = "running"
        logger.info(f"Input data: {jobs[job_id]['input_data']}")

        # Execute the AI task
        result = await execute_crew_task(jobs[job_id]["input_data"])
        print(f"Result: {result}")
        logger.info(f"Task completed for job {job_id}")
        
        # Convert result to string for payment completion
        result_string = result.raw if hasattr(result, "raw") else str(result)
        
        # Mark payment as completed on Masumi
        await payment_instances[job_id].complete_payment(payment_id, result_string)
        logger.info(f"Payment completed for job {job_id}")

        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["payment_status"] = "completed"
        jobs[job_id]["result"] = result

        # Stop monitoring payment status
        if job_id in payment_instances:
            payment_instances[job_id].stop_status_monitoring()
            del payment_instances[job_id]
    except Exception as e:
        print(f"Error processing payment {payment_id} for job {job_id}: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        
        if job_id in payment_instances:
            payment_instances[job_id].stop_status_monitoring()
            del payment_instances[job_id]

# ─────────────────────────────────────────────────────────────────────────────
# 3) Check Job and Payment Status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/status")
async def get_status(job_id: str):
    logger.info(f"Checking status for job {job_id}")
    if job_id not in jobs:
        logger.warning(f"Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Check latest payment status if payment instance exists
    if job_id in payment_instances:
        try:
            status = await payment_instances[job_id].check_payment_status()
            job["payment_status"] = status.get("data", {}).get("status")
            logger.info(f"Updated payment status for job {job_id}: {job['payment_status']}")
        except ValueError as e:
            logger.warning(f"Error checking payment status: {str(e)}")
            job["payment_status"] = "unknown"
        except Exception as e:
            logger.error(f"Error checking payment status: {str(e)}", exc_info=True)
            job["payment_status"] = "error"

    result_data = job.get("result")
    result = result_data.raw if result_data and hasattr(result_data, "raw") else result_data

    return {
        "job_id": job_id,
        "status": job["status"],
        "payment_status": job["payment_status"],
        "result": result
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4) Check Server Availability
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/availability")
async def check_availability():
    return {"status": "available", "type": "masumi-agent", "message": "Server operational."}

# ─────────────────────────────────────────────────────────────────────────────
# 5) Retrieve Input Schema
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/input_schema")
async def input_schema():
    return {
    "input_data": [
        {
            "id": "ad_category",
            "type": "string",
            "name": "Ad Category",
            "data": {"description": "e.g., Technology, Gaming", "placeholder": "Technology"}
        },
        {
            "id": "ad_content_url",
            "type": "string",
            "name": "Ad Image/Video Link",
            "data": {"description": "Link to IPFS or hosted ad", "placeholder": "https://..."}
        }
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# 6) Health Check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy"}

# ─────────────────────────────────────────────────────────────────────────────
# Main Logic if Called as a Script
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Default to API mode if no args, or if arg is 'api'
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] == "api"):
        port = int(os.environ.get("API_PORT", 8000))
        host = os.environ.get("API_HOST", "127.0.0.1")

        print("\n" + "=" * 70)
        print("🚀 Starting FastAPI server with Masumi integration...")
        print("=" * 70)
        print(f"API Documentation:        http://{host}:{port}/docs")
        uvicorn.run(app, host=host, port=port, log_level="info")
    else:
        # NOTE: This standalone mode still uses the OLD logic from your original code.
        # It requires 'ResearchCrew' to be working.
        
        def main():
            import os
            os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'
            print("🚀 Running CrewAI agents locally...")
            try:
                from crew_definition import ResearchCrew
                input_data = {"text": "The impact of AI on the job market"}
                crew = ResearchCrew(verbose=True)
                result = crew.crew.kickoff(inputs=input_data)
                print(result)
            except ImportError:
                print("Error: crew_definition.py not found. Cannot run standalone mode.")

        main()