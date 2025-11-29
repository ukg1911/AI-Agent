import os
import uvicorn
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from masumi.config import Config
from masumi.payment import Payment, Amount
from logging_config import setup_logging

# Configure logging
logger = setup_logging()

# Load environment variables
load_dotenv(override=True)

# Retrieve API Keys and URLs
PAYMENT_SERVICE_URL = os.getenv("PAYMENT_SERVICE_URL")
PAYMENT_API_KEY = os.getenv("PAYMENT_API_KEY")
NETWORK = os.getenv("NETWORK")
AGENT_IDENTIFIER = os.getenv("AGENT_IDENTIFIER")

logger.info("Starting application with configuration:")
logger.info(f"PAYMENT_SERVICE_URL: {PAYMENT_SERVICE_URL}")
logger.info(f"NETWORK: {NETWORK}")

# Initialize FastAPI
app = FastAPI(
    title="Aura Network User Agent",
    description="Agent that filters ads based on user preferences and bid amount.",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory storage
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
# Pydantic Models (Corrected for Ad Offers)
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
                "identifier_from_purchaser": "1234567890abcdef12345678",
                "input_data": {
                    "ad_category": "Technology",
                    "bid_amount": "5.0",
                    "ad_content_url": "https://ipfs.io/ipfs/example"
                }
            }
        }

# ─────────────────────────────────────────────────────────────────────────────
# Logic: The "Gatekeeper" Function
# ─────────────────────────────────────────────────────────────────────────────
# Mock user preferences (In a real app, load this from a DB)
USER_PREFERENCES = {
    "interested_categories": ["Technology", "Gaming", "DeFi"],
    "min_bid": 0.5 
}

async def execute_crew_task(input_data: dict) -> str:
    """ 
    Acts as the 'Gatekeeper'. 
    Checks if the Ad Offer matches the User's Preferences. 
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
        
        # Log the input
        input_desc = f"Ad Offer: {data.input_data.ad_category}"
        logger.info(f"Received job request: '{input_desc}'")
        logger.info(f"Starting job {job_id} with agent {AGENT_IDENTIFIER}")

        # Convert Pydantic model to dict for storage and Masumi SDK
        input_data_dict = data.input_data.model_dump()

        # Define payment amounts
        payment_amount = os.getenv("PAYMENT_AMOUNT", "5000000")  # Default 5 ADA
        payment_unit = os.getenv("PAYMENT_UNIT", "lovelace")

        # Create Amount Object
        amounts = [Amount(amount=payment_amount, unit=payment_unit)]
        logger.info(f"Using payment amount: {payment_amount} {payment_unit}")
        
        # Create a payment request using Masumi
        # --- THIS IS THE CRITICAL FIX: amounts=amounts is included ---
        payment = Payment(
            agent_identifier=AGENT_IDENTIFIER,
            amounts=amounts,
            config=config,
            identifier_from_purchaser=data.identifier_from_purchaser,
            input_data=input_data_dict, 
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
            "input_data": input_data_dict,
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
            "amounts": amounts, # Included for verification
            "payByTime": payment_request["data"]["payByTime"],
        }
    except Exception as e:
        logger.error(f"Error in start_job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Payment request failed: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Process Payment and Execute Logic
# ─────────────────────────────────────────────────────────────────────────────
async def handle_payment_status(job_id: str, payment_id: str) -> None:
    """ Executes Task after payment confirmation """
    try:
        logger.info(f"Payment {payment_id} completed for job {job_id}, executing task...")
        
        # Update job status to running
        jobs[job_id]["status"] = "running"
        logger.info(f"Input data: {jobs[job_id]['input_data']}")

        # Execute the AI task (Gatekeeper Logic)
        result = await execute_crew_task(jobs[job_id]["input_data"])
        print(f"Result: {result}")
        logger.info(f"Task completed for job {job_id}")
        
        # Convert result to string
        result_string = str(result)
        
        # Mark payment as completed on Masumi
        await payment_instances[job_id].complete_payment(payment_id, result_string)
        logger.info(f"Payment completed for job {job_id}")

        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["payment_status"] = "completed"
        jobs[job_id]["result"] = result

        # Stop monitoring
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
# 3) Check Status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/status")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Refresh payment status
    if job_id in payment_instances:
        try:
            status = await payment_instances[job_id].check_payment_status()
            job["payment_status"] = status.get("data", {}).get("status")
        except:
            pass

    return {
        "job_id": job_id,
        "status": job["status"],
        "payment_status": job["payment_status"],
        "result": job.get("result")
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4) Availability & Schema
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/availability")
async def check_availability():
    return {"status": "available", "type": "masumi-agent", "message": "Server operational."}

@app.get("/input_schema")
async def input_schema():
    return {
        "input_data": [
            {
                "id": "ad_category",
                "type": "string",
                "name": "Ad Category",
                "data": {"description": "e.g., Technology", "placeholder": "Technology"}
            },
            {
                "id": "ad_content_url",
                "type": "string",
                "name": "Ad Content URL",
                "data": {"description": "IPFS Link", "placeholder": "https://..."}
            }
        ]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ─────────────────────────────────────────────────────────────────────────────
# Main Runner (PORT 8001)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔥🔥🔥 I AM THE NEW CODE ON PORT 8001 🔥🔥🔥")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")