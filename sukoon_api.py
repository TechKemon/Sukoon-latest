from fastapi import FastAPI, HTTPException, Query, Depends
# Additional imports for handling CORS (Cross-Origin Resource Sharing)
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List

import logging
# import httpx
from typing import Literal
import os

# from fastapi import Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# import json
# from datetime import datetime

# from sukoon import chat
# from myca_supa import chat # Assumes chat is a synchronous function; convert to async if needed.
from simple import chat
from utils.supabase_manager import SupabaseManager

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("myca_api.log"),  # Log messages are written to this file
        logging.StreamHandler()  # Also write to the console
        ]
    )
logger = logging.getLogger(__name__)  # New change: Using a logger instead of print

# Create a global instance once and reuse it
# supabase_manager = SupabaseManager()
# Instantiate the Supabase manager once and use Dependency Injection for scalability
def get_supabase_manager() -> SupabaseManager:
    return SupabaseManager()

app = FastAPI(
    title="MYCA", 
    description="API for the MYCA mental health support system",
    version="1.0"
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ["https://yourdomain.com", "https://anotherdomain.com"],  # Only allow trusted domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],  #  => Only allow specific HTTP methods ELSE # Allows all headers using ["*"]
    allow_headers=["Authorization", "Content-Type"],  # Only allow necessary headers ELSE # Allows all headers using ["*"]
)

class MYCARequest(BaseModel):
    mobile: str = Field(
        ...,
        pattern=r"^\d{10}$", # ensuring mobile is a 10-digit number
        description="a 10-digit mobile number"
    )
    input: str = Field(..., description="User Chat Message")

class MYCAResponse(BaseModel):
    output: str = Field(..., description="Chatbot response")

# Request model for logging chat conversation
class ChatRequest(BaseModel):
    mobile: str = Field(
        ...,
        pattern=r"^\d{10}$", # ensuring mobile is a 10-digit number
        description="a 10-digit mobile number"
    ) # mobile: Annotated[str, Query(..., min_length=10, max_length=10)]
    user: str = Field(..., description="User input")
    response: str = Field(..., description="Chatbot response")

class ChatResponse(BaseModel):
    messages: List[dict] = Field([], description="List of past chat messaages")

# API ENDPOINT USED IN CHATBOT
@app.post("/query", response_model=MYCAResponse)
async def process_query(request: MYCARequest, supabase: SupabaseManager = Depends(get_supabase_manager)):
    try:
        """
        Using Dependency Injection to pass in our SupabaseManager.
        """
        # New change: Externalize config if necessary (e.g., read from environment variables)
        # config = {"configurable": {"thread_id": "1", "user_id": "1"}} # CHECK THESE VALUES
        user_input = request.input
        mobile= request.mobile
        
        # Process chat. If chat() is blocking, consider running it in a threadpool using run_in_executor.
        history = supabase.get_chat_history(mobile=mobile)
        logger.info("Retrieved chat history for mobile %s: %s", mobile, history)
        response = chat(user_input, history)
        # chat_response = response.content
        
        # Log chat asynchronously if the underlying supabase.log_chat supports it,
        # otherwise, consider running it in the background.
        try:
            supabase.log_chat(
                mobile=mobile,
                user=user_input,
                response=response
            )
            logger.info("Chat logged successfully for mobile: %s", mobile)
        except Exception as log_error:
            logger.error("Failed to log chat for mobile %s: %s", mobile, log_error)
        
        return MYCAResponse(output=response)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

@app.get("/fetch_convo", response_model=ChatResponse)
async def fetch_history(
    mobile: str = Query(..., pattern=r"^\d{10}$", description="10 digit mobile number"),
    supabase: SupabaseManager = Depends(get_supabase_manager)
):
    """
    Endpoint to fetch chat conversation history for a given mobile.
    New change: Validate mobile number through Query parameters.
    """
    try:
        history = supabase.get_chat_history(mobile=mobile)
        return ChatResponse(messages=history or [])
    except Exception as error:
        logger.exception("Error fetching conversation history for mobile: %s", mobile)
        raise HTTPException(status_code=500, detail="Failed to fetch conversation history")

@app.post("/log_convo")
async def save_history(request: ChatRequest, supabase: SupabaseManager = Depends(get_supabase_manager)):
    try:
        status = supabase.log_chat(
            mobile=request.mobile,
            user=request.user,
            response=request.response
        )
        if status:
            return {"success": True}
        else:
            logger.warning("Chat log returned a false status for mobile: %s", request.mobile)
            return {"success": False}
    except Exception as error:
        logger.exception("Error saving conversation history for mobile: %s", request.mobile)
        raise HTTPException(status_code=500, detail="Failed to log conversation history")


@app.get("/")
async def root():
    """
    Root endpoint to welcome users.
    """
    return {"message": "Welcome to the MYCA API. Use the /docs or /query endpoint to interact with the system."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Only run if this module is executed as the main script
if __name__ == "__main__":
    # New change: Read host/port from environment variables if deployed in different environments.
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
    
# @app.post("/docs")
# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")
 
# class FeedbackRequest(BaseModel):
#     feedback: str = Field(..., description='Feedback must be "like" or "dislike"') # Literal["like", "dislike"] # 
#     message: str
#     message_id: str

# # New change: Use a validator to ensure only allowed values for feedback
# @classmethod
# def __get_validators__(cls):
#     yield cls.validate_feedback

# @classmethod
# def validate_feedback(cls, value):
#     allowed = {"like", "dislike"}
#     if value not in allowed:
#         raise ValueError("Feedback must be either 'like' or 'dislike'")
#     return value

# with langgraph code
# @app.post("/query", response_model=MYCAResponse)
# async def process_query(request: MYCARequest, supabase: SupabaseManager = Depends(get_supabase_manager)):
#     try:
#         """
#         Using Dependency Injection to pass in our SupabaseManager.
#         """
#         # New change: Externalize config if necessary (e.g., read from environment variables)
#         config = {"configurable": {"thread_id": "1", "user_id": "1"}} # CHECK THESE VALUES
#         user_input = request.input
#         mobile= request.mobile
        
#         # Process chat. If chat() is blocking, consider running it in a threadpool using run_in_executor.
#         history = supabase.get_chat_history(mobile=mobile)
#         logger.info("Retrieved chat history for mobile %s: %s", mobile, history)
#         response = chat(user_input, config, history)
#         chat_response = response.content
        
#         # Log chat asynchronously if the underlying supabase.log_chat supports it,
#         # otherwise, consider running it in the background.
#         try:
#             supabase.log_chat(
#                 mobile=mobile,
#                 user=user_input,
#                 response=chat_response
#             )
#             logger.info("Chat logged successfully for mobile: %s", mobile)
#         except Exception as log_error:
#             logger.error("Failed to log chat for mobile %s: %s", mobile, log_error)
        
#         return MYCAResponse(output=chat_response)
#     except Exception as error:
#         raise HTTPException(status_code=500, detail=str(error))