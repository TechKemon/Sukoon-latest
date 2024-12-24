from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import TypedDict, Annotated
import uvicorn
import httpx
from typing import Literal
import os

# from fastapi import Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# import json
# from datetime import datetime

# from sukoon import chat
from myca import chat

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MYCA", description="API for the MYCA mental health support system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class MYCARequest(BaseModel):
    input: str

class MYCAResponse(BaseModel):
    output: str

class FeedbackRequest(BaseModel):
    feedback: Literal["like", "dislike"]
    message: str
    message_id: str

@app.post("/query", response_model=MYCAResponse)
async def process_query(request: MYCARequest):
    try:
        config = {"configurable": {"thread_id": "1", "user_id": "1"}}
        user_input = request.input
        response = chat(user_input, config)
        return MYCAResponse(output=response.content)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    
@app.get("/query", response_model=MYCAResponse)
async def process_query_get(input: str = Query(..., description="Please tell what brings you here?")):
    try:
        config = {"configurable": {"thread_id": "1", "user_id": "1"}}
        response = chat(input, config)
        return MYCAResponse(output=response.content)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

@app.get("/")
async def root():
    return {"message": "Welcome to the MYCA API. Use the /docs or /query endpoint to interact with the system."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# @app.post("/docs")
# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port=8000)

# @app.post("/feedback", response_model=MYCAResponse)
# async def submit_feedback(request: FeedbackRequest):
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.post(
#                 "https://supabase.pplus.ai/rest/v1/Feedback",
#                 headers={
#                     'Content-Type': 'application/json',
#                     'apikey': os.getenv('SUPABASE_API_KEY'),
#                     'Authorization': f"Bearer {os.getenv('SUPABASE_AUTHORIZATION_TOKEN')}",
#                     'Prefer': 'return=minimal'
#                 },
#                 json={
#                     'action': request.feedback,
#                     'feedback': request.message,
#                     'message_id': request.message_id
#                 },
#                 timeout=10.0
#             )
            
#             if response.status_code != 201:
#                 raise HTTPException(
#                     status_code=response.status_code,
#                     detail="Failed to submit feedback to Supabase"
#                 )
                
#             return MYCAResponse(output="Feedback submitted successfully")
            
#     except httpx.TimeoutException:
#         raise HTTPException(status_code=504, detail="Request to Supabase timed out")
#     except Exception as error:
#         raise HTTPException(status_code=500, detail=str(error))
        
    
    # try {
    #     const response = await fetch("https://supabase.pplus.ai/rest/v1/Feedback", {
    #     method: 'POST',
    #     headers: {
    #         'Content-Type': 'application/json',
    #         'apikey': SUPABASE_API_KEY,
    #         'Authorization': SUPABASE_AUTHORIZATION_TOKEN,
    #         'Prefer': 'return=minimal'
    #     },
    #     body: JSON.stringify({
    #         'action': feedback.feedback,
    #         'feedback': feedbackMessage,
    #     }),
    #     });

    #     if (response.status !== 201) {
    #     throw new Error(`HTTP error! Status: ${response.status}`);
    #     }

    #     setFeedback("");
    #     setFeedbackMessage("");

    #     if (feedback.feedback === "like") {
    #     // remove the message from the list of dislikedMessages, if it was added earlier (only for local state)
    #     if (dislikedMessages.includes(feedback.messageId) === true) {
    #         setDislikedMessages(dislikedMessages => dislikedMessages.filter(item => item !== feedback.messageId));
    #     }
    #     setLikedMessages(likedMessages => [...likedMessages, feedback.messageId]);
    #     } else {
    #     // remove the message from the list of setLikedMessages, if it was added earlier (only for local state)
    #     if (likedMessages.includes(feedback.messageId) === true) {
    #         setLikedMessages(likedMessages => likedMessages.filter(item => item !== feedback.messageId));
    #     }
    #     setDislikedMessages(dislikedMessages => [...dislikedMessages, feedback.messageId]);
    #     }
    # } catch (error) {
    #     console.error('Error:', error);
    # }
    # }
    


# for google analytics
# templates = Jinja2Templates(directory="templates")

# # Track conversation events
# async def track_conversation_event(conversation_id: str, event_type: str, data: dict):
#     # Send to Google Analytics
#     event = {
#         'conversation_id': conversation_id,
#         'event_type': event_type,
#         'timestamp': datetime.utcnow().isoformat(),
#         'data': data
#     }
#     # Log for analysis
#     print(json.dumps(event))

# @app.post("/chat")
# async def chat_endpoint(request: Request):
#     # Your existing chat logic
#     conversation_id = "unique_id"  # Generate unique ID
    
#     # Track conversation start
#     await track_conversation_event(
#         conversation_id=conversation_id,
#         event_type="conversation_start",
#         data={
#             'user_id': request.client.host,
#             'timestamp': datetime.utcnow().isoformat()
#         }
#     )