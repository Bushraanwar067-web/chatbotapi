import os
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from datetime import datetime
import uuid

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

if not GROQ_API_KEY:
    raise ValueError("API key for Groq is missing. Please set the GROQ_API_KEY in the .env file.") 

app = FastAPI(title="Groq Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatbotfrontend-liart.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Client
client = MongoClient(MONGODB_URI)
db = client["groq_chatbot"]
conversations_collection = db["conversations"]

# Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

class UserInput(BaseModel):
    message: str
    role: str = "user"
    conversation_id: str
    get_or_create_conversation: str

class Conversation:
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a useful AI assistant."}
        ]
        self.active: bool = True
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()

def get_or_create_conversation(conversation_id: str) -> Conversation:
    # Try to load from MongoDB
    conversation_data = conversations_collection.find_one({"conversation_id": conversation_id})
    
    if conversation_data:
        conversation = Conversation(conversation_id)
        conversation.messages = conversation_data["messages"]
        conversation.active = conversation_data["active"]
        conversation.created_at = conversation_data["created_at"]
        return conversation
    else:
        # Create new conversation
        conversation = Conversation(conversation_id)
        # Save to MongoDB
        conversations_collection.insert_one({
            "conversation_id": conversation.conversation_id,
            "messages": conversation.messages,
            "active": conversation.active,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at
        })
        return conversation

def save_conversation(conversation: Conversation):
    conversations_collection.update_one(
        {"conversation_id": conversation.conversation_id},
        {
            "$set": {
                "messages": conversation.messages,
                "active": conversation.active,
                "updated_at": datetime.utcnow()
            }
        }
    )

def query_groq_api(conversation: Conversation) -> str:
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation.messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
                
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with Groq API: {str(e)}")

@app.post("/chat/")
async def chat(input: UserInput):
    # Retrieve or create the conversation
    conversation = get_or_create_conversation(input.conversation_id)

    if not conversation.active:
        raise HTTPException(
            status_code=400,
            detail="The chat session has ended. Please start a new session."
        )
    
    try:
        # Append the user's message to the conversation
        conversation.messages.append({
            "role": input.role,
            "content": input.message
        })
        
        response = query_groq_api(conversation)
        
        conversation.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Save updated conversation to MongoDB
        save_conversation(conversation)
        
        return {
            "response": response,
            "conversation_id": input.conversation_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation_data = conversations_collection.find_one({"conversation_id": conversation_id})
    if not conversation_data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_data["conversation_id"],
        "messages": conversation_data["messages"],
        "created_at": conversation_data["created_at"],
        "updated_at": conversation_data["updated_at"]
    }

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    result = conversations_collection.delete_one({"conversation_id": conversation_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Conversation deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



    

# run  uvicorn main:app --reload  
