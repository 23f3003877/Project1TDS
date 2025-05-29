import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import json
import os
import chromadb
from chromadb.config import Settings
import base64

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_URL = "https://aipipe.org/openrouter/v1/chat/completions"
OPENAI_EMBEDDING_URL = "https://aipipe.org/openai/v1/embeddings"

app = FastAPI()
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "embeddings_data"
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {OPENAI_API_KEY}'
}

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_context(results):
    """Format ChromaDB results into readable context for LLM"""
    context_parts = []
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        # Calculate relevance score (higher is better)
        relevance = 1 - dist
        source = meta.get('username') or meta.get('type', 'Unknown')
        post_url = meta.get('post_url', '')
        
        context_parts.append(f"### CONTEXT {i+1} (Relevance: {relevance:.2f}, Source: {source}) ###")
        context_parts.append(doc)
        if post_url:
            context_parts.append(f"Source URL: {post_url}")
        context_parts.append("")  # Empty line between items
    
    return "\n".join(context_parts)

@app.post("/post")
def post_data(data: dict):
    try:
        question = data["question"]
        imagebase64 = data.get("image", None)
        
        # Generate embedding for the question
        embedding_payload = {
            "model": "text-embedding-3-small",
            "input": [question]  # Note: input must be a list
        }
        embedding_response = requests.post(
            OPENAI_EMBEDDING_URL, 
            headers=headers, 
            json=embedding_payload
        )
        embedding_response.raise_for_status()
        
        # Extract embedding from response
        embedding_data = embedding_response.json().get("data", [])
        if not embedding_data:
            return JSONResponse(
                status_code=500,
                content={"error": "No embedding data returned from API"}
            )
            
        query_embedding = embedding_data[0]["embedding"]
        
        # Query ChromaDB with the embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,  # Get top 5 most relevant results
            include=["documents", "metadatas", "distances"]
        )
        
        # Format context for the LLM
        context = format_context(results)
        
        # System message with instructions
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful teaching assistant for the TDS course. Answer questions based ONLY on the provided context. "
                "If the question isn't related to the course or context doesn't contain the answer, politely decline to answer. "
                "ALWAYS include relevant source links from the context in your response. "
                "Your response MUST be in JSON format with the following structure:\n"
                "{\n"
                "  \"answer\": \"Your detailed answer here\",\n"
                "  \"links\": [\n"
                "    {\"url\": \"https://example.com/1\", \"text\": \"Description 1\"},\n"
                "    {\"url\": \"https://example.com/2\", \"text\": \"Description 2\"}\n"
                "  ]\n"
                "}"
            )
        }
        
        # Build user message with context
        user_content = f"QUESTION: {question}\n\n"
        if imagebase64:
            user_content += "NOTE: An image was provided with this question\n\n"
        user_content += "CONTEXT FROM COURSE MATERIALS:\n" + context
        
        user_message = {"role": "user", "content": user_content}
        
        # Generate response using OpenAI
        chat_payload = {
            "model": "gpt-4.1-nano",
            "messages": [system_message, user_message],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}  # Force JSON output
        }
        
        chat_response = requests.post(
            OPENAI_CHAT_URL,
            headers=headers,
            json=chat_payload
        )
        chat_response.raise_for_status()
        
        # Parse and return the JSON response
        answer_content = chat_response.json()["choices"][0]["message"]["content"]
        answer_data = json.loads(answer_content)
        
        return JSONResponse(status_code=200, content=answer_data)

    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=500, content={"error": f"API request failed: {str(e)}"})
    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"error": "Invalid JSON response from OpenAI API"})
    except KeyError as e:
        return JSONResponse(status_code=500, content={"error": f"Missing expected data: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)