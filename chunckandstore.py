import requests
import json
import os
import chromadb
from dotenv import load_dotenv
from transformers import GPT2TokenizerFast
import tiktoken
import time

# Load environment variables
load_dotenv()

# Load tokenizer and disable length warnings
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.model_max_length = int(1e30)  # Disable sequence length warnings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

OPENAI_EMBEDDING_URL = "https://aipipe.org/openai/v1/embeddings"
course_content = "scraped_data/course_content.json"
topics = "scraped_data/topics.json"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "embeddings_data"
SAFE_MODE = True  # Set to False to disable safety checks

# Chunking configuration
MAX_TOKENS = 1024
OVERLAP = 100
EMBEDDING_MODEL_TOKEN_LIMIT = 8191  # for text-embedding-3-small

def chunk_text(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    """Chunk text with token overlap and return chunks"""
    tokens = tokenizer.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + max_tokens]
        decoded = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(decoded)
        i += max_tokens - overlap
    return chunks

def clean_metadata(metadata):
    """Ensure all metadata values are non-null and of accepted types"""
    cleaned = {}
    for key, value in metadata.items():
        # Handle null values first
        if value is None:
            if key in ["postnumber", "reply_to_post_number", "topic_id"]:
                cleaned[key] = 0  # Default integer for numeric fields
            else:
                cleaned[key] = ""  # Default string for other fields
        
        # Handle existing values
        elif key in ["postnumber", "reply_to_post_number", "topic_id"]:
            # Ensure numeric fields are integers
            try:
                cleaned[key] = int(value)
            except (ValueError, TypeError):
                cleaned[key] = 0
        
        # Handle string fields
        elif key in ["username", "post_url", "date", "type"]:
            cleaned[key] = str(value)
        
        # Handle all other fields
        else:
            cleaned[key] = str(value) if value is not None else ""
    
    return cleaned

# Prepare data
print("Loading data files...")
with open(topics, "r", encoding="utf-8") as file:
    topics_data = json.load(file)
    print(f"Loaded {len(topics_data)} topics")

with open(course_content, "r", encoding="utf-8") as file:
    course_content_data = json.load(file)
    print("Loaded course content")

post_texts = []
post_metadata = []
post_ids = []

# Process forum topics
print("Processing forum topics...")
for index, topic in enumerate(topics_data):
    content = topic.get("topic_content", "").strip()
    if content:
        chunks = chunk_text(content)
        for j, chunk in enumerate(chunks):
            # Create metadata with explicit type handling
            meta = {
                "topic_id": topic.get("topic_id"),
                "username": topic.get("username"),
                "post_url": topic.get("post_url"),
                "postnumber": topic.get("postnumber"),
                "reply_to_post_number": topic.get("reply_to_post_number"),
                "date": topic.get("date")
            }
            cleaned_meta = clean_metadata(meta)
            
            post_texts.append(chunk)
            post_metadata.append(cleaned_meta)
            post_ids.append(f"post_{topic['topic_id']}_{index}_{j}")

# Process course content
print("Processing course content...")
course_text = course_content_data.get("course_content", "").strip()
if course_text:
    chunks = chunk_text(course_text)
    for j, chunk in enumerate(chunks):
        post_texts.append(chunk)
        post_metadata.append(clean_metadata({"type": "course_content"}))
        post_ids.append(f"course_001_{j}")

print(f"Total chunks generated: {len(post_texts)}")

# ===== SAFETY CHECKS =====
if SAFE_MODE:
    print("\n" + "="*50)
    print("SAFE MODE ENABLED - VERIFYING BEFORE PROCEEDING")
    print("="*50)
    
    # 1. Show sample chunks and metadata
    print("\nSample chunks and metadata (first 3):")
    for i in range(3):
        print(f"Chunk {i+1}: {post_texts[i][:100]}...")
        print(f"Metadata: {post_metadata[i]}")
    
    # 2. Check for null values in metadata
    print("\nChecking for null values in metadata...")
    null_found = False
    for i, meta in enumerate(post_metadata):
        for key, value in meta.items():
            if value is None:
                print(f"⚠️ Null found in metadata at index {i}, key '{key}'")
                null_found = True
    if not null_found:
        print("✅ No null values found in metadata")
    
    # 3. Estimate token usage and cost
    embedding_encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(embedding_encoding.encode(text)) for text in post_texts)
    estimated_cost = total_tokens * (0.00002 / 1000)  # $0.00002 per 1k tokens
    
    print(f"\nEstimated token usage: {total_tokens:,}")
    print(f"Estimated cost: ${estimated_cost:.4f}")
    
    # 4. Verify with user
    proceed = input("\nDo you want to proceed with embeddings? (y/n): ").lower()
    if proceed != 'y':
        print("Aborted by user")
        exit()

# Embedding function with rate limiting
def get_embeddings(texts):
    """Get embeddings with rate limiting and error handling"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }

    embeddings = []
    batch_size = 10  # Reduced batch size for safety
    max_chunk_tokens = EMBEDDING_MODEL_TOKEN_LIMIT
    embedding_encoding = tiktoken.get_encoding("cl100k_base")
    request_count = 0

    print(f"\nGenerating embeddings in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        request_count += 1
        
        # Rate limiting: 3 requests per second
        if request_count % 3 == 0:
            time.sleep(1)
        
        filtered_batch = []
        for text in batch:
            token_len = len(embedding_encoding.encode(text))
            if token_len <= max_chunk_tokens:
                filtered_batch.append(text)
            else:
                print(f"⚠️ Skipping chunk with {token_len} tokens (too long): {text[:100]}...")

        if not filtered_batch:
            continue

        payload = {
            "model": "text-embedding-3-small",
            "input": filtered_batch
        }

        try:
            response = requests.post(OPENAI_EMBEDDING_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json().get("data", [])
            batch_embeddings = [item["embedding"] for item in data]
            embeddings.extend(batch_embeddings)
            print(f"Processed batch {request_count}: {i+batch_size}/{len(texts)} chunks")
        except requests.exceptions.RequestException as e:
            print(f"🚨 Error processing batch {request_count}: {str(e)}")
            print("Skipping batch and continuing...")
        except json.JSONDecodeError:
            print(f"🚨 Invalid JSON response for batch {request_count}")
            print("Skipping batch and continuing...")

    return embeddings

# Get embeddings
print(f"\nStarting embedding generation for {len(post_texts)} chunks...")
embeddings = get_embeddings(post_texts)
print(f"Successfully generated {len(embeddings)} embeddings")

# Store in ChromaDB
print(f"\nStoring in ChromaDB at {CHROMA_DIR}...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Verify embeddings match texts
if len(embeddings) != len(post_texts):
    print(f"⚠️ Warning: Embeddings count ({len(embeddings)}) doesn't match text count ({len(post_texts)})")
    min_length = min(len(embeddings), len(post_texts))
    embeddings = embeddings[:min_length]
    post_texts = post_texts[:min_length]
    post_metadata = post_metadata[:min_length]
    post_ids = post_ids[:min_length]

# Add a final metadata cleaning step
clean_metadatas = [clean_metadata(meta) for meta in post_metadata]

# Double-check for nulls before storing
for i, meta in enumerate(clean_metadatas):
    for key, value in meta.items():
        if value is None:
            print(f"🚨 CRITICAL: Null found in cleaned metadata at index {i}, key '{key}'")
            # Convert to safe default
            if key in ["postnumber", "reply_to_post_number", "topic_id"]:
                clean_metadatas[i][key] = 0
            else:
                clean_metadatas[i][key] = ""

collection.add(
    documents=post_texts,
    embeddings=embeddings,
    metadatas=clean_metadatas,
    ids=post_ids
)

print(f"Successfully stored {len(embeddings)} vectors in collection '{COLLECTION_NAME}'")
print("Persistence complete!")

# Final verification
print("\nVerifying storage...")
count = collection.count()
print(f"Collection contains {count} items")
if count == len(embeddings):
    print("✅ All embeddings stored successfully")
else:
    print(f"⚠️ Mismatch: Expected {len(embeddings)} items, found {count}")