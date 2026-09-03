import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Load environment variables from .env
load_dotenv()

# Get MongoDB URI from .env
mongodb_uri = os.getenv("MONGODB_URI")

print("=" * 45)
print("MediCare AI — MongoDB Connection Test")
print("=" * 45)

# Check if URI is set
if not mongodb_uri or mongodb_uri == "your_mongodb_uri_here":
    print("ERROR: MongoDB URI not set in .env file")
    exit()

try:
    # Create MongoDB client
    client = MongoClient(mongodb_uri)

    # Test the connection
    client.admin.command("ping")
    print("Connection     : SUCCESS")

    # Access our database
    db = client["medicare_rag"]
    print(f"Database       : {db.name}")

    # Access our collection
    collection = db["medical_chunks"]
    print(f"Collection     : {collection.name}")

    # Count documents (should be 0 right now)
    count = collection.count_documents({})
    print(f"Documents      : {count} (empty, expected)")

    print("=" * 45)
    print("MongoDB is ready for MediCare AI")
    print("=" * 45)

except ConnectionFailure as e:
    print(f"Connection FAILED: {e}")
except Exception as e:
    print(f"Error: {e}")