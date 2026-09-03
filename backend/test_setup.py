import os
from dotenv import load_dotenv

# This reads your .env file
# and loads the variables into Python
load_dotenv()

print("=" * 40)
print("MediCare AI — Setup Verification")
print("=" * 40)

# Check Google API Key
google_key = os.getenv("GOOGLE_API_KEY")
if google_key and google_key != "your_google_api_key_here":
    print(f"Google API Key : FOUND ({google_key[:8]}...)")
else:
    print("Google API Key : NOT SET")

# Check MongoDB URI
mongo_uri = os.getenv("MONGODB_URI")
if mongo_uri and mongo_uri != "your_mongodb_uri_here":
    print(f"MongoDB URI    : FOUND ({mongo_uri[:20]}...)")
else:
    print("MongoDB URI    : NOT SET")

print("=" * 40)
print("Setup check complete.")
print("=" * 40)