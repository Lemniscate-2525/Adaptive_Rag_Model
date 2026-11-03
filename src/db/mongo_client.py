"""MongoDB client initialization from environment configuration."""

import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGO_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "adaptive_rag")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
