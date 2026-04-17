import os
from supabase import create_client, Client
from dotenv import load_dotenv
from database import supabase

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://lqvrzqryfjtsytcahnpb.supabase.co")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxxdnJ6cXJ5Zmp0c3l0Y2FobnBiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ4OTIwNjUsImV4cCI6MjA5MDQ2ODA2NX0.XLpqKFOLtEXHTsylOn936WBCplB4pGv8gJsnldKZ5Xw")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def get_property_data(property_id: str):
    response = supabase.table("properties") \
        .select("*") \
        .eq("id", property_id) \
        .single() \
        .execute()

    if response.data:
        return response.data
    else:
        raise ValueError(f"Property {property_id} not found")


def save_prediction(property_id: str, predicted_price: float):
    response = supabase.table("predictions").insert({
        "property_id": property_id,
        "predicted_price": predicted_price,
        "created_at": "now()"
    }).execute()

    return response.data