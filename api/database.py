import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError(
        "Missing SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables. "
        "Set them in your .env file or Railway dashboard."
    )

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
