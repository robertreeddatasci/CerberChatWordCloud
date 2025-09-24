import os
import requests
import json
from dotenv import load_dotenv

#THIS DOESN'T WORK, USE THE ONLINE TOOL! https://www.twitchchatdownloader.com/video



# --- Load environment variables ---
load_dotenv()
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Missing TWITCH_CLIENT_ID or TWITCH_CLIENT_SECRET in .env file.")

# --- Step 1: Get OAuth token ---
def get_oauth_token():
    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    resp = requests.post(url, params=params)
    resp.raise_for_status()
    return resp.json()["access_token"]

# --- Step 2: Fetch chat messages for a VOD ---
def fetch_chat(video_id, oauth_token):
    url = f"https://api.twitch.tv/v5/videos/{video_id}/comments"
    headers = {
        "Client-ID": CLIENT_ID,
        "Accept": "application/vnd.twitchtv.v5+json",
        "Authorization": f"Bearer {oauth_token}"
    }

    messages = []
    cursor = None

    while True:
        params = {"cursor": cursor} if cursor else {}
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        for comment in data.get("comments", []):
            messages.append({
                "timestamp": comment["created_at"],
                "user": comment["commenter"]["display_name"],
                "message": comment["message"]["body"]
            })

        # Pagination: stop if no more comments
        cursor = data.get("_next")
        if not cursor:
            break

    return messages

# --- Step 3: Run the script ---
if __name__ == "__main__":
    # Replace with the VOD ID you want to fetch
    VOD_ID = "2573020908"

    print(f"Fetching chat for VOD {VOD_ID}...")
    token = get_oauth_token()
    chat_messages = fetch_chat(VOD_ID, token)

    print(f"Fetched {len(chat_messages)} messages.")

    # Save to JSON
    output_file = f"chat_{VOD_ID}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chat_messages, f, ensure_ascii=False, indent=2)

    print(f"Saved chat to {output_file}")
