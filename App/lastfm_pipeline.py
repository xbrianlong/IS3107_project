import datetime
import pandas as pd
import requests
import os
import json

def get_lastfm_user_tracks(username, api_key, limit=50):
    url = f"http://ws.audioscrobbler.com/2.0/?method=user.getrecenttracks&user={username}&api_key={api_key}&format=json&limit={limit}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        data = response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"Error fetching tracks for {username}: {e}")
        return []

    tracks = []
    for track in data.get("recenttracks", {}).get("track", []):
        song = track.get("name", "Unknown")
        artist = track.get("artist", {}).get("#text", "Unknown")
        album = track.get("album", {}).get("#text", "Unknown")
        listen_date = track.get("date", {}).get("#text", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        genre = get_lastfm_track_genre(artist, song, api_key)
        duration = get_lastfm_track_duration(artist, song, api_key)
        tracks.append((username, song, artist, album, "played", listen_date, genre, duration))

    return tracks

def get_lastfm_track_genre(artist, song, api_key):
    url = f"http://ws.audioscrobbler.com/2.0/?method=track.gettoptags&artist={artist}&track={song}&api_key={api_key}&format=json"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, json.JSONDecodeError):
        return "Unknown"

    tags = data.get("toptags", {}).get("tag", [])
    return tags[0]["name"] if tags else "Unknown"

def get_lastfm_track_duration(artist, song, api_key):
    url = f"http://ws.audioscrobbler.com/2.0/?method=track.getinfo&artist={artist}&track={song}&api_key={api_key}&format=json"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, json.JSONDecodeError):
        return "Unknown"

    return int(data.get("track", {}).get("duration", 0)) // 1000 if data.get("track", {}).get("duration") else "Unknown"

def generate_lastfm_data(usernames, api_key, limit=50, output_file="lastfm_data.json", refresh=False):
    if os.path.exists(output_file) and not refresh:
        return pd.read_json(output_file)

    interactions = []
    for user in usernames:
        print(f"Fetching data for: {user}")
        user_tracks = get_lastfm_user_tracks(user, api_key, limit)
        interactions.extend(user_tracks)

    df = pd.DataFrame(interactions, columns=["User", "Song", "Artist", "Album", "Interaction", "Listen_Date", "Genre", "Song_Length"])
    df.to_json(output_file, orient="records", date_format="iso")
    return df

# Load API key from environment variables (security improvement)
api_key = os.getenv("LASTFM_API_KEY", "e822c69f73c6929a7df2b53fe25fc629")  # Replace with your actual API key
file_path = os.path.join("src", "data", "raw_data", "lastfm_active_users.csv")

# Read the CSV into a DataFrame
active_users_df = pd.read_csv(file_path)

# Optional: Preview the data
print(active_users_df.head())
usernames = active_users_df["username"].dropna().unique().tolist()[:300]

df = generate_lastfm_data(usernames, api_key, refresh=True)  # Set `refresh=True` to force an update
# print(df.head())


