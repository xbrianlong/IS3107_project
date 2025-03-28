import pandas as pd
from bs4 import BeautifulSoup
import re
import requests
import time
from datetime import datetime
import os

from settings import LASTFM_API_KEY, LASTFM_BASE_URL, TOP_GLOBAL_SONGS_URL


# Last.fm API configuration
API_KEY = LASTFM_API_KEY
BASE_URL = LASTFM_BASE_URL
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'raw_data')
INPUT_CSV = os.path.join(DATA_DIR, 'lastfm_active_users.csv')

def get_lastfm_users(filename):
    """Read Last.fm usernames from a CSV file."""
    try:
        df = pd.read_csv(filename)
        return df['username'].tolist()  # Assuming column is named 'username'
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def get_user_top_tracks(username, limit=10):
    """Get a user's top tracks for the week from Last.fm API."""
    params = {
        'method': 'user.gettoptracks',
        'user': username,
        'period': '7day',
        'limit': limit,
        'api_key': API_KEY,
        'format': 'json'
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'toptracks' in data and 'track' in data['toptracks']:
            tracks = []
            for track in data['toptracks']['track']:
                track_info = {
                    'username': username,
                    'track_name': track.get('name', ''),
                    'artist': track.get('artist', {}).get('name', ''),
                    'rank': int(track.get('@attr', {}).get('rank', 0)),
                }
                tracks.append(track_info)
            return tracks
        else:
            print(f"No track data found for user: {username}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"API request failed for user {username}: {e}")
        return []

def get_top_tracks():
    # Read active users from CSV
    users = get_lastfm_users(INPUT_CSV)
    if not users:
        print("No users found. Exiting.")
        return
    
    # Collect top tracks for all users
    all_tracks = []
    count = 0
    for user in users:
        print(count)
        print(f"Fetching top tracks for user: {user}")
        user_tracks = get_user_top_tracks(user)
        all_tracks.extend(user_tracks)
        count += 1
        time.sleep(0.2)  # Respect Last.fm API rate limits
    
    # Create DataFrame
    if all_tracks:
        df = pd.DataFrame(all_tracks)
        
        # Reorder columns
        df = df[['username', 'rank', 'track_name', 'artist']]
        
        # Save to CSV with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_filename = os.path.join(DATA_DIR, f'lastfm_top_tracks_{timestamp}.csv')
        df.to_csv(output_filename, index=False)
        
        print(f"\nSuccessfully extracted data for {len(users)} users.")
        print(f"Total tracks collected: {len(df)}")
        print(f"Data saved to: {output_filename}")
        return output_filename
    else:
        print("No track data was collected.")
        return pd.DataFrame()
    
def get_global_top_songs():
    """Scrape the global top songs from kworb.net"""
    url = TOP_GLOBAL_SONGS_URL
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    top_songs = set()
    table = soup.find('table')
    for row in table.find_all('tr')[1:]:  # Skip header row
        cols = row.find_all('td')
        artist_song = cols[2].text.strip()
        top_songs.add(artist_song.lower())
    return top_songs

def clean_track_name(text):
    """Remove (feat. ...), (with ...), (w/ ...) and similar patterns from track names for consistency"""
    return re.sub(r'\([^)]*\)', '', text).strip()

def filter_out_global_songs(user_data_filename):
    """Remove songs that appear in the global top chart"""
    global_top = get_global_top_songs()
    user_data_df = pd.read_csv(user_data_filename)
    
    # Create a combined artist-track string for comparison
    user_data_df['artist_track'] = (user_data_df['artist'].str.lower() + " - " + 
                                   user_data_df['track_name'].str.lower())
    
    # Clean both user data and global top songs
    user_data_df['artist_track'] = user_data_df['artist_track'].apply(clean_track_name)
    cleaned_global_top = {clean_track_name(song) for song in global_top}
    
    # Filter out songs that appear in global top
    filtered_df = user_data_df[~user_data_df['artist_track'].isin(cleaned_global_top)]
    songs_removed = user_data_df.shape[0] - filtered_df.shape[0]
    print(f"Removed {songs_removed} popular entries")
    
    # Drop the temporary column
    filtered_df = filtered_df.drop(columns=['artist_track'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_filename = os.path.join(DATA_DIR, f'filtered_top_tracks_{timestamp}.csv')
    filtered_df.to_csv(output_filename, index=False)

    
    return filtered_df

# To run as a script
if __name__ == '__main__': 
    output_file = get_top_tracks()
    filter_out_global_songs(output_file)