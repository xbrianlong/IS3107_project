import pandas as pd
from bs4 import BeautifulSoup
import re
import requests
import time
from datetime import datetime
import os

from config.settings import LASTFM_API_KEY, LASTFM_BASE_URL, TOP_GLOBAL_SONGS_URL


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

    
    return filtered_df, output_filename

def convert_to_lightgcn_format(
    csv_path,
    output_dir='lightgcn/data/music', # might have to double check dir paths
    interaction_filename='user_track_interactions.txt',
    user_map_filename='id2user.json',
    item_map_filename='id2track.json'
):
    import json

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['track_full_name'] = df['track_name'] + ' - ' + df['artist']
    user2id = {username: idx for idx, username in enumerate(df['username'].unique())}
    track2id = {track: idx for idx, track in enumerate(df['track_full_name'].unique())}
    id2user = {idx: user for user, idx in user2id.items()}
    id2track = {idx: track for track, idx in track2id.items()}
    df['user_id'] = df['username'].map(user2id)
    df['item_id'] = df['track_full_name'].map(track2id)
    interaction_df = df[['user_id', 'item_id']].drop_duplicates()
    interaction_path = os.path.join(output_dir, interaction_filename)
    interaction_df.to_csv(interaction_path, sep=' ', index=False, header=False)
    with open(os.path.join(output_dir, user_map_filename), 'w') as f:
        json.dump(id2user, f)
    with open(os.path.join(output_dir, item_map_filename), 'w') as f:
        json.dump(id2track, f)
    print(f"[✅] Interactions saved to: {interaction_path}")
    print(f"[💾] ID maps saved to: {user_map_filename}, {item_map_filename}")
    print(f"[👥] Total users: {len(user2id)} | 🎵 Tracks: {len(track2id)}")

def split_lightgcn_train_test(
    input_path='lightgcn/data/music/user_track_interactions.txt',
    output_dir='lightgcn/data/music',
    train_filename='train.txt',
    test_filename='test.txt',
    test_ratio=0.2,
    random_state=42
):
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_path, sep=' ', names=['user', 'item'])
    train_rows, test_rows = [], []

    for user, group in df.groupby('user'):
        if len(group) < 2:
            train_rows.append(group)
            continue
        train, test = train_test_split(group, test_size=test_ratio, random_state=random_state)
        train_rows.append(train)
        test_rows.append(test)

    train_df, test_df = pd.concat(train_rows), pd.concat(test_rows)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, train_filename)
    test_path = os.path.join(output_dir, test_filename)
    train_df.to_csv(train_path, sep=' ', index=False, header=False)
    test_df.to_csv(test_path, sep=' ', index=False, header=False)

    print(f"[✅] Train/Test split complete.")
    print(f"📂 Train: {train_path} ({len(train_df)} rows)")
    print(f"📂 Test: {test_path} ({len(test_df)} rows)")

# To run as a script
if __name__ == '__main__': 
    output_file = get_top_tracks()
    if output_file:
        filtered_file = filter_out_global_songs(output_file)
        if isinstance(filtered_file, pd.DataFrame):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            temp_filtered_csv = os.path.join(DATA_DIR, f'filtered_temp_{timestamp}.csv')
            filtered_file.to_csv(temp_filtered_csv, index=False)

            convert_to_lightgcn_format(csv_path=temp_filtered_csv)
            split_lightgcn_train_test()