import pandas as pd
from bs4 import BeautifulSoup
import re
import requests
import time
import pickle
from datetime import datetime
import os
import sys

# Add project root to path and get absolute paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from config.settings import LASTFM_API_KEY, LASTFM_BASE_URL, TOP_GLOBAL_SONGS_URL
from src.database.db_utils import MusicDB

API_KEY = LASTFM_API_KEY
BASE_URL = LASTFM_BASE_URL
DATA_DIR = os.path.join(project_root, 'src', 'data', 'raw_data')
INPUT_CSV = os.path.join(DATA_DIR, 'lastfm_active_users.csv')

def get_lastfm_users(filename):
    try:
        df = pd.read_csv(filename)
        return df['username'].tolist()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def get_user_top_tracks(username, limit=10):
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
                tracks.append({
                    'username': username,
                    'track_name': track.get('name', ''),
                    'artist': track.get('artist', {}).get('name', ''),
                    'rank': int(track.get('@attr', {}).get('rank', 0)),
                    'playcount': int(track.get('playcount', 0))
                })
            return tracks
        else:
            print(f"No track data found for user: {username}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"API request failed for user {username}: {e}")
        return []


# def get_top_tracks(db_instance=None):
#     users = get_lastfm_users(INPUT_CSV)
#     if not users:
#         print("No users found. Exiting.")
#         return

#     if db_instance:
#         db_instance.init_db()

#     all_tracks = []
#     for count, user in enumerate(users):
#         print(f"[{count + 1}/{len(users)}] Fetching top tracks for: {user}")
#         user_tracks = get_user_top_tracks(user)

#         if db_instance and user_tracks:
#             for track in user_tracks:
#                 user_id = db_instance.insert_or_get_user(track['username'])
#                 song_id = db_instance.insert_or_get_song(track['track_name'], track['artist'])

#                 listen_week = datetime.now().strftime("%Y-%U")
#                 playcount = track.get('playcount', 1)

#                 db_instance.insert_listening_data(
#                     user_id=user_id,
#                     song_id=song_id,
#                     listen_week=listen_week,
#                     playcount=playcount
#                 )

#         all_tracks.extend(user_tracks)
#         time.sleep(0.2)

#     if all_tracks:
#         print(f"\n Extracted data for {len(users)} users.")
#         print(f" Total tracks collected: {len(all_tracks)}")
#         return pd.DataFrame(all_tracks)
#     else:
#         print("No track data collected.")
#         return pd.DataFrame()

def get_top_tracks():
    users = get_lastfm_users(INPUT_CSV)
    if not users:
        print("No users found. Exiting.")
        return pd.DataFrame()

    all_tracks = []
    for count, user in enumerate(users):
        print(f"[{count + 1}/{len(users)}] Fetching top tracks for: {user}")
        user_tracks = get_user_top_tracks(user)
        all_tracks.extend(user_tracks)
        time.sleep(0.2)

    if all_tracks:
        print(f"\nExtracted data for {len(users)} users.")
        print(f"Total tracks collected: {len(all_tracks)}")
        return pd.DataFrame(all_tracks)
    else:
        print("No track data collected.")
        return pd.DataFrame()
    
def load_tracks_to_db(tracks_df, db_instance=None):
    if not isinstance(tracks_df, pd.DataFrame) or tracks_df.empty:
        print("No tracks to load into database.")
        return

    if db_instance is None:
        db_instance = MusicDB()

    db_instance.init_db()

    for _, track in tracks_df.iterrows():
        user_id = db_instance.insert_or_get_user(track['username'])
        song_id = db_instance.insert_or_get_song(track['track_name'], track['artist'])

        listen_week = datetime.now().strftime("%Y-%U")
        playcount = track.get('playcount', 1)

        db_instance.insert_listening_data(
            user_id=user_id,
            song_id=song_id,
            listen_week=listen_week,
            playcount=playcount
        )

    print(f"Loaded {len(tracks_df)} tracks into database.")

def get_global_top_songs():
    response = requests.get(TOP_GLOBAL_SONGS_URL)
    soup = BeautifulSoup(response.text, 'html.parser')
    top_songs = set()

    table = soup.find('table')
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        artist_song = cols[2].text.strip()
        top_songs.add(artist_song.lower())
    return top_songs

def clean_track_name(text):
    return re.sub(r'\([^)]*\)', '', text).strip()

def filter_out_global_songs(user_data_df_filepath, global_top_filepath):
    user_data_df = pd.read_csv(user_data_df_filepath)  # Read CSV directly
    with open(global_top_filepath, 'rb') as f:
        global_top = pickle.load(f)

    user_data_df['artist_track'] = (user_data_df['artist'].str.lower() + " - " +
                                   user_data_df['track_name'].str.lower())
    user_data_df['artist_track'] = user_data_df['artist_track'].apply(clean_track_name)
    cleaned_global_top = {clean_track_name(song) for song in global_top}

    filtered_df = user_data_df[~user_data_df['artist_track'].isin(cleaned_global_top)]
    songs_removed = user_data_df.shape[0] - filtered_df.shape[0]
    print(f"Removed {songs_removed} global top songs")

    filtered_df = filtered_df.drop(columns=['artist_track'])
    return filtered_df


def convert_to_lightgcn_format(
    output_dir=None,
    interaction_filename='user_track_interactions.txt',
    db_instance=None
):
    if output_dir is None:
        output_dir = os.path.join(project_root, 'src', 'lightgcn', 'data', 'music')
    
    if db_instance is None:
        db_instance = MusicDB()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    interaction_path = os.path.join(output_dir, interaction_filename)
    if os.path.exists(interaction_path):
        os.remove(interaction_path)

    # Query all user-song interactions directly from the DB
    interactions = db_instance.get_all_user_song_interactions()
    print(f"Found {len(interactions)} interactions in database")

    if not interactions:
        print("Warning: No interactions found in database!")
        return

    # Convert to LightGCN 0-indexed format
    interaction_path = os.path.join(output_dir, interaction_filename)
    
    try:
        with open(interaction_path, 'w') as f:
            for user_id, song_id in interactions:
                f.write(f"{user_id - 1} {song_id - 1}\n")
        print(f"Successfully wrote interactions to: {interaction_path}")
        
    except Exception as e:
        print(f"Error writing files: {e}")
        raise

    return interaction_path


def split_lightgcn_train_test(
    input_path=None,
    output_dir=None,
    train_filename='train.txt',
    test_filename='test.txt',
    test_ratio=0.2,
    random_state=42
):
    if input_path is None:
        input_path = os.path.join(project_root, 'src', 'lightgcn', 'data', 'music', 'user_track_interactions.txt')
    if output_dir is None:
        output_dir = os.path.join(project_root, 'src', 'lightgcn', 'data', 'music')

    from sklearn.model_selection import train_test_split

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, train_filename)
    test_path = os.path.join(output_dir, test_filename)
    for path in [train_path, test_path]:
        if os.path.exists(path):
            os.remove(path)

    df = pd.read_csv(input_path, sep=' ', names=['user', 'item'])
    train_rows, test_rows = [], []

    for user, group in df.groupby('user'):
        if len(group) < 2:
            train_rows.append(group)
        else:
            train, test = train_test_split(group, test_size=test_ratio, random_state=random_state)
            train_rows.append(train)
            test_rows.append(test)

    # Ensure we have data before concatenation
    if train_rows:
        train_df = pd.concat(train_rows, ignore_index=True)
    else:
        train_df = pd.DataFrame(columns=['user', 'item'])

    if test_rows:
        test_df = pd.concat(test_rows, ignore_index=True)
    else:
        test_df = pd.DataFrame(columns=['user', 'item'])

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, train_filename)
    test_path = os.path.join(output_dir, test_filename)
    train_df.to_csv(train_path, sep=' ', index=False, header=False)
    test_df.to_csv(test_path, sep=' ', index=False, header=False)

    print(f"Train/Test split complete.")
    print(f"Train: {train_path} ({len(train_df)} rows)")
    print(f"Test: {test_path} ({len(test_df)} rows)")


# Optional script entry
if __name__ == '__main__':
    db = MusicDB()
    df = get_top_tracks(db_instance=db)

    if isinstance(df, pd.DataFrame) and not df.empty:
        filtered_df = filter_out_global_songs(df)
        if not filtered_df.empty:
            convert_to_lightgcn_format(db_instance=db)
            split_lightgcn_train_test()
