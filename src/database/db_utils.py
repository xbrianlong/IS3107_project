import sqlite3
from datetime import datetime
import os

class MusicDB:
    def __init__(self, db_path="/opt/airflow/outputs/music.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS songs (
                    song_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_name TEXT NOT NULL,
                    artist_name TEXT NOT NULL,
                    album_name TEXT,
                    genre TEXT,
                    song_length INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(song_name, artist_name)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS listening_data (
                    listening_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    song_id INTEGER NOT NULL,
                    listen_week TEXT NOT NULL,
                    playcount INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (song_id) REFERENCES songs(song_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    song_id INTEGER NOT NULL,
                    rank INTEGER NOT NULL,
                    score FLOAT NOT NULL,
                    batch_id TEXT NOT NULL,
                    generated_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (song_id) REFERENCES songs(song_id)
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_listening_user ON listening_data(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_listening_song ON listening_data(song_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_batch ON recommendations(batch_id)')

            conn.commit()

    def insert_or_get_user(self, username):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if user already exists
            cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            if result:
                return result[0]

            # Assign next user_id manually
            cursor.execute("SELECT MAX(user_id) FROM users")
            max_id = cursor.fetchone()[0] or 0
            next_id = max_id + 1

            # Insert user and commit
            cursor.execute("INSERT INTO users (user_id, username) VALUES (?, ?)", (next_id, username))
            conn.commit()
            return next_id
            
    def insert_or_get_song(self, song_name, artist_name, album_name=None, genre=None, song_length=None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if song already exists
            cursor.execute("""
                SELECT song_id FROM songs 
                WHERE song_name = ? AND artist_name = ?
            """, (song_name, artist_name))
            result = cursor.fetchone()
            if result:
                return result[0]

            # Assign next song_id manually
            cursor.execute("SELECT MAX(song_id) FROM songs")
            max_id = cursor.fetchone()[0] or 0
            next_id = max_id + 1

            # Insert song and commit
            cursor.execute("""
                INSERT INTO songs (song_id, song_name, artist_name, album_name, genre, song_length)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (next_id, song_name, artist_name, album_name, genre, song_length))
            conn.commit()
            return next_id

    def insert_listening_data(self, user_id, song_id, listen_week, playcount):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO listening_data (user_id, song_id, listen_week, playcount)
                VALUES (?, ?, ?, ?)
            """, (user_id, song_id, listen_week, playcount))
            conn.commit()

    def get_all_user_song_interactions(self):
        """Fetch all user-song interactions from the DB"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, song_id FROM listening_data")
            return cursor.fetchall()

    def save_recommendations(self, recommendations, batch_id=None):
        if batch_id is None:
            batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            current_time = datetime.now()
            data = [
                (rec['user_id'], rec['song_id'], rec['rank'], rec['score'], batch_id, current_time)
                for rec in recommendations
            ]
            cursor.executemany("""
                INSERT INTO recommendations (user_id, song_id, rank, score, batch_id, generated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
            return batch_id

    def get_user_recommendations(self, username, batch_id=None, limit=10):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = """
                SELECT s.song_name, s.artist_name, s.album_name, r.rank, r.score, r.generated_at
                FROM recommendations r
                JOIN users u ON r.user_id = u.user_id
                JOIN songs s ON r.song_id = s.song_id
                WHERE u.username = ?
            """
            params = [username]
            if batch_id:
                query += " AND r.batch_id = ?"
                params.append(batch_id)
            query += " ORDER BY r.rank LIMIT ?"
            params.append(limit)
            cursor.execute(query, params)
            return cursor.fetchall()

    def get_user_listening_history(self, username, limit=10):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.song_name, s.artist_name, s.album_name, l.listen_week, l.playcount
                FROM listening_data l
                JOIN users u ON l.user_id = u.user_id
                JOIN songs s ON l.song_id = s.song_id
                WHERE u.username = ?
                ORDER BY l.created_at DESC
                LIMIT ?
            """, (username, limit))
            return cursor.fetchall()

    def get_username_by_id(self, user_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_song_by_id(self, song_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT song_name, artist_name FROM songs WHERE song_id = ?", (song_id,))
            result = cursor.fetchone()
            return {'name': result[0], 'artist': result[1]} if result else None

    def get_latest_batch_id(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT batch_id FROM recommendations
                ORDER BY generated_at DESC LIMIT 1
            """)
            result = cursor.fetchone()
            return result[0] if result else None

    def create_lightgcn_input_files(self, output_dir="src/lightgcn/data/music"):
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, "test_execution.txt") # Define test file path

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT user_id, song_id FROM listening_data")
            interactions_file = os.path.join(output_dir, "user_track_interactions.txt")
            with open(interactions_file, "w") as f:
                for user_id, song_id in cursor:
                    f.write(f"{user_id - 1} {song_id - 1}\n")
            with open(test_file, "w") as f:
                f.write("Execution successful. Data processing complete.\n")
            print(f"Successfully created interaction file: {interactions_file}")
            print(f"Successfully created test file: {test_file}")
            return interactions_file

    def get_db_stats(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            stats = {}
            cursor.execute("SELECT COUNT(*) FROM users")
            stats['total_users'] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM songs")
            stats['total_songs'] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM listening_data")
            stats['total_interactions'] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT batch_id) FROM recommendations")
            stats['total_recommendation_batches'] = cursor.fetchone()[0]
            return stats
