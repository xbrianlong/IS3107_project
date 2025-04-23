import sqlite3
from datetime import datetime
import os

class RecommendationDB:
    def __init__(self, db_path="/opt/airflow/outputs/recommendations.db"):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    track_name TEXT NOT NULL,
                    score FLOAT NOT NULL,
                    generated_at TIMESTAMP NOT NULL,
                    batch_id TEXT NOT NULL
                )
            ''')
            conn.commit()

    def save_recommendations(self, recommendations, batch_id=None):
        """Save a batch of recommendations to the database"""
        if batch_id is None:
            batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            current_time = datetime.now()
            
            data = [
                (rec['user'], rec['track'], rec['score'], current_time, batch_id)
                for rec in recommendations
            ]
            
            cursor.executemany(
                'INSERT INTO recommendations (username, track_name, score, generated_at, batch_id) VALUES (?, ?, ?, ?, ?)', 
                data
            )
            conn.commit()
            return batch_id

    def get_user_recommendations(self, username, limit=10):
        """Get the most recent recommendations for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT track_name, score, generated_at 
                FROM recommendations 
                WHERE username = ? 
                ORDER BY generated_at DESC, score DESC 
                LIMIT ?
            ''', (username, limit))
            return cursor.fetchall()