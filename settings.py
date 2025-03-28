import os
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_BASE_URL = 'http://ws.audioscrobbler.com/2.0/'
TOP_GLOBAL_SONGS_URL = "https://kworb.net/spotify/country/global_weekly.html"
