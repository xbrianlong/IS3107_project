import streamlit as st
import random
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import requests
import math
import os
import ast
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from database.db_utils import MusicDB
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY") 
API_ROOT = "http://ws.audioscrobbler.com/2.0/"

st.set_page_config(layout="wide")

# def load_recommendations():
#     # Read the file with correct encoding
#     with open("top3_recommendations.txt", "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     data = []

#     for line in lines:
#         if ':' not in line:
#             continue  # skip malformed lines
#         username, recs_str = line.strip().split(":", 1)
#         try:
#             recs = ast.literal_eval(recs_str.strip())
#             # Take only the first 3 recommendations
#             row = [username] + recs[:3]
#             data.append(row)
#         except:
#             continue  # skip lines that fail to parse

#     df = pd.DataFrame(data, columns=["username", "rec_1", "rec_2", "rec_3"])
#     return df


# recommendations = load_recommendations()


def generate_listening_data_from_db(usernames, refresh=False):
    music_db = MusicDB(db_path="../outputs/music.db")  # update to your relative path
    all_records = []

    for username in usernames:
        user_history = music_db.get_user_listening_history(username, limit= 100)  # large limit to get everything
        for record in user_history:
            all_records.append({
                "username": username,
                "song_name": record[0],
                "artist_name": record[1],
                "album_name": record[2],
                "listen_week": record[3],
                "playcount": record[4],
                "Genre": record[5],
                
            })

    df = pd.DataFrame(all_records)
    return df

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# Example usage
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Construct the relative path to the CSV file
csv_path = os.path.join(base_path, "src", "data", "raw_data", "lastfm_active_users.csv")

active_users_df = pd.read_csv(csv_path)
usernames = active_users_df["username"].dropna().unique().tolist()[:300]

df = generate_listening_data_from_db(usernames, refresh=True)


df = df.rename(columns={
    "username": "User",
    "song_name": "Song",
    "artist_name": "Artist",
    "listen_week": "Listen_Date"
})
    
# df = pd.read_json("lastfm_data.json")
# df['Listen_Date'] = pd.to_datetime(df['Listen_Date'], format='mixed', dayfirst=True, errors='coerce').dt.date
# st.write(df.head())
# st.write(df.album_name.value_counts())

def get_top_artists_with_images(limit=10, genre_filter=None):
    if genre_filter and genre_filter != "All":
        filtered_df = df[df["Genre"] == genre_filter]
    else:
        filtered_df = df
        
    top_artists = filtered_df["Artist"].value_counts().head(limit).reset_index()
    top_artists.columns = ["Artist", "Count"]
    top_artists_sorted = top_artists.sort_values(by="Count", ascending=False)
    

    
    
    return top_artists_sorted

# Create tabs
tab1, tab2= st.tabs(["📊 Overall Dashboard", "👤 User Dashboard"])

with tab1:
    st.session_state.current_tab = "overall"
    st.title("📊 Overall Dashboard")
    top_artists = get_top_artists_with_images(10)  # Make sure this function also uses 'playcount'

    # Create two columns for side-by-side charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Most Popular Songs")
        top_songs = (
            df.groupby(["Song", "Artist"])["playcount"]
            .sum()
            .reset_index()
            .sort_values(by="playcount", ascending=False)
            .head(10)
        )

        # Create combined labels
        top_songs["Song_Artist"] = top_songs["Song"] + " (" + top_songs["Artist"] + ")"

# Truncate long labels for the x-axis
        top_songs["Song_Artist_Short"] = top_songs["Song_Artist"].str.slice(0, 25) + "..."

        fig = go.Figure(data=[
            go.Bar(
                x=top_songs["Song_Artist_Short"],
                y=top_songs["playcount"],
                text=top_songs["playcount"],
                hovertext=top_songs["Song_Artist"],  # Full info on hover
                hoverinfo="text+y",
                textposition='auto',
                marker_color='rgb(107,174,214)'
            )
        ])
        fig.update_layout(
            xaxis_title="Song (Artist)",
            yaxis_title="Play Count",
        )

        st.plotly_chart(fig, use_container_width=True)

    

    with col2:
        st.subheader("Top Artists")
        top_artists = df.groupby("Artist", as_index=False)["playcount"].sum()
        top_artists = top_artists.sort_values(by="playcount", ascending=False)

        artist_fig = go.Figure(data=[
            go.Bar(
                x=top_artists["Artist"].head(10),
                y=top_artists["playcount"].head(10),
                text=top_artists["playcount"].head(10),
                textposition='auto',
                marker=dict(
                    color='rgb(158,202,225)',
                    line=dict(color='rgb(8,48,107)', width=1.5)
                )
            )
        ])
        artist_fig.update_layout(
            xaxis_title="Artist",
            yaxis_title="Play Count",
        )
        st.plotly_chart(artist_fig, use_container_width=True)




with tab2:
    st.session_state.current_tab = "user"
    
    # Header section with user selection
    st.title("👤 User Personal Dashboard")
    st.markdown("Explore your music preferences and discover new songs based on your listening habits.")
    
    selected_user = st.selectbox("Select a User", sorted(df.User.unique().tolist()))
    user_df = df[df["User"] == selected_user]
    
    # Check if user has data
    if user_df.empty:
        st.warning(f"No listening data found for {selected_user}.")
        st.stop()
    
    # ==========================================
    # SECTION 1: Music Taste Analysis
    # ==========================================
    st.markdown("---")
    st.header(f"🎭 {selected_user}'s Music Taste Profile")
    # st.write(df)
    # Create tabs for different music insights
    
    
    
    col1, col2 = st.columns(2)
    
    with col2:
        st.subheader(f"{selected_user}'s Top Songs")

        # Aggregate top songs by playcount and include artist info
        user_top_songs = (
            user_df.groupby(["Song", "Artist"])["playcount"]
            .sum()
            .reset_index()
            .sort_values("playcount", ascending=False)
            .head(10)
        )
        
        # Create a label that combines song and artist
        user_top_songs["Song_Artist"] = user_top_songs["Song"] + " (" + user_top_songs["Artist"] + ")"

        fig = px.bar(
            x=user_top_songs["playcount"],
            y=user_top_songs["Song_Artist"],
            orientation='h',
            labels={"playcount": "Play Count", "Song_Artist": "Song"},
            color=user_top_songs["playcount"],
            color_continuous_scale=px.colors.sequential.Viridis
        )

        fig.update_layout(
            height=400,
            margin=dict(t=10, b=0, l=0, r=0),
            xaxis_title="Number of Listens",  
            yaxis_title="Song (Artist)",
            yaxis=dict(autorange="reversed")   
        )

        st.plotly_chart(fig, use_container_width=True)
                

    with col1:
        # Add user's top artists
        st.subheader(f"{selected_user}'s Top Artists")

        # Sum playcount per artist for this user
        user_top_artists = user_df.groupby("Artist", as_index=False)["playcount"].sum()
        user_top_artists = user_top_artists.sort_values(by="playcount", ascending=False).head(10)

        fig = px.bar(
            user_top_artists,
            x="playcount",
            y="Artist",
            orientation='h',
            labels={"playcount": "Total Play Count", "Artist": "Artist"},
            color="playcount",
            color_continuous_scale=px.colors.sequential.Viridis
        )

        fig.update_layout(
            height=400,
            margin=dict(t=10, b=0, l=0, r=0),
            yaxis=dict(autorange="reversed")  # Put highest values at top
        )

        st.plotly_chart(fig, use_container_width=True)
    
    
    # ==========================================
    # SECTION 2: Personalized Recommendations
    # ==========================================
    # st.markdown("---")
    st.subheader("🎵 Personalized Music Recommendations")

    music_db = MusicDB(db_path="../outputs/music.db")
    
    rec_tab1, rec_tab2 = st.tabs(["AI Top Picks", "Community Recommendations"])
    
    with rec_tab1:
    # Show personalized recommendations using the music_db.get_user_recommendations function
        user_recommendations = music_db.get_user_recommendations(selected_user, limit=10)
        if user_recommendations:
            # Display recommendations in a more visually appealing way
            st.subheader(f"🤖 {selected_user}'s Song Recommendations")
            
            # Sort by score (descending)
            sorted_recommendations = sorted(user_recommendations[:5], key=lambda x: x[4], reverse=True)
            
            # Use columns for cleaner layout - display up to 5 recommendations
            cols = st.columns(min(5, len(sorted_recommendations)))
            for i, rec in enumerate(sorted_recommendations):
                song_name, artist_name, album_name, rank, score, generated_at = rec
                
                with cols[i]:
                    st.markdown(f"### #{i+1}")
                    st.markdown(f"🎶 **{song_name}**")
                    st.markdown(f"👤 *{artist_name}*")
                    if album_name:
                        st.markdown(f"💿 *Album:* {album_name}")
            
            # If there are more than 5 recommendations, display them in a table
        else:
            st.warning(f"No personalized recommendations found for {selected_user}.")

        
    with rec_tab2:
        st.subheader("🔍 Explore Tracks Popular in your Community")
        
        # Calculate and display top artists for the selected user
        user_artist_counts = df.groupby(['User', 'Artist']).size().reset_index(name='Count')
        top_artists_per_user = {}
        for user in df.User.unique():
            user_artists = user_artist_counts[user_artist_counts['User'] == user]
            top_5 = user_artists.sort_values('Count', ascending=False).head(5)['Artist'].tolist()
            top_artists_per_user[user] = top_5
        
        # Get top 5 artists for selected user
        selected_user_top_artists = top_artists_per_user.get(selected_user, [])
        
        if selected_user_top_artists:
            # Find users who share top artists with the selected user
            connected_users = []
            similarity_scores = {}
            
            for user, artists in top_artists_per_user.items():
                if user != selected_user:
                    # Count how many top artists are shared
                    shared_artists = [artist for artist in artists if artist in selected_user_top_artists]
                    if shared_artists:
                        connected_users.append(user)
                        # Calculate similarity score (number of shared artists)
                        similarity_scores[user] = len(shared_artists)
            
            # Get songs from connected users who listen to your top artists
            connected_user_songs = df[
                (df['User'].isin(connected_users)) & 
                (df['Artist'].isin(selected_user_top_artists))
            ]
            
            # Remove songs that the selected user has already listened to
            selected_user_songs = set(user_df['Song'].unique())
            new_songs = connected_user_songs[~connected_user_songs['Song'].isin(selected_user_songs)]
            
            # Get top 10 most popular songs among similar users that you haven't heard
            top_songs = new_songs.groupby('Song').size().reset_index(name='Listen Count')
            top_songs = top_songs.sort_values('Listen Count', ascending=False).head(10)
            
            # Add artist information to the top songs
            song_artists = new_songs[['Song', 'Artist']].drop_duplicates()
            top_songs = top_songs.merge(song_artists, on='Song', how='left')
            
            # Create a combined string for display
            top_songs['Song - Artist'] = top_songs['Song'] + " (" + top_songs['Artist'] + ")"
            
            if not top_songs.empty:
                # Create a horizontal bar chart
                fig_top_songs = px.bar(
                    top_songs,
                    y='Song - Artist',
                    x='Listen Count',
                    color='Listen Count',
                    color_continuous_scale='Oranges',
                    orientation='h'
                )
                
                fig_top_songs.update_layout(
                    title="Songs you haven't heard yet, but similar users love!",
                    xaxis_title="Number of Listens by Similar Users",
                    yaxis_title="",
                    yaxis={'categoryorder':'total ascending'},
                    height=500
                )
                
                st.plotly_chart(fig_top_songs, use_container_width=True)
            else:
                st.info("No new song recommendations found from similar users.")
        else:
            st.warning("No listening data found to make similarity-based recommendations.")

    # ==========================================
    # SECTION 3: Music Community Insights
    # ==========================================
    st.markdown("---")
    st.header("🌐 Your Music Community")
    
    community_tab1, community_tab2 = st.tabs(["Similar Users", "Network Graphs"])
    
    with community_tab1:
        if similarity_scores:
            # Sort users by similarity score
            similar_users_df = pd.DataFrame({
                'User': list(similarity_scores.keys()),
                'Shared Artists': list(similarity_scores.values())
            }).sort_values('Shared Artists', ascending=False).head(10)
            
            st.subheader("Users With Taste Similar To Yours")
            
            # Create a bar chart
            fig_similar_users = px.bar(
                similar_users_df,
                x='User', 
                y='Shared Artists',
                color='Shared Artists',
                color_continuous_scale='Viridis',
                labels={'Shared Artists': 'Number of Shared Top Artists'}
            )
            
            fig_similar_users.update_layout(
                xaxis_title="User",
                yaxis_title="Shared Top Artists",
                yaxis_range=[0, 5],  # Max is 5 since we're looking at top 5 artists
                height=400
            )
            
            st.plotly_chart(fig_similar_users, use_container_width=True)
            
            st.markdown("These users share your taste in artists! Connect with them to discover more music.")
        else:
            st.info("No users with similar tastes found.")
    
    with community_tab2:
        
        network_choice = st.radio(
            "Choose a network visualization:",
            ["User-Artist Network", "Recommendation Network"],
            horizontal=True
        )
        
        if network_choice == "User-Artist Network":
            # Display the User-Artist network visualization
            if selected_user_top_artists and connected_users:
                # Build the network graph as in your original code
                # (Most of your original 3D network visualization code would go here)
                G = nx.Graph()
                
                # Add the selected user node
                G.add_node(selected_user, type="user", is_selected=True)
                
                # Add artist nodes and edges from selected user to artists
                for artist in selected_user_top_artists:
                    G.add_node(artist, type="artist")
                    G.add_edge(selected_user, artist)
                
                # Add other user nodes and their connections to the same artists
                for user in connected_users:
                    G.add_node(user, type="user", is_selected=False)
                    user_artists = top_artists_per_user[user]
                    for artist in user_artists:
                        if artist in selected_user_top_artists:
                            # Only add edges to artists that the selected user also has in their top 5
                            G.add_edge(user, artist)
                
                # Generate positions in 3D space (from your original code)
                # Create 3D visualization (from your original code)
                # (Your existing code for 3D positions and visualization)
                
                # Position the selected user prominently
                pos_3d = {}
                # Top layer: selected user
                pos_3d[selected_user] = (0, 0, 3.0)

                # Middle layer: artists in a circle
                r = 1.5  # Radius of the circle
                num_artists = len(selected_user_top_artists)
                for i, artist in enumerate(selected_user_top_artists):
                    theta = (2 * math.pi * i) / num_artists
                    pos_3d[artist] = (r * math.cos(theta), r * math.sin(theta), 0.0)

                # Bottom layer: connected users in a wider circle
                r_users = 3.0
                for i, user in enumerate(connected_users):
                    theta = (2 * math.pi * i) / (len(connected_users) or 1)  # Avoid division by zero
                    pos_3d[user] = (r_users * math.cos(theta), r_users * math.sin(theta), -3.0)

                # Create edges for the 3D plot
                edge_x_3d, edge_y_3d, edge_z_3d = [], [], []
                for edge in G.edges():
                    x0, y0, z0 = pos_3d[edge[0]]
                    x1, y1, z1 = pos_3d[edge[1]]
                    edge_x_3d.extend([x0, x1, None])
                    edge_y_3d.extend([y0, y1, None])
                    edge_z_3d.extend([z0, z1, None])

                edge_trace_3d = go.Scatter3d(
                    x=edge_x_3d, y=edge_y_3d, z=edge_z_3d,
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                # Create separate traces for different node types
                # Selected user node (green)
                selected_user_x = [pos_3d[selected_user][0]]
                selected_user_y = [pos_3d[selected_user][1]]
                selected_user_z = [pos_3d[selected_user][2]]
                selected_user_trace = go.Scatter3d(
                    x=selected_user_x, y=selected_user_y, z=selected_user_z,
                    mode='markers+text',
                    marker=dict(size=15, color='green', line=dict(width=2, color='white')),
                    text=[f"👤 {selected_user} (YOU)"],
                    textposition="top center",
                    hoverinfo='text',
                    name='Selected User'
                )
                
                # Artist nodes (orange)
                artist_x = [pos_3d[artist][0] for artist in selected_user_top_artists]
                artist_y = [pos_3d[artist][1] for artist in selected_user_top_artists]
                artist_z = [pos_3d[artist][2] for artist in selected_user_top_artists]
                artist_text = [f"🎵 {artist}" for artist in selected_user_top_artists]
                artist_trace = go.Scatter3d(
                    x=artist_x, y=artist_y, z=artist_z,
                    mode='markers+text',
                    marker=dict(size=12, color='orange', line=dict(width=1)),
                    text=artist_text,
                    textposition="bottom center",
                    hoverinfo='text',
                    name='Top Artists'
                )
                
                # Other user nodes (blue)
                other_user_trace = None
                if connected_users:
                    other_user_x = [pos_3d[user][0] for user in connected_users]
                    other_user_y = [pos_3d[user][1] for user in connected_users]
                    other_user_z = [pos_3d[user][2] for user in connected_users]
                    other_user_text = [f"👤 {user}" for user in connected_users]
                    other_user_trace = go.Scatter3d(
                        x=other_user_x, y=other_user_y, z=other_user_z,
                        mode='markers+text',
                        marker=dict(size=10, color='blue', line=dict(width=1)),
                        text=other_user_text,
                        textposition="top center",
                        hoverinfo='text',
                        name='Similar Users'
                    )
                
                # Combine traces for the figure
                traces = [edge_trace_3d, selected_user_trace, artist_trace]
                if other_user_trace:
                    traces.append(other_user_trace)
                
                fig_3d = go.Figure(
                    data=traces,
                    layout=go.Layout(
                        title='Users Who Share Your Musical Taste',
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        scene=dict(
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                    )
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                with st.expander("How to use this visualization"):
                    st.markdown("""
                    - **Green node**: You
                    - **Orange nodes**: Your top artists
                    - **Blue nodes**: Other users who share your taste
                    - **Lines**: Connections between users and artists
                    
                    You can rotate, zoom, and pan the visualization to explore connections.
                    """)
            else:
                st.info("Not enough data to create a User-Artist network.")
                
        else:  # Recommendation Network
            
            
            # Get recommendations for the selected user
            selected_user_recs = music_db.get_user_recommendations(selected_user, limit=5)
            
            if not selected_user_recs:
                st.info(f"No recommendations found for {selected_user}.")
            else:
                # Create a list of songs recommended to the selected user
                selected_user_songs = [(rec[0], rec[1]) for rec in selected_user_recs]  # (song_name, artist_name)
                selected_user_song_ids = [f"{song} - {artist}" for song, artist in selected_user_songs]
                
                # Find other users with the same recommendations
                shared_recommendations = []
                all_users = df.User.unique().tolist()
                all_users.remove(selected_user)  # Remove the selected user
                

                # For each sample user, check if they share recommendations with the selected user
                for other_user in all_users:
                    other_user_recs = music_db.get_user_recommendations(other_user, limit=5)
                    if other_user_recs:
                        other_user_song_ids = [f"{rec[0]} - {rec[1]}" for rec in other_user_recs]
                        
                        # Find shared songs
                        shared_songs = set(selected_user_song_ids) & set(other_user_song_ids)
                        
                        if shared_songs:
                            for song_id in shared_songs:
                                shared_recommendations.append({
                                    "User": other_user,
                                    "Song_Artist": song_id
                                })
                
                # Add the selected user's recommendations to the shared recommendations
                for song_id in selected_user_song_ids:
                    shared_recommendations.append({
                        "User": selected_user,
                        "Song_Artist": song_id
                    })
                
                # Convert to DataFrame
                shared_rec_df = pd.DataFrame(shared_recommendations)
                
                if len(shared_rec_df) > 1:
                    # Build the network graph
                    G = nx.Graph()
                    
                    # Add nodes and edges
                    for _, row in shared_rec_df.iterrows():
                        G.add_node(row["User"], type="user")
                        G.add_node(row["Song_Artist"], type="song")
                        G.add_edge(row["User"], row["Song_Artist"])
                    
                    # Generate 3D layout for nodes
                    # Place selected user at center
                    pos_3d = {}

                    # Selected user at origin (bottom layer)
                    pos_3d[selected_user] = (0, 0, 0)

                    # Position songs in a horizontal ring above the selected user (middle layer)
                    song_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'song']
                    num_songs = len(song_nodes)
                    radius_song = 3.0  # Slightly wider radius for visibility
                    z_song = 6.0       # Increased height
                    angle_gap_song = 2 * math.pi / (num_songs or 1)

                    for i, song in enumerate(song_nodes):
                        angle = i * angle_gap_song
                        pos_3d[song] = (
                            radius_song * math.cos(angle),
                            radius_song * math.sin(angle),
                            z_song
                        )

                    # Position other users in a horizontal ring above the songs (top layer)
                    other_users = [node for node in G.nodes() if G.nodes[node]['type'] == 'user' and node != selected_user]
                    num_users = len(other_users)
                    radius_user = 5.0   # Slightly wider
                    z_user = 12.0       # More height difference
                    angle_gap_user = 2 * math.pi / (num_users or 1)

                    for i, user in enumerate(other_users):
                        angle = i * angle_gap_user
                        pos_3d[user] = (
                            radius_user * math.cos(angle),
                            radius_user * math.sin(angle),
                            z_user
                        )

                    # Create edges for the 3D plot
                    edge_x_3d, edge_y_3d, edge_z_3d = [], [], []
                    for edge in G.edges():
                        x0, y0, z0 = pos_3d[edge[0]]
                        x1, y1, z1 = pos_3d[edge[1]]
                        edge_x_3d.extend([x0, x1, None])
                        edge_y_3d.extend([y0, y1, None])
                        edge_z_3d.extend([z0, z1, None])
                    
                    edge_trace_3d = go.Scatter3d(
                        x=edge_x_3d, y=edge_y_3d, z=edge_z_3d,
                        line=dict(width=1, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    
                    # Create selected user node trace
                    selected_user_x = [pos_3d[selected_user][0]]
                    selected_user_y = [pos_3d[selected_user][1]]
                    selected_user_z = [pos_3d[selected_user][2]]
                    selected_user_text = [f"👤 {selected_user} (YOU)"]
                    
                    selected_user_trace = go.Scatter3d(
                        x=selected_user_x, y=selected_user_y, z=selected_user_z,
                        mode='markers+text',
                        marker=dict(size=15, color='green', line=dict(width=2, color='white')),
                        text=selected_user_text,
                        textposition="top center",
                        hoverinfo='text',
                        name='Selected User'
                    )
                    
                    # Create other user node trace
                    other_user_trace = None
                    if other_users:
                        other_user_x = [pos_3d[user][0] for user in other_users]
                        other_user_y = [pos_3d[user][1] for user in other_users]
                        other_user_z = [pos_3d[user][2] for user in other_users]
                        other_user_text = [f"👤 {user}" for user in other_users]
                        
                        other_user_trace = go.Scatter3d(
                            x=other_user_x, y=other_user_y, z=other_user_z,
                            mode='markers+text',
                            marker=dict(size=12, color='royalblue', line=dict(width=1)),
                            text=other_user_text,
                            textposition="top center",
                            hoverinfo='text',
                            name='Similar Users'
                        )
                    
                    # Create song node trace
                    song_x = [pos_3d[song][0] for song in song_nodes]
                    song_y = [pos_3d[song][1] for song in song_nodes]
                    song_z = [pos_3d[song][2] for song in song_nodes]
                    # Trim song names if they're too long
                    song_text = [f"🎵 {song[:20]}..." if len(song) > 20 else f"🎵 {song}" for song in song_nodes]
                    
                    song_trace = go.Scatter3d(
                        x=song_x, y=song_y, z=song_z,
                        mode='markers+text',
                        marker=dict(size=10, color='gold', line=dict(width=1)),
                        text=song_text,
                        textposition="bottom center",
                        hoverinfo='text',
                        name='Songs'
                    )
                    
                    # Combine traces for the figure
                    traces = [edge_trace_3d, selected_user_trace, song_trace]
                    if other_user_trace:
                        traces.append(other_user_trace)
                    
                    fig_3d = go.Figure(
                        data=traces,
                        layout=go.Layout(
                            title='Users who received the same song recommendations as you',
                            showlegend=True,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            scene=dict(
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            )
                        )
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    with st.expander("How to use this visualization"):
                        st.markdown("""
                        - **Green node**: You
                        - **Gold nodes**: Recommended songs
                        - **Blue nodes**: Other users who received the same recommendations
                        - **Lines**: Connections between users and their song recommendations
                        
                        This visualization shows which of your recommended songs are also recommended to other users!
                        """)
                else:
                    st.info("No overlapping recommended songs with other users were found.")
                
                def check_shared_recommendations(music_db):
                    """
                    Check if any users share the same recommended songs.
                    Returns a dictionary with shared songs and the users who received them.
                    """
                    import sqlite3
                    import pandas as pd
                    
                    # Get all users
                    with sqlite3.connect(music_db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT username FROM users")
                        all_users = df.User.unique().tolist()
                       
                    
                    # Dictionary to store songs and the users who received them
                    song_to_users = {}
                    
                    # For each user, get their recommendations
                    for user in all_users:
                        recommendations = music_db.get_user_recommendations(user, limit=5)
                        
                        # Add each recommendation to the dictionary
                        for rec in recommendations:
                            song_name, artist_name = rec[0], rec[1]
                            song_key = f"{song_name} - {artist_name}"
                            
                            if song_key not in song_to_users:
                                song_to_users[song_key] = []
                            
                            song_to_users[song_key].append(user)
                    
                    # Filter to only songs recommended to multiple users
                    shared_songs = {song: users for song, users in song_to_users.items() if len(users) > 1}
                    
                    # Print summary
                    print(f"Found {len(shared_songs)} songs that are recommended to multiple users")
                    
                    # Show details of shared songs
                    shared_songs_df = pd.DataFrame([
                        {"Song": song, "Users": len(users), "User List": ", ".join(users[:5]) + ("..." if len(users) > 5 else "")}
                        for song, users in shared_songs.items()
                    ]).sort_values("Users", ascending=False)
                    
                    return shared_songs, shared_songs_df

                # Usage in your Streamlit app:
                if st.checkbox("Check for shared recommendations (Just for verfication purposes)"):
                    with st.spinner("Analyzing recommendation patterns..."):
                        shared_songs, shared_songs_df = check_shared_recommendations(music_db)
                        
                        if shared_songs:
                            st.success(f"Found {len(shared_songs)} songs that are recommended to multiple users!")
                            st.dataframe(shared_songs_df)
                            
                            # Get the most shared song
                            most_shared = shared_songs_df.iloc[0]
                            st.info(f"The most widely recommended song is '{most_shared['Song']}', " 
                                f"which was recommended to {most_shared['Users']} different users.")
                        else:
                            st.warning("No shared recommendations found. Each user has unique song recommendations.")



