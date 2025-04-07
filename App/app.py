import streamlit as st
import random
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os
import ast
from dotenv import load_dotenv

load_dotenv()


LASTFM_API_KEY = os.getenv("LASTFM_API_KEY") 
API_ROOT = "http://ws.audioscrobbler.com/2.0/"

st.set_page_config(layout="wide")

def load_recommendations():
    # Read the file with correct encoding
    with open("top3_recommendations.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []

    for line in lines:
        if ':' not in line:
            continue  # skip malformed lines
        username, recs_str = line.strip().split(":", 1)
        try:
            recs = ast.literal_eval(recs_str.strip())
            # Take only the first 3 recommendations
            row = [username] + recs[:3]
            data.append(row)
        except:
            continue  # skip lines that fail to parse

    df = pd.DataFrame(data, columns=["username", "rec_1", "rec_2", "rec_3"])
    return df


recommendations = load_recommendations()

# Function to get artist image
def get_artist_image(artist_name):
    try:
        params = {
            'method': 'artist.getinfo',
            'artist': artist_name,
            'api_key': LASTFM_API_KEY,
            'format': 'json'
        }
        response = requests.get(API_ROOT, params=params)
        data = response.json()
        
        # Get medium-sized image
        if 'artist' in data and 'image' in data['artist']:
            for img in data['artist']['image']:
                if img['size'] == 'large':
                    if img['#text']:
                        return img['#text']
        
        return "https://via.placeholder.com/150"
    except Exception as e:
        print(f"Error fetching image for {artist_name}: {e}")
        return "https://via.placeholder.com/150"
    
df = pd.read_json("lastfm_data.json")

df["Listen_Date"] = pd.to_datetime(df["Listen_Date"], errors='coerce')

df["Listen_Date"] = df["Listen_Date"].dt.date
df = df.dropna(subset=["Listen_Date"])

@st.cache_data
def get_top_artists_with_images(limit=10, genre_filter=None):
    if genre_filter and genre_filter != "All":
        filtered_df = df[df["Genre"] == genre_filter]
    else:
        filtered_df = df
        
    top_artists = filtered_df["Artist"].value_counts().head(limit).reset_index()
    top_artists.columns = ["Artist", "Count"]
    top_artists_sorted = top_artists.sort_values(by="Count", ascending=False)
    

    top_artists_sorted["ImageURL"] = top_artists_sorted["Artist"].apply(get_artist_image)
    
    return top_artists_sorted

def get_songs_by_artist(artist_name, genre_filter=None):
    if genre_filter and genre_filter != "All":
        filtered_df = df[df["Genre"] == genre_filter]
    else:
        filtered_df = df
        
    artist_songs = filtered_df[filtered_df["Artist"] == artist_name]
    song_counts = artist_songs["Song"].value_counts().reset_index()
    song_counts.columns = ["Song", "Count"]
    song_counts_sorted = song_counts.sort_values(by="Count", ascending=False)
    return song_counts_sorted

# Create session state to track clicked artist
if 'selected_artist' not in st.session_state:
    st.session_state.selected_artist = None
if 'show_artist_details' not in st.session_state:
    st.session_state.show_artist_details = False
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "overall"
if 'genre_filter' not in st.session_state:
    st.session_state.genre_filter = "All"

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overall Dashboard", "👤 User Dashboard", "📈 Trend Analysis", "🎵 Artist Intelligence"])

with tab1:
    st.session_state.current_tab = "overall"
    st.title("📊 Overall Dashboard")
    
    # Create tabs within the overall dashboard for better organization
    overall_tabs = st.tabs(["Top Artists", "Song Analytics"])
    
    # TAB 1: TOP ARTISTS
    with overall_tabs[0]:
        st.header("🎸 Top Artists Leaderboard")
        
        # Get top artists data
        top_artists = get_top_artists_with_images(10)
        
        # Display artist grid in a more compact way
        st.write("Click on an artist to see their details")
        
        # Use 5 columns for desktop, fewer on mobile (responsive)
        cols = st.columns(5)
        for i, (index, artist) in enumerate(top_artists.iterrows()):
            col_index = i % 5
            with cols[col_index]:
                # Make images smaller for less clutter
                st.image(artist["ImageURL"], width=120)
                
                # Simplified button label
                if st.button(f"**{artist['Artist']}**\n{artist['Count']}", key=f"artist_btn_{i}_tab1"):
                    st.session_state.selected_artist = artist['Artist']
                    st.session_state.show_artist_details = True
                    st.session_state.genre_filter = "All"
                    st.rerun()
        
        # Show artist details in a dedicated container for visual separation
        if st.session_state.show_artist_details and st.session_state.current_tab == "overall":
            with st.container():
                st.markdown("---")
                artist_name = st.session_state.selected_artist
                
                # Create columns for artist image and info
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    artist_image = get_artist_image(artist_name)
                    st.image(artist_image, width=180)
                    
                    # Add a button to close artist details
                    if st.button("Close Artist Details"):
                        st.session_state.show_artist_details = False
                        st.session_state.selected_artist = None
                        st.rerun()
                        
                with col2:
                    st.header(f"{artist_name}")
                    
                    # Get songs by this artist
                    artist_songs = get_songs_by_artist(artist_name)
                    
                    # Display song counts as a bar chart
                    if not artist_songs.empty:
                        # Limit to top 8 songs for better visualization
                        display_songs = artist_songs.head(8)
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=display_songs["Song"],
                                y=display_songs["Count"],
                                text=display_songs["Count"],
                                textposition='auto',
                                marker_color='rgb(158,202,225)'
                            )
                        ])
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            xaxis_title="Song",
                            yaxis_title="Play Count",
                            height=350,
                            margin=dict(l=40, r=40, t=30, b=80)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show full data in an expander to reduce clutter
                        with st.expander("View All Songs Data"):
                            st.dataframe(artist_songs)
                    else:
                        st.write("No song data available for this artist.")
    
    # TAB 2: SONG ANALYTICS
    with overall_tabs[1]:
        # Create two columns for side-by-side charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Most Popular Songs")
            top_songs = df["Song"].value_counts().head(10).reset_index()
            top_songs.columns = ["Song", "Count"]
            top_songs_sorted = top_songs.sort_values(by="Count", ascending=False)
            
            # Use plotly instead of st.bar_chart for better customization
            fig = go.Figure(data=[
                go.Bar(
                    x=top_songs_sorted["Song"],
                    y=top_songs_sorted["Count"],
                    text=top_songs_sorted["Count"],
                    textposition='auto',
                    marker_color='rgb(107,174,214)'
                )
            ])
            fig.update_layout(
                xaxis_tickangle=-45,
                height=400,
                margin=dict(l=40, r=40, t=10, b=80)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Genre Distribution")
            genre_counts = df["Genre"].value_counts().head(10).reset_index()
            genre_counts.columns = ["Genre", "Count"]
            genre_counts_sorted = genre_counts.sort_values(by="Count", ascending=False)
            
            # Use plotly pie chart for genre distribution
            fig = go.Figure(data=[
                go.Pie(
                    labels=genre_counts_sorted["Genre"],
                    values=genre_counts_sorted["Count"],
                    hole=.3
                )
            ])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Artist stats in a cleaner format
        st.subheader("Top Artists")
        
        # Create a more visually appealing bar chart
        artist_fig = go.Figure(data=[
            go.Bar(
                x=top_artists["Artist"].head(10),
                y=top_artists["Count"].head(10),
                text=top_artists["Count"].head(10),
                textposition='auto',
                marker=dict(
                    color='rgb(158,202,225)',
                    line=dict(color='rgb(8,48,107)', width=1.5)
                )
            )
        ])
        artist_fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Artist",
            yaxis_title="Play Count",
            height=400,
            margin=dict(l=40, r=40, t=40, b=80)
        )
        st.plotly_chart(artist_fig, use_container_width=True)
    

with tab2:
    st.session_state.current_tab = "user"
    st.title("👤 User Personal Dashboard")
    selected_user = st.selectbox("Select a User", df.User.unique().tolist())
    user_df = df[df["User"] == selected_user]
    
    
    # Add music taste breakdown
    st.markdown("---")
    st.subheader("🎭 Music Taste Analysis")

    # Genre distribution
    col1, col2 = st.columns(2)

    with col1:
        genre_counts = user_df["Genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]
        total_listens = genre_counts["Count"].sum()
        genre_counts["Percentage"] = (genre_counts["Count"] / total_listens * 100).round(1)
        
        fig = go.Figure(data=[
            go.Pie(
                labels=genre_counts["Genre"],
                values=genre_counts["Count"],
                textinfo='label+percent',
                insidetextorientation='radial'
            )
        ])
        
        fig.update_layout(
            title="Genre Distribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


    with col2:
        
        # Add user's top artists
        st.write("### Top Artists")
        user_top_artists = user_df["Artist"].value_counts().head(5)
        st.bar_chart(user_top_artists)
    
    user_df["Listen_Date"] = pd.to_datetime(user_df["Listen_Date"])
    st.markdown("---")
    col2a, col2b = st.columns(2)

    with col2a:
        st.write("### Listening Heatmap")
        if "Listen_Date" in user_df.columns:
            user_df["Listen_Date"] = pd.to_datetime(user_df["Listen_Date"])
            user_df_sorted = user_df.sort_values("Listen_Date")
            user_df_sorted["Listen_Date"] = user_df_sorted["Listen_Date"].dt.date
            user_listen_counts = user_df_sorted.groupby("Listen_Date").size().reset_index(name="Count")

            years = user_df["Listen_Date"].dt.year.unique()
            if len(years) > 0:
                selected_year = st.selectbox("Select Year for Heatmap", sorted(years, reverse=True))

                # Filter data for selected year
                user_df_year = user_df[user_df["Listen_Date"].dt.year == selected_year].sort_values("Listen_Date").copy()
                user_df_year["Listen_Date"] = user_df_year["Listen_Date"].dt.date
                user_listen_counts_year = user_df_year.groupby("Listen_Date").size()

                # Generate a date range covering the full year
                date_range = pd.date_range(start=f"{selected_year}-01-01", end=f"{selected_year}-12-31")
                heatmap_data = user_listen_counts_year.reindex(date_range, fill_value=0).reset_index()
                heatmap_data.columns = ["Date", "Count"]

                # Extract week and weekday information
                heatmap_data["Week"] = heatmap_data["Date"].dt.strftime("%U-%b")  # Week of the year format: Jan-1, Jan-2, etc.
                heatmap_data["Weekday"] = heatmap_data["Date"].dt.weekday  # Monday=0, Sunday=6

                # Aggregate data to avoid duplicates (sum counts for same weekday and week)
                heatmap_data_agg = heatmap_data.groupby(["Weekday", "Week"], as_index=False)["Count"].sum()

                # Create a pivot table with weekdays as rows and weeks as columns
                pivot_table = heatmap_data_agg.pivot(index="Weekday", columns="Week", values="Count").fillna(0)

                # Plot the heatmap
                fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 5)) # Adjust figure size as needed
                sns.heatmap(pivot_table, cmap="Greens", linewidths=0.5, ax=ax_heatmap, cbar=True, square=True,
                            xticklabels=True,
                            yticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

                ax_heatmap.set_xlabel("Week of Year")
                ax_heatmap.set_ylabel("Day of Week")
                ax_heatmap.set_title(f"Listening Activity of {selected_user} - {selected_year}")
                st.pyplot(fig_heatmap)
            else:
                st.warning("No listening data with dates available for this user.")
        else:
            st.warning("The 'Listen_Date' column is not available in the data.")

    with col2b:
        st.write("### User-Song Interaction Network")
        df["Song_Artist"] = df["Song"] + " - " + df["Artist"]

        # Step 2: Count number of users per (Song, Artist)
        song_artist_user_count = df.groupby("Song_Artist")["User"].nunique()

        # Step 3: Filter (Song, Artist) pairs with at least 2 users
        valid_songs = song_artist_user_count[song_artist_user_count >= 2].index

        # Step 4: Filter DataFrame to keep only valid (Song, Artist) pairs
        filtered_df = df[df["Song_Artist"].isin(valid_songs)]

        user_songs = filtered_df[filtered_df["User"] == selected_user]["Song_Artist"].unique()

        # Filter interactions where at least one other user has listened to the same song
        sub_df = filtered_df[filtered_df["Song_Artist"].isin(user_songs)]

        if len(sub_df) > 1:
            # Build Network Graph
            G = nx.Graph()
            for _, row in sub_df.iterrows():
                G.add_node(row["User"], type="user")
                G.add_node(row["Song_Artist"], type="song")
                G.add_edge(row["User"], row["Song_Artist"])

            # 3D Network Visualization
            pos_3d = {node: (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) for node in G.nodes()}
            edge_x_3d, edge_y_3d, edge_z_3d = [], [], []
            for edge in G.edges():
                x0, y0, z0 = pos_3d[edge[0]]
                x1, y1, z1 = pos_3d[edge[1]]
                edge_x_3d.extend([x0, x1, None])
                edge_y_3d.extend([y0, y1, None])
                edge_z_3d.extend([z0, z1, None])

            edge_trace_3d = go.Scatter3d(x=edge_x_3d, y=edge_y_3d, z=edge_z_3d,
                                        line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')

            # Create separate traces for each node type
            # Selected user node
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
            
            # Other user nodes
            other_users = [node for node in G.nodes() if G.nodes[node]['type'] == 'user' and node != selected_user]
            if other_users:
                other_user_x = [pos_3d[node][0] for node in other_users]
                other_user_y = [pos_3d[node][1] for node in other_users]
                other_user_z = [pos_3d[node][2] for node in other_users]
                other_user_text = [f"👤 {node}" for node in other_users]
                
                other_user_trace = go.Scatter3d(
                    x=other_user_x, y=other_user_y, z=other_user_z,
                    mode='markers+text',
                    marker=dict(size=12, color='blue', line=dict(width=1)),
                    text=other_user_text,
                    textposition="top center",
                    hoverinfo='text',
                    name='Other Users'
                )
            else:
                other_user_trace = None
            
            # Song nodes
            song_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'song']
            song_x = [pos_3d[node][0] for node in song_nodes]
            song_y = [pos_3d[node][1] for node in song_nodes]
            song_z = [pos_3d[node][2] for node in song_nodes]
            # Trim song names if they're too long
            song_text = [f"🎵 {node[:20]}..." if len(node) > 20 else f"🎵 {node}" for node in song_nodes]
            
            song_trace = go.Scatter3d(
                x=song_x, y=song_y, z=song_z,
                mode='markers+text',
                marker=dict(size=10, color='orange', line=dict(width=1)),
                text=song_text,
                textposition="bottom center",
                hoverinfo='text',
                name='Songs'
            )

            # Combine traces for the figure
            traces = [edge_trace_3d, selected_user_trace, song_trace]
            if other_user_trace:
                traces.append(other_user_trace)

            fig_3d = go.Figure(data=traces,
                            layout=go.Layout(title='3D User-Song Network', 
                                                showlegend=True,
                                                hovermode='closest', 
                                                margin=dict(b=20, l=5, r=5, t=40),
                                                scene=dict(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False))))

            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.warning("No connections found for the selected user to display the network.")

    st.markdown("---")
    col3a, col3b = st.columns(2)
    with col3a:
        
        st.subheader("🎧 Top 3 Recommended Songs")

        # Show personalized recommendations
        if selected_user in list(recommendations['username']):
            st.success(f"Here are your **Top 3 Picks**, *{selected_user}* 🎵")

            # Get the row corresponding to the selected user
            user_row = recommendations[recommendations['username'] == selected_user].iloc[0]
            
            # Extract top 3 recommended songs
            rec_list = [user_row['rec_1'], user_row['rec_2'], user_row['rec_3']]

            # Use columns for cleaner layout
            cols = st.columns(3)
            for i, song in enumerate(rec_list):
                with cols[i]:
                    st.markdown(f"### #{i+1}")
                    song_title = song.split(' - ')[0].strip()
                    artist = song.split(' - ')[1].split('(score')[0].strip()
                    score = song.split('score: ')[1].replace(')', '')
                    
                    st.markdown(f"🎶 **{song_title}**")
                    st.markdown(f"👤 *{artist}*")
                    st.markdown(f"⭐ Score: `{score}`")

        else:
            st.warning(f"🚫 No recommendations found for **{selected_user}**.")

    with col3b:
        long_recs = recommendations.melt(id_vars="username", value_vars=["rec_1", "rec_2", "rec_3"], 
                                    var_name="Rank", value_name="Full_Rec")

        # Step 2: Extract Song and Artist
        long_recs["Song"] = long_recs["Full_Rec"].apply(lambda x: x.split(" - ")[0].strip())
        long_recs["Artist"] = long_recs["Full_Rec"].apply(lambda x: x.split(" - ")[1].split("(score")[0].strip())
        long_recs["Song_Artist"] = long_recs["Song"] + " - " + long_recs["Artist"]
        long_recs.rename(columns={"username": "User"}, inplace=True)

        # Step 3: Count number of users per song
        song_user_count = long_recs.groupby("Song_Artist")["User"].nunique()

        # Step 4: Keep only songs recommended to at least 2 users
        valid_songs = song_user_count[song_user_count >= 2].index
        filtered_df = long_recs[long_recs["Song_Artist"].isin(valid_songs)]

        # Step 5: Filter only songs connected to the selected user
        user_songs = filtered_df[filtered_df["User"] == selected_user]["Song_Artist"].unique()
        sub_df = filtered_df[filtered_df["Song_Artist"].isin(user_songs)]

        # Step 6: Build and visualize network
        st.markdown("### 🌐 User-Song Recommendation Network")

        if len(sub_df) > 1:
            G = nx.Graph()

            for _, row in sub_df.iterrows():
                G.add_node(row["User"], type="user")
                G.add_node(row["Song_Artist"], type="song")
                G.add_edge(row["User"], row["Song_Artist"])

            # Random 3D positions
            pos_3d = {node: (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) for node in G.nodes()}

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
            # Selected user node
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
            
            # Other user nodes
            other_users = [node for node in G.nodes() if G.nodes[node]['type'] == 'user' and node != selected_user]
            if other_users:
                other_user_x = [pos_3d[node][0] for node in other_users]
                other_user_y = [pos_3d[node][1] for node in other_users]
                other_user_z = [pos_3d[node][2] for node in other_users]
                other_user_text = [f"👤 {node}" for node in other_users]
                
                other_user_trace = go.Scatter3d(
                    x=other_user_x, y=other_user_y, z=other_user_z,
                    mode='markers+text',
                    marker=dict(size=12, color='royalblue', line=dict(width=1)),
                    text=other_user_text,
                    textposition="top center",
                    hoverinfo='text',
                    name='Other Users'
                )
            else:
                other_user_trace = None
            
            # Song nodes
            song_nodes = [node for node in G.nodes() if G.nodes[node]['type'] == 'song']
            song_x = [pos_3d[node][0] for node in song_nodes]
            song_y = [pos_3d[node][1] for node in song_nodes]
            song_z = [pos_3d[node][2] for node in song_nodes]
            # Trim song names if they're too long
            song_text = [f"🎵 {node[:20]}..." if len(node) > 20 else f"🎵 {node}" for node in song_nodes]
            
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

            fig_3d = go.Figure(data=traces,
                            layout=go.Layout(
                                title=f'3D Network for {selected_user} & Shared Recommendations',
                                showlegend=True,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                scene=dict(
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                )
                            ))

            st.plotly_chart(fig_3d, use_container_width=True)

        else:
            st.info("No overlapping recommended songs with other users were found.")



with tab3:
    st.session_state.current_tab = "trends"
    st.title("📈 Trend Analysis")
    
    if not pd.api.types.is_datetime64_any_dtype(df["Listen_Date"]):
        df["Listen_Date"] = pd.to_datetime(df["Listen_Date"], errors='coerce')
    
    # Add month and year columns
    df["Month"] = df["Listen_Date"].dt.month
    df["Year"] = df["Listen_Date"].dt.year
    df["YearMonth"] = df["Listen_Date"].dt.strftime("%Y-%m")
    
    # Create monthly trend analysis
    st.header("🔍 Monthly Listening Trends")
    
    # Let user choose what to analyze
    trend_type = st.selectbox("Select trend to analyze:", 
                             ["Genre Popularity", "Artist Popularity"])
    
    date_range = st.date_input("Select date range:", 
                              [df["Listen_Date"].min(), df["Listen_Date"].max()],
                              min_value=df["Listen_Date"].min(),
                              max_value=df["Listen_Date"].max())
    
    # Filter by date range if specified
    if len(date_range) == 2:
        start_date, end_date = [pd.to_datetime(d) for d in date_range]  # Convert to datetime64[ns]
        filtered_df = df[(df["Listen_Date"] >= start_date) & (df["Listen_Date"] <= end_date)]
    else:
        filtered_df = df
    if trend_type == "Genre Popularity":
        # Get top genres for the period
        top_genres = filtered_df["Genre"].value_counts().nlargest(30).index.tolist()
        selected_genres = st.multiselect("Select genres to compare:", top_genres, default=top_genres[:3])
        
        if selected_genres:
            genre_df = filtered_df[filtered_df["Genre"].isin(selected_genres)]
            
            genre_monthly = genre_df.groupby(["YearMonth", "Genre"]).size().reset_index(name="Count")
            
            fig = go.Figure()
            
            for genre in selected_genres:
                genre_data = genre_monthly[genre_monthly["Genre"] == genre]
                if not genre_data.empty:
                    fig.add_trace(go.Scatter(
                        x=genre_data["YearMonth"],
                        y=genre_data["Count"],
                        mode='lines+markers',
                        name=genre
                    ))
            
            fig.update_layout(
                title="Genre Popularity Trends Over Time",
                xaxis_title="Month",
                yaxis_title="Listen Count",
                xaxis_tickangle=-45,
                legend_title="Genre",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a normalized view to see relative popularity
            if st.checkbox("Show normalized trends (percentage)"):
                # Calculate percentage of each genre per month
                monthly_totals = genre_monthly.groupby("YearMonth")["Count"].sum().reset_index()
                genre_monthly = genre_monthly.merge(monthly_totals, on="YearMonth", suffixes=("", "_total"))
                genre_monthly["Percentage"] = (genre_monthly["Count"] / genre_monthly["Count_total"]) * 100
                
                fig_norm = go.Figure()
                
                for genre in selected_genres:
                    genre_data = genre_monthly[genre_monthly["Genre"] == genre]
                    if not genre_data.empty:
                        fig_norm.add_trace(go.Scatter(
                            x=genre_data["YearMonth"],
                            y=genre_data["Percentage"],
                            mode='lines+markers',
                            name=genre
                        ))
                
                fig_norm.update_layout(
                    title="Relative Genre Popularity Trends (% of Monthly Listens)",
                    xaxis_title="Month",
                    yaxis_title="Percentage (%)",
                    xaxis_tickangle=-45,
                    legend_title="Genre",
                    height=500
                )
                
                st.plotly_chart(fig_norm, use_container_width=True)
    
    elif trend_type == "Artist Popularity":
        # Get top artists for the period
        top_artists = filtered_df["Artist"].value_counts().nlargest(10).index.tolist()
        selected_artists = st.multiselect("Select artists to compare:", top_artists, default=top_artists[:3])
        
        if selected_artists:
            artist_df = filtered_df[filtered_df["Artist"].isin(selected_artists)]
            
            artist_monthly = artist_df.groupby(["YearMonth", "Artist"]).size().reset_index(name="Count")
            
            fig = go.Figure()
            
            for artist in selected_artists:
                artist_data = artist_monthly[artist_monthly["Artist"] == artist]
                if not artist_data.empty:
                    fig.add_trace(go.Scatter(
                        x=artist_data["YearMonth"],
                        y=artist_data["Count"],
                        mode='lines+markers',
                        name=artist
                    ))
            
            fig.update_layout(
                title="Artist Popularity Trends Over Time",
                xaxis_title="Month",
                yaxis_title="Listen Count",
                xaxis_tickangle=-45,
                legend_title="Artist",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    

with tab4:
    st.session_state.current_tab = "artist_intelligence"
    st.title("🎵 Artist Intelligence Dashboard")

    # Artist Selection
    all_artists = df["Artist"].unique().tolist()
    selected_artist = st.selectbox("Select Artist to Analyze:", all_artists)
    artist_df = df[df["Artist"] == selected_artist]

    # --- Artist Overview Section ---
    st.header(f"Overview of {selected_artist}")
    col_overview_1, col_overview_2, col_overview_3 = st.columns([1, 2, 1])

    with col_overview_1:
        artist_image = get_artist_image(selected_artist)
        st.image(artist_image, width=200)

        st.subheader("Key Stats")
        total_plays = len(artist_df)
        unique_listeners = artist_df["User"].nunique()
        unique_songs = artist_df["Song"].nunique()
        plays_per_user = artist_df.groupby("User").size().mean()

        st.metric("Total Plays", total_plays)
        st.metric("Unique Listeners", unique_listeners)
        st.metric("Catalog Size", unique_songs)
        st.metric("Avg. Plays per Listener", f"{plays_per_user:.1f}")

    with col_overview_2:
        # Top Songs Chart
        st.subheader("Top Songs")
        top_songs = artist_df["Song"].value_counts().head(10).reset_index()
        top_songs.columns = ["Song", "Plays"]

        fig_top_songs = go.Figure(data=[
            go.Bar(
                x=top_songs["Plays"],
                y=top_songs["Song"],
                orientation='h',
                marker_color='rgba(50, 171, 96, 0.7)'
            )
        ])

        fig_top_songs.update_layout(
            title_text="Top Songs by Play Count",
            xaxis_title="Play Count",
            yaxis_title="Song",
            height=300 
        )
        st.plotly_chart(fig_top_songs, use_container_width=True)

        # Temporal Listening Trend
        if "Listen_Date" in artist_df.columns:
            artist_df["YearMonth"] = artist_df["Listen_Date"].dt.strftime("%Y-%m")
            monthly_listens = artist_df.groupby("YearMonth").size().reset_index(name="Listens")

            fig_monthly_trend = go.Figure()
            fig_monthly_trend.add_trace(go.Scatter(
                x=monthly_listens["YearMonth"],
                y=monthly_listens["Listens"],
                mode='lines+markers',
                line=dict(color='rgba(131, 58, 180, 0.7)', width=2),
                marker=dict(size=8)
            ))

            fig_monthly_trend.update_layout(
                title_text=f"Monthly Listening Trend for {selected_artist}",
                xaxis_title="Month",
                yaxis_title="Listen Count",
                xaxis_tickangle=-45,
                height=350
            )

            st.plotly_chart(fig_monthly_trend, use_container_width=True)
        else:
            st.warning("Listen Date information is not available for temporal analysis.")

    with col_overview_3:
        # Listener Engagement Funnel
        st.subheader("Listener Engagement")
        if "Listen_Date" in artist_df.columns:
            user_listen_counts = artist_df.groupby("User").size().reset_index(name="ListenCount")

            listen_categories = [
                (1, "One-time Listeners"),
                (2, 5, "Casual Listeners"),
                (6, 15, "Regular Listeners"),
                (16, float('inf'), "Super Fans")
            ]

            funnel_data = []
            for category in listen_categories:
                if len(category) == 2:
                    count = len(user_listen_counts[user_listen_counts["ListenCount"] == category[0]])
                    funnel_data.append({"Category": category[1], "Count": count})
                else:
                    count = len(user_listen_counts[(user_listen_counts["ListenCount"] >= category[0]) &
                                                    (user_listen_counts["ListenCount"] <= category[1])])
                    funnel_data.append({"Category": category[2], "Count": count})

            funnel_df = pd.DataFrame(funnel_data)

            fig_engagement_funnel = go.Figure(data=[
                go.Funnel(
                    y=funnel_df["Category"],
                    x=funnel_df["Count"],
                    textinfo="value+percent initial"
                )
            ])

            fig_engagement_funnel.update_layout(
                title_text="Engagement Funnel",
                height=700  # Make it taller to match the combined height of the two charts beside it
            )

            st.plotly_chart(fig_engagement_funnel, use_container_width=True)
        else:
            st.warning("Listen Date information is not available for engagement analysis.")

    st.markdown("---") 

    # --- Audience Insights Section ---
    st.header(f"Audience Insights for {selected_artist}'s Listeners")

    artist_users = artist_df["User"].unique()
    users_listening_data = df[df["User"].isin(artist_users)]

    col_audience_1, col_audience_2 = st.columns(2)

    with col_audience_1:
        st.subheader("Genre Preferences")
        genre_affinity = users_listening_data.groupby("Genre").size().reset_index(name="Count")
        genre_affinity = genre_affinity.sort_values("Count", ascending=False).head(10)

        fig_genre_affinity = go.Figure(data=[
            go.Bar(
                x=genre_affinity["Genre"],
                y=genre_affinity["Count"],
                marker_color='rgba(71, 58, 131, 0.7)'
            )
        ])

        fig_genre_affinity.update_layout(
            title_text="Top Genre Preferences of Audience",
            xaxis_title="Genre",
            yaxis_title="Listen Count",
            xaxis_tickangle=-45,
            height=350 
        )
        st.plotly_chart(fig_genre_affinity, use_container_width=True)

    with col_audience_2:
        st.subheader("Other Artists Liked")
        other_artists = users_listening_data[users_listening_data["Artist"] != selected_artist]["Artist"].value_counts().head(10)
        other_artists = other_artists.reset_index()
        other_artists.columns = ["Artist", "Count"]

        fig_other_artists = go.Figure(data=[
            go.Bar(
                x=other_artists["Artist"],
                y=other_artists["Count"],
                marker_color='rgba(58, 71, 131, 0.7)'
            )
        ])

        fig_other_artists.update_layout(
            title_text="Other Artists Popular Among Fans",
            xaxis_title="Artist",
            yaxis_title="Listen Count",
            xaxis_tickangle=-45,
            height=350 
        )
        st.plotly_chart(fig_other_artists, use_container_width=True)
        
        