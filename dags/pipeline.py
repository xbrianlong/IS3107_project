from airflow import DAG
from airflow.operators.python import PythonOperator
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
import pandas as pd
import pickle
import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data.data_extraction import (
    get_global_top_songs,
    filter_out_global_songs,
    get_top_tracks,
    load_tracks_to_db,
    convert_to_lightgcn_format,
    split_lightgcn_train_test
)

def get_global_top_songs_task(**kwargs):
    global_top_songs_file_path = os.path.join(project_root, 'src', 'lightgcn', 'data', 'temp', 'global_top_songs.pkl')
    top_tracks_file = get_global_top_songs()
    with open(global_top_songs_file_path, 'wb') as f:
        pickle.dump(top_tracks_file, f)
    kwargs['ti'].xcom_push(key='global_top_songs_filepath', value=global_top_songs_file_path)

def get_top_tracks_task(**kwargs):
    # from src.database.db_utils import MusicDB
    # db = MusicDB()
    # output_file = get_top_tracks(db_instance=db)
    # kwargs['ti'].xcom_push(key='top_tracks_filepath', value=output_file)
    users_top_tracks_file_path = os.path.join(project_root, 'src', 'lightgcn', 'data', 'temp', 'users_top_tracks.csv')
    users_top_tracks = get_top_tracks()
    users_top_tracks.to_csv(users_top_tracks_file_path, index=False, header=True)
    kwargs['ti'].xcom_push(key='users_top_tracks_filepath', value=users_top_tracks_file_path)

def filter_top_tracks_task(**kwargs):
    global_top_songs_file_path = kwargs['ti'].xcom_pull(task_ids='get_global_top_songs', key='global_top_songs_filepath')
    users_top_tracks_file_path = kwargs['ti'].xcom_pull(task_ids='get_top_tracks', key='users_top_tracks_filepath')

    filtered_df = filter_out_global_songs(user_data_df_filepath=users_top_tracks_file_path, global_top_filepath=global_top_songs_file_path)
    if not filtered_df.empty:
        output_file = os.path.join(project_root, 'src', 'lightgcn', 'data', 'temp', 'users_top_tracks_filtered.csv')
        filtered_df.to_csv(output_file, index=False, header=True)
        kwargs['ti'].xcom_push(key='filtered_top_tracks_filepath', value=output_file)
        os.remove(global_top_songs_file_path)
        os.remove(users_top_tracks_file_path)
    else:
        print("No valid tracks found after filtering.")

def load_to_db_task(**kwargs):
    from src.database.db_utils import MusicDB
    filtered_top_tracks_file_path = kwargs['ti'].xcom_pull(task_ids='filter_top_tracks', key='filtered_top_tracks_filepath')
    if filtered_top_tracks_file_path:
        filtered_tracks = pd.read_csv(filtered_top_tracks_file_path)
    
        db = MusicDB()
        load_tracks_to_db(tracks_df=filtered_tracks, db_instance=db)
        os.remove(filtered_top_tracks_file_path)
    else:
        print("No filtered top tracks file found. No data loaded into DB.")

def convert_task(**kwargs):
    from src.database.db_utils import MusicDB
    output_path = os.path.join(project_root, 'src', 'lightgcn', 'data', 'music')
    print(f"Writing files to: {output_path}")
    db = MusicDB()
    
    # Get database stats before conversion
    stats = db.get_db_stats()
    print(f"Database stats before conversion: {stats}")
    
    convert_to_lightgcn_format(
        db_instance=db,
        output_dir=output_path
    )

def split_task(**kwargs):
    input_path = os.path.join(project_root, 'src', 'lightgcn', 'data', 'music', 'user_track_interactions.txt')
    output_dir = os.path.join(project_root, 'src', 'lightgcn', 'data', 'music')
    split_lightgcn_train_test(input_path=input_path, output_dir=output_dir)

def train_lgcn(**kwargs):
    import time
    sys.path.append(os.path.abspath("src/lightgcn/code"))

    import world
    import utils
    import model
    import register
    import Procedure

    config = {
        'bpr_batch': 2048,
        'recdim': 64,
        'layer': 3,
        'lr': 0.001,
        'decay': 1e-4,
        'dropout': 0,
        'keep_prob': 0.6,
        'a_fold': 100,
        'testbatch': 100,
        'dataset': 'music',
        'path': os.path.abspath("src/lightgcn/data"),
        'topks': [20],
        'tensorboard': 1,
        'comment': 'lightgcn',
        'load': 0,
        'epochs': 1000,
        'multicore': 0,
        'pretrain': 0,
        'seed': 2020,
        'model': 'lgn',
        'batch_size': 4096,
        'bpr_batch_size': 2048,
        'latent_dim_rec': 64,
        'lightGCN_n_layers': 3,
        'A_n_fold': 100,
        'test_u_batch_size': 100,
        'A_split': False,
        'bigdata': False,
    }

    world.config = config
    world.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world.model_name = config['model']
    world.dataset = config['dataset']
    world.TRAIN_epochs = config['epochs']
    world.PATH = config['path']
    world.topks = config['topks']
    world.LOAD = config['load']
    world.comment = config['comment']
    world.seed = config['seed']

    dataset = register.dataset
    Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
    bpr_loss = utils.BPRLoss(Recmodel, world.config)
    weight_file = utils.getFileName()

    w = SummaryWriter(os.path.join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)) if world.tensorboard else None

    for epoch in range(world.TRAIN_epochs):
        if epoch % 10 == 0:
            Procedure.Test(dataset, Recmodel, epoch, w=w, multicore=world.config['multicore'])

        output_info = Procedure.BPR_train_original(dataset, Recmodel, bpr_loss, epoch, neg_k=1, w=w)
        print(f"[{epoch+1}/{world.TRAIN_epochs}] {output_info}")
        torch.save(Recmodel.state_dict(), weight_file)

    kwargs['ti'].xcom_push(key='trained_model_path', value=weight_file)
    print("Training complete.")

def predict_lgcn(**kwargs):
    weight_file = kwargs['ti'].xcom_pull(task_ids='train_lightgcn', key='trained_model_path')
    sys.path.append(os.path.abspath("src/lightgcn/code"))

    import world
    import model
    import register
    from src.database.db_utils import MusicDB

    db = MusicDB()

    dataset = register.dataset
    Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
    Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
    Recmodel.eval()

    users = torch.arange(dataset.n_users, device=world.device)
    with torch.no_grad():
        scores = Recmodel.getUsersRating(users)

    # Mask already listened songs
    allPos = dataset.allPos  # list of lists: songs listened by each user
    scores = scores.clone()  # Important to avoid modifying original tensor

    for user_id, pos_items in enumerate(allPos):
        if len(pos_items) > 0:
            scores[user_id, pos_items] = -1e9  # Large negative so they won't appear in top-k

    top_k = 5
    top_scores, top_items = torch.topk(scores, k=top_k, dim=1)

    results = []
    for uid, (items, scores) in enumerate(zip(top_items.tolist(), top_scores.tolist())):
        username = db.get_username_by_id(uid + 1)
        if not username:
            print(f"Warning: No username found for user_id {uid + 1}")
            continue

        user_id = db.insert_or_get_user(username)
        user_recs = []

        for iid, score in zip(items, scores):
            song_info = db.get_song_by_id(iid + 1)
            if not song_info:
                print(f"Warning: No song found for song_id {iid + 1}")
                continue

            song_id = db.insert_or_get_song(song_info['name'], song_info['artist'])
            user_recs.append({
                'user_id': user_id,
                'song_id': song_id,
                'rank': 0,
                'score': float(score)
            })

        user_recs.sort(key=lambda x: x['score'], reverse=True)
        for rank, rec in enumerate(user_recs, 1):
            rec['rank'] = rank
            results.append(rec)

    try:
        batch_id = db.save_recommendations(results)
        kwargs['ti'].xcom_push(key='recommendation_batch_id', value=batch_id)
        print(f"[predict_lgcn] Saved {len(results)} recommendations to database with batch_id: {batch_id}")
    except Exception as e:
        print(f"Error saving recommendations: {e}")
        raise

with DAG(
    'IS3107_Project',
    description='Process Last.fm user tracks and train LightGCN',
    schedule_interval=timedelta(days=7),
    start_date=datetime(2025, 4, 1),
    catchup=False,
) as dag:

    global_top_songs = PythonOperator(
        task_id='get_global_top_songs',
        python_callable=get_global_top_songs_task,
        provide_context=True,
    )

    user_top_tracks = PythonOperator(
        task_id='get_top_tracks',
        python_callable=get_top_tracks_task,
        provide_context=True,
    )

    filter_top_tracks = PythonOperator(
        task_id='filter_top_tracks',
        python_callable=filter_top_tracks_task,
        provide_context=True,
    )

    load_to_db = PythonOperator(
        task_id='load_to_db',
        python_callable=load_to_db_task,
        provide_context=True,
    )

    convert_to_lightgcn = PythonOperator(
        task_id='convert_to_lightgcn',
        python_callable=convert_task,
        provide_context=True,
    )

    split_train_test = PythonOperator(
        task_id='split_train_test',
        python_callable=split_task,
        provide_context=True,
    )

    train_task = PythonOperator(
        task_id="train_lightgcn",
        python_callable=train_lgcn,
        provide_context=True,
    )

    predict_lightgcn = PythonOperator(
        task_id="predict_lightgcn",
        python_callable=predict_lgcn,
        provide_context=True,
    )

    [global_top_songs, user_top_tracks] >> filter_top_tracks >> load_to_db >> convert_to_lightgcn >> split_train_test >> train_task >> predict_lightgcn