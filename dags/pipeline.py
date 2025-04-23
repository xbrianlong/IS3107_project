from airflow import DAG
from airflow.operators.python import PythonOperator
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
import pandas as pd
import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data.data_extraction import (
    get_top_tracks,
    filter_out_global_songs,
    convert_to_lightgcn_format,
    split_lightgcn_train_test
)

#OUTPUT_DIR = os.path.join(project_root, 'outputs')

def get_top_tracks_task(**kwargs):
    output_file = get_top_tracks() # --> get_top_tracks_returns a filepath
    kwargs['ti'].xcom_push(key='top_tracks_filepath', value=output_file)  

def filter_task(**kwargs):
    # Get the output file from the previous task
    top_tracks_file = kwargs['ti'].xcom_pull(task_ids='fetch_top_tracks', key='top_tracks_filepath')
    print(f"Top tracks file path: {top_tracks_file}")
    filtered_out_global_songs= filter_out_global_songs(top_tracks_file)[1]
    #filtered_out_global_songs_df.to_csv(output_filepath, index=False)
    kwargs['ti'].xcom_push(key='filtered_global_songs', value=filtered_out_global_songs)       

def convert_task(**kwargs):
    filtered_file = kwargs['ti'].xcom_pull(task_ids='filter_global_songs', key='filtered_global_songs')
    print(f"Filtered file path: {filtered_file}")
    output_path = os.path.join(project_root, 'lightgcn/data/music')
    convert_to_lightgcn_format(csv_path=filtered_file, output_dir=output_path)

def split_task(**kwargs):
    input_path = os.path.join(project_root, 'lightgcn/data/music/user_track_interactions.txt')
    output_dir = os.path.join(project_root, 'lightgcn/data/music')
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
            'path': os.path.abspath("src/lightgcn/data"),  # Point to src/lightgcn/data
            'topks': [20],  # Match world.py
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
    print(f"Current working directory: {os.getcwd()}")
    print(f"Config path: {config['path']}")
    print(f"Train file exists: {os.path.exists(os.path.join(config['path'], 'music', 'train.txt'))}")
    
    # Update world.config with the custom configuration
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
    
    #setting up model & data
    dataset = register.dataset
    Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
    bpr_loss = utils.BPRLoss(Recmodel, world.config)

    weight_file = utils.getFileName()
    print(f"Training LightGCN model... Saving to {weight_file}")
    print(f"World tensorboard: {world.tensorboard}")

    if world.tensorboard:
        w: SummaryWriter = SummaryWriter(
            os.path.join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")

    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            print(f"[TEST] Epoch {epoch}")
            Procedure.Test(dataset, Recmodel, epoch, w=w, multicore=world.config['multicore'])

        output_info = Procedure.BPR_train_original(dataset, Recmodel, bpr_loss, epoch, neg_k=1, w=w)
        print(f"[{epoch+1}/{world.TRAIN_epochs}] {output_info}")
        torch.save(Recmodel.state_dict(), weight_file)
    kwargs['ti'].xcom_push(key='trained_model_path', value=weight_file)  
    print("Training complete.")

def predict_lgcn(**kwargs):
    weight_file = kwargs['ti'].xcom_pull(task_ids='train_lightgcn', key='trained_model_path')
    print(f"Trained model file path: {weight_file}")
    sys.path.append(os.path.abspath("src/lightgcn/code"))

    import world
    import model
    import register

    
    # loading the dataset and model
    dataset = register.dataset
    Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)

    #loading the trianed weights in checkpoints/ folder
    print(f"[predict_lgcn] Loading weights from: {weight_file}")
    Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
    Recmodel.eval()

    # formulate predictions
    users = torch.arange(dataset.n_users, device=world.device)
    with torch.no_grad():
        scores = Recmodel.getUsersRating(users)

    # get the top 10 songs
    top_k = 10
    top_scores, top_items = torch.topk(scores, k=top_k, dim=1)

    results = []
    for uid, (items, scores) in enumerate(zip(top_items.tolist(), top_scores.tolist())):
        for iid, score in zip(items, scores):
            results.append((uid, iid, score))

    df = pd.DataFrame(results, columns=["user_id", "item_id", "score"])
    out_path = "/opt/airflow/outputs/predictions.csv"       # assuming this is where we want to store the res
    df.to_csv(out_path, index=False)
    print(f"[predict_lgcn] Wrote predictions to {out_path}")



# Initialize the DAG
with DAG(
    'IS3107_Project',
    description='Process Last.fm user tracks and prepare for LightGCN',
    schedule_interval=timedelta(days=7),
    start_date=datetime(2025, 4, 1),
    catchup=False,
) as dag:

    fetch_top_tracks = PythonOperator(
        task_id='fetch_top_tracks',
        python_callable=get_top_tracks_task,
    )

    # Task 2: Filter out global top songs

    filter_global_songs = PythonOperator(
        task_id='filter_global_songs',
        python_callable=filter_task,
    )

    convert_to_lightgcn = PythonOperator(
        task_id='convert_to_lightgcn',
        python_callable=convert_task,
        provide_context=True,
    )

    # Task 4: Split into train/test sets
    split_train_test = PythonOperator(
        task_id='split_train_test',
        python_callable=split_lightgcn_train_test,
    )

    train_task = PythonOperator(
        task_id="train_lightgcn",
        python_callable=train_lgcn,
        provide_context=True,
        dag=dag
    )

    predict_lightgcn = PythonOperator(
        task_id="predict_lightgcn",
        python_callable=predict_lgcn,
        dag=dag,
    )


    # Define task dependencies
    fetch_top_tracks >> filter_global_songs >> convert_to_lightgcn >> split_train_test >> train_task >> predict_lightgcn
