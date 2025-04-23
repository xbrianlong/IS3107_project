import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import json

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()

    # === LOAD MAPPINGS ===
    try:
        with open("..src/lightgcn/data/music/id2user.json") as f: # might have to double check dir paths
            id2user = json.load(f)
        with open("..src/lightgcn/data/music/id2track.json") as f: # might have to double check dir paths
            id2track = json.load(f)
    except Exception as e:
        print("❌ Failed to load id2user or id2track mapping files:", e)
        id2user, id2track = {}, {}

    # === GENERATE TOP-3 RECOMMENDATIONS ===
    print("\n📦 Generating Top-3 Recommendations Per User with Scores...\n")
    Recmodel.eval()
    with torch.no_grad():
        users = torch.arange(dataset.n_users).to(world.device)
        rating_matrix = Recmodel.getUsersRating(users)

        # Mask items users have already interacted with
        for user in users:
            pos_items = dataset.getUserPosItems([user.item()])[0]
            rating_matrix[user, pos_items] = -1e9  # mask them

        top_k = 3
        top_scores, top_indices = torch.topk(rating_matrix, k=top_k)
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

    # === OUTPUT ===
    for user_id, (item_ids, scores) in enumerate(zip(top_indices, top_scores)):
        username = id2user.get(str(user_id), f"User_{user_id}")
        tracks = [id2track.get(str(i), f"Item_{i}") for i in item_ids]
        formatted = [f"{t} (score: {s:.3f})" for t, s in zip(tracks, scores)]
        print(f"👤 {username} ➜ {formatted}")

    # === OPTIONAL: Save to file ===
    with open("top3_recommendations.txt", "w") as f:
        for user_id, (item_ids, scores) in enumerate(zip(top_indices, top_scores)):
            username = id2user.get(str(user_id), f"User_{user_id}")
            tracks = [id2track.get(str(i), f"Item_{i}") for i in item_ids]
            formatted = [f"{t} (score: {s:.3f})" for t, s in zip(tracks, scores)]
            f.write(f"{username}: {formatted}\n")
    print("✅ Saved top-3 recommendations to top3_recommendations.txt")