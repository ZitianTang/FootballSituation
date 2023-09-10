import numpy as np
import torch
import json
import os

from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm

from model.dataprocess import FreezeFrame2EncoderInput, AEDataset
from model.zone import Zone

data_path = '/mnt/tangzitian/SoccerDatasets/StatsBomb/data' # Please modify this path
model = torch.load('./model/encoder.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
centers = [np.load(f'model/cluster_centers/zone_{i}.npy') for i in range(1, 31)]
n = np.cumsum([0] + [len(c) for c in centers])


def situation(event, feature):
    """
        Input:
         - event: 360 data sample
         - feature: encoder feature of the sample
        Output:
         - zone_id: which zone the event locates in (1~30)
         - cluster_id: which cluster of this zone the event belongs to (1~)
         - situation_id: the overall situation id (1~95)
    """
    zone_id = None
    for p in event['freeze_frame']:
        if p['actor']:
            zone_id = Zone(p['location'])
            break
    if zone_id is None:
        return None
    dists = (centers[zone_id - 1] - feature[None, :]) ** 2
    dists = dists.sum(axis=-1)
    cluster_id = dists.argmin() + 1
    situation_id = n[zone_id - 1] + cluster_id
    return {
        'zone_id': int(zone_id),
        'cluster_id': int(cluster_id),
        'situation_id': int(situation_id)
    }

@torch.no_grad()
def ProcessSingleFrame(event):
    x = FreezeFrame2EncoderInput(event).to(device)
    feature = model(x.unsqueeze(0))
    feature = feature.cpu().numpy()[0]
    return situation(event, feature)

@torch.no_grad()
def ProcessGame(game_id, batch_size=512, num_workers=32):
    """
        Given a game ID, compute the situation id of all the freeze frames and save them in ./data/situation/{game_id}.json
    """
    os.makedirs('data/situation', exist_ok=True)
    freeze_frames = json.load(open(f'{data_path}/three-sixty/{game_id}.json'))
    dataset = AEDataset(freeze_frames)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    situations = {}
    for step, x in enumerate(dataloader):
        feature = model(x.to(device))
        feature = feature.cpu().numpy()
        for i in range(len(feature)):
            event = freeze_frames[step * batch_size + i]
            sit = situation(event, feature[i])
            if sit is not None:
                situations[event['event_uuid']] = sit
    with open(f'data/situation/{game_id}.json', 'w') as f:
        f.write(json.dumps(situations, indent=4))
        f.close()

if __name__ == '__main__':
    for game in tqdm(json.load(open(f'{data_path}/matches/43/106.json'))):
        ProcessGame(game['match_id'])