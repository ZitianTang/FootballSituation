import torch
import numpy as np
import json

LATTICE = []
for x in range(64):
    for y in range(96):
        LATTICE.append((x + 0.5, y + 0.5))
LATTICE = np.array(LATTICE)

def draw_position(p):
    p = np.array(p)
    A = np.zeros((64, 96), dtype=int)
    if len(p) > 0:
        A[p[:, 0], p[:, 1]] = 1
    return A

def pos_players(event):
    """
        Positions of teammates and opponents
    """
    teammates = []
    opponents = []
    for p in event['freeze_frame']:
        x, y = p['location'][1], p['location'][0]
        if x < 0 or x > 80 or y < 0 or y > 120:
            continue
        x = int(x / 80 * 64)
        y = int(y / 120 * 96)
        if x == 64:
            x = 63
        if y == 96:
            y = 95
        if p['teammate']:
            teammates.append((x, y))
        else:
            opponents.append((x, y))
    return np.stack([
        draw_position(teammates),
        draw_position(opponents)
    ])

def pos_actor(event):
    """
        Position of the event actor
    """
    actor = []
    for p in event['freeze_frame']:
        x, y = p['location'][1], p['location'][0]
        if x < 0 or x > 80 or y < 0 or y > 120:
            continue
        x = int(x / 80 * 64)
        y = int(y / 120 * 96)
        if x == 64:
            x = 63
        if y == 96:
            y = 95
        if p['actor']:
            actor.append((x, y))
    return draw_position(actor)[None, ...]

def pos_keeper(event):
    """
        Positions of keepers
    """
    keeper = []
    for p in event['freeze_frame']:
        x, y = p['location'][1], p['location'][0]
        if x < 0 or x > 80 or y < 0 or y > 120:
            continue
        x = int(x / 80 * 64)
        y = int(y / 120 * 96)
        if x == 64:
            x = 63
        if y == 96:
            y = 95
        if p['keeper']:
            keeper.append((x, y))
    return draw_position(keeper)[None, ...]

def vis_area(event):
    """
        Visible area
    """
    points = np.array(event['visible_area']).reshape(-1, 2)[::-1, ::-1] / 80 * 64
    lattice = LATTICE
    v = lattice[:, None] - points[None, :-1] # n_points x n_lat x 2
    u = (points[1:] - points[:-1])[None, ...] # 1 x n_lat x 2
    prod = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0] # n_points x n_lat
    A = (prod>=0).all(axis=-1).astype(float)
    return A.reshape(1, 64, 96)

def position2ball(event):
    """
        x and y differences to the event actor, distance to the actor
    """
    actor = []
    for p in event['freeze_frame']:
        x, y = p['location'][1], p['location'][0]
        # x = max(x, 0)
        # x = min(x, 80)
        # y = max(y, 0)
        # y = min(y, 120)
        x = int(x / 80 * 64)
        y = int(y / 120 * 96)
        if x == 64:
            x = 63
        if y == 96:
            y = 95
        if p['actor']:
            actor.append((x, y))
            break
    if len(actor) == 0:
        actor = [(32, 48)]
    b = np.array(actor)
    lattice = LATTICE
    dif_x = (lattice[:, 0] - b[:, 0]).reshape(1, 64, 96)
    dif_y = (lattice[:, 1] - b[:, 1]).reshape(1, 64, 96)
    dis = np.sqrt(((lattice - b) ** 2).sum(-1)).reshape(1, 64, 96)
    return np.concatenate([dif_x, dif_y, dis], axis=0) / 64

def position2goal(event):
    """
        x and y differences to the opponent's goal, distance to the opponent's goal
    """
    b = np.array([[32, 96]])
    lattice = LATTICE
    dif_x = (lattice[:, 0] - b[:, 0]).reshape(1, 64, 96)
    dif_y = (lattice[:, 1] - b[:, 1]).reshape(1, 64, 96)
    dis = np.sqrt(((lattice - b) ** 2).sum(-1)).reshape(1, 64, 96)
    return np.concatenate([dif_x, dif_y, dis], axis=0) / 64

def FreezeFrame2EncoderInput(event):
    """
        Convert a freeze frame to the format of encoder input
    """
    x = np.concatenate([
        pos_players(event),
        pos_actor(event),
        pos_keeper(event),
        vis_area(event),
        position2ball(event),
        position2goal(event)
    ], axis=0)
    return torch.FloatTensor(x)

class AEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        freeze_frames
    ):
        super().__init__()
        self.freeze_frames = freeze_frames

    def __len__(self):
        return len(self.freeze_frames)

    def __getitem__(self, idx):
        return FreezeFrame2EncoderInput(self.freeze_frames[idx])
        

