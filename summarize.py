import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import os

data_path = '/mnt/tangzitian/SoccerDatasets/StatsBomb/data' # Please modify this path

def draw_heatmap(data, fn, x_labels, y_labels, cmap='Reds', unit=None):
    if unit is None:
        unit = 0
        while np.abs(data).max() >= 10:
            unit += 1
            data /= 10
        while np.abs(data).max() < 1:
            unit -= 1
            data *= 10
    else:
        if unit >= 0:
            data *= np.power(10, unit)
        else:
            data /= np.power(10, -unit)
    if cmap == 'Reds':
        vmax = None
        vmin = 0
    else:
        vmax = np.abs(data).max()
        vmin = -vmax
    plt.figure(figsize=(12, data.shape[0] * 2 // 3), dpi=600)
    plt.imshow(data, cmap=cmap, vmax=vmax, vmin=vmin)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, '%.2f'%data[i, j],
                ha="center", va="center", color="black", fontdict={'size':9})

    plt.xticks(np.arange(data.shape[1]), x_labels)
    plt.yticks(np.arange(data.shape[0]), y_labels)
    plt.title(f'Unit: 10^{unit}')
    plt.savefig(fn)
    plt.close()

def GetTeams(matches):
    """
        List all the teams in the provided matches
    """
    teams = {}
    for m in matches:
        teams[m['home_team']['home_team_id']] = {
            'id': m['home_team']['home_team_id'],
            'name': m['home_team']['home_team_name']
        }
        teams[m['away_team']['away_team_id']] = {
            'id': m['away_team']['away_team_id'],
            'name': m['away_team']['away_team_name']
        }
    return [teams[k] for k in teams]

def summarize_team(team_id, game_ids):
    """
        Summarize the performance of the team in the games
        Make sure that the team is involved in all the provided games, otherwise the result will be wrong
    """
    situation_cnt = [0] * 96
    situation_xT_add = [0] * 96
    conceded_situation_cnt = [0] * 96
    conceded_situation_xT_add = [0] * 96
    for game_id in game_ids:
        events = json.load(open(f'data/action_values/{game_id}.json'))
        for event_id in events:
            e = events[event_id]
            if e['team']['id'] == team_id:
                situation_cnt[e['situation_id']] += 1
                situation_xT_add[e['situation_id']] += e['xT_add']
            else:
                conceded_situation_cnt[e['situation_id']] += 1
                conceded_situation_xT_add[e['situation_id']] += e['xT_add']
    return {
        'n_games': len(game_ids),
        'situation_cnt': situation_cnt,
        'situation_xT_add': situation_xT_add,
        'conceded_situation_cnt': conceded_situation_cnt,
        'conceded_situation_xT_add': conceded_situation_xT_add
    }

def summarize(save_name, matches, teams=None):
    """
        Summarize the performance of the teams in the given games
        If 'teams' is None, it will be all the teams involved in the given games
        The results are saved in ./data/summarize/{save_name}
    """
    if teams is None:
        teams = GetTeams(matches)
    s = []
    for team in tqdm(teams):
        # find all games involving the team
        game_ids = []
        for m in matches:
            if m['home_team']['home_team_id'] == team['id'] or m['away_team']['away_team_id'] == team['id']:
                game_ids.append(m['match_id'])
        
        d = summarize_team(team['id'], game_ids)
        team.update(d)
        s.append(team)
    os.makedirs(f'data/summarize/{save_name}', exist_ok=True)
    with open(f'data/summarize/{save_name}/data.json', 'w') as f:
        f.write(json.dumps(s, indent=4))
        f.close()

    ## draw heatmaps
    print('Visualizing...')
    subfigures = [
        ('Zone_1-3', list(range(1, 7))),
        ('Zone_4-8', list(range(7, 22))),
        ('Zone_9-13', list(range(22, 37))),
        ('Zone_14-18', list(range(37, 54))),
        ('Zone_19-23', list(range(54, 74))),
        # ('Zone_24-30', list(range(74, 95))),
        ('Corner', list(range(74, 80)) + list(range(90, 96))),
        ('Penalty_Area', list(range(80, 90)))
    ]
    n_games = np.array([t['n_games'] for t in s])
    sit_cnt = np.array([t['situation_cnt'] for t in s]) / n_games[:, None]
    sit_xT = np.array([t['situation_xT_add'] for t in s]) / n_games[:, None]
    con_sit_cnt = np.array([t['conceded_situation_cnt'] for t in s]) / n_games[:, None]
    con_sit_xT = np.array([t['conceded_situation_xT_add'] for t in s]) / n_games[:, None]
    sit_names = json.load(open('model/names.json'))
    team_names = [t['name'] for t in s]
    os.makedirs(f'data/summarize/{save_name}/offensive', exist_ok=True)
    os.makedirs(f'data/summarize/{save_name}/defensive', exist_ok=True)
    for title, indices in subfigures:
        x = [sit_names[i] for i in indices]
        indices = np.array(indices)
        draw_heatmap(
            data = sit_cnt[:, indices],
            fn = f'data/summarize/{save_name}/offensive/{title}_situation_cnt.jpg',
            x_labels = x,
            y_labels = team_names,
            cmap = 'Reds',
            unit = 0
        )
        draw_heatmap(
            data = sit_xT[:, indices],
            fn = f'data/summarize/{save_name}/offensive/{title}_xT_add.jpg',
            x_labels = x,
            y_labels = team_names,
            cmap = 'seismic'
        )
        draw_heatmap(
            data = con_sit_cnt[:, indices],
            fn = f'data/summarize/{save_name}/defensive/{title}_conceded_situation_cnt.jpg',
            x_labels = x,
            y_labels = team_names,
            cmap = 'Reds',
            unit = 0
        )
        draw_heatmap(
            data = con_sit_xT[:, indices],
            fn = f'data/summarize/{save_name}/defensive/{title}_conceded_xT_add.jpg',
            x_labels = x,
            y_labels = team_names,
            cmap = 'seismic'
        )

if __name__ == '__main__':
    matches = json.load(open(f'{data_path}/matches/43/106.json'))
    summarize('World_Cup_2022', matches)