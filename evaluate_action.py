import json
from glob import glob
from tqdm import tqdm
import os

import numpy as np


data_path = '/mnt/tangzitian/SoccerDatasets/StatsBomb/data' # Please modify this path

def EvaluateActions(game_id):
    """
        Given a game ID, evaluate all the open-play pass, carry, and shot. The results are saved in ./data/action_values/{game_id}.json
    """
    situations = json.load(open(f'data/situation/{game_id}.json'))
    situations[''] = {'situation_id': 0} # event id '' indicates failed action
    xT = np.load('model/xT.npz')['xT']
    events = json.load(open(f'{data_path}/events/{game_id}.json'))
    all_events = {e['id']: e for e in events}
    events_360 = json.load(open(f'{data_path}/three-sixty/{game_id}.json'))
    actions = {}

    for event in events:
        if event['id'] not in situations:
            continue
        situation_id = situations[event['id']]['situation_id']
        if event['type']['name'] == 'Shot':
            if event['shot']['type']['name'] != 'Open Play':
                continue
            outcome = 1 if event['shot']['outcome']['name'] == 'Goal' else 0
            actions[event['id']] = {
                'team': event['team'],
                'player': event['player'],
                'type': 'shot',
                'situation_id': situation_id,
                'xT_add': outcome - xT[situation_id]
            }
        elif event['type']['name'] == 'Pass':
            if 'type' in event['pass']: # not open play
                continue
            ## find the corresponding receipt event
            receipt_id = None
            if 'related_events' in event:
                for eid in event['related_events']:
                    if all_events[eid]['type']['name'] == 'Ball Receipt*':
                        receipt_id = eid

            if receipt_id is None or receipt_id not in situations: # if not corresponding receipt event, deem the pass as failed
                end_event_id = ''
            else:
                if 'ball_receipt' in all_events[receipt_id] or 'outcome' in event['pass']: # the pass failed
                    end_event_id = ''
                else: # the pass succeeded
                    end_event_id = receipt_id
            end_situation_id = situations[end_event_id]['situation_id']
            actions[event['id']] = {
                    'team': event['team'],
                    'player': event['player'],
                    'type': 'pass',
                    'situation_id': situation_id,
                    'end_event_id': end_event_id,
                    'end_situation_id': end_situation_id,
                    'xT_add': xT[end_situation_id]  - xT[situation_id]
                }
        elif event['type']['name'] == 'Carry':
            # find the next event
            eids = {}
            if 'related_events' in event:
                for eid in event['related_events']:
                    if all_events[eid]['timestamp'] > event['timestamp']:
                        eids[all_events[eid]['type']['name']] = eid
            if 'Shot' in eids and eids['Shot'] in situations:
                end_event_id = eids['Shot']
            elif 'Pass' in eids and eids['Pass'] in situations:
                end_event_id = eids['Pass']
            elif 'Miscontrol' in eids or 'Dispossessed' in eids:
                end_event_id = ''
            elif 'Dribble' in eids and eids['Dribble'] in situations:
                end_event_id = eids['Dribble'] if all_events[eids['Dribble']]['dribble']['outcome']['name'] == 'Complete' else ''
            else:
                end_event_id = ''
            
            end_situation_id = situations[end_event_id]['situation_id']
            actions[event['id']] = {
                    'team': event['team'],
                    'player': event['player'],
                    'type': 'carry',
                    'situation_id': situation_id,
                    'end_event_id': end_event_id,
                    'end_situation_id': end_situation_id,
                    'xT_add': xT[end_situation_id]  - xT[situation_id]
                }
    os.makedirs(f'data/action_values', exist_ok=True)
    with open(f'data/action_values/{game_id}.json', 'w') as f:
        f.write(json.dumps(actions, indent=4))
        f.close() 

if __name__ == '__main__':
    for game in tqdm(json.load(open(f'{data_path}/matches/43/106.json'))):
        EvaluateActions(game['match_id'])