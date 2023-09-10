# Football Situation

**Clustering Football Game Situations via Deep Representation Learning, StatsBomb Conference 2023**

Zitian Tang, Xing Wang, Shaoliang Zhang

<br>

This repository releases the developed model our StatsBomb Conference 2023 work. In this work, we cluster freeze frames of open-play events in StatsBomb 360 data into 95 different situations in football (soccer) games. Moreover, we propose Situation Expected Threat (Situational xT) for action evaluation. This released model is developed using the data of the English Premier League 2021/22. You can use this model in any League you are interested in.

## Quickstart

After modifying the `data_path` in `identify_situation.py`, `evaluate_actoin.py`, and `summarize.py`, you can run our codes to analyze the performances of World Cup 2022 teams with the public [StatsBomb data](https://github.com/statsbomb/open-data).

1. Identify the situation of each event.

   ```
   python identify_situation.py
   ```

   The results will be in `./data/situation`.

2. Evaluate each action with Situational xT.

   ```
   python evaluate_action.py
   ```

   The results will be in `./data/action_values`.

3. Summarize the team performances and drawing heatmaps.

   ```
   python summarize.py
   ```

   The results will be in `./data/summarize`.

If you want to analyze other leagues, please just modify the match list in the codes.

## Citation

We will be happy if you find this model useful. Please cite our work if you use it.

```
@inproceedings{ZitianStatsBomb2023,
      title = {Clustering Football Game Situations via Deep Representation Learning}, 
      author = {
      	Zitian Tang and
      	Xing Wang and
      	Shaoliang Zhang},
      booktitle = {StatsBomb},
      year = {2023}
}
```

