import os
import collections
import pandas as pd
import numpy as np
import json


"""This script loads CSV data, adds a couple of new features from raw data in json"""
OUTPUT_NAME = 'data_prep'

# read raw data in json
def read_matches(matches_file):
    _, filename = os.path.split(matches_file)

    with open(matches_file) as fin:
        for line in fin:
            yield json.loads(line)


# function to compute new features with other func and concat them to the original df
def create_new_feat_df(matches_file, feat_func):
    new_features = []
    for match in read_matches(matches_file):
        new_feat = feat_func(match)
        new_features.append(new_feat)

    new_features_df = pd.DataFrame.from_records(new_features).set_index('match_id_hash')

    return new_features_df


# compute additional features proposed by organizers
def add_org_features(df_features, matches_file):
    # Process raw data and add new features
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']

        # Counting ruined towers for both teams
        radiant_tower_kills = 0
        dire_tower_kills = 0
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':
                if objective['team'] == 2:
                    radiant_tower_kills += 1
                if objective['team'] == 3:
                    dire_tower_kills += 1

        # Write new features
        df_features.loc[match_id_hash, 'Radiant_tower_kills'] = radiant_tower_kills
        df_features.loc[match_id_hash, 'Dire_tower_kills'] = dire_tower_kills
        df_features.loc[match_id_hash, 'Diff_tower_kills'] = radiant_tower_kills - dire_tower_kills
        if (radiant_tower_kills + dire_tower_kills) > 0:
            df_features.loc[match_id_hash, 'Diff2_tower_kills'] = (radiant_tower_kills - dire_tower_kills) \
                                                                  / (radiant_tower_kills + dire_tower_kills)
        else:
            df_features.loc[match_id_hash, 'Diff2_tower_kills'] = 0


# adding more player features
def add_new_features_1(match):
    row = [
        ('match_id_hash', match['match_id_hash']),
    ]

    for slot, player in enumerate(match['players']):

        if slot < 5:
            player_name = 'r%d' % (slot + 1)
        else:
            player_name = 'd%d' % (slot - 4)

        row.append((f'{player_name}_multi_kill_num', len(player['multi_kills'])))
        row.append((f'{player_name}_nearby_creep_death_count', player['nearby_creep_death_count']))

        gold_t, xp_t, ts = player['gold_t'], player['xp_t'], player['times']
        gold_vel_mean = np.mean(np.diff(gold_t) / np.diff(ts))
        xp_vel_mean = np.mean(np.diff(xp_t) / np.diff(ts))

        row.append((f'{player_name}_gold_vel', gold_vel_mean))
        row.append((f'{player_name}_xp_vel', xp_vel_mean))


    return collections.OrderedDict(row)


def comp_team_features(df, feat_list, func_list=('mean', 'std', 'sum', 'median')):
    # for each feature in feat_list compute statistic from func_list for each team
    player_num_cols = []
    for name in feat_list:
        for f in func_list:
            for team in ['r', 'd']:
                # select columns for each player of each team with feature 'name'
                cols = [f'{team}{i}_{name}' for i in range(1, 6)]
                player_num_cols.extend(cols)
                df[f'{team}_{name}__{f}'] = df[cols].agg(f, axis=1)

    df.drop(columns=player_num_cols, inplace=True)
    return df


def add_ratios(df, feat_list, func_list=('mean', 'std', 'sum', 'median')):
    for name in feat_list:
        for f in func_list:
            df[f'rd_{name}__{f}__ratio'] = (df[f'r_{name}__{f}'] - df[f'd_{name}__{f}']) / \
            (df[f'd_{name}__{f}'] + df[f'r_{name}__{f}'])
    return df


def main(PATH_TO_DATA):

    # Read original data
    train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), index_col='match_id_hash')
    test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), index_col='match_id_hash')

    # Org features
    add_org_features(train_df, os.path.join(PATH_TO_DATA, 'train_matches.jsonl'))
    add_org_features(test_df, os.path.join(PATH_TO_DATA, 'test_matches.jsonl'))

    # Other features need a bit more code, since we need to compute
    # team-based features out of player-based ones
    nf_train_df = create_new_feat_df(os.path.join(PATH_TO_DATA, 'train_matches.jsonl'), add_new_features_1)
    nf_test_df = create_new_feat_df(os.path.join(PATH_TO_DATA, 'test_matches.jsonl'), add_new_features_1)
    train_df = pd.concat((train_df, nf_train_df), axis=1)
    test_df = pd.concat((test_df, nf_test_df), axis=1)

    player_num_feats = [col.split('_', 1)[1]
                        for col in train_df.columns
                        if col.startswith('r')
                        if not col.endswith('hero_id')]

    for df in [train_df, test_df]:
        df = comp_team_features(df, player_num_feats, func_list=('mean', 'std', 'sum', 'median'))

    # add rations
    for df in [train_df, test_df]:
        df = add_ratios(df, player_num_feats, func_list=('mean', 'std', 'sum', 'median'))

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    # OHE of hero_ids
    rheros_cols = [f'r{i}_hero_id' for i in range(1, 6)]
    r_heros_ohe_train = 0
    for col in rheros_cols:
        r_heros_ohe_train += pd.get_dummies(train_df[col])
    r_heros_ohe_train.columns = [f'r_hero_{col}' for col in r_heros_ohe_train.columns]

    r_heros_ohe_test = 0
    for col in rheros_cols:
        r_heros_ohe_test += pd.get_dummies(test_df[col])
    r_heros_ohe_test.columns = [f'r_hero_{col}' for col in r_heros_ohe_test.columns]

    dheros_cols = [f'd{i}_hero_id' for i in range(1, 6)]
    d_heros_ohe_train = 0
    for col in dheros_cols:
        d_heros_ohe_train += pd.get_dummies(train_df[col])
    d_heros_ohe_train.columns = [f'd_hero_{col}' for col in d_heros_ohe_train.columns]

    d_heros_ohe_test = 0
    for col in dheros_cols:
        d_heros_ohe_test += pd.get_dummies(test_df[col])
    d_heros_ohe_test.columns = [f'd_hero_{col}' for col in d_heros_ohe_test.columns]

    # drop useless columns
    train_df.drop(columns=rheros_cols+dheros_cols, inplace=True)
    test_df.drop(columns=rheros_cols+dheros_cols, inplace=True)

    # concat OHE hero_ids
    train_df = pd.concat((r_heros_ohe_train, d_heros_ohe_train, train_df), axis=1)
    test_df = pd.concat((r_heros_ohe_test, d_heros_ohe_test, test_df), axis=1)

    # Let's find a projection of mean coordinate to the diagonal
    r_pos = np.dot(train_df[['r_x__mean', 'r_y__mean']].values,
                   np.full((2,1), 1/np.sqrt(2))).reshape(-1)
    d_pos = np.dot(train_df[['d_x__mean', 'd_y__mean']].values,
                   np.full((2,1), 1/np.sqrt(2))).reshape(-1)
    r_pos_test = np.dot(test_df[['r_x__mean', 'r_y__mean']].values,
                   np.full((2,1), 1/np.sqrt(2))).reshape(-1)
    d_pos_test = np.dot(test_df[['d_x__mean', 'd_y__mean']].values,
                   np.full((2,1), 1/np.sqrt(2))).reshape(-1)

    train_df['r_pos'] = r_pos
    train_df['d_pos'] = d_pos
    test_df['r_pos'] = r_pos_test
    test_df['d_pos'] = d_pos_test

    # save our processed data
    train_df.to_csv(os.path.join(PATH_TO_DATA, f'{OUTPUT_NAME}_train.csv'), index='match_id_hash')
    test_df.to_csv(os.path.join(PATH_TO_DATA, f'{OUTPUT_NAME}_test.csv'), index='match_id_hash')
