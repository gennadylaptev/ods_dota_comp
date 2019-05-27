import os
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def train_catboost(X_train, y_train, folds, params,
                   verbose=100, early_stop=100, metric=roc_auc_score):
    models = []
    cv_scores = []

    train_preds = np.zeros((X_train.shape[0]))

    for n_fold, (train_ind, valid_ind) in enumerate(folds.split(X_train, y_train)):
        print(f'Fold {n_fold+1}')

        model = CatBoostClassifier(**params)
        model.fit(X_train[train_ind], y_train[train_ind],
                  eval_set=(X_train[valid_ind], y_train[valid_ind]),
                  verbose_eval=verbose,
                  early_stopping_rounds=early_stop)
        models.append(model)
        train_preds[valid_ind] = model.predict_proba(X_train[valid_ind])[:,1]
        cv_score = metric(y_train[valid_ind], train_preds[valid_ind])
        cv_scores.append(cv_score)

    cv_scores = np.array(cv_scores)
    print(f'CV mean: {cv_scores.mean():.5f}, CV std: {cv_scores.std():.5f}')
    return models, cv_scores, train_preds


def make_preds_cat(models, X_test):
    preds = 0
    for model in models:
        preds += model.predict_proba(X_test)[:,1]
    preds /= len(models)

    return preds


def main(PATH_TO_DATA):
    train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'data_prep_train.csv'), index_col='match_id_hash')
    test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'data_prep_test.csv'), index_col='match_id_hash')
    train_df_F = pd.read_csv(os.path.join(PATH_TO_DATA, 'data_prep_flip_train.csv'), index_col='match_id_hash')
    train_df_F = train_df_F[train_df.columns]
    train_df = train_df.reset_index(drop=True)
    train_df_F = train_df_F.reset_index(drop=True)

    train_df = pd.concat((train_df, train_df_F), axis=0, sort=True)
    test_df = test_df[train_df.columns]

    target = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), index_col='match_id_hash')
    y = target['radiant_win'].values.astype(np.int32)
    y_F = np.abs(y - 1)
    y = np.hstack((y, y_F))

    cat_cols_game = ['game_mode', 'lobby_type']
    train_df.drop(columns=cat_cols_game, inplace=True)
    test_df.drop(columns=cat_cols_game, inplace=True)

    folds = StratifiedKFold(n_splits=10, random_state=17, shuffle=True)
    X_train = train_df.values

    catboost_params = {
                        'bagging_temperature': 1.03,
                        'border_count': 254,
                        'l2_leaf_reg': 25,
                        'random_strength': 10.0,
                        'rsm': 0.85,
                        'iterations': 10000,
                        'learning_rate': 0.03,
                        'depth': 4,
                        'loss_function': 'Logloss',
                        'eval_metric': 'AUC',
                        'use_best_model': True,
                        'random_seed': 1,
                        'thread_count': 12,
    }

    cat_models, cat_cv_scores, cat_train_preds = train_catboost(X_train, y, folds, catboost_params,
                                                                early_stop=500, verbose=False)

    X_test = test_df.values
    sub_preds = make_preds_cat(cat_models, X_test)
    submit = pd.read_csv(os.path.join(PATH_TO_DATA, 'sample_submission.csv'))
    submit['radiant_win_prob'] = sub_preds
    submit.to_csv('submission.csv', index=False)