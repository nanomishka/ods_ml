import numpy as np
import os
import pandas as pd
import seaborn as sns
import time

from contextlib import contextmanager
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


PATH_TO_DATA = '../../data/'
AUTHOR = 'Yury_Kashnitskiy'
SEED = 17
N_JOBS = 4
NUM_TIME_SPLITS = 10    # for time-based cross-validation
# TF_IDF
SITE_NGRAMS = (1, 2)    # site ngrams for "bag of sites"
DF_MAX = 0.8
DF_MIN = 2
MAX_FEATURES = 50000    # max features for "bag of sites"
BEST_LOGIT_C = 5.45559  # precomputed tuned C for logistic regression

 

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def prepare_sparse_features(path_to_train, path_to_test, vectorizer_params):
    times = ['time%s' % i for i in range(1, 11)]
    sites = ['site%s' % i for i in range(1, 11)]

    train_df = pd.read_csv(path_to_train, index_col='session_id', parse_dates=times)
    test_df = pd.read_csv(path_to_test, index_col='session_id', parse_dates=times)

    # Sort the data by time
    train_df = train_df.sort_values(by='time1')

    train_df[sites].fillna(0).astype('int').to_csv('train_sessions_text.txt', sep=' ', index=None, header=None)
    test_df[sites].fillna(0).astype('int').to_csv('test_sessions_text.txt', sep=' ', index=None, header=None)

    vectorizer = TfidfVectorizer(**vectorizer_params)
    with open('train_sessions_text.txt') as inp_train_file:
        X_train = vectorizer.fit_transform(inp_train_file)
    with open('test_sessions_text.txt') as inp_test_file:
        X_test = vectorizer.transform(inp_test_file)

    y_train = train_df['target'].astype('int').values
    
    train_times, test_times = train_df[times], test_df[times]
    
    return X_train, X_test, y_train, vectorizer, train_times, test_times


def add_features(times, X_sparse):
    hour = times['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    
    # sess_duration = (times.max(axis=1) - times.min(axis=1)).astype('timedelta64[s]')\
		  #  .astype('int').values.reshape(-1, 1)
    # day_of_week = times['time1'].apply(lambda t: t.weekday()).values.reshape(-1, 1)
    # month = times['time1'].apply(lambda t: t.month).values.reshape(-1, 1) 
    # year_month = times['time1'].apply(lambda t: 100 * t.year + t.month).values.reshape(-1, 1) / 1e5

    X = hstack([
        X_sparse,
        *[f.values.reshape(-1, 1) for f in
            (hour, morning, day, evening)
        ]])
    
    return X


with timer('Building sparse site features'):
    X_train_sites, X_test_sites, y_train, vectorizer, train_times, test_times = \
        prepare_sparse_features(
            path_to_train=os.path.join(PATH_TO_DATA, 'train_sessions.csv'),
            path_to_test=os.path.join(PATH_TO_DATA, 'test_sessions.csv'),
            vectorizer_params={'ngram_range': SITE_NGRAMS,
                               'max_features': MAX_FEATURES,
                               'max_df': DF_MAX,
                               'min_df': DF_MIN})


with timer('Building additional features'):
    X_train_final = add_features(train_times, X_train_sites)
    X_test_final = add_features(test_times, X_test_sites)


# with timer('Cross-validation'):
#     time_split = TimeSeriesSplit(n_splits=NUM_TIME_SPLITS)
#     logit = LogisticRegression(random_state=SEED, solver='liblinear')

#     # I've done cross-validation locally, and do not reproduce these heavy computations here,
#     # but this is the vest C that I've found
#     c_values = [BEST_LOGIT_C]

#     logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values},
#                                   scoring='roc_auc', n_jobs=N_JOBS, cv=time_split, verbose=1)
#     logit_grid_searcher.fit(X_train_final, y_train)
#     print('CV score', logit_grid_searcher.best_score_)


# with timer('Test prediction and submission'):
#     test_pred = logit_grid_searcher.predict_proba(X_test_final)[:, 1]
#     pred_df = pd.DataFrame(test_pred, index=np.arange(1, test_pred.shape[0] + 1),
#                        columns=['target'])
#     pred_df.to_csv(f'submission_alice_{AUTHOR}.csv', index_label='session_id')

