#!/usr/bin/env python
from neurocombat_sklearn import CombatModel
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             explained_variance_score, r2_score)

# reading given csv file
filename = '~/resilience/2yfu_partial.csv'
abcd = pd.read_csv(filename)

new_abcd = abcd.dropna()

new_abcd.columns = new_abcd.columns.astype(str)
X = new_abcd.drop(columns=['src_subject_id','eventname','site_id_l','rel_family_id', 'cbcl_scr_syn_totprob_r'], axis=1)

covars = new_abcd[['site_id_l']]

# Creating model
model = CombatModel()

# Fitting model
# make sure that your inputs are 2D, e.g. shape [n_samples, n_discrete_covariates]
model.fit(X, covars[['site_id_l']])

# Harmonize data
# could be performed together with fitt by using .fit_transform method
data_combat = model.transform(X, covars[['site_id_l']])

X = data_combat
X = pd.DataFrame(X)

y = pd.DataFrame(new_abcd, columns=['cbcl_scr_syn_totprob_r'])
groups = pd.DataFrame(new_abcd, columns=['rel_family_id'])

# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
# Define a Standard Scaler to normalize inputs
scaler = StandardScaler()
estimator = RandomForestRegressor(criterion='squared_error',
                                  n_jobs=1, verbose=1, random_state=1)
pipe = Pipeline(steps=[("scaler", scaler), ("estimator", estimator)])
#Conducting LOSOCV
cv = GroupKFold(n_splits=10)
# Parameters of pipelines can be set using '__' separated parameter names:
param_grid = {'estimator__max_depth': [5, 10, 20, 40, None],
              'estimator__max_features': [1, 5, 'log2', 'sqrt', None]}
grid_search = GridSearchCV(pipe, param_grid=param_grid,
                           cv=5, verbose=1)
metrics = []
data = []
data_generalization = []
def predict_collect_save(data_pred, data_collect, y_true, test_index,
                         split, save_type):
    scores = {}
    pred_ = grid_search.predict(data_pred)
    y_true_ = y_true.iloc[test_index]
    predictions = pd.DataFrame(pred_, columns=['predicted'],
			       index=y_true_.index)
    predictions['true'] = y_true_
    predictions['test_indices'] = pd.DataFrame(test_index,
					       columns=['test_indices'],
					       index=y_true_.index)
    predictions['fold'] = pd.Series([split] * len(predictions),
				    index=predictions.index)
    data_collect.append(predictions)
    scores['mae'] = mean_absolute_error(y_true_, pred_)
    scores['squared_error'] = mean_squared_error(y_true_, pred_)
    scores['ev'] = explained_variance_score(y_true_, pred_)
    scores['r2_score'] = r2_score(y_true_, pred_)
    scores['fold'] = split
    scores['Estimator'] = 'RandomForest'
    scores['Permuted'] = 'no'
    scores['model_testing'] = save_type
    scores['modality'] = 'SDoH'
    scores['target'] = 'CBCL'
    metrics.append(scores)
    return
for groups, (train_index, test_index) in enumerate(cv.split(X, y.to_numpy().flatten(), groups.to_numpy().flatten())):
    scores = {}
    grid_search.fit(X.iloc[train_index], y.iloc[train_index].values.ravel())

    predict_collect_save(data_pred=X.iloc[test_index], data_collect=data,
                         y_true=y, test_index=test_index,
			 split=groups, save_type='validation')
scores = pd.DataFrame(metrics)
scores.to_csv('~/resilience/250115_ComBatresults.csv')
full = pd.concat(data)
full.to_csv('~/resilience/ComBat_pred.csv')
