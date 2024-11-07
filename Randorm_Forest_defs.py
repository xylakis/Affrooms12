# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:28:28 2024

@author: Manos
"""

from sklearn.ensemble import RandomForestClassifier


def HP_tune(inputs,targets,inputs_val,targets_val):

    n_estimators = [50,100,150] # number of trees in the random forest
    criterion = ['gini','entropy','log_loss']
    max_features = ['sqrt','log2',None] # number of features in consideration at every split
    max_depth = [3,5,7,11] # maximum number of levels allowed in each decision tree
    min_samples_split = [2, 6, 10] # minimum sample number to split a node
    min_samples_leaf = [1, 3, 4, 9]
    bootstrap = [True]
    
    param_grid = {'n_estimators': n_estimators, 'criterion':criterion, 'max_features': max_features, 'min_samples_split':min_samples_split,
                  'min_samples_leaf':min_samples_leaf, 'max_depth': max_depth, 'bootstrap':bootstrap}
    
    best_score = 0
    for n_estimators in param_grid['n_estimators']:
        for criterion in param_grid['criterion']:
            for max_depth in param_grid['max_depth']:
                for max_features in param_grid['max_features']:
                    for bootstrap in param_grid['bootstrap']:
                        for MIN_SAMPLES_SPLIT in param_grid['min_samples_split']:
                            for MIN_SAMPLES_LEAF in param_grid['min_samples_leaf']:
                                # Train model with current hyperparameters
                                rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                                              min_samples_split=MIN_SAMPLES_SPLIT,
                                                              min_samples_leaf=MIN_SAMPLES_LEAF,
                                                              max_features=max_features,
                                                              bootstrap = bootstrap,
                                                              random_state=None)
                                rf.fit(inputs, targets.ravel())
                                
                                # Evaluate performance on training set
                                score = rf.score(inputs_val, targets_val.ravel())
    
    
                                # Update best hyperparameters if score is better
                                if score > best_score:
                                    #print(score,n_estimators,max_depth,max_features,bootstrap,MIN_SAMPLES_SPLIT,MIN_SAMPLES_LEAF,len(my_features))
                                    best_hyperparameters = {'n_estimators': n_estimators,
                                                            'criterion': criterion,
                                                            'max_depth': max_depth,
                                                            'max_features': max_features,
                                                            'bootstrap': bootstrap,
                                                            'min_samples_split': MIN_SAMPLES_SPLIT,
                                                            'min_samples_leaf': MIN_SAMPLES_LEAF
                                                            }
                                    best_score = score
                                
    
    return  best_hyperparameters