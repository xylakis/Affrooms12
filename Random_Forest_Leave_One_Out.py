from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import os
from datetime import datetime
import scipy.stats
from Randorm_Forest_defs import HP_tune

metrics = ["delta_gradient_vol","delta_minmax","delta_mean"]

affects = ["arousal","pleasure"]

MEMORIES = [1,3,5,11]

tolerances = [0.0,0.5,0.66,0.75]

thresholds = [0.0,0.05,0.10]

results_columns = ['conditions','accuracy','f1score','precision','recall','estimators','criterion','max_depth','min_samples_split','min_samples_leaf','bootstrap']
results = pd.DataFrame(columns = results_columns)

results_all_columns = ['conditions','participant','accuracy_train','accuracy_test','f1score_train','f1score_test','precision_train','precision_test','recall_train','recall_test','dataset_train','dataset_test']
results_all = pd.DataFrame(columns = results_all_columns)

TIMES_TO_RUN_EXPERIMENT = 10

#VALDIATION PARTICIPANTS
# pleasure_participants_validation = ["4BA02813-5D38-8AC1-3173-F0BFE86FC0A2","B43A40B3-4347-8C0B-A27F-1F27D88869CE","00BFC6FA-F83D-5D8C-DD38-A04850971782"]
# arousal_participants_validation = ["49FD3BC7-FEAA-8003-88F5-AD1ACFEC2633","E38168B7-5D12-0CE0-6723-1261EA05942F","48461B53-FDF8-FC54-57F7-AD0B021C19BE"]

for f in range (0,len(affects)): 
    
    affect = affects[f]
    
    path = "Ordinal_labels/"+affect+"/Train_Test/"
    val_path = "Ordinal_labels/"+affect+"/Validation/"
    
    participants_for_clasification = os.listdir(path)
    
    for metric in metrics:
        for MEMORY in MEMORIES:
            for tolerance in tolerances:
                for THRESHOLD in thresholds:
                    
                    testing_correct_sum = 0 
                    testing_number_of_samples_sum = 0 
                    
                    #VALIDATION SET
                    my_features = pd.read_csv(val_path+"features__"+metric+"_"+str(THRESHOLD)+"_"+"None_"+str(MEMORY)+".csv", index_col=0)
                    inputs_val = my_features.to_numpy()
                    my_targets = pd.read_csv(val_path+"targets__"+metric+"_"+str(THRESHOLD)+"_"+"None_"+str(MEMORY)+".csv", index_col=0)
                    targets_val = my_targets.to_numpy()
                    
                    #We use only the 1st participant in the set for HP tuning 
                    participant = participants_for_clasification[0]
                    folder = path+str(participant)+"/"
                    
                    #TRAINING SET
                    my_features = pd.read_csv(folder+"/TRAIN_features__"+metric+"_"+str(THRESHOLD)+"_"+str(tolerance)+"_"+str(MEMORY)+".csv", index_col=0)
                    inputs = my_features.to_numpy()
                    my_targets = pd.read_csv(folder+"/TRAIN_targets__"+metric+"_"+str(THRESHOLD)+"_"+str(tolerance)+"_"+str(MEMORY)+".csv", index_col=0)
                    targets = my_targets.to_numpy()
                    
                    
                    best_hyperparameters = HP_tune(inputs, targets, inputs_val, targets_val)
                                                    
                    #print(affect,best_hyperparameters)
                
                    N_ESTIMATORS = best_hyperparameters['n_estimators']
                    CRITERION = best_hyperparameters['criterion']
                    MAX_FEATURES = best_hyperparameters['max_features']
                    MAX_DEPTH = best_hyperparameters['max_depth']
                    BOOTSTRAP = best_hyperparameters['bootstrap']
                    MIN_SAMPLES_SPLIT  = best_hyperparameters['min_samples_split']
                    MIN_SAMPLES_LEAF  = best_hyperparameters['min_samples_split']
                
                    acc_pid = 0.0
                    f1_pid = 0.0
                    prec_pid = 0.0
                    rec_pid = 0.0
                    
                    acc_train_pid = 0.0
                    f1_train_pid = 0.0
                    prec_train_pid = 0.0
                    rec_train_pid = 0.0
                    
                    participant_losses = 0
                    
                    for i,participant in enumerate(participants_for_clasification): 
                                
                        acc = 0.0
                        f1 = 0.0
                        prec = 0.0
                        rec = 0.0
                        
                        acc_train = 0.0
                        f1_train = 0.0
                        prec_train = 0.0
                        rec_train = 0.0
                        
                        folder = path+str(participant)+"/"
                        
                        for time in range (0, TIMES_TO_RUN_EXPERIMENT):
                            # print(time,datetime.now())

                            #TRAINING SET
                            my_features = pd.read_csv(folder+"/TRAIN_features__"+metric+"_"+str(THRESHOLD)+"_"+str(tolerance)+"_"+str(MEMORY)+".csv", index_col=0)
                            inputs = my_features.to_numpy()
                            my_targets = pd.read_csv(folder+"/TRAIN_targets__"+metric+"_"+str(THRESHOLD)+"_"+str(tolerance)+"_"+str(MEMORY)+".csv", index_col=0)
                            targets = my_targets.to_numpy()
                            
                            #TEST SET
                            my_features_test = pd.read_csv(folder+"/TEST_features__"+metric+"_"+str(THRESHOLD)+"_"+str("None")+"_"+str(MEMORY)+".csv", index_col=0)
                            inputs_test = my_features_test.to_numpy()
                            my_targets_test = pd.read_csv(folder+"/TEST_targets__"+metric+"_"+str(THRESHOLD)+"_"+str("None")+"_"+str(MEMORY)+".csv", index_col=0)
                            targets_test = my_targets_test.to_numpy()
        
                            random_seed = None
        
                            optclf = RandomForestClassifier(n_estimators = N_ESTIMATORS, min_samples_split = MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF, 
                                                            max_features = MAX_FEATURES, max_depth = MAX_DEPTH, bootstrap=BOOTSTRAP,random_state=random_seed)
                            
                            # Train your model on the training set using the best hyperparameters
                            optclf.fit(inputs, targets.ravel())
                            
                            testing_acc = accuracy_score(targets_test,optclf.predict(inputs_test),normalize=False)
                            testing_correct_sum += testing_acc
                            
                            testing_number_of_samples_sum += len(targets_test)
                            
                            acc +=  accuracy_score(targets_test,optclf.predict(inputs_test))
                            f1 += f1_score(targets_test,optclf.predict(inputs_test))
                            prec += precision_score(targets_test,optclf.predict(inputs_test))
                            rec += recall_score(targets_test,optclf.predict(inputs_test))
                            
                            acc_train +=  accuracy_score(targets,optclf.predict(inputs))
                            f1_train += f1_score(targets,optclf.predict(inputs))
                            prec_train += precision_score(targets,optclf.predict(inputs))
                            rec_train += recall_score(targets,optclf.predict(inputs))
                            
                            importances = optclf.feature_importances_
                            
                            feature_columns = my_features.columns
                            random_forest_importances = pd.DataFrame(columns=['feature','importance'])
                            random_forest_importances['feature'] = feature_columns
                            random_forest_importances['importance'] = importances
                            
                        acc_train_pid += acc_train/TIMES_TO_RUN_EXPERIMENT
                        f1_train_pid += f1_train/TIMES_TO_RUN_EXPERIMENT
                        prec_train_pid += prec_train/TIMES_TO_RUN_EXPERIMENT
                        rec_train_pid +=  rec_train/TIMES_TO_RUN_EXPERIMENT
                            
                        acc_pid += acc/TIMES_TO_RUN_EXPERIMENT
                        f1_pid += f1/TIMES_TO_RUN_EXPERIMENT
                        prec_pid += prec/TIMES_TO_RUN_EXPERIMENT
                        rec_pid +=  rec/TIMES_TO_RUN_EXPERIMENT
                        
                        string = affect+"_"+str(metric)+"_Tolerance:"+str(tolerance)+"_Threshold:"+str(THRESHOLD)+"_Memory:"+str(MEMORY)
                        
                        new_row_all = {'conditions': string,'participant': participant,'accuracy_train':round(acc_train/TIMES_TO_RUN_EXPERIMENT,3),'accuracy_test': round(acc/TIMES_TO_RUN_EXPERIMENT,3),
                                    'f1score_train': round(f1_train/TIMES_TO_RUN_EXPERIMENT,3),'f1score_test': round(f1/TIMES_TO_RUN_EXPERIMENT,3),'precision_train':round(prec_train/TIMES_TO_RUN_EXPERIMENT,3),
                                    'precision_test': round(prec/TIMES_TO_RUN_EXPERIMENT,3),'recall_train': round(rec_train/TIMES_TO_RUN_EXPERIMENT,3),'recall_test': round(rec/TIMES_TO_RUN_EXPERIMENT,3),
                                    'dataset_train':len(targets),'dataset_1': np.count_nonzero(targets == 1),'dataset_0': np.count_nonzero(targets == 0),'dataset_test':len(targets_test)
                                    }
                        holder = pd.DataFrame.from_dict([new_row_all])
                        results_all = pd.concat([results_all, holder], axis=0, ignore_index=True)
                    
                    binom_testing = scipy.stats.binomtest(testing_correct_sum, n=testing_number_of_samples_sum, p=0.5)
                    
                    if binom_testing.pvalue<=0.05:
                        binom_check = True
                    else:
                        binom_check = False
                    
                    string = affect+"_"+str(metric)+"_Tolerance:"+str(tolerance)+"_Threshold:"+str(THRESHOLD)+"_Memory:"+str(MEMORY)
                    print(string, "ADDDEEEEDD",datetime.now())
                    new_row = {'conditions': string, 
                                'accuracy':round(acc_pid/(len(participants_for_clasification)),3),'f1score':round(f1_pid/(len(participants_for_clasification)),3),
                                'precision':round(prec_pid/(len(participants_for_clasification)),3),'recall': round(rec_pid/(len(participants_for_clasification)),3),
                                'accuracy_train':round(acc_train_pid/(len(participants_for_clasification)),3),'f1score_train':round(f1_train_pid/(len(participants_for_clasification)),3),
                                'precision_train':round(prec_train_pid/(len(participants_for_clasification)),3),'recall_train': round(rec_train_pid/(len(participants_for_clasification)),3),
                                'estimators':N_ESTIMATORS,'criterion':CRITERION,'max_depth':MAX_DEPTH,'max_features':MAX_FEATURES,'min_samples_split':MIN_SAMPLES_SPLIT,'min_samples_leaf':MIN_SAMPLES_LEAF,
                                'bootstrap': BOOTSTRAP,'nr_correct_preds': testing_correct_sum,'total_nr_of_samples': testing_number_of_samples_sum,'ratio': str(testing_correct_sum)+"/"+str(testing_number_of_samples_sum),'binom_testing': binom_check
                                }
                    holder = pd.DataFrame.from_dict([new_row])
                    results = pd.concat([results, holder], axis=0, ignore_index=True)

                        
