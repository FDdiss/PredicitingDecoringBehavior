'''
Versionen:

v1: Input: /DATASET_INTERVALL_v6.csv
v2: Input: /DATASET_INTERVALL_v9.csv, pickel code snippets entfernt
v3: Input: /DATASET_INTERVALL_v10.csv
v4: Featuresets 1 bis 12 eingepflegt. Schleife aufgebaut um alle Featuresets durchzurechnen, globale modelscore print to csv
v5: Datenauswahl Funktion ausgegliedert in eigene py-Datei, XGBoost über sklearn implementiert. Ausleiten der XGB/shap importance in csv. Zeitstempel hinzugefügt
v6: Input: /DATASET_INTERVALL_v12.csv, Automatische Featureauswahl mittels SelectFromModel. SHAP Plot export, Featureliste nun eigener Export. Erstellung RMSE für NN. Zahl Datenpunkte in Export hinzugefügt. SHAP Decision Plot hinzugefügt. 
v7: komplette Überarbeitung, gridsearch in normale Modelberechnung integriert (ohne KF), modelausspeicherung, Scalierung angepasst, Export angepasst, scores angepasst, input_dataset geändert auf he_marked (für skalierung benötigt, damit he_Werte nicht skaliert werden), XGB_Filterung integriert, und wahrscheinlich noch mehr
v8: Datenzusammensetzung verändert (v14, Q-Werte nachgetragen), x-fache random initialisierung eingebaut mit ausleitung der besten modelle und Durchschnittsbildung für die scores

'''

### Import
#from tkinter.font import names
#from jinja2.filters import V
#from numba.cuda import target

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb
from xgboost import DMatrix
#from xgboost import train
from xgboost import plot_importance

import joblib
from itertools import product
import random
import math
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from xgboost.rabit import init

from preparedata_v3 import prepare_data
from preparedata_RIEGEL_v3 import prepare_data_riegel

import shap

##############################################################################################
# MAIN CODE #
##############################################################################################

path="D:/EntkernKI"
runname="run05"
folderexplanation="baseline"

continue_trigger=False

#########

featuresets=[1,2,3,4,5,6,7,8,9,10,11,12]
#featuresets=[2]

XGBfilter_trigger=True

#featureparas=[7,6,0.5,0.1]
#XGB_parallel_tree=[featureparas[0]]  #XGB_parallel_tree=[2,4,6,8,10,50]
#XGB_maxdepth=[featureparas[1]] #XGB_maxdepth=[2,3,4,5,6,7,8,9,10,16,32]
#XGB_eta=[featureparas[2]] #O1:0.1 (objective, eta: 0.1783) #XGB_eta=[0.05,0.1,0.2,0.3] #O1:0.1 (objective, eta: 0.1783)
#XGB_alpha=[featureparas[3]]  #XGB_alpha=[0,0.25,0.5,1.0,2.0]



### Zielgröße auswählen # [Name,max-Ausreißer,min-Ausreißer,max,min,mean,median]
targetparam_intervall=["06_i_m_progress",100,0,0,0,0,0]
#targetparam=["07_i_d_progress",150,0,0,0,0,0]
#targetparam=["08_i_m_progress_r",1,0,0,0,0,0] 
#targetparam=["09_i_d_progress_r",1,0,0,0,0,0] 


targetparam_hs=["10_b_n_hammerblows",8000,0,0,0,0,0]

gramm_trigger=True
hs_trigger=True



system_trigger=True
systemwahl=["D"] ###alle außer ""A","G","M","F","Q","L","E","P","B","N""

loss_curve_trigger=False
importance_trigger=False
SHAP_trigger = False
savemodel_trigger = False
best_export_trigger = True

#################################

scalerlist=["",preprocessing.RobustScaler,preprocessing.MinMaxScaler,preprocessing.StandardScaler]
scalerlist=scalerlist[2:3]
scalernamelist=["none","robust","minmax","standard"]
scalernamelist=scalernamelist[2:3]
print(scalernamelist)


XGBfilterfraction_list=[100,90,80,70,60,50,40,30,20,10,7,4,1]  ### in %
#XGBfilterfraction_list=[100]  ### in %

XGBfiltermodel_fe = ""
XGBfiltermodel_fe = ""
XGBfiltermodel_tsfresh = "20230216-201211_XGB_model_run02_base_tsfresh_data_NIO_2000_5_6_0.1"

NN_trigger = True
XGB_trigger = True


random.seed(3)

### XGBoost Parameter

XGB_objective=["reg:squarederror"] #O1:log (objective, eta:0.1783)  #XGB_objective=["reg:logistic","reg:squarederror","reg:squaredlogerror","reg:pseudohubererror"] #O1:log (objective, eta:0.1783)

XGB_verbosity=[0]

XGB_eta=[0.1] #O1:0.1 (objective, eta: 0.1783) #XGB_eta=[0.05,0.1,0.2,0.3] #O1:0.1 (objective, eta: 0.1783)
XGB_alpha=[0.0001]  #XGB_alpha=[0,0.25,0.5,1.0,2.0]
XGB_gamma=[0]  #XGB_gamma=[0.01,0.05,0.1,0.2,0.3,1.0,2.0,5.0,10]
XGB_lambda=[2]  #XGB_lambda=[1,2,4,8]
XGB_minchild_weight=[1]   #XGB_minchild_weight=[0.01,0.05,0.1,0.2,0.3,1.0,2.0,5.0,10]
XGB_max_delta_step=[0]  #XGB_max_delta_step=[0.5,1,2,4,8,16]

XGB_n_estimators=[2000] #500
XGB_earlystoppingrounds=[50]
XGB_parallel_tree=[1,3,5,7]  #XGB_parallel_tree=[2,4,6,8,10,50]
XGB_maxdepth=[2,4,6] #XGB_maxdepth=[2,3,4,5,6,7,8,9,10,16,32]

XGB_subsample=[0.75]   #XGB_subsample=[0.25,0.5,0.75,1.0]
XGB_colsample_bytree=[0.75]    #XGB_colsample_bytree=[0.25,0.5,0.75,1.0]
XGB_colsample_bylevel=[0.75]   #XGB_colsample_bylevel=[0.25,0.5,0.75,1.0]
XGB_colsample_bynode=[0.75]   #XGB_colsample_bynode=[0.25,0.5,0.75,1.0]
# for XGB_objective_i,XGB_seed_i,XGB_eta_i,XGB_alpha_i,XGB_gamma_i,XGB_lambda_i,XGB_minchild_weight_i,XGB_max_delta_step_i,XGB_n_estimators_i,XGB_earlystoppingrounds_i,XGB_parallel_tree_i,XGB_maxdepth_i,XGB_subsample_i,XGB_colsample_bytree_i,XGB_colsample_bylevel_i,XGB_colsample_bynode_i in XGB_objective,XGB_seed,XGB_eta,XGB_alpha,XGB_gamma,XGB_lambda,XGB_minchild_weight,XGB_max_delta_step,XGB_n_estimators,XGB_earlystoppingrounds,XGB_parallel_tree,XGB_maxdepth,XGB_subsample,XGB_colsample_bytree,XGB_colsample_bylevel,XGB_colsample_bynode
#NN Parameter

NN_verbose=[True]

NN_layers=[(10,10),(20,20),(30,30),(40,40),(50,50),(100,100),(200,200),(400,400),(800,800),
           (10,5),(20,10),(30,15),(40,20),(50,25),(100,50),(200,100),(400,200),(800,400),
           (10,10,5),(20,20,10),(30,30,15),(40,40,20),(50,50,25),(100,100,50),(200,200,100),(400,400,200),(800,800,400)
           ]
NN_layers_names=[
    "10x10","20x20","30x30","40x40","50x50","100x100","200x200","400x400","800x800",
    "10x5","20x10","30x15","40x20","50x25","100x50","200x100","400x200","800x400",
    "10x10x5","20x20x10","30x30x15","40x40x20","50x50x25","100x100x50","200x200x100","400x400x200","800x800x400"
    ]
NN_activation=["relu"] #default='relu'
NN_solver=["adam"] #default='adam'
NN_alpha=[0.0001] # default=0.0001
#NN_batch_size=["auto"] # default='auto' # Ersetzt durch eine primzahlenberechnung, um 
NN_minibatch_size = [30] # integer, not a list! wird zur Berechnung der genauen minibatchgröße benutzt, um eine möglichst ideale Teilung zu ermöglichen
NN_learning_rate=["constant"] #default='constant'
NN_learning_rate_init=[0.0001] #default=0.001 ### 0.00001,0.001

NN_early_stopping=[True]
NN_tol=[0.000001] #default 1e-4
NN_n_iter_no_change = [50] #default 10 ###

NN_validation_fraction=[0.1] #default 0.1
NN_max_fun=[15000] # default=15000
NN_max_iter=[2000]#default =200
NN_beta_1=[0.9]#default =0.9, 
NN_beta_2=[0.999]#default =0.999, 
NN_epsilon=[1e-08]#default =1e-08, 

NN_momentum=[0.9]
NN_nesterovs_momentum=[True]

NN_power_t=[0.5] #default 0.5
NN_shuffle=[True] #default True





##############################################################################################
# Start MAIN
##############################################################################################
globalscores=[]
### Ordner erstellen
if folderexplanation != "":
    savepath=path+"/"+runname+"_"+folderexplanation
else:
    savepath=path+"/"+runname
if not os.path.exists(savepath):
      
    # if the demo_folder directory is not present 
    # then create it.
    os.makedirs(savepath)
elif continue_trigger == True:
    lastcsv="none"
    scorefile_searchstring = "scores"

    for filename in os.listdir(savepath):
        if scorefile_searchstring in filename:
            lastcsv=str(savepath + os.sep +filename)
    if lastcsv != "none":
        with open(lastcsv, 'r') as fh:
            csvdata=pd.read_csv(fh)
        csvdata.drop(csvdata.columns[0],axis=1,inplace=True)
        globalscores = csvdata.values.tolist()
        print(globalscores)

###### Primzahlen finden
primelist=[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,
           227,229,233,239,241,251,257,263,269,271,277,281,283,293]

##############################################################################################################################
#Datenseterzeugung
##############################################################################################################################

#if True in [xgb_base_trigger,xgb_kok_base_trigger,xgb_nest_base_trigger]:
#    clf = joblib.load(savepath+os.sep+XGBfiltermodel_base)       
#    selectmodel = SelectFromModel(clf, prefit=True,threshold=-np.inf,max_features=int(len(base_data)*xgbfraction/100))
#    xgb_base_data=base_data.drop("FEHLERART_BEZ",axis=1)
#    xgbcolumnnames=selectmodel.get_feature_names_out(xgb_base_data.columns)
#    xgb_base_data = pd.DataFrame(columns=xgbcolumnnames,data=selectmodel.transform(xgb_base_data))
#    xgb_base_data["FEHLERART_BEZ"] = base_data["FEHLERART_BEZ"]

columnnames=[
"model","runname","dataname","targetfeature","validation_systems","datacount_all","datacount_test","datacount_train","datacount_val","featurecount","XGBfilterfraction",
"score_test_mean","RMSE_test_mean","MAE_rescaled_test_mean",
"score_test_best","RMSE_test_best","MAE_rescaled_test_best",
"score_val_best","RMSE_val_best","MAE_rescaled_val_best",

"RMSE_rescaled_test_mean","MAE_test_mean",
"score_all_mean","RMSE_all_mean","RMSE_rescaled_all_mean","MAE_all_mean","MAE_rescaled_all_mean",
"score_val_mean","RMSE_val_mean","RMSE_rescaled_val_mean","MAE_val_mean","MAE_rescaled_val_mean",
"last_training_loss_mean","last_validation_loss_mean",

"RMSE_rescaled_test_best","MAE_test_best",
"score_all_best","RMSE_all_best","RMSE_rescaled_all_best","MAE_all_best","MAE_rescaled_all_best",
"RMSE_rescaled_val_best","MAE_val_best",
"last_training_loss_best","last_validation_loss_best","random_state_best",

"targetvalue_test_mean_best","targetvalue_test_max_best","targetvalue_test_min_best","targetvalue_test_median_best",
"targetvalue_train_mean_best","targetvalue_train_max_best","targetvalue_train_min_best","targetvalue_train_median_best",
"targetvalue_val_mean_best","targetvalue_val_max_best","targetvalue_val_min_best","targetvalue_val_median_best",

"scalername","randomseeds",
"NN_solver","NN_activation","NN_layers","NN_learning_rate_init","NN_alpha","NN_minibatchsize","NN_max_iter","NN_n_iter_no_change","NN_tol",
"NN_validation_fraction","NN_beta_1","NN_beta_2","NN_epsilon","NN_max_fun","NN_learning_rate","NN_shuffle","NN_early_stopping","NN_power_t","NN_momentum","NN_nesterovs_momentum",
"XGB_objective","XGB_n_estimators","XGB_parallel_tree","XGB_maxdepth","XGB_eta","XGB_earlystoppingrounds","XGB_minchild_weight","XGB_max_delta_step",
"XGB_subsample","XGB_colsample_bytree","XGB_colsample_bylevel","XGB_colsample_bynode","XGB_alpha","XGB_lambda","XGB_gamma"
    ]

XGBfilter_modelnamebase_hs="_10_b_n_hammerblows_XGB_filtermodel"
XGBfilter_modelnamebase_gramm="_06_i_m_progress_XGB_filtermodel"


##############################################################################################################################
#start Rechnungen
##############################################################################################################################
randomseeds=random.sample(range(1, 1000), 10)
randomseeds_str='|'.join([str(elem) for elem in randomseeds])

best_dict={}
best_score_dict={}
for i in featuresets:
    for k in XGBfilterfraction_list:
        entryname="featureset_"+str(i).zfill(2)+"_"+str(k).zfill(3)+"_"+targetparam_intervall[0]+"_NN"
        best_dict[entryname]=9999
        entryname="featureset_"+str(i).zfill(2)+"_"+str(k).zfill(3)+"_"+targetparam_hs[0]+"_NN"
        best_dict[entryname]=9999
        entryname="featureset_"+str(i).zfill(2)+"_"+str(k).zfill(3)+"_"+targetparam_intervall[0]+"_XGB"
        best_dict[entryname]=9999
        entryname="featureset_"+str(i).zfill(2)+"_"+str(k).zfill(3)+"_"+targetparam_hs[0]+"_XGB"
        best_dict[entryname]=9999




saveslotcounter=1
gramm_or_hs=[]
if gramm_trigger==True:
    gramm_or_hs=gramm_or_hs+["gramm"]
if hs_trigger==True:
    gramm_or_hs=gramm_or_hs+["hs"]

if system_trigger==False:
    modelscore_val_best="none"
    modelRMSE_val_best="none"   
    modelRMSE_rescaled_val_best="none"       
    modelMAE_val_best="none"   
    modelMAE_rescaled_val_best="none"  
    modelscore_val="none"
    modelRMSE_val="none"
    modelRMSE_rescaled_val="none"
    modelMAE_val="none"
    modelMAE_rescaled_val="none"
    valsystem_str="none"
    rowcount_val=0 
else:
    valsystem_str='|'.join([str(elem) for elem in systemwahl])

for r in gramm_or_hs:
    targetlabel=r
    if targetlabel == "gramm":
        targetparam=targetparam_intervall
        with open(path+"/DATASET_INTERVALL_he_v14_he_marked.csv") as fh:
            orgdata=pd.read_csv(fh)
        orgdata.drop(orgdata.columns[0],axis=1,inplace=True)
    if targetlabel == "hs":
        targetparam=targetparam_hs
        with open(path+"/DATASET_RIEGEL_he_v14_he_marked.csv") as fh:
            orgdata=pd.read_csv(fh)
        orgdata.drop(orgdata.columns[0],axis=1,inplace=True)
    targetlabel = targetparam[0]
    for k in range(len(featuresets)):
        featureset=featuresets[k]
        dataname=str("featureset_"+str(featureset).zfill(2))
        #Datenvorbereitung
        
        if XGBfilter_trigger == False:
            XGBfilterfraction_list=[100]
        for XGBfilterfraction in XGBfilterfraction_list:
            data=orgdata.copy()
            if r == "gramm":
                data,y,X_valsystem_org,y_valsystem_org,targetparam=prepare_data(data,targetparam,system_trigger,systemwahl,featureset)
                print("gramm data loaded")
            if r == "hs":
                data,y,X_valsystem_org,y_valsystem_org,targetparam=prepare_data_riegel(data,targetparam,system_trigger,systemwahl,featureset)
                print("hs data loaded")
            targetvalue_val_mean=y_valsystem_org.mean()
            targetvalue_val_median=y_valsystem_org.median()
            targetvalue_val_max=y_valsystem_org.max()
            targetvalue_val_min=y_valsystem_org.min()   
            rowcount_val=y_valsystem_org.shape[0]          
            print("Dataset", featureset, "Zahl features:", data.shape[1], "Zahl datenpunkte:", data.shape[0])
            
            if ((XGBfilter_trigger == True) and (XGBfilterfraction != 100)):
                if r == "gramm":
                    clf = joblib.load(savepath+os.sep+"featureset_"+str(featureset).zfill(2)+"_100_"+targetparam_intervall[0]+"_XGB_bestmodel") 
                if r == "hs":
                    clf = joblib.load(savepath+os.sep+"featureset_"+str(featureset).zfill(2)+"_100_"+targetparam_hs[0]+"_XGB_bestmodel") 
                selectmodel = SelectFromModel(clf, prefit=True,threshold=-np.inf,max_features=int(math.ceil(data.shape[1]*XGBfilterfraction/100)))
                print(int(len(data)*XGBfilterfraction/100))
                print(data.columns)
                xgbcolumnnames=selectmodel.get_feature_names_out(data.columns)
                data = pd.DataFrame(columns=xgbcolumnnames,data=selectmodel.transform(data))
                print(data.columns)
                if system_trigger == True:
                    X_valsystem_org = pd.DataFrame(columns=xgbcolumnnames,data=selectmodel.transform(X_valsystem_org))
                print(X_valsystem_org.columns)                
            print("jetzt:", dataname,targetlabel)
            #if data

            for i in range(len(scalerlist)):
                scalername=scalernamelist[i]
                print("jetzt:",dataname,r,scalername,targetlabel)
                if NN_trigger == True:
                    for l in range(len(NN_layers)):
                        for NN_activation_i,NN_solver_i,NN_alpha_i,NN_minibatch_size_i,NN_learning_rate_i,NN_learning_rate_init_i,NN_early_stopping_i,NN_tol_i,NN_n_iter_no_change_i,NN_validation_fraction_i,NN_max_fun_i,NN_max_iter_i,NN_beta_1_i,NN_beta_2_i,NN_epsilon_i,NN_momentum_i,NN_nesterovs_momentum_i,NN_power_t_i,NN_shuffle_i in product(NN_activation,NN_solver,NN_alpha,NN_minibatch_size,NN_learning_rate,NN_learning_rate_init,NN_early_stopping,NN_tol,NN_n_iter_no_change,NN_validation_fraction,NN_max_fun,NN_max_iter,NN_beta_1,NN_beta_2,NN_epsilon,NN_momentum,NN_nesterovs_momentum,NN_power_t,NN_shuffle):
                            print("jetzt: NN",NN_layers_names[l],dataname,targetlabel)
                            best_trigger=False
                            RMSE_test_rand_list=[]
                            RMSE_rescaled_test_rand_list=[]                            
                            RMSE_all_rand_list=[]
                            RMSE_rescaled_all_rand_list=[]                                                           
                            MAE_test_rand_list=[]
                            MAE_rescaled_test_rand_list=[]                            
                            MAE_all_rand_list=[]
                            MAE_rescaled_all_rand_list=[]                            
                            modelscore_test_rand_list=[]
                            modelscore_all_rand_list=[]
                            modeltrainingloss_rand_list=[]
                            modelvalidationloss_rand_list=[]
                            rowcount_all_rand_list=[]
                            rowcount_test_rand_list=[]
                            rowcount_train_rand_list=[]                
                            if system_trigger==True:
                                RMSE_val_rand_list=[]
                                RMSE_rescaled_val_rand_list=[]    
                                MAE_val_rand_list=[]
                                MAE_rescaled_val_rand_list=[]     
                                modelscore_val_rand_list=[]                                                                                        
                            for NN_random_state_i in randomseeds:
                                X_train, X_test, y_train, y_test = train_test_split(data, y, test_size =0.2, random_state = NN_random_state_i)
                                targetvalue_test_mean=y_test.mean()
                                targetvalue_test_median=y_test.median()
                                targetvalue_test_max=y_test.max()
                                targetvalue_test_min=y_test.min()
                                targetvalue_train_mean=y_train.mean()
                                targetvalue_train_median=y_train.median()
                                targetvalue_train_max=y_train.max()
                                targetvalue_train_min=y_train.min()   
                                                              
                                if scalernamelist[i]!="none":
                                    X_valsystem=X_valsystem_org.copy()
                                    y_valsystem=y_valsystem_org.copy()
                                    he_columnnames = list(X_train.filter(regex='_he', axis=1).columns)
                                    scaler = scalerlist[i]()
                                    scaler.fit(X_train)

                                    puffer=X_train[he_columnnames].copy().reset_index()
                                    X_train = pd.DataFrame(scaler.transform(X_train),columns=data.columns)
                                    X_train[he_columnnames]=puffer[he_columnnames]

                                    puffer=X_test[he_columnnames].copy().reset_index()
                                    X_test = pd.DataFrame(scaler.transform(X_test),columns=data.columns)
                                    X_test[he_columnnames]=puffer[he_columnnames]

                                    puffer=data[he_columnnames].copy().reset_index()
                                    X_all = pd.DataFrame(scaler.transform(data),columns=data.columns)
                                    X_all[he_columnnames]=puffer[he_columnnames]

                                    y_scaler = scalerlist[i]()
                                    y_scaler.fit(np.reshape(y_train.values,(-1, 1)))
                                    y_train = pd.DataFrame(y_scaler.transform(np.reshape(y_train.values,(-1, 1))),columns=[targetparam[0]])
                                    y_test = pd.DataFrame(y_scaler.transform(np.reshape(y_test.values,(-1, 1))),columns=[targetparam[0]])
                                    y_all = pd.DataFrame(y_scaler.transform(np.reshape(y.values,(-1, 1))),columns=[targetparam[0]])

                                    if system_trigger==True:
                                        puffer=X_valsystem[he_columnnames].copy().reset_index()
                                        X_valsystem = pd.DataFrame(scaler.transform(X_valsystem),columns=data.columns)
                                        X_valsystem[he_columnnames]=puffer[he_columnnames]
                                        y_valsystem = pd.DataFrame(y_scaler.transform(np.reshape(y_valsystem.values,(-1, 1))),columns=[targetparam[0]])
                                else:
                                    modeldata=data     
                                maxdivisor=1
                                if X_train.shape[0]>2*NN_minibatch_size_i:
                                    for m in primelist:                        
                                        if  X_train.shape[0]/m < NN_minibatch_size_i:
                                            break
                                        maxdivisor=m
                                        if ((X_train.shape[0]/m >= NN_minibatch_size_i) and (X_train.shape[0] % m)) == 0:
                                            maxdivisor=m
                                if X_train.shape[0] % maxdivisor == 0:
                                    NN_batch_size=int(X_train.shape[0]/maxdivisor)
                                else:
                                    NN_batch_size=int(math.ceil(X_train.shape[0]/maxdivisor))                           
                                
                                NNmodel_rand=MLPRegressor(
                                    hidden_layer_sizes=NN_layers[l],
                                    activation=NN_activation_i,
                                    solver=NN_solver_i,
                                    alpha=NN_alpha_i,
                                    batch_size=NN_batch_size,
                                    learning_rate=NN_learning_rate_i,
                                    learning_rate_init=NN_learning_rate_init_i,
                                    power_t=NN_power_t_i,
                                    max_iter=NN_max_iter_i,
                                    shuffle=NN_shuffle_i,
                                    random_state=NN_random_state_i,
                                    tol=NN_tol_i,
                                    verbose=NN_verbose[0],
                                    early_stopping=NN_early_stopping_i,
                                    validation_fraction=NN_validation_fraction_i,
                                    beta_1=NN_beta_1_i,
                                    beta_2=NN_beta_2_i,
                                    epsilon=NN_epsilon_i,
                                    n_iter_no_change=NN_n_iter_no_change_i,
                                    max_fun=NN_max_fun_i,
                                    momentum=NN_momentum_i,
                                    nesterovs_momentum=NN_nesterovs_momentum_i

                                    )
                                NNmodel_rand.fit(X_train.values,y_train.values)
                                modeltrainingloss=NNmodel_rand.loss_curve_[-1]
                                modelvalidationloss=NNmodel_rand.validation_scores_[-1]

                                predicitions_test = NNmodel_rand.predict(X_test.values)
                                modelscore_test=NNmodel_rand.score(X_test.values,y_test.values)
                                modelRMSE_test=mean_squared_error(y_test.values,predicitions_test)**(1/2)
                                modelRMSE_rescaled_test=modelRMSE_test*(targetparam[3]-targetparam[4])
                                modelMAE_test=mean_absolute_error(y_test.values,predicitions_test)
                                modelMAE_rescaled_test=modelMAE_test*(targetparam[3]-targetparam[4])

                                predicitions_all = NNmodel_rand.predict(X_all.values)
                                modelscore_all=NNmodel_rand.score(X_all,y_all)
                                modelRMSE_all=mean_squared_error(y_all,predicitions_all)**(1/2)
                                modelRMSE_rescaled_all=y_scaler.inverse_transform([[modelRMSE_all]])[0][0]
                                modelMAE_all=mean_absolute_error(y_all,predicitions_all)
                                modelMAE_rescaled_all=y_scaler.inverse_transform([[modelMAE_all]])[0][0]

                                rowcount_all=X_all.shape[0]
                                rowcount_test=X_test.shape[0]
                                rowcount_train=X_train.shape[0]

                                modeltrainingloss_rand_list=modeltrainingloss_rand_list+[modeltrainingloss]
                                modelvalidationloss_rand_list=modelvalidationloss_rand_list+[modelvalidationloss]


                                modelscore_test_rand_list=modelscore_test_rand_list+[modelscore_test]
                                RMSE_test_rand_list=RMSE_test_rand_list+[modelRMSE_test]
                                RMSE_rescaled_test_rand_list=RMSE_rescaled_test_rand_list+[modelRMSE_rescaled_test]
                                MAE_test_rand_list=MAE_test_rand_list+[modelMAE_test]
                                MAE_rescaled_test_rand_list=MAE_rescaled_test_rand_list+[modelMAE_rescaled_test]

                                modelscore_all_rand_list=modelscore_all_rand_list+[modelscore_all]
                                RMSE_all_rand_list=RMSE_all_rand_list+[modelRMSE_all]
                                RMSE_rescaled_all_rand_list=RMSE_rescaled_all_rand_list+[modelRMSE_rescaled_all]
                                MAE_all_rand_list=MAE_all_rand_list+[modelMAE_all]
                                MAE_rescaled_all_rand_list=MAE_rescaled_all_rand_list+[modelMAE_rescaled_all]

                                rowcount_all_rand_list=rowcount_all_rand_list+rowcount_all
                                rowcount_test_rand_list=rowcount_test_rand_list+rowcount_test
                                rowcount_train_rand_list=rowcount_train_rand_list+rowcount_train

                                if system_trigger==True:
                                    predicitions_val = NNmodel_rand.predict(X_valsystem.values)
                                    modelscore_val=NNmodel_rand.score(X_valsystem,y_valsystem)
                                    modelRMSE_val=mean_squared_error(y_valsystem,predicitions_val)**(1/2)
                                    modelMAE_val=mean_absolute_error(y_valsystem,predicitions_val)
                                    if modelRMSE_val < 10:
                                        modelRMSE_rescaled_val=y_scaler.inverse_transform([[modelRMSE_val]])[0][0]
                                    else:         
                                        modelRMSE_rescaled_val=999999  
                                    if modelMAE_val < 10:
                                        modelMAE_rescaled_val=y_scaler.inverse_transform([[modelMAE_val]])[0][0]          
                                    else:
                                        modelMAE_rescaled_val=999999                                  
                                    modelscore_val_rand_list=modelscore_val_rand_list+[modelscore_val]
                                    RMSE_val_rand_list=RMSE_val_rand_list+[modelRMSE_val]
                                    RMSE_rescaled_val_rand_list=RMSE_rescaled_val_rand_list+[modelRMSE_rescaled_val]
                                    MAE_val_rand_list=MAE_val_rand_list+[modelMAE_val]
                                    MAE_rescaled_val_rand_list=MAE_rescaled_val_rand_list+[modelMAE_rescaled_val]


                                if modelRMSE_test <= min(RMSE_test_rand_list):
                                    NNmodel=NNmodel_rand
                                    modelscore_test_best=modelscore_test
                                    modelRMSE_test_best=modelRMSE_test
                                    modelRMSE_rescaled_test_best=modelRMSE_rescaled_test
                                    modelMAE_test_best=modelMAE_test
                                    modelMAE_rescaled_test_best=modelMAE_rescaled_test    
                                    modelscore_all_best=modelscore_all   
                                    modelRMSE_all_best=modelRMSE_all   
                                    modelRMSE_rescaled_all_best=modelRMSE_rescaled_all       
                                    modelMAE_all_best=modelMAE_all    
                                    modelMAE_rescaled_all_best=modelMAE_rescaled_all                                    
                                    random_state_best=NN_random_state_i
                                    modeltrainingloss_best=modeltrainingloss
                                    modelvalidationloss_best=modelvalidationloss
                                    targetvalue_test_mean_best=targetvalue_test_mean
                                    targetvalue_test_median_best=targetvalue_test_median
                                    targetvalue_test_max_best=targetvalue_test_max
                                    targetvalue_test_min_best=targetvalue_test_min
                                    targetvalue_train_mean_best=targetvalue_train_mean
                                    targetvalue_train_median_best=targetvalue_train_median
                                    targetvalue_train_max_best=targetvalue_train_max
                                    targetvalue_train_min_best=targetvalue_train_min
                                    rowcount_all_best=X_all.shape[0]
                                    rowcount_test_best=X_test.shape[0]
                                    rowcount_train_best=X_train.shape[0]
                                    
                                    if system_trigger==True:
                                        modelscore_val_best=modelscore_val   
                                        modelRMSE_val_best=modelRMSE_val   
                                        modelRMSE_rescaled_val_best=modelRMSE_rescaled_val       
                                        modelMAE_val_best=modelMAE_val   
                                        modelMAE_rescaled_val_best=modelMAE_rescaled_val  
                                                                                                     

                            rowcount_all_best=np.mean(rowcount_all_rand_list)
                            rowcount_test_best=np.mean(rowcount_test_rand_list)
                            rowcount_train_best=np.mean(rowcount_train_rand_list)
                            featurecount=X_train.shape[1]

                            modeltrainingloss=np.mean(modeltrainingloss_rand_list)
                            modelvalidationloss=np.mean(modelvalidationloss_rand_list)

                            modelscore_test=np.mean(modelscore_test_rand_list)
                            modelRMSE_test=np.mean(RMSE_test_rand_list)
                            modelRMSE_rescaled_test=np.mean(RMSE_rescaled_test_rand_list)
                            modelMAE_test=np.mean(MAE_test_rand_list)
                            modelMAE_rescaled_test=np.mean(MAE_rescaled_test_rand_list)


                            modelscore_all=np.mean(modelscore_all_rand_list)
                            modelRMSE_all=np.mean(RMSE_all_rand_list)
                            modelRMSE_rescaled_all=np.mean(RMSE_rescaled_all_rand_list)
                            modelMAE_all=np.mean(MAE_all_rand_list)
                            modelMAE_rescaled_all=np.mean(MAE_rescaled_all_rand_list)
                            
                            if system_trigger==True:
                                modelscore_val=np.mean(modelscore_val_rand_list)
                                modelRMSE_val=np.mean(RMSE_val_rand_list)
                                modelRMSE_rescaled_val=np.mean(RMSE_rescaled_val_rand_list)
                                modelMAE_val=np.mean(MAE_val_rand_list)
                                modelMAE_rescaled_val=np.mean(MAE_rescaled_val_rand_list)

                            scores=[
                                ["NN"]+[runname]+[dataname]+[targetlabel]+[valsystem_str]+[rowcount_all]+[rowcount_test]+[rowcount_train]+[rowcount_val]+[featurecount]+[XGBfilterfraction]+
                                [modelscore_test]+[modelRMSE_test]+[modelMAE_rescaled_test]+
                                [modelscore_test_best]+[modelRMSE_test_best]+[modelMAE_rescaled_test_best]+
                                [modelscore_val_best]+[modelRMSE_val_best]+[modelMAE_rescaled_val_best]+

                                [modelRMSE_rescaled_test]+[modelMAE_test]+
                                [modelscore_all]+[modelRMSE_all]+[modelRMSE_rescaled_all]+[modelMAE_all]+[modelMAE_rescaled_all]+
                                [modelscore_val]+[modelRMSE_val]+[modelRMSE_rescaled_val]+[modelMAE_val]+[modelMAE_rescaled_val]+
                                [modeltrainingloss]+[modelvalidationloss]+

                                [modelRMSE_rescaled_test_best]+[modelMAE_test_best]+
                                [modelscore_all_best]+[modelRMSE_all_best]+[modelRMSE_rescaled_all_best]+[modelMAE_all_best]+[modelMAE_rescaled_all_best]+
                                [modelRMSE_rescaled_val_best]+[modelMAE_val_best]+
                                [modeltrainingloss_best]+[modelvalidationloss_best]+[random_state_best]+   
                                                            
                                [targetvalue_test_mean_best]+[targetvalue_test_max_best]+[targetvalue_test_min_best]+[targetvalue_test_median_best]+
                                [targetvalue_train_mean_best]+[targetvalue_train_max_best]+[targetvalue_train_min_best]+[targetvalue_train_median_best]+        
                                [targetvalue_val_mean]+[targetvalue_val_max]+[targetvalue_val_min]+[targetvalue_val_median]+                                                       
                                [scalername]+[randomseeds_str]+
                                [NN_solver_i]+[NN_activation_i]+[NN_layers_names[l]]+[NN_learning_rate_init_i]+[NN_alpha_i]+[NN_minibatch_size_i]+[NN_max_iter_i]+[NN_n_iter_no_change_i]+[NN_tol_i]+
                                [NN_validation_fraction_i]+[NN_beta_1_i]+[NN_beta_2_i]+[NN_epsilon_i]+[NN_max_fun_i]+[NN_learning_rate_i]+[NN_shuffle_i]+[NN_early_stopping_i]+[NN_power_t_i]+[NN_momentum_i]+[NN_nesterovs_momentum_i]+
                                ["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+ ["none"]+["none"]+["none"]+["none"]+["none"]+["none"]
                            ]
                            if best_export_trigger == True:
                                entryname="featureset_"+str(featureset).zfill(2)+"_"+str(XGBfilterfraction).zfill(3)+"_"+str(targetlabel)+"_NN"
                                if modelRMSE_test < best_dict[entryname]:
                                    best_dict[entryname]=modelRMSE_test.copy()
                                    best_score_dict[entryname+"_scorelist"]=scores[0].copy()
                                    exportdata=pd.DataFrame.from_dict(best_score_dict,orient='index',columns=columnnames)
                                    with open(savepath+r"\\best_scores.csv", mode='w', newline='\n') as f:
                                        exportdata.to_csv(f, float_format='%.6f',index=False) 
                                    best_trigger=True
                            globalscores=globalscores+scores
                            pd_dic={"training loss" :NNmodel.loss_curve_,"validation loss":NNmodel.validation_scores_}
                            pd_dataset=pd.DataFrame(pd_dic)        
                            str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                            if loss_curve_trigger == True:
                                with open(savepath+r"\\"+str_time+"_losscurves_NN_"+NN_layers_names[l]+"_"+runname+"_"+dataname+"_"+targetlabel+".csv", mode='w', newline='\n') as f:
                                    pd_dataset.to_csv(f, float_format='%.6f',index=True) 
                            pl.figure(figsize=(1, 1), dpi=80)
                            pl.plot(list(NNmodel.loss_curve_),label='training')
                            pl.plot(list(NNmodel.validation_scores_),label='validation')
                            pl.xlabel('iteration')
                            pl.ylabel('loss')
                            pl.title("losscurves_NN_"+NN_layers_names[l]+"_"+runname+"_"+dataname+"_"+targetlabel)
                            pl.legend()
                            fig = pl.gcf()
                            fig.set_size_inches(18.5, 10.5)
                            fig.tight_layout()
                            str_time=datetime.now().strftime('%Y%m%d-%H%M%S') 
                            if best_trigger == True:
                                with open(savepath+r"\\"+entryname+"_best_losscurve.eps", mode='w') as f:
                                    pl.savefig(f,bbox_inches='tight', format='eps')
                                with open(savepath+r"\\"+entryname+"_best_losscurve.png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')   
                                with open(savepath+r"\\"+entryname+"_bestmodel", mode='wb') as f:
                                    joblib.dump(NNmodel, f)                                      
                            if loss_curve_trigger == True:
                                with open(savepath+r"\\"+str_time+"_losscurves_NN_"+NN_layers_names[l]+"_"+runname+"_"+dataname+"_"+targetlabel+".eps", mode='w') as f:
                                    pl.savefig(f,bbox_inches='tight', format='eps')
                                with open(savepath+r"\\"+str_time+"_losscurves_NN_"+NN_layers_names[l]+"_"+runname+"_"+dataname+"_"+targetlabel+".png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')
                            pl.close(True)
                            print("Ende NN")
                            pd_dataset=pd.DataFrame(columns=columnnames,data=globalscores)
                            str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                            with open(savepath+r"\\scores_saveslot_"+str(saveslotcounter)+".csv", mode='w', newline='\n') as f:
                                pd_dataset.to_csv(f, float_format='%.6f',index=True) 
                            saveslotcounter=saveslotcounter+1
                            if saveslotcounter>10:
                                saveslotcounter=1
                
                if XGB_trigger == True:
                    for XGB_objective_i,XGB_eta_i,XGB_alpha_i,XGB_gamma_i,XGB_lambda_i,XGB_minchild_weight_i,XGB_max_delta_step_i,XGB_n_estimators_i,XGB_earlystoppingrounds_i,XGB_parallel_tree_i,XGB_maxdepth_i,XGB_subsample_i,XGB_colsample_bytree_i,XGB_colsample_bylevel_i,XGB_colsample_bynode_i in product(XGB_objective,XGB_eta,XGB_alpha,XGB_gamma,XGB_lambda,XGB_minchild_weight,XGB_max_delta_step,XGB_n_estimators,XGB_earlystoppingrounds,XGB_parallel_tree,XGB_maxdepth,XGB_subsample,XGB_colsample_bytree,XGB_colsample_bylevel,XGB_colsample_bynode):
                        print("jetzt: XGB",dataname,targetlabel)
                        best_trigger=False
                        RMSE_test_rand_list=[]
                        RMSE_rescaled_test_rand_list=[]                            
                        RMSE_all_rand_list=[]
                        RMSE_rescaled_all_rand_list=[]                            
                        MAE_test_rand_list=[]
                        MAE_rescaled_test_rand_list=[]                            
                        MAE_all_rand_list=[]
                        MAE_rescaled_all_rand_list=[]                            
                        modelscore_test_rand_list=[]
                        modelscore_all_rand_list=[]
                        modeltrainingloss_rand_list=[]
                        modelvalidationloss_rand_list=[]
                        rowcount_all_rand_list=[]
                        rowcount_test_rand_list=[]
                        rowcount_train_rand_list=[]                         
                        if system_trigger==True:
                            RMSE_val_rand_list=[]
                            RMSE_rescaled_val_rand_list=[]    
                            MAE_val_rand_list=[]
                            MAE_rescaled_val_rand_list=[]     
                            modelscore_val_rand_list=[]                          
                        for XGB_seed_i in randomseeds:
                            X_train, X_test, y_train, y_test = train_test_split(data, y, test_size =0.2, random_state = XGB_seed_i)
                            targetvalue_test_mean=y_test.mean()
                            targetvalue_test_median=y_test.median()
                            targetvalue_test_max=y_test.max()
                            targetvalue_test_min=y_test.min()
                            targetvalue_train_mean=y_train.mean()
                            targetvalue_train_median=y_train.median()
                            targetvalue_train_max=y_train.max()
                            targetvalue_train_min=y_train.min()    
                            if scalernamelist[i]!="none":
                                X_valsystem=X_valsystem_org.copy()
                                y_valsystem=y_valsystem_org.copy()                                
                                he_columnnames = list(X_train.filter(regex='_he', axis=1).columns)
                                scaler = scalerlist[i]()
                                scaler.fit(X_train)

                                puffer=X_train[he_columnnames].copy().reset_index()
                                X_train = pd.DataFrame(scaler.transform(X_train),columns=data.columns)
                                X_train[he_columnnames]=puffer[he_columnnames]

                                puffer=X_test[he_columnnames].copy().reset_index()
                                X_test = pd.DataFrame(scaler.transform(X_test),columns=data.columns)
                                X_test[he_columnnames]=puffer[he_columnnames]

                                puffer=data[he_columnnames].copy().reset_index()
                                X_all = pd.DataFrame(scaler.transform(data),columns=data.columns)
                                X_all[he_columnnames]=puffer[he_columnnames]

                                y_scaler = scalerlist[i]()
                                y_scaler.fit(np.reshape(y_train.values,(-1, 1)))
                                y_train = pd.DataFrame(y_scaler.transform(np.reshape(y_train.values,(-1, 1))),columns=[targetparam[0]])
                                y_test = pd.DataFrame(y_scaler.transform(np.reshape(y_test.values,(-1, 1))),columns=[targetparam[0]])
                                y_all = pd.DataFrame(y_scaler.transform(np.reshape(y.values,(-1, 1))),columns=[targetparam[0]])

                                if system_trigger==True:
                                    puffer=X_valsystem[he_columnnames].copy().reset_index()
                                    X_valsystem = pd.DataFrame(scaler.transform(X_valsystem),columns=data.columns)
                                    X_valsystem[he_columnnames]=puffer[he_columnnames]
                                    y_valsystem = pd.DataFrame(y_scaler.transform(np.reshape(y_valsystem.values,(-1, 1))),columns=[targetparam[0]])                                

                            else:
                                X_all=data
                            XGBmodel_rand = xgb.XGBRegressor(max_depth=XGB_maxdepth_i,
                                                learning_rate=XGB_eta_i, 
                                                n_estimators=XGB_n_estimators_i,
                                                verbosity=XGB_verbosity[0],
                                                objective=XGB_objective_i,
                                                gamma=XGB_gamma_i,
                                                min_child_weight=XGB_minchild_weight_i,
                                                max_delta_step=XGB_max_delta_step_i, 
                                                subsample=XGB_subsample_i,
                                                colsample_bytree=XGB_colsample_bytree_i, 
                                                colsample_bylevel=XGB_colsample_bylevel_i,
                                                colsample_bynode=XGB_colsample_bynode_i, 
                                                reg_alpha=XGB_alpha_i, 
                                                reg_lambda=XGB_lambda_i,
                                                random_state=XGB_seed_i,
                                                num_parallel_tree=XGB_parallel_tree_i,
                                                importance_type="gain",
                                                early_stopping_round=XGB_earlystoppingrounds_i
                                                )
                            XGBmodel_rand.fit(X_train,y_train,eval_set = [(X_train, y_train), (X_test,y_test)],early_stopping_rounds=XGB_earlystoppingrounds[0])
                            XGBlossplotdata = XGBmodel_rand.evals_result()

                            modeltrainingloss=XGBlossplotdata['validation_0']['rmse'][-1]
                            modelvalidationloss=XGBlossplotdata['validation_1']['rmse'][-1]

                            predicitions_test = XGBmodel_rand.predict(X_test.values)
                            modelscore_test=XGBmodel_rand.score(X_test,y_test)
                            modelRMSE_test=mean_squared_error(y_test,predicitions_test)**(1/2)
                            modelRMSE_rescaled_test=modelRMSE_test*(targetparam[3]-targetparam[4])
                            modelMAE_test=mean_absolute_error(y_test,predicitions_test)
                            modelMAE_rescaled_test=modelMAE_test*(targetparam[3]-targetparam[4])

                            predicitions_all = XGBmodel_rand.predict(X_all.values)
                            modelscore_all=XGBmodel_rand.score(X_all,y_all)
                            modelRMSE_all=mean_squared_error(y_all,predicitions_all)**(1/2)
                            modelRMSE_rescaled_all=y_scaler.inverse_transform([[modelRMSE_all]])[0][0]
                            modelMAE_all=mean_absolute_error(y_all,predicitions_all)
                            modelMAE_rescaled_all=y_scaler.inverse_transform([[modelMAE_all]])[0][0]
                            modeltrainingloss_rand_list=modeltrainingloss_rand_list+[modeltrainingloss]
                            modelvalidationloss_rand_list=modelvalidationloss_rand_list+[modelvalidationloss]

                            rowcount_all=X_all.shape[0]
                            rowcount_test=X_test.shape[0]
                            rowcount_train=X_train.shape[0]

                            modelscore_test_rand_list=modelscore_test_rand_list+[modelscore_test]
                            RMSE_test_rand_list=RMSE_test_rand_list+[modelRMSE_test]
                            RMSE_rescaled_test_rand_list=RMSE_rescaled_test_rand_list+[modelRMSE_rescaled_test]
                            MAE_test_rand_list=MAE_test_rand_list+[modelMAE_test]
                            MAE_rescaled_test_rand_list=MAE_rescaled_test_rand_list+[modelMAE_rescaled_test]

                            modelscore_all_rand_list=modelscore_all_rand_list+[modelscore_all]
                            RMSE_all_rand_list=RMSE_all_rand_list+[modelRMSE_all]
                            RMSE_rescaled_all_rand_list=RMSE_rescaled_all_rand_list+[modelRMSE_rescaled_all]
                            MAE_all_rand_list=MAE_all_rand_list+[modelMAE_all]
                            MAE_rescaled_all_rand_list=MAE_rescaled_all_rand_list+[modelMAE_rescaled_all]

                            rowcount_all_rand_list=rowcount_all_rand_list+rowcount_all
                            rowcount_test_rand_list=rowcount_test_rand_list+rowcount_test
                            rowcount_train_rand_list=rowcount_train_rand_list+rowcount_train
                            if system_trigger==True:
                                predicitions_val = XGBmodel_rand.predict(X_valsystem.values)
                                modelscore_val=XGBmodel_rand.score(X_valsystem,y_valsystem)
                                modelRMSE_val=mean_squared_error(y_valsystem,predicitions_val)**(1/2)
                                modelMAE_val=mean_absolute_error(y_valsystem,predicitions_val)
                                if modelRMSE_val < 10:
                                    modelRMSE_rescaled_val=y_scaler.inverse_transform([[modelRMSE_val]])[0][0]
                                else:         
                                    modelRMSE_rescaled_val=999999  
                                if modelMAE_val < 10:
                                    modelMAE_rescaled_val=y_scaler.inverse_transform([[modelMAE_val]])[0][0]          
                                else:
                                    modelMAE_rescaled_val=999999       
                                  
                                modelscore_val_rand_list=modelscore_val_rand_list+[modelscore_val]
                                RMSE_val_rand_list=RMSE_val_rand_list+[modelRMSE_val]
                                RMSE_rescaled_val_rand_list=RMSE_rescaled_val_rand_list+[modelRMSE_rescaled_val]
                                MAE_val_rand_list=MAE_val_rand_list+[modelMAE_val]
                                MAE_rescaled_val_rand_list=MAE_rescaled_val_rand_list+[modelMAE_rescaled_val]

                            if modelRMSE_test <= min(RMSE_test_rand_list):
                                XGBmodel=XGBmodel_rand
                                modelscore_test_best=modelscore_test
                                modelRMSE_test_best=modelRMSE_test
                                modelRMSE_rescaled_test_best=modelRMSE_rescaled_test
                                modelMAE_test_best=modelMAE_test
                                modelMAE_rescaled_test_best=modelMAE_rescaled_test    
                                modelscore_all_best=modelscore_all   
                                modelRMSE_all_best=modelRMSE_all   
                                modelRMSE_rescaled_all_best=modelRMSE_rescaled_all       
                                modelMAE_all_best=modelMAE_all    
                                modelMAE_rescaled_all_best=modelMAE_rescaled_all   
                                random_state_best=XGB_seed_i
                                modeltrainingloss_best=modeltrainingloss
                                modelvalidationloss_best=modelvalidationloss
                                targetvalue_test_mean_best=targetvalue_test_mean
                                targetvalue_test_median_best=targetvalue_test_median
                                targetvalue_test_max_best=targetvalue_test_max
                                targetvalue_test_min_best=targetvalue_test_min
                                targetvalue_train_mean_best=targetvalue_train_mean
                                targetvalue_train_median_best=targetvalue_train_median
                                targetvalue_train_max_best=targetvalue_train_max
                                targetvalue_train_min_best=targetvalue_train_min   
                                rowcount_all_best=X_all.shape[0]
                                rowcount_test_best=X_test.shape[0]
                                rowcount_train_best=X_train.shape[0]                                    
                                if system_trigger==True:
                                    modelscore_val_best=modelscore_val   
                                    modelRMSE_val_best=modelRMSE_val   
                                    modelRMSE_rescaled_val_best=modelRMSE_rescaled_val       
                                    modelMAE_val_best=modelMAE_val   
                                    modelMAE_rescaled_val_best=modelMAE_rescaled_val                                                            


                        modeltrainingloss=np.mean(modeltrainingloss_rand_list)
                        modelvalidationloss=np.mean(modelvalidationloss_rand_list)

                        modelscore_test=np.mean(modelscore_test_rand_list)
                        modelRMSE_test=np.mean(RMSE_test_rand_list)
                        modelRMSE_rescaled_test=np.mean(RMSE_rescaled_test_rand_list)
                        modelMAE_test=np.mean(MAE_test_rand_list)
                        modelMAE_rescaled_test=np.mean(MAE_rescaled_test_rand_list)


                        modelscore_all=np.mean(modelscore_all_rand_list)
                        modelRMSE_all=np.mean(RMSE_all_rand_list)
                        modelRMSE_rescaled_all=np.mean(RMSE_rescaled_all_rand_list)
                        modelMAE_all=np.mean(MAE_all_rand_list)
                        modelMAE_rescaled_all=np.mean(MAE_rescaled_all_rand_list)    

                        rowcount_all_best=np.mean(rowcount_all_rand_list)
                        rowcount_test_best=np.mean(rowcount_test_rand_list)
                        rowcount_train_best=np.mean(rowcount_train_rand_list)
                        featurecount=X_train.shape[1]

                        if system_trigger==True:
                            modelscore_val=np.mean(modelscore_val_rand_list)
                            modelRMSE_val=np.mean(RMSE_val_rand_list)
                            modelRMSE_rescaled_val=np.mean(RMSE_rescaled_val_rand_list)
                            modelMAE_val=np.mean(MAE_val_rand_list)
                            modelMAE_rescaled_val=np.mean(MAE_rescaled_val_rand_list)

                        scores=[
                            ["XGB"]+[runname]+[dataname]+[targetlabel]+[valsystem_str]+[rowcount_all]+[rowcount_test]+[rowcount_train]+[rowcount_val]+[featurecount]+[XGBfilterfraction]+
                            [modelscore_test]+[modelRMSE_test]+[modelMAE_rescaled_test]+
                            [modelscore_test_best]+[modelRMSE_test_best]+[modelMAE_rescaled_test_best]+
                            [modelscore_val_best]+[modelRMSE_val_best]+[modelMAE_rescaled_val_best]+

                            [modelRMSE_rescaled_test]+[modelMAE_test]+
                            [modelscore_all]+[modelRMSE_all]+[modelRMSE_rescaled_all]+[modelMAE_all]+[modelMAE_rescaled_all]+
                            [modelscore_val]+[modelRMSE_val]+[modelRMSE_rescaled_val]+[modelMAE_val]+[modelMAE_rescaled_val]+
                            [modeltrainingloss]+[modelvalidationloss]+

                            [modelRMSE_rescaled_test_best]+[modelMAE_test_best]+
                            [modelscore_all_best]+[modelRMSE_all_best]+[modelRMSE_rescaled_all_best]+[modelMAE_all_best]+[modelMAE_rescaled_all_best]+
                            [modelRMSE_rescaled_val_best]+[modelMAE_val_best]+
                            [modeltrainingloss_best]+[modelvalidationloss_best]+[random_state_best]+   
                                                        
                            [targetvalue_test_mean_best]+[targetvalue_test_max_best]+[targetvalue_test_min_best]+[targetvalue_test_median_best]+
                            [targetvalue_train_mean_best]+[targetvalue_train_max_best]+[targetvalue_train_min_best]+[targetvalue_train_median_best]+  
                            [targetvalue_val_mean]+[targetvalue_val_max]+[targetvalue_val_min]+[targetvalue_val_median]+                               
                            [scalername]+[randomseeds_str]+
                            ["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+["none"]+
                            [XGB_objective_i]+[XGB_n_estimators_i]+[XGB_parallel_tree_i]+[XGB_maxdepth_i]+[XGB_eta_i]+[XGB_earlystoppingrounds_i]+[XGB_minchild_weight_i]+[XGB_max_delta_step_i]+
                            [XGB_subsample_i]+[XGB_colsample_bytree_i]+[XGB_colsample_bylevel_i]+[XGB_colsample_bynode_i]+[XGB_alpha_i]+[XGB_lambda_i]+[XGB_gamma_i]
                        ]
                        if best_export_trigger == True:
                            entryname="featureset_"+str(featureset).zfill(2)+"_"+str(XGBfilterfraction).zfill(3)+"_"+str(targetlabel)+"_XGB"
                            if modelRMSE_test < best_dict[entryname]:
                                best_dict[entryname]=modelRMSE_test.copy()
                                best_score_dict[entryname+"_scorelist"]=scores[0].copy()
                                exportdata=pd.DataFrame.from_dict(best_score_dict,orient='index',columns=columnnames)
                                with open(savepath+r"\\best_scores.csv", mode='w', newline='\n') as f:
                                    exportdata.to_csv(f, float_format='%.6f',index=False) 
                                best_trigger=True


                        #with open(savepath+r"\\"+str_time+"_XGB_filtermodel_"+runname+"_"+dataname+"_"+targetlabel+"_eta"+str(XGB_eta)+"_alpha"+str(XGB_alpha)+"_paralleltree"+str(XGB_parallel_tree)+"maxdepth"+str(XGB_maxdepth), mode='wb') as f:
                        #    joblib.dump(XGBmodel, f)
                        if savemodel_trigger==True:
                            with open(savepath+r"\\"+str_time+"_XGB_filtermodel_"+runname+"_"+dataname+"_"+targetlabel, mode='wb') as f:
                                joblib.dump(XGBmodel, f)
                        globalscores=globalscores+scores
                        pl.figure(figsize=(1, 1), dpi=80)
                        pl.plot(XGBlossplotdata['validation_0']['rmse'], label='train')
                        pl.plot(XGBlossplotdata['validation_1']['rmse'], label='test')
                        pl.xlabel('iteration')
                        pl.ylabel('loss')
                        pl.title("losscurves_XGB_"+runname+"_"+dataname+"_"+targetlabel+"_"+scalername)
                        pl.legend()
                        fig = pl.gcf()
                        fig.set_size_inches(18.5, 10.5)
                        fig.tight_layout()
                        str_time=datetime.now().strftime('%Y%m%d-%H%M%S') 
                        if loss_curve_trigger == True:
                            with open(savepath+r"\\"+str_time+"_losscurves_XGB_"+runname+"_"+dataname+"_"+targetlabel+"_eta"+str(XGB_eta_i)+"_alpha"+str(XGB_alpha_i)+"_paralleltree"+str(XGB_parallel_tree_i)+"maxdepth"+str(XGB_maxdepth_i)+".eps", mode='w') as f:
                                pl.savefig(f,bbox_inches='tight', format='eps')
                            with open(savepath+r"\\"+str_time+"_losscurves_XGB_"+runname+"_"+dataname+"_"+targetlabel+"_eta"+str(XGB_eta_i)+"_alpha"+str(XGB_alpha_i)+"_paralleltree"+str(XGB_parallel_tree_i)+"maxdepth"+str(XGB_maxdepth_i)+".png", mode='wb') as f:
                                pl.savefig(f,bbox_inches='tight', format='png')
                        if best_trigger == True:
                            with open(savepath+r"\\"+entryname+"_best_losscurve.eps", mode='w') as f:
                                pl.savefig(f,bbox_inches='tight', format='eps')
                            with open(savepath+r"\\"+entryname+"_best_losscurve.png", mode='wb') as f:
                                pl.savefig(f,bbox_inches='tight', format='png')       
                            with open(savepath+r"\\"+entryname+"_bestmodel", mode='wb') as f:
                                joblib.dump(XGBmodel, f)                     
                        pl.close(True)
                        explainer = shap.TreeExplainer(XGBmodel)
                        shap_values = explainer.shap_values(X_all.values)
                        shap_values_abs=np.sum(abs(shap_values),axis=0)

                        xgbidentifier=str("XGB_"+runname+"_"+dataname+"_"+targetlabel+"_"+scalername)
                        pd_dic={"xgbidentifier":xgbidentifier,"importance gain" :XGBmodel.feature_importances_,"shap value":shap_values_abs}
                        pd_dataset=pd.DataFrame(pd_dic)        
                        str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                        if importance_trigger == True:
                            with open(savepath+r"\\"+str_time+"_importance_XGB_"+runname+"_"+dataname+"_"+targetlabel+"_eta"+str(XGB_eta_i)+"_alpha"+str(XGB_alpha_i)+"_paralleltree"+str(XGB_parallel_tree_i)+"maxdepth"+str(XGB_maxdepth_i)+".csv", mode='w', newline='\n') as f:
                                pd_dataset.to_csv(f, float_format='%.6f',index=True) 

                        str_time=datetime.now().strftime('%Y%m%d-%H%M%S') 



                        if ((SHAP_trigger ==True) or (best_trigger==True)):

                            #SHAP PLOT
                            pl.figure(figsize=(8.3,11.7))
                            pl.title("SHAP XGB_"+runname+"_"+dataname+"_"+targetlabel+"_"+scalername+", count datapoints:"+str(X_all.shape[0]))
                            shap.summary_plot(shap_values,pd.DataFrame(columns=X_all.columns,data=X_all.values),max_display=30,show=False)
                            fig = pl.gcf()
                            fig.set_size_inches(8.3, 11.7)
                            #fig.tight_layout()
                            str_time=datetime.now().strftime('%Y%m%d-%H%M%S')

                            if SHAP_trigger == True:
                                with open(savepath+r"\\"+str_time+"_SHAPplot_XGB_"+runname+"_"+dataname+"_"+targetlabel+"_eta"+str(XGB_eta_i)+"_alpha"+str(XGB_alpha_i)+"_paralleltree"+str(XGB_parallel_tree_i)+"maxdepth"+str(XGB_maxdepth_i)+".png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')
                                #with open(savepath+r"\\"+str_time+"_SHAPplot_XGB_"+runname+"_"+dataname+"_"+targetlabel+"_eta"+str(XGB_eta_i)+"_alpha"+str(XGB_alpha_i)+"_paralleltree"+str(XGB_parallel_tree_i)+"maxdepth"+str(XGB_maxdepth_i)+".eps", mode='w') as f:
                            #       pl.savefig(f,bbox_inches='tight', format='eps')
                            if best_trigger == True:
                                with open(savepath+r"\\"+entryname+"_best_SHAP_plot.png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')
                                with open(savepath+r"\\"+entryname+"_best_SHAP_plot.eps", mode='w') as f:
                                    pl.savefig(f,bbox_inches='tight', format='eps')
                            pl.close(True)

                            #SHAP DECISION PLOT
                            pl.figure(figsize=(8.3,11.7))
                            pl.title("SHAP decision plot XGB_"+runname+"_"+dataname+"_"+targetlabel+", count datapoints:"+str(X_all.shape[0]))
                            #s=list(np.argsort(-(abs(shap_values)).mean(0)))
                            #shap_data=X_pd.iloc[:,s]
                            #shap_values = explainer.shap_values(shap_data)
                            expected_value = explainer.expected_value
                            
                            shapdpsubset=list(random.sample(range(len(shap_values)), 20))
                            shap.decision_plot(expected_value,shap_values[shapdpsubset],X_all.iloc[shapdpsubset],show=False,feature_order="importance",feature_display_range=slice(-1, -31, -1))
                            fig = pl.gcf()
                            fig.set_size_inches(8.3, 11.7)            
                            str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                            #with open(savepath+r"\\"+str_time+"_SHAPdecisionplot_XGB_"+runname+"_"+dataname+"_"+targetlabel+"_eta"+str(XGB_eta_i)+"_alpha"+str(XGB_alpha_i)+"_paralleltree"+str(XGB_parallel_tree_i)+"maxdepth"+str(XGB_maxdepth_i)+".eps", mode='w') as f:
                            #    pl.savefig(f,bbox_inches='tight', format='eps')
                            if SHAP_trigger == True:
                                with open(savepath+r"\\"+str_time+"_SHAPdecisionplot_XGB_"+runname+"_"+dataname+"_"+targetlabel+"_eta"+str(XGB_eta_i)+"_alpha"+str(XGB_alpha_i)+"_paralleltree"+str(XGB_parallel_tree_i)+"maxdepth"+str(XGB_maxdepth_i)+".png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')      
                            if best_trigger == True:                          
                                with open(savepath+r"\\"+entryname+"_best_SHAP_decision_plot.png", mode='wb') as f:
                                    pl.savefig(f,bbox_inches='tight', format='png')
                                with open(savepath+r"\\"+entryname+"_best_SHAP_decision_plot.eps", mode='w') as f:
                                   pl.savefig(f,bbox_inches='tight', format='eps')
                            pl.close(True)
                        pd_dataset=pd.DataFrame(columns=columnnames,data=globalscores)
                        str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
                        with open(savepath+r"\\scores_saveslot_"+str(saveslotcounter)+".csv", mode='w', newline='\n') as f:
                            pd_dataset.to_csv(f, float_format='%.6f',index=True) 
                        saveslotcounter=saveslotcounter+1
                        if saveslotcounter>10:
                            saveslotcounter=1
                        print("Ende XGBoost")


if ((XGB_trigger == True) & (NN_trigger == False)):
    pd_dataset=pd.DataFrame(columns=columnnames,data=globalscores)
    str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
    with open(savepath+r"\\"+str_time+"_scores_"+runname+"_"+folderexplanation+".csv", mode='w', newline='\n') as f:
        pd_dataset.to_csv(f, float_format='%.6f',index=True) 
    "Ende, XGB Modell"
elif ((XGB_trigger == False) & (NN_trigger == True)):
    pd_dataset=pd.DataFrame(columns=columnnames,data=globalscores)
    str_time=datetime.now().strftime('%Y%m%d-%H%M%S')
    with open(savepath+r"\\"+str_time+"_scores_"+runname+"_"+folderexplanation+".csv", mode='w', newline='\n') as f:
        pd_dataset.to_csv(f, float_format='%.6f',index=True) 
    "Ende, NN Modell"
elif ((XGB_trigger == True) & (NN_trigger == True)):
    pd_dataset=pd.DataFrame(columns=columnnames,data=globalscores)
    str_time=datetime.now().strftime('%Y%m%d-%H%M%S')   
    with open(savepath+r"\\"+str_time+"_scores_"+runname+"_"+folderexplanation+".csv", mode='w', newline='\n') as f:
        pd_dataset.to_csv(f, float_format='%.6f',index=True)
    "Ende, XGB und NN Modell"
else:
    "Ende, kein Modell"
