# -*- coding: utf-8 -*-

print("====== Model python file has been hit======")

import argparse
import pickle
import pandas as pd
from utils import modelPerformance ,GridSearch
from azureml.core import Run
from azureml.core.model import Model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix ,classification_report
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

########Model Building ########

########## Split Features and Target Varible ############

#Access from local system
ws=Workspace.from_config()
ws.get_details()


#Create Azure ML Experiment
exp=Experiment(workspace=ws,name='ModelTraining')

#Start logging metrics
import datetime
run=exp.start_logging()
run.log("Experiement start time",str(datetime.datetime.now()))



print("====Model args1=====")
parser = argparse.ArgumentParser()
print("====Model args2=====")
parser.add_argument("--input",dest="input")
print("====Model args3=====")
parser.add_argument("--output",dest="output")
print("====Model args4=====")
args = parser.parse_args()
print('args.input :',(args.input))
print('args.output :',(args.output))

data=pd.read_csv(args.input)

#data=pd.read_csv('C:/Visionet/Projects/LoanApprovalModel/Data/data.csv')
print('data columns :',data.columns)

idCols=['LeadID','DateAdded']
targCol=['Approved']

X = data.drop(columns=idCols+targCol)
y = data['Approved']
    
print('X columns :',X .columns)

################# Splitting into Train -Test Data #######

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify =y,random_state =42)

X_train=pd.DataFrame(data=X_train)
X_test=pd.DataFrame(X_test)
y_train=pd.DataFrame(y_train)
y_test =pd.DataFrame(y_test)

# Random forest model creation

#Model performance on CV Training Data

param_grid = {
              'max_depth': [2, 3,4 ,5],
              'min_samples_leaf': [1,3, 5, 10,],
              'min_samples_split':[2,3,4],
              'max_features':['auto','sqrt'],
              'n_estimators':[100,200,300,400]
              }


# random forest model creation
rfc = RandomForestClassifier(class_weight="balanced",random_state=123)

best_params=GridSearch(rfc,param_grid,3,X_train,y_train)


print("====== Best params are found in model file======")

rfc = RandomForestClassifier(
                             max_depth=best_params['max_depth'], 
                             max_features=best_params['max_features'],
                             min_samples_split=best_params['min_samples_split'], 
                             min_samples_leaf=best_params['min_samples_leaf'],
                             n_estimators=best_params['n_estimators'],
                             class_weight="balanced"
                            )
rfc.fit(X_train,y_train)


#pickle.dump(rfc, open('C://Projects/LoanApprovalModel/outputs/model.pkl' ,'wb'))



#precision,recall,f1score=modelPerformance(rfc,X_train,y_train,'Train')






#print("precision,recall,f1score :",precision,recall,f1score)
    

# =============================================================================
# #Model performance on Test Data
# modelPerformance(rfc,X_test,y_test,'Test')
# 
# =============================================================================


# save the model to disk
filename = args.output
pickle.dump(rfc, open(filename, 'wb'))


print("====== Model pickle file hass been generated======")

print("====== Mworkspace details have been taken======")


#Register the model
model=Model.register(model_path='./outputs/rfc_Model.sav',
                     model_name="RandomForestModel",
                     tags={"Key":"1"},
                     description='Loan Approval Prediction',
                     workspace=ws)

print("model name :",model.name)
print("model version :",model.version)

print("====== Model has been registered======")














