# -*- coding: utf-8 -*-
import pandas as pd
import pickle
# Label encode features
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support
import sqlite3

from sklearn.metrics import confusion_matrix ,classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV




def trimWhiteSpace(data,catVars):
    for col in data[catVars]:
        data[col]=[entry.strip() for entry in data[col]]
    return data


def missingValImputation(data,idCols):
    ###### Impute missing Values
    # Preprocessing for categorical data
    numVars=['TotalLoanAmount', 'CreditScore', 'CLTV', 'DTI', 'BorrowerAge', 'TotalIncome']
    catVars=['LeadSourceGroup', 'LoanType', 'BorrowerOwnRent','Education','ZipCode']
    
    numerical_transformer = SimpleImputer(strategy = "mean")
    categorical_transformer = SimpleImputer(strategy='most_frequent')
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numVars),
            ('cat', categorical_transformer, catVars)
        ])
    
    #print('numVars Summary :',data[numVars].isna().sum())
    #print('catVars Summary :',data[catVars].isna().sum())
    
    newData=pd.concat([data[idCols],pd.DataFrame(data=preprocessor.fit_transform(data[numVars+catVars]),columns=numVars+catVars),data['Approved']],axis=1)
    #print('New numVars Summary :',data[numVars].describe(),'\n')
    #print('New catVars Summary :',data[catVars].describe())
    #newData['Approved']=data['Approved']
    print('preprocessed data columns :',newData.columns)
    newData=newData.infer_objects()
    #newData[numVars] = newData[numVars].apply(pd.to_numeric) 
    
    #print('describe data :', newData.describe())
    #Change to right numeric data type
    #numVars=['TotalLoanAmount', 'CreditScore', 'CLTV', 'DTI', 'BorrowerAge', 'TotalIncome']
    #newData[numVars]=newData[numVars].apply(pd.to_numeric,errors='ignore')
    
    print('new numVars DataType :',newData.dtypes)
    
    return newData , preprocessor


def groupLSR(data):
    #Regroup LeadSourceGroup
    LSR = ['Internet', 'TV','Radio','Repeat Client' ]
    data['LeadSourceGroup'] = [v if v in LSR else 'Other' for v in data.LeadSourceGroup]
    return data

def groupZipCode(data):
    zips = ['75', '76', '77', '78', '79']
    data['ZipCode'] = [str(zp)[:2] for zp in data['ZipCode']]
    data['ZipCode'] = [v if v in zips else 'OtherZP' for v in data.ZipCode]
    print(data['ZipCode'].unique())
    return data

def treatNegCol(data,col,dataType):
    #Impute negative value & missing values with mode
    if dataType=='Cat':
        data[col][data[col]<=0]=data[col].mode()
    elif dataType=='Num':
        data[col][data[col]<=0]=data[col].mean()
    else:
        pass
    return data


def groupEdu(data):
    #Create bins for Education
    bins=[0,12,16,18,data.BorrowerYearsInSchool.max()]
    if bins[-1]==bins[-2]:
        bins[-1]=30
    group=['Higher School','UnderGrad','PostGrad','PHD']
    data['Education']=pd.cut(data['BorrowerYearsInSchool'],bins,labels=group)
    return data

def encodeNew(data,nominal,ordinal,pipeline):
    encDf = pd.get_dummies(data[nominal] , drop_first=True)
    
    if pipeline=='Train':
        encols=encDf.columns.values
        pickle.dump(encols, open('./outputs/encols.pkl' ,'wb'))
    else:
        encols = pickle.load(encols, open('./outputs/encols.pkl' ,'wb'))
        encDf = encDf.reindex(encols,axis='columns')
    return encDf

def encode(df,nominal,ordinal):
    catVars= nominal + ordinal
    
    #Define encoder
    encoder = ColumnTransformer(transformers =[ 
            ('OHEenc', OneHotEncoder(sparse = False, handle_unknown ='ignore'), nominal),
            ('labelenc',   OrdinalEncoder() ,ordinal) ], remainder ='passthrough')
    
    #Encode catVars
    encDf=pd.DataFrame(encoder.fit_transform(df[catVars])).reset_index(drop=True)
    encDf.columns=[entry.split('_')[-1] for entry in list(encoder.named_transformers_['OHEenc'].get_feature_names())]+ordinal
    print('shape1',encDf.shape)
    
    NumDf = df.drop(catVars, axis=1).reset_index(drop=True)
    print('shape2',NumDf.shape)
    df_ = pd.concat([encDf, NumDf], axis=1)
    df_.columns=[entry.strip() for entry in df_.columns]       
    print('shape3',df_.shape)
    return df_ ,encoder


def encodeTest(encoder,df,catVars):
    ordinal=['Education']
    encDf=pd.DataFrame(encoder.transform(df))
    encDf.columns=[entry.split('_')[-1] for entry in list(encoder.named_transformers_['OHEenc'].get_feature_names())]+ordinal
    print('Encoded Data columns :',encDf.columns)
    NumDf = df.drop(catVars, axis=1).reset_index(drop=True)
    print('Numerical Data columns',NumDf.columns)
    df_ = pd.concat([encDf, NumDf], axis=1)
    df_.columns=[entry.strip() for entry in df_.columns]
    print('Total Data columns',df_.columns)
    return df_


# Write a function for Grid Search 
def GridSearch(model,param_grid,cv,X,y,scoring=None):
    # Random search of parameters
    gcv = GridSearchCV(estimator = model, param_grid=param_grid , refit = True, cv = cv, verbose=1,  n_jobs = -1)
    # Fit the model
    gcv.fit(X,y)

    return gcv.best_params_

def modelPerformance(model,X,y,dataType):
    print("=== Model Performance report on ",dataType," Data ===")
    y_pred = model.predict(X)
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y, y_pred))
    precision,recall,f1score,xyz=precision_recall_fscore_support(y, y_pred, average='weighted')
    print('\n')
    
    print("=== Classification Report ===")
    print(classification_report(y, y_pred))
    return precision,recall,f1score
    
    

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn













