# -*- coding: utf-8 -*-



print("========Hello World========")
# -*- coding: utf-8 -*-

print("=====prepareScore file has been hit====")



import yaml
import pickle
import pyodbc
import pandas as pd
import sqlite3
import random
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from azureml.core.model import Model
from azureml.core import Run
from utils import *
#from utils import modelPerformance ,GridSearch ,encode,groupEdu,groupLSR,treatNegCol,groupZipCode,groupZipCode


## Load configuration file ###
with open(r"./configs/config.yaml",'r') as file:
    config=yaml.load(file,Loader=yaml.FullLoader)
    
params=config['params']

db=config['dbDetails']['dbName']


if os.path.isfile(db):
    print('Database already exists')
else:
    create_connection(db)
    print('Database created successfully')

conn=sqlite3.connect(db)
cur=conn.cursor()

#get the count of tables with the name
cur.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='ScoringTable' ''')




### Read data from DB

query="""select * from ScoringTable"""

leftDf = pd.read_sql(query, conn)

leftDf.shape
leftDf.columns

grpd=result.groupby(['Run_id'])['TP','FP','TN','FN'].sum()


#### Evaluate Accuracy ,  Precision, Recall ,F1 score .

grpd['Accuracy']=((grpd['TP']+grpd['TN'])/(grpd['TP']+grpd['FP']+grpd['FN']+grpd['TN']))
grpd['Precision']=((grpd['TP'])/(grpd['TP']+grpd['FP']))
grpd['Recall']=((grpd['TP'])/(grpd['TP']+grpd['FN']))
grpd['Specificity']=((grpd['TN'])/(grpd['TN']+grpd['FP']))
grpd['F1_Score']=2*((grpd['Precision']*grpd['Recall'])/(grpd['Precision']+grpd['Recall']))
grpd=grpd.round(2)
grpd.reset_index(inplace=True)

grpd.columns


#if the count is 1, then table exists
if cur.fetchone()[0]==1 :
    print('Table exists already')
else:
    # Create table - Jobsdata
    # Create table - Jobsdata
    cur.execute('''CREATE TABLE ModelPerformance
                 ( [Run_id] INTEGER ,  [TP] INTEGER ,  [FP] INTEGER ,  
                  [TN] INTEGER ,  [FN] INTEGER ,  [Accuracy] INTEGER ,  
                  [Precision] INTEGER ,  [Recall] INTEGER ,[Specificity] INTEGER ,
                  [F1_Score] INTEGER''')
                 
                 
 
grpd.to_sql('ModelPerformance',conn, if_exists='append', index = False) # Insert the values from the csv file into the table 'jobsData' 


cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur.fetchall())

conn.commit()
conn.close()


















