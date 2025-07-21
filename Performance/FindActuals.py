# -*- coding: utf-8 -*-



print("========Hello World========")
# -*- coding: utf-8 -*-

print("=====actual file has been hit====")

##Import Azure SDK modules
import azureml.core
from azureml.core import Workspace ,Experiment
from azureml.core.model import Model
from azureml.core import ScriptRunConfig
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule
from azureml.core import Run

import yaml
import pickle
import pyodbc
import pandas as pd
import sqlite3
import urllib
import random
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from azureml.core.model import Model
from azureml.core import Run
from utils import *
#from utils import modelPerformance ,GridSearch ,encode,groupEdu,groupLSR,treatNegCol,groupZipCode,groupZipCode


#Access from local system
ws=Workspace.from_config()
ws.get_details()


#Create Azure ML Experiment
exp=Experiment(workspace=ws,name='LAPexpv02')

#Start logging metrics
import datetime

run=exp.start_logging()
run.log("Experiement start time",str(datetime.datetime.now()))


print(pyodbc.drivers())

## Load configuration file ###
with open(r"./configs/config.yaml",'r') as file:
    config=yaml.load(file,Loader=yaml.FullLoader)
    
params=config['params']

server = params['server']
database = params['database']
username = params['username']
password = params['password']
driver= params['driver']
print("-----===Driver", driver, server)
params = urllib.parse.quote_plus('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)




#Establish session with DB
Session = sessionmaker(bind=engine)
session = Session()

print("===== Server paramas have been loaded====")


print('Connection has been established')

query = """WITH CTE_DateAdded
    AS (
    SELECT f.[DateAdded],f.[LeadID],f.[LeadSourceGroup],f.[User],f.[Group],f.[ZipCode],f.LQB_LoanNumber AS LoanNumber
    FROM VLF.DimLead AS f WHERE f.isActive = 1),CTE_FL as (select Tfact.LoanNumber,(case when ApprovalDateID != -1 then
    CONVERT(date, CONVERT(varchar(8), ApprovalDateID), 110) end) as ApprovalDate, BranchCode,LoanPurpose,LoanType,
    TotalLoanAmount,CreditScore,CLTV,DTI, BorrowerAge,BorrowerYearsInSchool,BorrowerTotalMonthlyIncome,BorrowerOwnRent,
    BorrowerCity, BorrowerState,IsCoBorrowerower,CoBorrowerYearsInSchool,CoBorrowerAge,CoBorrowerTotalMonthlyIncome
    from LQB.DimBorrower as Tdimbor,LQB.DimSubjectProperty as TdimSub,LQB.FactLoan as Tfact,LQB.lkpBranch as Tbran,
    LQB.lkpLoanPurpose as Tloanp,LQB.lkpLoanType as Tloant,LQB.lkpMileStone as Tmile
    where
    Tbran.BranchID = Tfact.BranchID
    and Tloanp.LoanPurposeID = Tfact.LoanPurposeID
    and Tloant.LoanTypeID = Tfact.LoanTypeID
    and Tmile.MileStoneID = Tfact.MileStoneID
    and Tfact.BorrowerID = Tdimbor.BorrowerID
    and Tfact.PropertyID = TdimSub.PropertyID
    )
    ,
    final_table as
    (
    select C.[LeadID]
    ,C.[DateAdded]
    ,C.[LeadSourceGroup]
    ,C.[User]
    ,C.[Group]
    ,C.[ZipCode]
    ,A.*
    from CTE_DateAdded AS C
    INNER JOIN CTE_FL AS A ON C.LoanNumber=A.LoanNumber
    )

    select * from final_table where DateAdded >= DATEADD(DAY,-90, GETDATE())"""

data = pd.read_sql(query, engine)

# =============================================================================
# minDate='2021-01-01'
# maxDate='2021-01-31'
# data=data[(data['DateAdded']>minDate) & (data['DateAdded']<maxDate)]
# 
# =============================================================================

print("Data has been loaded successfully")
#df_=data.copy(deep=True)

######## Create Approved Target column #############
data['Approved']='Yes'
data['Approved'][data.ApprovalDate.isna()]='No'

#Calculate difference between current date and DateAdded in days
data['Diff']=pd.Timestamp.now().normalize()-data['DateAdded'][data.ApprovalDate.isna()]
data['Diff']=[entry.days for entry in data['Diff']]

data['Approved']=['No' if ((x>=70) & (y=='No')) else 'Yes' for (x,y) in zip(data.Diff,data.Approved)  ]

# Change data type to category
data['Approved']=data['Approved'].astype('category')

# Map Target column for correlation Matrix
data.Approved.replace({'Yes':1,'No':0},inplace=True)

data.Approved.unique()
data.Approved.value_counts()
data=data[['LeadID','Approved']]

data.rename(columns={'Approved':'Actual'},inplace=True)


### Read data from DB
tableName='AzTableGH'

query='''select * from '''+ tableName

leftDf = pd.read_sql(query, engine)

leftDf.shape
print('Columns of leftDf :',leftDf.columns)
print('Columns of data :',data.columns)

result = pd.merge(leftDf.drop(['Actual'],axis=1),data, on="LeadID",how="left")

result.shape
print('1st result.columns :',result.columns)
    
## Build Evaluation Matrix

thre=0.50
result['TP'],result['TN'],result['FP'],result['FN']=0,0,0,0
result['TP'].loc[(result['Prediction']>=thre) & (result['Actual']==1)]=1
result['TN'].loc[(result['Prediction']<thre) & (result['Actual']==0)]=1
result['FP'].loc[(result['Prediction']>=thre) & (result['Actual']==0)]=1
result['FN'].loc[(result['Prediction']<thre) & (result['Actual']==1)]=1

print('result.columns :',result.columns)

print('head of result :',result[['Prediction', 'ModelVersion', 'Run_id','Actual']].head())
 

print('Columns of result :',result.columns)
print('Missing values of result :',result.isna().sum())

session.execute('''SET IDENTITY_INSERT dbo.''' + tableName + " ON")
session.commit()

session.execute('''TRUNCATE TABLE '''+tableName)
session.commit()

result.to_sql( tableName , con=engine,schema='dbo', if_exists='append', index = False) 

#result['Run_id']=random.choices(range(1000, 1010),  k = result.shape[0])

grpd=result.groupby(['Model Run Date'])['TP','FP','TN','FN'].sum()

#### Evaluate Accuracy ,  Precision, Recall ,F1 score .

grpd['Accuracy']=((grpd['TP']+grpd['TN'])/(grpd['TP']+grpd['FP']+grpd['FN']+grpd['TN']))
grpd['Precision']=((grpd['TP'])/(grpd['TP']+grpd['FP']))
grpd['Recall']=((grpd['TP'])/(grpd['TP']+grpd['FN']))
grpd['Specificity']=((grpd['TN'])/(grpd['TN']+grpd['FP']))
grpd['F1_Score']=2*((grpd['Precision']*grpd['Recall'])/(grpd['Precision']+grpd['Recall']))
grpd=grpd.round(2)

grpd.reset_index(inplace=True)

print('columns of grpd data :',grpd.columns)

print('head of grpd data :',grpd.head())


tableName='AzModelPerformanceGH'
if not engine.dialect.has_table(engine, tableName):  # If table don't exist, Create.
    metadata = MetaData(engine)
    # Create a table with the appropriate Columns
    Table(tableName, metadata,
          Column('Model Run Date', Date, primary_key=True, nullable=False), 
          Column('TP', Integer), Column('FP', Integer), 
          Column('TN', Integer), Column('FN', Integer), 
          Column('Accuracy', Float), Column('Precision', Float), 
          Column('Recall', Float), Column('Specificity', Float), 
          Column('F1_Score', Float)          
          )
    # Implement the creation
    metadata.create_all()

query='''select * from ''' +   tableName  
df=pd.read_sql(query,engine)

print("columns of table :",df.columns)    

grpd.to_sql(tableName,con=engine,schema='dbo',if_exists='append', index = False) # Insert the values from the csv file into the table 'jobsData' 

print(" results have been pushed in table")
