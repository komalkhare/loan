

print("========Hello World========")
# -*- coding: utf-8 -*-

print("=====prepareScore file has been hit====")



import yaml
import pickle
import pyodbc
import pandas as pd
import sqlite3
import numpy as np
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
import urllib
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from azureml.core.model import Model
from azureml.core import Run
from utils import *
from datetime import datetime
from datetime import date
#from utils import modelPerformance ,GridSearch ,encode,groupEdu,groupLSR,treatNegCol,groupZipCode,groupZipCode


## Load configuration file ###
with open(r"./configs/config.yaml",'r') as file:
    config=yaml.load(file,Loader=yaml.FullLoader)
    
params=config['params']

server = params['server']
database = params['database']
username = params['username']
password = params['password']
driver= params['driver']

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

    select * from final_table where DateAdded >= DATEADD(DAY,-40, GETDATE())"""

data = pd.read_sql(query, engine)
#data=pd.read_csv('./Data/Required_Data.csv',parse_dates=['DateAdded','ApprovalDate'])
data=data.infer_objects()

#Just keep dates from datetime column called DateAdded
data['DateAdded']=data['DateAdded'].dt.date
data.DateAdded=pd.to_datetime(data.DateAdded, format="%Y-%m-%d")
print("Shape of  data :",data.shape)

#filter data for last 7 days only
newData=data[data.DateAdded.isin(pd.date_range(pd.Timedelta(-9, unit='d')+pd.datetime.today().date(), periods=10))] 
newData.reset_index(drop=True,inplace=True)

# =============================================================================
# minDate='2021-01-01'
# maxDate='2021-01-31'
# 
# print("Stats afer loading ")
# 
# print('min Date :',newData.DateAdded.min(),'max Date :',newData.DateAdded.max())
# print('Size of newData :',newData.shape)
# print("dtypes of  newData :",newData.dtypes)
# 
# print("Stats afer filtering ")
# newData=newData[(newData['DateAdded']>minDate) & (newData['DateAdded']<maxDate)]
# print('Size of newData :',newData.shape)
# print('head of newData :',newData.head())
# 
# 
# print('min Date :',newData.DateAdded.min(),'max Date :',newData.DateAdded.max())
# print('head of newData :',newData.head())
# =============================================================================

newData['TotalIncome']=newData.BorrowerTotalMonthlyIncome+newData.CoBorrowerTotalMonthlyIncome
newData=treatNegCol(newData,'BorrowerYearsInSchool','Num')

print("Shape of  newData :",newData.shape)
print("dtypes of  newData :",newData.dtypes)
#print('BorrowerYearsInSchool.unique() :',list(newData['BorrowerYearsInSchool'].unique())

#newData['BorrowerYearsInSchool'][newData.BorrowerYearsInSchool.isna()]=newData.BorrowerYearsInSchool.mode().values


#Regroup education
newData=groupEdu(newData)

print("dtypes of  newData after Regroup education :",newData.dtypes)

# Get the workspace details
ws = Run.get_context().experiment.workspace


### Load Model pickle file #####
amlModel = Model(ws, name="RandomForestModel")
print("amlModel.version :",amlModel.version)
print("properties of amlModel :",dir(amlModel))

#Create Azure ML Experiment

run=Run.get_context().experiment.start_logging()
run.log("Experiement start time",str(datetime.now()))

details = run.get_details()

print("details :",details)

model_path = amlModel.get_model_path('RandomForestModel')

print('model_path :->',model_path)
model=pickle.load(open(model_path,'rb'))

##### Load encoder pickle file #####
encoder_path=Model.get_model_path('encoder')
print('encoder_path :->',encoder_path)
encoder=pickle.load(open(encoder_path,'rb'))

##### Load preprocessor pickle file #####
preprocessor_path=Model.get_model_path('preprocessor')
print('preprocessor_path :->',preprocessor_path)
preprocessor=pickle.load(open(preprocessor_path,'rb'))


# =============================================================================
# model = pickle.load(open('C:/Visionet/Projects/LoanApprovalModel/outputs/model.pkl','rb'))
# encoder = pickle.load(open('C:/Visionet/Projects/LoanApprovalModel/outputs/encoder.pkl','rb'))
# preprocessor = pickle.load(open('C:/Visionet/Projects/LoanApprovalModel/outputs/preprocessor.pkl','rb'))
# 
# =============================================================================

# Impute Missing values
numVars=['TotalLoanAmount', 'CreditScore', 'CLTV', 'DTI', 'BorrowerAge', 'TotalIncome','BorrowerYearsInSchool']
catVars=['LeadSourceGroup', 'LoanType', 'BorrowerOwnRent','Education','ZipCode']
idCols=['LeadID','DateAdded']

newData[numVars] = newData[numVars].apply(pd.to_numeric) 

print("dtypes of  newData before preprocessor :",newData.dtypes)


# Get the workspace details
ws = Run.get_context().experiment.workspace

print("unique values of zipcode 720:735 :",newData[0:5]['ZipCode'])
print('uniue values of new data zipcode :',newData['ZipCode'].mode().values)
print(' index min-max :',newData.index.min(),newData.index.max())
print(' data shape :',newData.shape)
for col in numVars+catVars :
    if col in numVars:
        print(" unique values of", col ," :",newData[col].unique())        
        newData[col][newData[col].isna()]=newData[col].mean()
    elif col in catVars:
        newData[col] = newData[col].str.strip()
        print(" unique values of", col ," :",newData[col].unique())
        newData[col][newData[col].isna()]=newData[col].mode().values[0]
        
    else:
        pass

numVars.remove('BorrowerYearsInSchool')

print(' Missing values have been imputed')
newData=newData[idCols+numVars+catVars]
#newData=pd.concat([newData[idCols],pd.DataFrame(data=preprocessor.transform(newData),columns=numVars+catVars)], axis=1)

newData.head()
print("dtypes of  newData after preprocessor :",newData.dtypes)
print('missing values in newData after preprocessor :',newData.isna().sum())

newData=groupLSR(newData)
newData=groupZipCode(newData)

print('newData has been regrouped')

#newData[numVars] = newData[numVars].apply(pd.to_numeric) 

print("dtypes of  newData before encoder:",newData.dtypes)
print('missing values in newData :',newData.isna().sum())

print("stats on newData :",newData.describe())


newData=trimWhiteSpace(newData,catVars)

newDataBkp=newData.copy(deep=True)

#Encode categorical data
print('numVars :',numVars)
newData=encodeTest(encoder,newData,catVars)

print('missing values in newData :',newData.isna().sum())
print('Columns of newData :',newData.columns)

#Reorder columns
cols = idCols + [col for col in newData if col not in idCols]
newData=newData[cols]


#newData['prediction']=model.predict(newData)
#### Predict the score data ####
newDataBkp['Prediction']=model.predict_proba(newData.drop(idCols,axis=1))[:,1].round(2)


today=date.today()

today=today.strftime("%Y-%m-%d")

newDataBkp['ModelVersion'],newDataBkp['Run_id'],newDataBkp['Model Run Date']=amlModel.version,details['runId'],today
newDataBkp['Model Run Date']=pd.to_datetime(newDataBkp['Model Run Date'], format="%Y-%m-%d")

print("newData['ModelVersion'] :",newDataBkp['ModelVersion'].unique(),"newData['Run_id'] :",newDataBkp['Run_id'].unique())


#### Truncate Table ####

#result.drop_duplicates(inplace=True)
newDataBkp.drop_duplicates(subset=['LeadID'])
print("Duplicate records have been dropped")


# Create table if it doesn't exist

tableName='demoTableV07'
if not engine.dialect.has_table(engine, tableName):  # If table don't exist, Create.
    metadata = MetaData(engine)
    # Create a table with the appropriate Columns
    Table(tableName, metadata,
          Column('LeadID', Integer, primary_key=True, nullable=False), 
          Column('DateAdded', Date), Column('LeadSourceGroup', Text),
          Column('LoanType', Text), Column('BorrowerOwnRent', Text), 
          Column('Education', Text), Column('ZipCode', Text), 
          Column('TotalLoanAmount', Integer), Column('CreditScore', Integer), 
          Column('CLTV', Integer), Column('DTI', Integer), 
          Column('BorrowerAge', Integer), Column('TotalIncome', Integer),
          Column('BorrowerYearsInSchool', Integer),Column('TotalLoanAmount', Integer),
          Column('CreditScore', Integer), Column('Prediction', Integer),
          Column('Actual', Integer),Column('Model Run Date', Date),
          Column('ModelVersion', Integer), Column('Run_id', Text),
          Column('TP', Integer),Column('FP', Integer),
          Column('TN', Integer),Column('FN', Integer),
          
          )
    # Implement the creation
    metadata.create_all()

session.execute('''SET IDENTITY_INSERT dbo.''' + tableName + " ON")
session.commit()

session.execute('''TRUNCATE TABLE '''+tableName)
session.commit()
print("Table has been truncated")

newDataBkp.to_sql(tableName, con=engine,schema='dbo', if_exists='append', index=False)

print('prediction has been made')
print("#### score file has been executed successfully####")


######## Create Approved Target column #############
print("columns of data :", data.columns)
print('Head of data :',data.head(2))
data['Approved']='Yes'
data['Approved'][data.ApprovalDate.isna()]='No'

# Change data type to category
data['Approved']=data['Approved'].astype('category')

# Map Target column for correlation Matrix
data.Approved.replace({'Yes':1,'No':0},inplace=True)

data=data[['LeadID','Approved']]

data.rename(columns={'Approved':'Actual'},inplace=True)


######## Pull all Data from Prediction Table #############

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
result['TP'],result['FP']=0,0
result['TP'].loc[(result['Prediction']>=thre) & (result['Actual']==1)]=1
result['FP'].loc[(result['Prediction']>=thre) & (result['Actual']==0)]=1

#### Truncate Table ####

result.drop_duplicates(subset=['LeadID'])
session.execute('''TRUNCATE TABLE '''+tableName)
session.commit()
print('Columns of newData :',result.columns)

#Push data in to PredictionTable back
result.to_sql(tableName,con=engine,schema='dbo', if_exists='append', index = False) 

session.commit()
session.close()
