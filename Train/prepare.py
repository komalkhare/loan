# -*- coding: utf-8 -*-

print("====== Prepare python file has been hit======")


import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

from sklearn.model_selection import KFold 
from sklearn.linear_model import  LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV


# -*- coding: utf-8 -*-
import pandas as pd
# Label encode features
from azureml.core.model import Model
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder

print("====== Sklearn & pandas have been imported======")
from azureml.core import Run

import pickle
import argparse
import yaml
import pyodbc
print("====== pickle & argparse have been imported======")

print("====args1=====")

parser = argparse.ArgumentParser()
print("====args2=====")
parser.add_argument("--output",dest="output")
print("====args3=====")
print("====parser.parse_args()====:",parser.parse_args())
args = parser.parse_args()
print("=====args.output=====",args.output)


from utils import *

print("====== All libraries have been imported======")

#Load data
pd.set_option('display.max_columns', None) #Show all columns
# pd.set_option("max_rows", None) #Show all rows
# pd.set_option(‘precision’, 2) # Take only last 2 values from decimal place


## Load configuration file ###
with open(r"./configs/config.yaml",'r') as file:
    config=yaml.load(file,Loader=yaml.FullLoader)
    
params=config['params']

server = params['server']
database = params['database']
username = params['username']
password = params['password']
driver= params['driver']

cnxn  = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
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

    select * from final_table where DateAdded >= DATEADD(YEAR,-2, GETDATE())"""

#data = pd.read_sql(query, cnxn)

data=pd.read_csv('./Data/Required_Data.csv',parse_dates=['DateAdded','ApprovalDate'])

data=data.infer_objects()
data.dtypes

print("====Data have been loaded successfully====")

#Add Borrower & Co-Borrower's income
data['TotalIncome']=data.BorrowerTotalMonthlyIncome + data.CoBorrowerTotalMonthlyIncome

######## Create Approved Target column #############
data['Approved']='Yes'
data['Approved'][data.ApprovalDate.isna()]='No'

#Recategorize Loan Purpose column
LP = ['Refinance','Purchase']
data['NewLP']= [v if v in LP else 'Refinance' for v in data.LoanPurpose]

#Calculate difference between current date and DateAdded in days
data['Diff']=pd.Timestamp.now().normalize()-data['DateAdded'][data.ApprovalDate.isna()]
data['Diff']=[entry.days for entry in data['Diff']]

data['Approved']=['No' if ((x>=90) & (y=='Purchase')) else 'Yes' for (x,y) in zip(data.Diff,data.NewLP)]
data['Approved']=['No' if ((x>=45) & (y=='Refinance')) else 'Yes' for (x,y) in zip(data.Diff,data.NewLP)]

# Change data type to category
data['Approved']=data['Approved'].astype('category')

######## Created Approved Target column #############

data=treatNegCol(data,'BorrowerYearsInSchool','Num')
data['BorrowerYearsInSchool'][data.BorrowerYearsInSchool.isna()]=data.BorrowerYearsInSchool.mode().values

#Regroup education
data=groupEdu(data)

#Missing values Imputation

df=data.copy(deep=True)

feats=['TotalLoanAmount', 'CreditScore', 'CLTV', 'DTI', 'BorrowerAge', 'TotalIncome','LeadSourceGroup', 'LoanType', 'BorrowerOwnRent','Education','ZipCode','Approved']
idCols=['LeadID','DateAdded']

data=data[idCols+feats]

print('Data Types :',data.dtypes)

print('Missing values :',data.isna().sum())

data,preprocessor=missingValImputation(data,idCols)


#Regroup stay type
Stay = ['Own','Rent']
data['BorrowerOwnRent']= [v if v in Stay else 'Own' for v in data.BorrowerOwnRent]

data=groupLSR(data)
data=groupZipCode(data)

## Filter out outliers

data=data[data.CLTV<110]
data=data[data.TotalLoanAmount<600000]
data=data[data.CreditScore>550]
data=data[data.TotalIncome<=30000]

# Map Target column for correlation Matrix
data.Approved.replace({'Yes':1,'No':0},inplace=True)



ordinal=['Education']
nominal=['LeadSourceGroup', 'LoanType', 'BorrowerOwnRent', 'ZipCode']

data,encoder=encode(data,nominal,ordinal)

print('columns :',data.columns)

#Reorder columns
cols = idCols + [col for col in data if col not in idCols]

data=data[cols]

# =============================================================================
# data=data['LeadID', 'DateAdded', 'Internet', 'Other', 'Radio', 'Repeat Client', 'TV', 'Conventional',
#        'FHA', 'USDA Rural', 'VA', 'Own', 'Rent', '75', '76', '77', '78', '79',
#        'OtherZP', 'Education','TotalLoanAmount',
#        'CreditScore', 'CLTV', 'DTI', 'BorrowerAge', 'TotalIncome','Approved']
# =============================================================================

print('Reorder columns :',data.columns)

#data=encodeNew(data,nominal,'Train')

print("=====Data have been encoded successfully=====")



#data.to_csv('C:/Visionet/Projects/LoanApprovalModel/Data/data.csv',index=False)

# Get the workspace details
ws = Run.get_context().experiment.workspace



#Register the Preprocessor file
preprocessor=Model.register(model_path='./outputs/preprocessor.pkl',
                     model_name="preprocessor",
                     tags={"Key":"1"},
                     description='Loan Approval Prediction preprocessor',
                     workspace=ws)

#Register the encoder file
encoder=Model.register(model_path='./outputs/encoder.pkl',
                     model_name="encoder",
                     tags={"Key":"1"},
                     description='Loan Approval Prediction encoder',
                     workspace=ws)

data.to_csv(args.output,index=False)


print("====== Prepare file has been executed=====")


