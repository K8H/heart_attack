import os
import pandas as pd

dataset_hf = pd.read_csv(os.path.abspath('../data/heart_failure_clinical_records_dataset.csv'))
dataset_clev = pd.read_csv('../data/heart_statlog_cleveland_hungary_final.csv')
dataset_risk_fact = pd.read_csv('../data/framingham.csv')
dataset_risk_fact_surv = pd.read_csv('../data/Behavioral_Risk_Factor_Surveillance_System__BRFSS__-__National_Cardiovascular_Disease_Surveillance_Data.csv')
dataset_medicare = pd.read_csv('../data/Center_for_Medicare___Medicaid_Services__CMS____Medicare_Claims_data.csv')
