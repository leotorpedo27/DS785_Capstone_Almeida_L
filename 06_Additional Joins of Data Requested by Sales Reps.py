# Pred Lead Gen: 3D Target

import pandas as pd
import numpy as np

fact = pd.read_csv('factRawLeadGenData_v2.csv')
dimInst = pd.read_csv('dimInstallBase3d2d.csv')
dimSla = pd.read_csv('dimLeadSlaActive3d2d.csv')
dimTech = pd.read_csv('dimTechCallsPredLead3d2d.csv')
dimRep = pd.read_excel('dimDxsReps.xlsx' ,sheet_name='Combined for Lookups')
dimOne = pd.read_csv('dimOneKeyDdr.csv')

dimNbl = pd.read_csv('dimNbImplant2023.csv')
dimId = pd.read_excel('dimId_implants_2023.xlsx')

dimDmiqId = pd.read_csv('dimDmiq_implants.csv')

dimidsg = pd.read_excel('SG Implant Direct 2023 User Base.xlsx')
dimnbsg = pd.read_excel('SG Nobel Implant Dollars.xlsx')
dfimpl = dimnbsg.merge(dimidsg, on='MDM Cust Key',how='outer')
col = ['MDM Cust Key', 'Customer Name_x', 'CONT_FULLPHONE_x',
       'MDM Street Addr_x', 'MDM Addr City_x', 'Bill Addr State_x',
       'MDM Addr Zip_x', 'CONT_EMAIL_x', 
       'Customer Name_y', 'CONT_FULLPHONE_y', 'MDM Street Addr_y',
       'MDM Addr City_y', 'Bill Addr State_y', 'MDM Addr Zip_y',
       'CONT_EMAIL_y', 'Implant_QTY','nobel Implant $$$']
dfimpl2 = dfimpl[col]
dfimpl3 = dfimpl2.merge(dimInst, left_on='MDM Cust Key',right_on='ship_to_mdm_cust_key',how='left')
dfimpl3.to_csv('sg_implant_list.csv')

# ## Aged 2D/3D Leads 
# * Only Purchased 2D/3D in 2017 or earlier.
# * Private Practice Only
# * Greater than 25% Chance of Lead Gen Probability

#PRIVATE PRACTICE ONLY
fact=fact[fact['SDS_CUST_CLASS1']=='STANDARD']
col = ['MDM_CUST_KEY','ACCT_OFFICE_NAME','ADDR_STREETADDR','ADDR_CITY','ADDR_STATE','ADDR_ZIP',
       'CONT_FULLPHONE','CONT_EMAIL','ADDR_LATITUDE','ADDR_LONGITUDE','lead_gen_prob',
       '2D_2D','IO_X-RAY_IO_X-RAY', 'IO-SENSOR_IO-SENSOR', 'SHAPING_SHAPING','FILL_/_OBTURATION_FILL_/_OBTURATION']
fact = fact[col]
fact2 = fact.merge(dimInst, left_on='MDM_CUST_KEY',right_on='ship_to_mdm_cust_key',how='inner')
fact2 = fact2.drop(columns=['Column1','ship_to_mdm_cust_key', '3D_maxdate', '2D_maxdate', '3D_min', '2D_min', 'clean_list', 'repl_2D', 'Column2'])
dimSla2 = dimSla.drop_duplicates(subset=['BILL_MDM_KEY'])
fact3 = fact2.merge(dimSla2,left_on='MDM_CUST_KEY',right_on='BILL_MDM_KEY',how='left')
fact3 = fact3.drop(columns=['BILL_MDM_KEY','SLA_SKU'])
dimTech = dimTech.drop_duplicates(subset=['MDM_ID'])
fact4 = fact3.merge(dimTech,left_on='MDM_CUST_KEY',right_on='MDM_ID',how='left')
fact4 = fact4.drop(columns=['MDM_ID'])
fact4['ZIP3'] = fact4['ADDR_ZIP'].str[:3]
fact5 = fact4.merge(dimRep,left_on='ZIP3',right_on='ZIP CODE' , how='left')
fact5 = fact5.drop(columns=['ZIP CODE','STATE / PROV','2024 RSD','2024 TERR','Interim TSM','DSS'])
dimOne = dimOne.drop_duplicates(subset=['mdm_site_key'])
fact6 = fact5.merge(dimOne,left_on='MDM_CUST_KEY',right_on='mdm_site_key',how='left')
fact6 = fact6.drop(columns=['Unnamed: 0','mdm_site_key'])
dimNbl = dimNbl.drop_duplicates(subset=['Row Labels'])
fact7 = fact6.merge(dimNbl,left_on='MDM_CUST_KEY',right_on='Row Labels',how='left')
fact7 = fact7.drop(columns=['Row Labels', 'Implant Brånemark System', 'Implant N1',
       'Implant Nobelactive', 'Implant Nobeldirect',
       'Implant Nobelparallel Conical Connection', 'Implant Nobelpearl',
       'Implant Nobelreplace', 'Implant Nobelspeedy', 'Implant Replace Select',
       'Trefoil'])
dimId = dimId.drop_duplicates(subset=['Unnamed: 0'])
fact8 = fact7.merge(dimId,left_on='MDM_CUST_KEY',right_on='Unnamed: 0',how='left')
fact8 = fact8.drop(columns=['Unnamed: 0'])
fact8.to_csv('00FlatLeadGen.csv')

# High Lead Gen Prob
# Private Practice Only
# Greater than 25% Chance of Lead Gen Probability

#PRIVATE PRACTICE ONLY
fact=fact[fact['SDS_CUST_CLASS1']=='STANDARD']
col = ['MDM_CUST_KEY','ACCT_OFFICE_NAME','ADDR_STREETADDR','ADDR_CITY','ADDR_STATE','ADDR_ZIP',
       'CONT_FULLPHONE','CONT_EMAIL','ADDR_LATITUDE','ADDR_LONGITUDE','lead_gen_prob',
       '2D_2D','IO_X-RAY_IO_X-RAY', 'IO-SENSOR_IO-SENSOR', 'SHAPING_SHAPING','FILL_/_OBTURATION_FILL_/_OBTURATION']
fact = fact[col]
fact2 = fact.merge(dimInst, left_on='MDM_CUST_KEY',right_on='ship_to_mdm_cust_key',how='left')
fact2 = fact2.drop(columns=['Column1','ship_to_mdm_cust_key', '3D_maxdate', '2D_maxdate', '3D_min', '2D_min', 'clean_list', 'repl_2D', 'Column2'])
dimSla2 = dimSla.drop_duplicates(subset=['BILL_MDM_KEY'])
fact3 = fact2.merge(dimSla2,left_on='MDM_CUST_KEY',right_on='BILL_MDM_KEY',how='left')
fact3 = fact3.drop(columns=['BILL_MDM_KEY','SLA_SKU'])
dimTech = dimTech.drop_duplicates(subset=['MDM_ID'])
fact4 = fact3.merge(dimTech,left_on='MDM_CUST_KEY',right_on='MDM_ID',how='left')
fact4 = fact4.drop(columns=['MDM_ID'])
fact4['ZIP3'] = fact4['ADDR_ZIP'].str[:3]
fact5 = fact4.merge(dimRep,left_on='ZIP3',right_on='ZIP CODE' , how='left')
fact5 = fact5.drop(columns=['ZIP CODE','STATE / PROV','2024 RSD','2024 TERR','Interim TSM','DSS'])
dimOne = dimOne.drop_duplicates(subset=['mdm_site_key'])
fact6 = fact5.merge(dimOne,left_on='MDM_CUST_KEY',right_on='mdm_site_key',how='left')
fact6 = fact6.drop(columns=['Unnamed: 0','mdm_site_key'])
dimNbl = dimNbl.drop_duplicates(subset=['Row Labels'])
fact7 = fact6.merge(dimNbl,left_on='MDM_CUST_KEY',right_on='Row Labels',how='left')
fact7 = fact7.drop(columns=['Row Labels', 'Implant Brånemark System', 'Implant N1',
       'Implant Nobelactive', 'Implant Nobeldirect',
       'Implant Nobelparallel Conical Connection', 'Implant Nobelpearl',
       'Implant Nobelreplace', 'Implant Nobelspeedy', 'Implant Replace Select',
       'Trefoil'])
dimId = dimId.drop_duplicates(subset=['Unnamed: 0'])
fact8 = fact7.merge(dimId,left_on='MDM_CUST_KEY',right_on='Unnamed: 0',how='left')
fact8 = fact8.drop(columns=['Unnamed: 0'])

fact8.to_csv('00FlatLeadGen_prospect.csv')
