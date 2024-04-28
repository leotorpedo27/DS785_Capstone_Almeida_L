from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier

from pyspark.ml.clustering import KMeans

from xgboost.spark import SparkXGBRegressor, SparkXGBClassifier

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import RandomOverSampler

import mlflow
import mlflow.spark

import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

from pyspark.sql.types import DoubleType
from math import sin, cos, sqrt, atan2, radians , log
from pyspark.sql.functions import col, round, expr
from pyspark.sql.types import DoubleType
from math import radians, sin, cos, sqrt, atan2
import pandas as pd

from mlflow.tracking.client import MlflowClient
from pyspark.ml.linalg import VectorUDT, DenseVector

#Load Data

file_path = """
SELECT CASE 
		WHEN COMPANY IN (
				'TOTALCARE'
				,'KERR'
				,'AXIS'
				,'SYBRONENDO'
				)
			THEN 'KERR'
		WHEN COMPANY IN (
				'DEXIS'
				,'GENDEX'
				,'INSTRUMENTARIUM'
				,'ISI'
				,'KAVO IMAGING'
				,'NOMAD'
				)
			THEN 'DEXIS'
		ELSE COMPANY
		END AS OPCO
	,SHIP_TO_MDM_CUST_KEY AS MDM_CUST_KEY
    ,SHIP_TO_MDM_ADDR_LATITUDE AS LATITUDE 
    ,SHIP_TO_MDM_ADDR_LONGITUDE AS LONGITUDE
    ,SHIP_TO_ZIP AS ZIP
    ,L1_DESC_NEW AS L1_DESC
    ,L1_DESC_NEW AS L2_DESC
    ,CAL_445_YYYY AS YYYY
    ,SUM(EXT_IVC_COST_USD) AS SALES
FROM fuzzy_match.dw_dealer_cust_sales_vtab
WHERE CAL_445_YYYY IN (2019,2020,2021,2022,2023)
	AND COMPANY IN (
		'DEXIS'
		,'GENDEX'
		,'INSTRUMENTARIUM'
		,'ISI'
		,'KAVO IMAGING'
		,'NOMAD'
		,'TOTALCARE'
		,'KERR'
		,'AXIS'
		,'SYBRONENDO'
		)
	AND L0_DESC_NEW IN (
		'RESTORATIVES'
		,'ENDODONTIC'
		,'IMAGING EXTRAORAL'
		,'IMAGING INTRAORAL'
		)
	AND L1_DESC_NEW IN (
		'FILL / OBTURATION'
		,'SHAPING'
		,'GLIDE PATH / ACCESS'
		,'WIPES'
		,'SURFACE DISINFECTANT'
		,'BARRIERS'
		,'TIPS / ADAPTORS'
		,'STERILANTS'
		,'PRECLEANER'
		,'EVACUATION TRAPS / CLEANERS'
		,'PREVENTIVE'
		,'BONDING AGENTS'
		,'CEMENTS'
		,'COMPOSITE'
		,'DIAMONDS'
		,'IMPRESSION'
		,'CARBIDES'
		,'FINISHING AND POLISHING'
		,'3D'
		,'2D'
		,'IO-SENSOR'
		,'IO X-RAY'
		)
	AND L2_DESC_NEW NOT IN ('SPARE PARTS')
	AND (
		ITEM_COMMISSION_CAT IN (
			'SMALL EQUIPMENT'
			,'CONSUMABLES'
			,'TARGET PRODUCTS'
			,'CORE PRODUCTS'
			,'ROTARY'
			,'COMMISSIONABLE'
			,'SALES OPS COMM'
			)
		OR PRODUCT_NUMBER IN (
			'80002568'
			,'80002567'
			)
		AND 
			SDS_CUST_CLASS1 = 'STANDARD'	
		)
GROUP BY 
    CASE 
		WHEN COMPANY IN (
				'TOTALCARE'
				,'KERR'
				,'AXIS'
				,'SYBRONENDO'
				)
			THEN 'KERR'
		WHEN COMPANY IN (
				'DEXIS'
				,'GENDEX'
				,'INSTRUMENTARIUM'
				,'ISI'
				,'KAVO IMAGING'
				,'NOMAD'
				)
			THEN 'DEXIS'
		ELSE COMPANY
		END
	,SHIP_TO_MDM_CUST_KEY
    ,SHIP_TO_MDM_ADDR_LATITUDE
    ,SHIP_TO_MDM_ADDR_LONGITUDE
    ,SHIP_TO_ZIP
    ,L1_DESC_NEW
    ,L2_DESC_NEW
    ,CAL_445_YYYY 
    
UNION ALL

SELECT 'NOBEL' AS OPCO
, n.MDM_CUST_KEY AS MDM_CUST_KEY
, c.ADDR_LATITUDE AS LATITUDE
, c.ADDR_LONGITUDE AS LONGITUDE
, n.ADDR_ZIP AS ZIP
, n.P_CAT_L1_DESC AS L1_DESC
, n.P_CAT_L2_DESC AS L2_DESC
, n.CAL_445_YYYY AS YYYY
, SUM(n.GROSS_SALES_USD) AS SALES
FROM fuzzy_match.dw_nobel_copa_vtab n
LEFT JOIN fuzzy_match.mdm_customer_masters c ON n.MDM_CUST_KEY = c.MDM_CUST_KEY
WHERE CAL_445_YYYY IN (2019,2020,2021,2022,2023)
AND n.CUST_CLASS_L1='STANDARD'
AND n.P_CAT_L2_DESC IN ('IMPRESSION COPING',
'IMPLANT REPLICA',
'CONICAL CONNECTION-FINAL PROSTHETICS',
'TRI-CHANNEL-FINAL PROSTHETICS',
'TEMP ABUTMENT',
'CONICAL CONNECTION-DENTAL IMPLANTS',
'CONICAL CONNECTION-FINAL PROSTHETICS',
'RESORBABLE MEMBRANES-BARRIER',
'TRI-CHANNEL-DENTAL IMPLANTS',
'TRIOVAL CONICAL CONNECTION-DI',
'BONE GRAFTING',
'CREOS XENOGAIN')
GROUP BY 
'NOBEL' 
, n.MDM_CUST_KEY
, c.ADDR_LATITUDE 
, c.ADDR_LONGITUDE
, n.ADDR_ZIP
, n.P_CAT_L1_DESC
, n.P_CAT_L2_DESC
, n.CAL_445_YYYY
"""
rawDf = spark.sql(file_path)

display(rawDf)

# Cleanse Data

testCust = rawDf.filter(rawDf["MDM_CUST_KEY"] == '')
display(testCust)



filtered_df = rawDf \
                 .filter(col("MDM_CUST_KEY").isNotNull()) \
                 .withColumn("SALES", when(col("SALES") < 0, 0).otherwise(col("SALES")))

filtered_df = filtered_df \
                    .withColumn("SALES", when(col("SALES") == 0, None).otherwise(col("SALES"))) \
                    .na.drop(subset=["SALES"])
rawDf2_with_concat_col = filtered_df.withColumn("L1_L2_DESC", concat(col("L1_DESC"), lit(" "), col("L2_DESC")))

unpivoted_df = rawDf2_with_concat_col.select("MDM_CUST_KEY", "L1_L2_DESC", "SALES") \
    .groupBy("MDM_CUST_KEY") \
    .pivot("L1_L2_DESC") \
    .agg(sum("SALES")) 
updated_df = unpivoted_df.withColumn("3D_3D_mod", when(unpivoted_df["3D 3D"] < 50000, 0).otherwise(unpivoted_df["3D 3D"]))
new_df = updated_df.drop("3D 3D")
new_df= new_df[new_df["MDM_CUST_KEY"]!="Master-Unmatched"]
df_grouped2 = new_df.filter(updated_df["3D_3D_mod"] >= 50000)
df_grouped2 = df_grouped2.sort(desc("3D_3D_mod"))
df_grouped2_pandas = df_grouped2.toPandas()
Q1 = df_grouped2_pandas["3D_3D_mod"].quantile(0.25)
Q3 = df_grouped2_pandas["3D_3D_mod"].quantile(0.75)
IQR = Q3 - Q1
lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR
dxs3D = new_df.withColumn("3D_Sales", when(new_df["3D_3D_mod"] >= upper_whisker, 0).otherwise(new_df["3D_3D_mod"]))
dxs3D = dxs3D.drop("3D_3D_mod")

cols_to_transform = ['2D 2D', 'BONDING AGENTS BONDING AGENTS', 'CARBIDES CARBIDES', 'CEMENTS CEMENTS', 'COMPONENTS & KITS IMPLANT REPLICA', 'COMPONENTS & KITS IMPRESSION COPING', 'COMPOSITE COMPOSITE', 'DIAMONDS DIAMONDS', 'FILL / OBTURATION FILL / OBTURATION', 'FINAL PROSTHETICS CONICAL CONNECTION-FINAL PROSTHETICS', 'FINAL PROSTHETICS TRI-CHANNEL-FINAL PROSTHETICS', 'FINISHING AND POLISHING FINISHING AND POLISHING', 'GLIDE PATH / ACCESS GLIDE PATH / ACCESS', 'IMPRESSION IMPRESSION', 'IO X-RAY IO X-RAY', 'IO-SENSOR IO-SENSOR', 'SHAPING SHAPING', 'TEMPORARY PROSTHETICS TEMP ABUTMENT']

dxs3D = dxs3D.fillna(0)
dxs3D = dxs3D.withColumn("label", when(dxs3D["3D_Sales"] >= 50000, 1).otherwise(0))

#Call the Model and Add the Lead Gen Prob Field

logged_model = 'runs:/f0289e7dd37c46f5a937ca387f93927a/model'
loaded_model = mlflow.spark.load_model(logged_model)
predictions = loaded_model.transform(dxs3D)
@udf(returnType=DoubleType())
def extract_probability(probability):
    if isinstance(probability, DenseVector):
        return float(probability[1])
    else:
        return None

predictions = predictions.withColumn("lead_gen_prob", extract_probability("probability"))
predictions = predictions.filter((predictions['label'] == 0) & (predictions['lead_gen_prob'] > 0.25))

table_name = "fuzzy_match.predictions_v2"
cleaned_columns = [col.replace(" ", "_").replace(",", "_").replace(";", "_") for col in predictions.columns]
predictions = predictions.toDF(*cleaned_columns)
predictions.write.mode("overwrite").saveAsTable(table_name)

# Ideal / Recommended Process (Optional)
# ETL / Data Cleansing and Preparation
# Modeling and Testing
# Model Registry and Deployment to Staging
# Finalize in April
# When ready create a ticket for Promotion

