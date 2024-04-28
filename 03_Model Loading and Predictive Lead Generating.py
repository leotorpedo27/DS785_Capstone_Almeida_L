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

#Cleanse Data
# Filter out rows where MDM_CUST_KEY is null and convert negatives to zero
filtered_df = rawDf \
                 .filter(col("MDM_CUST_KEY").isNotNull()) \
                 .withColumn("SALES", when(col("SALES") < 0, 0).otherwise(col("SALES")))

# Replace all zeros with nulls
filtered_df = filtered_df \
                    .withColumn("SALES", when(col("SALES") == 0, None).otherwise(col("SALES"))) \
                    .na.drop(subset=["SALES"])

# Unpivot Data by L1 or L2
rawDf2_with_concat_col = filtered_df.withColumn("L1_L2_DESC", concat(col("L1_DESC"), lit(" "), col("L2_DESC")))

# Unpivot imp_df
unpivoted_df = rawDf2_with_concat_col.select("MDM_CUST_KEY", "L1_L2_DESC", "SALES") \
    .groupBy("MDM_CUST_KEY") \
    .pivot("L1_L2_DESC") \
    .agg(sum("SALES")) 

# Create a new column "3D 3D updated" with values replaced based on the condition
updated_df = unpivoted_df.withColumn("3D_3D_mod", when(unpivoted_df["3D 3D"] < 50000, 0).otherwise(unpivoted_df["3D 3D"]))

# Create a new DataFrame without the "3D 3D" column and remove Unmatched MDM Keys
new_df = updated_df.drop("3D 3D")
new_df= new_df[new_df["MDM_CUST_KEY"]!="Master-Unmatched"]

# Only MDM Customers with a purchase sum of igher than $50K for 3D, removes the noise
df_grouped2 = new_df.filter(updated_df["3D_3D_mod"] >= 50000)

df_grouped2 = df_grouped2.sort(desc("3D_3D_mod"))

# Convert PySpark DataFrame to Pandas DataFrame
df_grouped2_pandas = df_grouped2.toPandas()

# Calculate quartiles
Q1 = df_grouped2_pandas["3D_3D_mod"].quantile(0.25)
Q3 = df_grouped2_pandas["3D_3D_mod"].quantile(0.75)
IQR = Q3 - Q1

# Calculate whiskers
lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR

# Create a new column "3D_Sales" with values replaced based on the condition
dxs3D = new_df.withColumn("3D_Sales", when(new_df["3D_3D_mod"] >= upper_whisker, 0).otherwise(new_df["3D_3D_mod"]))

# Create a new DataFrame without the "3D 3D" column and remove Unmatched MDM Keys
dxs3D = dxs3D.drop("3D_3D_mod")

cols_to_transform = ['2D 2D', 'BONDING AGENTS BONDING AGENTS', 'CARBIDES CARBIDES', 'CEMENTS CEMENTS', 'COMPONENTS & KITS IMPLANT REPLICA', 'COMPONENTS & KITS IMPRESSION COPING', 'COMPOSITE COMPOSITE', 'DIAMONDS DIAMONDS', 'FILL / OBTURATION FILL / OBTURATION', 'FINAL PROSTHETICS CONICAL CONNECTION-FINAL PROSTHETICS', 'FINAL PROSTHETICS TRI-CHANNEL-FINAL PROSTHETICS', 'FINISHING AND POLISHING FINISHING AND POLISHING', 'GLIDE PATH / ACCESS GLIDE PATH / ACCESS', 'IMPRESSION IMPRESSION', 'IO X-RAY IO X-RAY', 'IO-SENSOR IO-SENSOR', 'SHAPING SHAPING', 'TEMPORARY PROSTHETICS TEMP ABUTMENT','DENTAL IMPLANTS CONICAL CONNECTION-DENTAL IMPLANTS','FINAL PROSTHETICS CONICAL CONNECTION-FINAL PROSTHETICS','BARRIER MEMBRANES RESORBABLE MEMBRANES-BARRIER','DENTAL IMPLANTS TRI-CHANNEL-DENTAL IMPLANTS','DENTAL IMPLANTS TRIOVAL CONICAL CONNECTION-DI','BIOMATERIALS BONE GRAFTING','BONE GRAFT SUBSTITUTE MATERIAL CREOS XENOGAIN']

# cols_to_transform = ['IO X-RAY IO X-RAY', 'IO-SENSOR IO-SENSOR', 'DIAMONDS DIAMONDS','CEMENTS CEMENTS','FINISHING AND POLISHING FINISHING AND POLISHING','TEMPORARY PROSTHETICS TEMP ABUTMENT','BONDING AGENTS BONDING AGENTS', 'FILL / OBTURATION FILL / OBTURATION']

# Convert the columns to log values
# for col_name in cols_to_transform:
#     dxs3D = dxs3D.withColumn(col_name, expr("log(`{}`)".format(col_name)).cast("float"))

dxs3D = dxs3D.fillna(0)

# Create a new column "3D_Sales" with values replaced based on the condition
dxs3D = dxs3D.withColumn("label", when(dxs3D["3D_Sales"] >= 50000, 1).otherwise(0))

display(dxs3D.limit(10))

#Training / Test Split 

train_df, test_df = dxs3D.randomSplit([.8, .2], seed=42)
train_repartition_df, test_repartition_df = (train_df
                                             .repartition(24)
                                             .randomSplit([.8, .2], seed=42))

feature_cols = ['IO X-RAY IO X-RAY', 'IO-SENSOR IO-SENSOR', 'DIAMONDS DIAMONDS','CEMENTS CEMENTS','FINISHING AND POLISHING FINISHING AND POLISHING','TEMPORARY PROSTHETICS TEMP ABUTMENT','BONDING AGENTS BONDING AGENTS', 'FILL / OBTURATION FILL / OBTURATION']

# Upsampling minority class
# Get minority class from vectorized df
# Use an Oversampling Technique like Random Over Sampler to address the imbalance

#Step 1 - Convert vector assembled trainig df to pandas
pdf = train_df.toPandas()

#Step 2 - Breakout your Label and Features into separate df's
X = pdf.drop('label', axis=1)  
y = pdf['label']

#Step 3 - Breakout your Label and Features into separate df's
X_np = np.array(X)  
ros = RandomOverSampler(   sampling_strategy='auto',   shrinkage=None)
X_res, y_res = ros.fit_resample(X, y)

#Step 4 - Union All the Oversampled Minority Data
pdf_oversampled = pd.DataFrame(X_res, columns=X.columns)
pdf_oversampled['label'] = y_res
train_df = spark.createDataFrame(pdf_oversampled)

# Create a vector assembled data frames
vec_assembler = VectorAssembler(inputCols=cols_to_transform, outputCol="features")
output = vec_assembler.transform(dxs3D)
vec_train_df = vec_assembler.transform(train_df) 

#Best Model Selection
xgboost = SparkXGBClassifier(num_workers=2, features_col="features", label_col="label", missing=0.0)
stages = [vec_assembler, xgboost]
pipeline = Pipeline(stages=stages)

param_grid = (ParamGridBuilder()
              .addGrid(xgboost.max_depth, [2, 4])
              .addGrid(xgboost.n_estimators, [10, 100])
              .build())

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3, seed=42)

xgboost = SparkXGBClassifier(num_workers=2, features_col="features", label_col="label", missing=0.0)
stages = [vec_assembler, xgboost]
pipeline = Pipeline(stages=stages)

param_grid = (ParamGridBuilder()
              .addGrid(xgboost.max_depth, [2, 4])
              .addGrid(xgboost.n_estimators, [10, 100])
              .build())

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3, seed=42)

cv_model = cv.fit(train_df)

best_model = cv_model.bestModel
predictions = best_model.transform(test_df)

accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
auc_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")

accuracy = accuracy_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
auc = auc_evaluator.evaluate(predictions)

with mlflow.start_run(run_name="XGBoost - 3D") as run:
    # Log parameters
    mlflow.log_param("label", "label")
    mlflow.log_param("features", "Imaging_Implants_Consumables")

    # Log model
    mlflow.spark.log_model(best_model, "model", input_example=train_df.limit(5).toPandas()) 

    # Log metrics
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision) 
    mlflow.log_metric("F1 Score", f1)

@udf(returnType=DoubleType())
def extract_probability(probability):
    if isinstance(probability, DenseVector):
        return float(probability[1])
    else:
        return None

lead_gen_prob_df = predictions.withColumn("lead_gen_prob", extract_probability("probability"))

display(lead_gen_prob_df.limit(10))

test_review = lead_gen_prob_df.toPandas()
test_review.head()

# Visualize the Test DF
# Who is the qualified lead?
# Who are the some customers?

purch_3D = test_review[test_review['label']==1]
opp_3D = test_review[test_review['label']==0]
opp_3D = opp_3D[opp_3D['lead_gen_prob']>.75]
len(opp_3D)

from pyspark.ml import PipelineModel
from pyspark.ml.util import MLWriter
from xgboost import plot_importance
import numpy as np

# Get the XGBoost model from the best model
xgb_model = best_model.stages[-1]

# Get the feature importances from the XGBoost model
feature_importances = xgb_model.get_booster().get_fscore()

# Get the feature names from the VectorAssembler
feature_names = vec_assembler.getInputCols()

# Create a list of tuples (feature name, importance score)
feature_importance_list = [(feature, feature_importances.get(feature, 0)) for feature in feature_names]

# Sort the feature importances in descending order
sorted_feature_importances = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

# Extract the top 5 features
top_7_features = [feature for feature, _ in sorted_feature_importances[:7]]

# Print the top 5 features
print("Top 7 Features:")
for feature in top_7_features:
    print(feature)

opp_75 = opp_3D[top_7_features]
opp_75

dbutils.data.summarize(opp_75)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Assuming you have the opp_75 DataFrame with the top 7 features

# Create a figure and axes for the subplots
fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(8, 20))

# Define a list of colors for each feature
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# Iterate over each feature and create a box plot
for i, feature in enumerate(top_7_features):
    axs[i].boxplot(opp_75[feature], vert=False, patch_artist=True, boxprops=dict(facecolor=colors[i]))
    axs[i].set_title(feature)
    axs[i].set_xlabel('Value')
    axs[i].set_ylabel('Feature')
    
    # Format x-axis labels as currency with "K"
    axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '${:,.0f}K'.format(x/1000)))
    
# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()

# Build Out Field Test Data
# Region=NORTHEAST
# Cust Class = Private Practice
# TSM = John Weyland & Mike Recor
# Lead Gen Prob > 75%

## Register the Model

run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"

model_name = "Predictive Lead Scoring Model - 3D"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)
client.update_registered_model(
    name=model_details.name,
    description="This model is a Predictive Lead Generator for DEXIS 3D leveraging DEXIS Imaging Equipment, KERR Consumables & Implant Direct Data"
)

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="This model version was built using the Classification Model within XGBoost"
)

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Staging"
)
model_version_details = client.get_model_version(
    name=model_details.name,
    version=model_details.version
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

