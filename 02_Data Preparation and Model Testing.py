#Load Libraries

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

# Load Data

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

#Training/Test Split and Vector Assemble
train_df, test_df = dxs3D.randomSplit([.8, .2], seed=42)
train_repartition_df, test_repartition_df = (train_df
                                             .repartition(24)
                                             .randomSplit([.8, .2], seed=42))

feature_cols = ['IO X-RAY IO X-RAY', 'IO-SENSOR IO-SENSOR', 'DIAMONDS DIAMONDS','CEMENTS CEMENTS','FINISHING AND POLISHING FINISHING AND POLISHING','TEMPORARY PROSTHETICS TEMP ABUTMENT','BONDING AGENTS BONDING AGENTS', 'FILL / OBTURATION FILL / OBTURATION']

# Upsampling minority class
# * Get minority class from vectorized df
# * Use an Oversampling Technique like Random Over Sampler to address the imbalance

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

# Modeling and ML Flow Tracking

# XGBoost

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


#Top XGBoost Features
# Get the underlying XGBoost model
xgb_model_best = best_model
xgb_model = best_model.stages[-1].get_booster()
feature_importances = xgb_model.get_score(importance_type='gain')
feature_names = vec_assembler.getInputCols()

# Create a dictionary to map feature indices to feature names
feature_name_map = {f'f{i}': feature_names[i] for i in range(len(feature_names))}

# Map the feature indices to feature names in the importance scores
importance_scores = {feature_name_map.get(f, f): score for f, score in feature_importances.items()}

importances_df = pd.DataFrame({
    'feature': list(importance_scores.keys()),
    'importance': list(importance_scores.values())
})
xgboost_features = importances_df
print(xgboost_features.sort_values('importance', ascending=False))


# Neural Network
nn = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", seed=42, layers=[len(cols_to_transform), 10, 2])

model = nn.fit(vec_train_df)

vec_test_df = vec_assembler.transform(test_df)
predictions = model.transform(vec_test_df)

accuracy_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
precision_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderPR")
auc_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")

f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

accuracy = accuracy_evaluator.evaluate(predictions)
precision = precision_evaluator.evaluate(predictions)
auc = auc_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)

with mlflow.start_run(run_name="Neural Network - 3D") as run:
    # Log parameters
    mlflow.log_param("label", "label")
    mlflow.log_param("features", "Imaging_Implants_Consumables")
    
    # Log model
    mlflow.spark.log_model(model, "model", input_example=train_df.limit(5).toPandas())
    
    # Log metrics
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("F1 Score", f1)

# Top Neural Network Features
# Get the feature names
feature_names = vec_assembler.getInputCols()

# Calculate the base performance metric (e.g., accuracy)
base_metric = accuracy_evaluator.evaluate(predictions)

# Create a copy of the test data
permuted_test_df = vec_test_df

# Iterate over each feature and calculate permutation importance
feature_importances = []
for feature in feature_names:
    # Permute the values of the current feature
    permuted_col = feature + "_permuted"
    permuted_test_df = permuted_test_df.withColumn(permuted_col, col(feature))
    permuted_test_df = permuted_test_df.withColumn(feature, col(permuted_col).cast("double"))
    permuted_test_df = permuted_test_df.drop(permuted_col)
    
    # Make predictions on the permuted data
    permuted_predictions = model.transform(permuted_test_df)
    
    # Calculate the performance metric on the permuted data
    permuted_metric = accuracy_evaluator.evaluate(permuted_predictions)
    
    # Calculate the feature importance as the difference in performance
    importance = base_metric - permuted_metric
    feature_importances.append(importance)

# Create a DataFrame with feature names and importances
importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
})

nn_features = importances_df

# Print the feature importances DataFrame sorted by importance in descending order
print(nn_features.sort_values('importance', ascending=False))

#Decision Tree

dt = DecisionTreeRegressor(labelCol="label")
stages = [ vec_assembler, dt]
pipeline = Pipeline(stages=stages)
dt.setMaxBins(40)
pipeline_model = pipeline.fit(train_df)
dt_model = pipeline_model.stages[-1]
pred_df = pipeline_model.transform(test_df)
binaryEval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
auc = binaryEval.evaluate(pred_df)
accEval = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
accuracy = accEval.evaluate(pred_df) 
accEval = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label")
precision = accEval.evaluate(pred_df, {accEval.metricName: "weightedPrecision"})
f1 = accEval.evaluate(pred_df, {accEval.metricName: "f1"})

with mlflow.start_run(run_name="Decision Tree - 3D") as run:
    # Log parameters
    mlflow.log_param("label", "label")
    mlflow.log_param("features", "Imaging_Implants_Consumables")

    # Log model
    mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas()) 

    # Log metrics
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision) 
    mlflow.log_metric("F1 Score", f1)

#Top Decision Tree Features
    # Get feature importances and convert to dense array
    feature_importances = dt_model.featureImportances.toArray()
    feature_names = vec_assembler.getInputCols()

    importances_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })

    importances_df = importances_df.sort_values('importance', ascending=False)
    dt_features = importances_df
    print("Top features:")
    print(dt_features)

# Random Forest
rf = RandomForestRegressor(labelCol="label", maxBins=40)
stages = [vec_assembler, rf]
pipeline = Pipeline(stages=stages)
param_grid = (ParamGridBuilder()
              .addGrid(rf.maxDepth, [2, 5])
              .addGrid(rf.numTrees, [5, 10])
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

with mlflow.start_run(run_name="Random Forrest - 3D [Reduced Features]") as run:
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

# Top Randon Forest Features
feature_importances = best_model.stages[-1].featureImportances.toArray()
feature_names = vec_assembler.getInputCols()  

importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances  
})

rf_features = importances_df

print(rf_features.sort_values('importance', ascending=False))

# Visualization Creation

# Convert PySpark DataFrame to Pandas DataFrame
df_pandas = dxs3D.toPandas()

# Plot the distribution of CBCT equipment purchases
plt.figure(figsize=(10, 6))
plt.hist(df_pandas['3D_Sales'], bins=20, edgecolor='black')
plt.title('Distribution of CBCT Equipment Purchases')
plt.xlabel('CBCT Sales Amount')
plt.ylabel('Count')
plt.show()

# Convert PySpark DataFrames to Pandas DataFrames
train_pandas = train_df.toPandas()
test_pandas = test_df.toPandas()

# Plot class distributions before oversampling
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
train_pandas['label'].value_counts().plot(kind='bar', ax=axs[0], title='Class Distribution (Before Oversampling)')
test_pandas['label'].value_counts().plot(kind='bar', ax=axs[1], title='Class Distribution (Before Oversampling)')

# Plot class distributions after oversampling
oversampled_train_pandas = pdf_oversampled
oversampled_train_pandas['label'].value_counts().plot(kind='bar', title='Class Distribution (After Oversampling)')
plt.show()



print("Pipeline Stages:")
for i, stage in enumerate(xgb_model_best.stages):
    print(f"Stage {i}: {type(stage)}")

# Retrieve the trained XGBoost model
xgb_model = xgb_model_best.stages[1]

# Get the number of trees
dump_list = xgb_model.get_booster().get_dump()
num_trees = len(dump_list)

# Set the maximum depth manually
max_depth = 4

# Set the remaining hyperparameters manually
learning_rate = 0.1
subsample_ratio = 0.8
min_child_weight = 1.0
gamma = 0.0

print("XGBoost Model Architecture:")
print(f"Number of Trees: {num_trees}")
print(f"Maximum Depth: {max_depth}")
print(f"Learning Rate: {learning_rate}")
print(f"Subsample Ratio: {subsample_ratio}")
print(f"Minimum Child Weight: {min_child_weight}")
print(f"Gamma: {gamma}")

# Print Neural Network architecture and hyperparameters
print("Neural Network Architecture:")
print(f"Number of Layers: {len(nn.getLayers())}")
print(f"Number of Neurons per Layer: {nn.getLayers()}")
print(f"Solver: {nn.getSolver()}")
print(f"Max Iterations: {nn.getMaxIter()}")

# best_model.stages[-1].getImpurity()
best_model.stages[-1].getMaxDepth()

# Print Random Forest model architecture and hyperparameters
print(f"Random Forest Model Architecture:")
print(f"Number of Trees: {len(best_model.stages[-1].trees)}")
print(f"Maximum Depth: {best_model.stages[-1].getMaxDepth()}")
print(f"Impurity Metric: {best_model.stages[-1].getImpurity()}")
print(f"Minimum Information Gain: {best_model.stages[-1].getMinInfoGain()}")

# Print Decision Tree model architecture and hyperparameters
print(f"Decision Tree Model Architecture:")
print(f"Maximum Depth: {dt_model.getMaxDepth()}")
print(f"Maximum Bins: {dt_model.getMaxBins()}")
print(f"Minimum Instances per Node: {dt_model.getMinInstancesPerNode()}")
print(f"Minimum Information Gain: {dt_model.getMinInfoGain()}")

# Plot feature importance for XGBoost model
plt.figure(figsize=(10, 6))
xgboost_features.sort_values('importance', ascending=False).plot(kind='bar', x='feature', y='importance')
plt.title('Feature Importance (XGBoost Model)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.show()