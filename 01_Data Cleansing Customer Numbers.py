from pyspark.sql.functions import col, udf, round, radians, sin, cos, atan2, sqrt
from pyspark.sql.types import DoubleType, FloatType

@udf(FloatType())  
def dist_udf(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1 = radians(float(lat1))
    phi2 = radians(float(lat2))
    delta_phi = radians(float(lat2 - lat1))
    delta_lambda = radians(float(lon2 - lon1))
    
    a = (sin(delta_phi/2.0)**2 +  
         cos(phi1) * cos(phi2) * sin(delta_lambda/2.0)**2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = r * c

    return d * 1000

# Load raw data
file_path = """
SELECT CASE 
        WHEN COMPANY IN ('TOTALCARE', 'KERR', 'AXIS', 'SYBRONENDO')
            THEN 'KERR'
        WHEN COMPANY IN ('DEXIS', 'GENDEX', 'INSTRUMENTARIUM', 'ISI', 'KAVO IMAGING', 'NOMAD')
            THEN 'DEXIS'
        ELSE COMPANY
        END AS OPCO,
    SHIP_TO_MDM_CUST_KEY AS MDM_CUST_KEY,
    SHIP_TO_MDM_ADDR_LATITUDE AS LATITUDE,
    SHIP_TO_MDM_ADDR_LONGITUDE AS LONGITUDE,
    SHIP_TO_ZIP AS ZIP,
    L1_DESC_NEW AS L1_DESC,
    L2_DESC_NEW AS L2_DESC,
    CAL_445_YYYY AS YYYY,
    SUM(EXT_IVC_COST_USD) AS SALES
FROM fuzzy_match.dw_dealer_cust_sales_vtab
WHERE CAL_445_YYYY IN (2019, 2020, 2021, 2022, 2023)
    AND COMPANY IN ('DEXIS', 'GENDEX', 'INSTRUMENTARIUM', 'ISI', 'KAVO IMAGING', 'NOMAD', 
                    'TOTALCARE', 'KERR', 'AXIS', 'SYBRONENDO')
    AND L0_DESC_NEW IN ('RESTORATIVES', 'ENDODONTIC', 'IMAGING EXTRAORAL', 'IMAGING INTRAORAL')
    AND L1_DESC_NEW IN ('FILL / OBTURATION', 'SHAPING', 'GLIDE PATH / ACCESS', 'WIPES', 
                        'SURFACE DISINFECTANT', 'BARRIERS', 'TIPS / ADAPTORS', 'STERILANTS', 
                        'PRECLEANER', 'EVACUATION TRAPS / CLEANERS', 'PREVENTIVE', 
                        'BONDING AGENTS', 'CEMENTS', 'COMPOSITE', 'DIAMONDS', 'IMPRESSION', 
                        'CARBIDES', 'FINISHING AND POLISHING', '3D', '2D', 'IO-SENSOR', 'IO X-RAY')
    AND (ITEM_COMMISSION_CAT IN ('SMALL EQUIPMENT', 'CONSUMABLES', 'TARGET PRODUCTS', 
                                 'CORE PRODUCTS', 'ROTARY', 'COMMISSIONABLE', 'SALES OPS COMM')
         OR PRODUCT_NUMBER IN ('80002568', '80002567')
         AND SDS_CUST_CLASS1 = 'STANDARD')
GROUP BY 
    CASE 
        WHEN COMPANY IN ('TOTALCARE', 'KERR', 'AXIS', 'SYBRONENDO')
            THEN 'KERR'
        WHEN COMPANY IN ('DEXIS', 'GENDEX', 'INSTRUMENTARIUM', 'ISI', 'KAVO IMAGING', 'NOMAD')
            THEN 'DEXIS'
        ELSE COMPANY
    END,
    SHIP_TO_MDM_CUST_KEY,
    SHIP_TO_MDM_ADDR_LATITUDE,
    SHIP_TO_MDM_ADDR_LONGITUDE,
    SHIP_TO_ZIP,
    L1_DESC_NEW,
    L2_DESC_NEW,
    CAL_445_YYYY
    
UNION ALL

SELECT 'NOBEL' AS OPCO,
    n.MDM_CUST_KEY AS MDM_CUST_KEY,
    c.ADDR_LATITUDE AS LATITUDE,
    c.ADDR_LONGITUDE AS LONGITUDE,
    n.ADDR_ZIP AS ZIP,
    n.P_CAT_L1_DESC AS L1_DESC,
    n.P_CAT_L2_DESC AS L2_DESC,
    n.CAL_445_YYYY AS YYYY,
    SUM(n.GROSS_SALES_USD) AS SALES
FROM fuzzy_match.dw_nobel_copa_vtab n
LEFT JOIN fuzzy_match.mdm_customer_masters c ON n.MDM_CUST_KEY = c.MDM_CUST_KEY
WHERE CAL_445_YYYY IN (2019, 2020, 2021, 2022, 2023)
    AND n.CUST_CLASS_L1 = 'STANDARD'
    AND n.P_CAT_L2_DESC IN ('IMPRESSION COPING', 'IMPLANT REPLICA', 'CONICAL CONNECTION-FINAL PROSTHETICS',
                            'TRI-CHANNEL-FINAL PROSTHETICS', 'TEMP ABUTMENT')
GROUP BY
    'NOBEL',
    n.MDM_CUST_KEY,
    c.ADDR_LATITUDE,
    c.ADDR_LONGITUDE, 
    n.ADDR_ZIP,
    n.P_CAT_L1_DESC,
    n.P_CAT_L2_DESC,
    n.CAL_445_YYYY
"""
rawDf = spark.sql(file_path)

# Cache raw DF with more partitions 
rawDf = rawDf.repartition(1000).cache()

# Prepare dataframes for cartesian join
df1 = rawDf.alias("df1") 
df1 = df1.drop("OPCO", "ZIP", "L1_DESC", "L2_DESC", "SALES", "YYYY")
df1 = df1.dropDuplicates(["MDM_CUST_KEY"])
df1 = df1.withColumnRenamed("MDM_CUST_KEY", "MDM_CUST_KEY_1") \
         .withColumnRenamed("LATITUDE", "LATITUDE_1") \
         .withColumnRenamed("LONGITUDE", "LONGITUDE_1")  
df1 = df1.dropna()

df2 = rawDf.alias("df2") 
df2 = df2.drop("OPCO", "ZIP", "L1_DESC", "L2_DESC", "SALES", "YYYY")
df2 = df2.dropDuplicates(["MDM_CUST_KEY"])
df2 = df2.withColumnRenamed("MDM_CUST_KEY", "MDM_CUST_KEY_2") \
         .withColumnRenamed("LATITUDE", "LATITUDE_2") \
         .withColumnRenamed("LONGITUDE", "LONGITUDE_2")
df2 = df2.dropna()

# Perform cartesian join and filtering
df_cartesian = df1.crossJoin(df2)    
df_cartesian = df_cartesian.withColumn("lat_round1", round(col("LATITUDE_1"), 0)) \
    .withColumn("lon_round1", round(col("LONGITUDE_1"), 0)) \
    .withColumn("lat_round2", round(col("LATITUDE_2"), 0)) \
    .withColumn("lon_round2", round(col("LONGITUDE_2"), 0))
df_cartesian_filtered = df_cartesian.filter(col("MDM_CUST_KEY_1") != col("MDM_CUST_KEY_2"))
df_cartesian_filtered = df_cartesian_filtered.drop("lat_round1", "lon_round1", "lat_round2", "lon_round2")

# Calculate distance between coordinates
df_filtered = df_cartesian_filtered.withColumn("DISTANCE", dist_udf(col("LATITUDE_1"), 
                                                                    col("LONGITUDE_1"), 
                                                                    col("LATITUDE_2"), 
                                                                    col("LONGITUDE_2")))
df_filtered = df_filtered.filter(col("DISTANCE") < 5)

# Save results
df_filtered.write.mode('overwrite').saveAsTable('fuzzy_match.dimMdmMatch')