# importing modules
import pandas as pd
import numpy as np
from pyspark.ml.fpm import FPGrowth
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import findspark
findspark.init()

# check if the preprocessed dataset is available
if not os.path.exists('processed_dataset.xlsx'):
    print('======> Please, run preprocess.py to generate a clean dataset first!')
    exit()

# loading preprocessed dataset
df = pd.read_excel('processed_dataset.xlsx')

# trim out [ at the beginning and ] at the end.
for ch in ['[', ']', '\n']:
    df['StockCode'] = df['StockCode'].str.replace(ch, '')

# initializing spark
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# convert pandas dataframe to spark dataframe
import pyspark.sql.types as pyt
mySchema = pyt.StructType([ pyt.StructField("InvoiceNo", pyt.StringType(), True)\
                       ,pyt.StructField("StockCode", pyt.StringType(), True)])
sdf = spark.createDataFrame(df,schema=mySchema)

# split the string on 'StockCode' row into a list of strings
from pyspark.sql.functions import split as sp
sdf = sdf.withColumn("StockCode", sp(sdf.StockCode, "\s+"))

# importing pyspark module
from pyspark.ml.fpm import FPGrowth
# apply frequent pattern growth model
fpGrowth = FPGrowth(itemsCol="StockCode", minSupport=0.01, minConfidence=0.5)
model = fpGrowth.fit(sdf)
# Display frequent itemsets.
model.freqItemsets.show(40)

# Display generated association rules.
model.associationRules.show()

# Display transformation predicts
model.transform(sdf).show(200)

# writing frequent setitems to excel file
freq_df = model.freqItemsets.toPandas()
writer = pd.ExcelWriter('frequent_itemsets.xlsx')
freq_df.to_excel(writer,'Sheet1', startcol = 0, startrow = 0)
writer.save()

# writing association rules to excel file
ar_pandas = model.associationRules.toPandas()
writer = pd.ExcelWriter('asscoiation_rules.xlsx')
ar_pandas.to_excel(writer,'Sheet1', startcol = 0, startrow = 0)
writer.save()

# writing predictions to excel file
transformation_df = model.transform(sdf).toPandas()
writer = pd.ExcelWriter('predictions.xlsx')
transformation_df.to_excel(writer,'Sheet1', startcol = 0, startrow = 0)
writer.save()

# stop spark context
spark.stop()