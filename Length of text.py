# Databricks notebook source
# Load in the table
df = spark.sql("select * from default.reviews_train")#.sample(0.01)

df = df.cache()

print((df.count(), len(df.columns)))

# COMMAND ----------

from pyspark.sql.functions import year, month, dayofweek
import numpy as np
# For our intitial modeling efforts, we are not going to use the following features
#drop_list = ['summary','asin', 'reviewID', 'reviewerID', 'unixReviewTime','reviewTime', 'image', 'style', 'reviewerName']
#df = df.select([column for column in df.columns if column not in drop_list])

# Drop duplicates
print("Before duplication removal: ", df.count())
df = df.dropDuplicates(['reviewerID', 'asin'])
print("After duplication removal: ", df.count())


# Convert Unix timestamp to readable date & create date columns
from pyspark.sql.functions import from_unixtime, to_date
from pyspark.sql.types import *
df = df.withColumn("reviewTime", to_date(from_unixtime(df.unixReviewTime))) \
                                                .drop("unixReviewTime")
df = df.withColumn('reviewTime', df['reviewTime'].cast('date'))

# Extract year, month, and day of the week into separate columns
df = df.withColumn('year', year(df['reviewTime']))
df = df.withColumn('month', month(df['reviewTime']))
df = df.withColumn('day_of_week', dayofweek(df['reviewTime']))

df = df.withColumn("age", 2023 - year(df['reviewTime']))

# drop NA
df = df.na.drop(subset=["reviewText", "label"])


#Combine reviewText and summary as reviewTot

from pyspark.sql.functions import concat_ws, col

# Assuming you have a DataFrame named "df" containing the "reviewText" and "summary" columns
df = df.withColumn("reviewTot", concat_ws(" ", col("reviewText"), col("summary")))

display(df)
print((df.count(), len(df.columns)))

# COMMAND ----------

####### Lowercase

from pyspark.sql.functions import lower
# Convert the "review" column to lowercase
df = df.withColumn("reviewTot", lower(df["reviewTot"]))

display(df)

# COMMAND ----------

# Removing Characters

from pyspark.sql.functions import regexp_replace, col

# Assuming your DataFrame is named 'df'

# Define a regular expression pattern to match hyphens, single quotes, and other punctuation characters (excluding white spaces)
# The pattern "[-']{1,}|[.,]+|[!\"#$%&()*+,/:;<=>?@[\\]^_`{|}~]+" matches one or more occurrences of hyphen (-), single quote ('), period (.), comma (,), and other punctuation (excluding white spaces)
pattern_to_remove = "[-? ! "", .]+"

# Remove the specified characters and other punctuation in the "reviewTot" column
df = df.withColumn("reviewTot", regexp_replace(col("reviewTot"), pattern_to_remove, " "))

# Show the updated DataFrame
display(df)


# COMMAND ----------

from pyspark.sql.functions import length

# Assuming your DataFrame is named 'df'

# Create a new column 'reviewTot_length' based on the length of 'reviewTot'
df = df.withColumn("reviewTot_length", length(col("reviewTot")))

# Show the updated DataFrame
display(df)

# COMMAND ----------

# Define segmentation conditions
conditions = [
    (col("reviewTot_length") < 500),
    (col("reviewTot_length") >= 500) & (col("reviewTot_length") < 2000),
    (col("reviewTot_length") >= 2000) & (col("reviewTot_length") < 4000),
    (col("reviewTot_length") >= 4000) & (col("reviewTot_length") < 6000),
    (col("reviewTot_length") >= 6000) & (col("reviewTot_length") < 8000),
    (col("reviewTot_length") >= 8000) & (col("reviewTot_length") < 10000),
    (col("reviewTot_length") >= 10000) & (col("reviewTot_length") < 12000),
    (col("reviewTot_length") >= 12000) & (col("reviewTot_length") < 14000),
    (col("reviewTot_length") >= 14000) & (col("reviewTot_length") < 16000),
    (col("reviewTot_length") >= 16000) & (col("reviewTot_length") < 18000),
    (col("reviewTot_length") >= 18000) & (col("reviewTot_length") < 20000),
    (col("reviewTot_length") >= 20000),
]

# Define corresponding labels for each segment
labels = ['1.<500', '2.500 to 2k', '3.2k to 4k', '4.4k to 6k', '5.6k to 8k', '6.8k to 10K','7.10K to 12k','8.12K to 14k',
          '9.14K to 16k','10.16K to 18k','11.18K to 20k','12.20k+','13.Other'
          ]

# Use the 'when' and 'otherwise' functions to create a new column
df = df.withColumn(
    "reviewTot_segment",
    when(conditions[0], labels[0])
    .when(conditions[1], labels[1])
    .when(conditions[2], labels[2])
    .when(conditions[3], labels[3])
    .when(conditions[4], labels[4])
    .when(conditions[5], labels[5])
    .when(conditions[6], labels[6])
    .when(conditions[7], labels[7])
    .when(conditions[8], labels[8])
    .when(conditions[9], labels[9])
    .when(conditions[10], labels[10])
    .when(conditions[11], labels[11])
    .otherwise(labels[12])
)

# Show the updated DataFrame
df.show()

# COMMAND ----------

display(df)

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

agg_df = df.groupBy("reviewTot_segment").agg({"label": "mean"})

# Convert the PySpark DataFrame to a Pandas DataFrame for plotting
plot_df = agg_df.toPandas()

# Define the desired order for the x-axis labels
desired_order = ['1.<500', '2.500 to 2k', '3.2k to 4k', '4.4k to 6k', '5.6k to 8k', '6.8k to 10K','7.10K to 12k','8.12K to 14k',
          '9.14K to 16k','10.16K to 18k','11.18K to 20k','12.20k+','13.Other']

# Reorder the DataFrame based on the desired order
plot_df['reviewTot_segment'] = pd.Categorical(plot_df['reviewTot_segment'], categories=desired_order, ordered=True)
plot_df = plot_df.sort_values('reviewTot_segment')

# Create the bar chart using Matplotlib with the desired order
plt.figure(figsize=(8, 6))
plt.bar(plot_df["reviewTot_segment"], plot_df["avg(label)"])
plt.xlabel("Review Length")
plt.ylabel("Mean Label Value")
plt.title("Bar Chart of Review Length vs. Mean Label Value")
plt.xticks(rotation=45)  # Rotate x-axis labels for readability if needed
plt.show()


# COMMAND ----------

segment_counts = df.groupBy("reviewTot_segment").count().orderBy("reviewTot_segment")
segment_counts.show()

# COMMAND ----------

from pyspark.sql.functions import corr

# Assuming your DataFrame is named 'df'

# Calculate the correlation between 'reviewTot_length' and 'label'
correlation = df.select(corr("reviewTot_length", "label")).collect()[0][0]

# Print the correlation value
print("Correlation between 'reviewTot_length' and 'label':", correlation)

# COMMAND ----------

length_stats = df.describe("reviewTot_length")

# Show the statistics
length_stats.show()

# COMMAND ----------

# Calculate and display percentiles (25%, 50%, and 75%)
percentiles = df.selectExpr("percentile_approx(reviewTot_length, array(0.25, 0.5, 0.75)) as percentiles")

# Show the percentiles
percentiles.show(truncate=False)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Assuming your DataFrame is named 'df'

# Create a VectorAssembler to assemble features
assembler = VectorAssembler(inputCols=["reviewTot_length"], outputCol="features")

# Create a LogisticRegression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Create a pipeline that assembles features and trains the logistic regression model
pipeline = Pipeline(stages=[assembler, lr])

(trainingData, testData) = df.randomSplit([0.8, 0.2])

# Fit the pipeline on your DataFrame
model = pipeline.fit(trainingData)

# Make predictions using the model
predictions = model.transform(testData)

# Show the predictions
predictions.select("label", "reviewTot_length", "prediction", "probability").show()

# COMMAND ----------

# Calculate AUC for train/test split

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print("AUC = %g" % auc)
