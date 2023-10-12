# Databricks notebook source
# Load in the table
df = spark.sql("select * from default.reviews_train")#.sample(0.01)

df = df.cache()

print((df.count(), len(df.columns)))

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your PySpark DataFrame

# Convert PySpark DataFrame to Pandas DataFrame for plotting
pandas_df = df.select('label').toPandas()

# Calculate value counts
value_counts = pandas_df['label'].value_counts()

# Plot the distribution as a bar chart
plt.figure(figsize=(8, 6))
value_counts.plot(kind='bar')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.show()

# COMMAND ----------

# Sentiment Analysis
import nltk

nltk.download('all')

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

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent = SentimentIntensityAnalyzer()

# define a function
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Define a UDF to calculate compound sentiment score
def check_sentiment_udf(text):
    return sent.polarity_scores(text)['compound']

# Register the UDF
check_sentiment = udf(check_sentiment_udf, FloatType())

# Apply the UDF to the DataFrame
df = df.withColumn("compound_score", check_sentiment(df["reviewTot"]))

display(df)


# COMMAND ----------

# Sentiment & Polarity Analysis features
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StringType, StructType, StructField
from textblob import TextBlob

def calculate_polarity(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
polarity_udf = udf(calculate_polarity, FloatType())
df = df.withColumn("polarity_score", polarity_udf(df["reviewTot"]))

def calculate_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.subjectivity
sentiment_udf = udf(calculate_sentiment, FloatType())
df = df.withColumn("sentiment_score", sentiment_udf(df["reviewTot"]))

# COMMAND ----------

display(df)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Collect the compound scores into a local list
compound_scores = df.select("compound_score").rdd.flatMap(lambda x: x).collect()

# Create a histogram using Matplotlib
plt.figure(figsize=(12, 6))
plt.hist(compound_scores, bins=50, color='blue', alpha=0.7)
plt.xlabel("Compound Score")
plt.ylabel("Frequency")
plt.title("Histogram of Compound Scores")
plt.grid(True)
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Collect the compound scores into a local list
compound_scores = df.select("polarity_score").rdd.flatMap(lambda x: x).collect()

# Create a histogram using Matplotlib
plt.figure(figsize=(12, 6))
plt.hist(compound_scores, bins=50, color='blue', alpha=0.7)
plt.xlabel("polarity_score")
plt.ylabel("Frequency")
plt.title("Histogram of polarity_score")
plt.grid(True)
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Collect the compound scores into a local list
compound_scores = df.select("sentiment_score").rdd.flatMap(lambda x: x).collect()

# Create a histogram using Matplotlib
plt.figure(figsize=(12, 6))
plt.hist(compound_scores, bins=50, color='blue', alpha=0.7)
plt.xlabel("sentiment_score")
plt.ylabel("Frequency")
plt.title("Histogram of subjectivity_score")
plt.grid(True)
plt.show()

# COMMAND ----------

# generate label function
from pyspark.sql.functions import when, col

# Use the when() function to create a new column based on conditions
df = df.withColumn("Predicted_Label", when((col("compound_score") > 0.5) | (col("compound_score") < -0.5), 1).otherwise(0))


display(df)

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your PySpark DataFrame

# Select 'label' and 'Predicted_label' columns and convert to Pandas DataFrame
pandas_df = df.select('label', 'Predicted_Label').toPandas()

# Calculate value counts for both 'label' and 'Predicted_label'
value_counts_label = pandas_df['label'].value_counts()
value_counts_predicted_label = pandas_df['Predicted_Label'].value_counts()

# Create a bar chart with both distributions side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot 'label' distribution
axes[0].bar(value_counts_label.index, value_counts_label.values)
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Label')

# Plot 'Predicted_label' distribution
axes[1].bar(value_counts_predicted_label.index, value_counts_predicted_label.values)
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Predicted Label')

plt.tight_layout()
plt.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
df = df.withColumn("Bin", when(col("compound_score") < 1, 999).otherwise(col("compound_score")))


df = df.withColumn("Bin", 
                    when(col("compound_score") <= -0.75, 1)
                   .when((col("compound_score") > -0.75) & (col("compound_score") <= -0.5), 2) 
                   .when((col("compound_score") > -0.5) & (col("compound_score") <= -0.25), 3) 
                   .when((col("compound_score") > -0.25) & (col("compound_score") <= 0), 4)
                   .when((col("compound_score") >0 ) & (col("compound_score") <= 0.25), 5)
                   .when((col("compound_score") >0.25 ) & (col("compound_score") <= 0.5), 6)
                   .when((col("compound_score") >0.5 ) & (col("compound_score") <= 0.75), 7)
                   .when(col("compound_score") > 0.75, 8 )
                   .otherwise("Unknown")) 

# COMMAND ----------

bin_label_count = df.groupBy('Bin', 'Label').count().orderBy('Bin', 'Label')

# Convert the Spark DataFrame to a Pandas DataFrame for plotting
histogram_data = bin_label_count.toPandas()

# Pivot the DataFrame to have separate columns for label 0 and label 1 counts
pivoted_data = histogram_data.pivot(index='Bin', columns='Label', values='count')
pivoted_data = pivoted_data.fillna(0).astype(int)

# Sort the index (X-axis values) in ascending order
pivoted_data = pivoted_data.sort_index()

# Create a bar plot to visualize the histogram
ax = pivoted_data.plot(kind='bar', stacked=False)
plt.xlabel('Compound_Score')
plt.ylabel('Count')
plt.title('Histogram of Label Count by Compound_Score')
plt.legend(title='Label', labels=['Label 0', 'Label 1'])

# Relabel the X-axis (Bins)
new_labels = ['<-0.75','-0.75->-0.5','-0.5->-0.25','-0.25->0', '0->0.25', '0.25->0.5','0.5->0.75','0.75->1']  # Replace with your desired labels
ax.set_xticklabels(new_labels)
# Rotate the existing X-axis labels for all bins
#ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.xticks(rotation=45)

plt.show()

# COMMAND ----------

from pyspark.sql.functions import col
display(df.groupBy("Bin", "Label").count().orderBy("Bin", "Label"))

# COMMAND ----------


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Convert PySpark DataFrame to Pandas DataFrame
pandas_df = df.toPandas()

# Calculate accuracy
accuracy = accuracy_score(pandas_df['label'], pandas_df['Predicted_Label'])

# Generate classification report
classification_rep = classification_report(pandas_df['label'], pandas_df['Predicted_Label'])

# Generate confusion matrix
confusion_mat = confusion_matrix(pandas_df['label'], pandas_df['Predicted_Label'])

# COMMAND ----------

print(accuracy)
print(confusion_mat )
print(classification_rep)

# COMMAND ----------

from pyspark.sql.functions import corr

# Assuming your DataFrame is named 'df'

# Calculate the correlation between 'reviewTot_length' and 'label'
correlation = df.select(corr("compound_score", "label")).collect()[0][0]

# Print the correlation value
print("Correlation between 'compound_score' and 'label':", correlation)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Assuming your DataFrame is named 'df'

# Create a VectorAssembler to assemble features
assembler = VectorAssembler(inputCols=["compound_score"], outputCol="features")

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
predictions.select("label", "compound_score", "prediction", "probability").show()

# COMMAND ----------

# Calculate AUC for train/test split

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
auc = evaluator.evaluate(predictions)
print("AUC = %g" % auc)
print("Test Accuracy       = %g" % (acc_evaluator.evaluate(predictions)))

# COMMAND ----------

######################################### Check Sentiment using TextBlob



# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Assuming your DataFrame is named 'df'

# Create a VectorAssembler to assemble features
assembler = VectorAssembler(inputCols=["polarity_score"], outputCol="features")

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
predictions.select("label", "polarity_score", "prediction", "probability").show()

# COMMAND ----------

# Calculate AUC for train/test split

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
auc = evaluator.evaluate(predictions)
print("AUC = %g" % auc)
print("Test Accuracy       = %g" % (acc_evaluator.evaluate(predictions)))

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
df = df.withColumn("Bin_Pol", when(col("polarity_score") < 1, 999).otherwise(col("polarity_score")))


df = df.withColumn("Bin_Pol", 
                    when(col("polarity_score") <= 0, 1)
                   .when((col("polarity_score") > 0) & (col("polarity_score") <= 0.25), 2) 
                   .when((col("polarity_score") >0.25 ) & (col("polarity_score") <= 0.5), 3)
                   .when((col("polarity_score") >0.5 ) & (col("polarity_score") <= 0.75), 4)
                   .when(col("polarity_score") > 0.75, 5 )
                   .otherwise("Unknown")) 

# COMMAND ----------

from pyspark.sql.functions import col
display(df.groupBy("Bin_Pol", "Label").count().orderBy("Bin_Pol", "Label"))

# COMMAND ----------

#########################################Subjectivity######################################
'''Sentiment Subjectivity (sentiment.subjectivity):

Definition: Sentiment subjectivity measures the degree to which a text expresses opinions, emotions, or personal feelings, as opposed to objective, factual information.
Values: Sentiment subjectivity is typically quantified on a scale ranging from 0 to 1, where:
0 represents a highly objective or factual text.
1 represents a highly subjective or emotional text.'''

'''Example Text: "The weather today is 25 degrees Celsius."

sentiment.polarity: 0 (neutral)
sentiment.subjectivity: 0 (objective)
Example Text: "I love the beautiful weather today!"

sentiment.polarity: 0.5 (positive)
sentiment.subjectivity: 1 (highly subjective)''''

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Assuming your DataFrame is named 'df'

# Create a VectorAssembler to assemble features
assembler2 = VectorAssembler(inputCols=["sentiment_score"], outputCol="features")

# Create a LogisticRegression model
lr2 = LogisticRegression(featuresCol="features", labelCol="label")

# Create a pipeline that assembles features and trains the logistic regression model
pipeline2 = Pipeline(stages=[assembler2, lr2])

(trainingData, testData) = df.randomSplit([0.8, 0.2])

# Fit the pipeline on your DataFrame
model2 = pipeline2.fit(trainingData)

# Make predictions using the model
predictions = model2.transform(testData)

# Show the predictions
predictions.select("label", "sentiment_score", "prediction", "probability").show()

# COMMAND ----------

# Calculate AUC for train/test split

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
auc = evaluator.evaluate(predictions)
print("AUC = %g" % auc)
print("Test Accuracy       = %g" % (acc_evaluator.evaluate(predictions)))

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
df = df.withColumn("Bin_Sen", when(col("sentiment_score") < 1, 999).otherwise(col("sentiment_score")))


df = df.withColumn("Bin_Sen", 
                    when(col("sentiment_score") <= 0.2, 1)
                   .when((col("sentiment_score") > 0.2) & (col("sentiment_score") <= 0.4), 2) 
                   .when((col("sentiment_score") >0.4 ) & (col("sentiment_score") <= 0.6), 3)
                   .when((col("sentiment_score") >0.6 ) & (col("sentiment_score") <= 0.8), 4)
                   .when(col("sentiment_score") > 0.8, 5 )
                   .otherwise("Unknown")) 

# COMMAND ----------

from pyspark.sql.functions import col
display(df.groupBy("Bin_Sen", "Label").count().orderBy("Bin_Sen", "Label"))
