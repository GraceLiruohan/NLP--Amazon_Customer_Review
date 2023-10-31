# Databricks notebook source
# DBTITLE 1,Load Data
# Load in the table
df = spark.sql("select * from default.reviews_train").sample(0.7)

df = df.cache()

print((df.count(), len(df.columns)))

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import NGram
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, NGram, Word2Vec, StringIndexer, RegexTokenizer, StopWordsRemover
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Stemmer, LemmatizerModel, SentimentDLModel
# Create NGram transformer
ngram = NGram(n=2, inputCol="reviewText", outputCol="ngrams")

# Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["ngrams"]) \
    .setOutputCols(["features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# Create a LightGBM classifier
lgb_classifier = GBTClassifier(maxIter=10, featuresCol="features")

# Create a pipeline
pipeline = Pipeline(stages=[ngram, finisher, lgb_classifier])

# Fit the pipeline
pipeline_model = pipeline.fit(df)

# Get the LightGBM model from the pipeline
lgb_model = pipeline_model.stages[-1]

# Get feature importances
feature_importance = lgb_model.featureImportances
print(feature_importance)

# COMMAND ----------

from pyspark.sql.functions import col

# Assuming you have a Spark DataFrame named 'df' with a boolean column 'verified'
# Convert 'verified' to an integer column
df = df.withColumn("verified", col("verified").cast("integer"))

# Show the updated DataFrame
df.show()
from pyspark.sql.functions import col, sum
from pyspark.sql import functions as F

# Group by "reviewID" and count the occurrences
review_counts = df.groupBy("reviewerID").agg(F.count("*").alias("review_count"))

# Left join sum_df to df by "reviewID" and select "sumbyreviewID" from sum_df
merged_df = df.join(review_counts.select("reviewerID","review_count"), on="reviewerID", how="left")

display(merged_df)
from pyspark.sql.functions import col, sum

# Ensure the "label" column is cast to an integer type

merged_df = merged_df.withColumnRenamed("asin", "productId")

# Group by "productId" and count the occurrences
Product_counts = merged_df.groupBy("productId").agg(F.count("*").alias("productId_count"))

# Left join sum_df to df by "productId" and select "sumbyreviewID" from sum_df
merged_df2 = merged_df.join(Product_counts.select("productId","productId_count"), on="productId", how="left")



display(merged_df2)
print((merged_df2.count(), len(merged_df2.columns)))
df = merged_df2.alias("df")

print((df.count(), len(df.columns)))
display(df)
df.printSchema()
display(df)

# COMMAND ----------

# DBTITLE 1,Data Wrangling/Prep
from pyspark.sql.functions import year, month, dayofweek
import numpy as np
from pyspark.sql.functions import concat_ws, regexp_replace, col, length, datediff, current_date, to_date, countDistinct, desc, row_number, min, max

# nltk import
import nltk
#nltk.download('all')
from nltk.corpus import words
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# For our intitial modeling efforts, we are not going to use the following features
#drop_list = ['summary','asin', 'reviewID', 'reviewerID', 'unixReviewTime','reviewTime', 'image', 'style', 'reviewerName']
#df = df.select([column for column in df.columns if column not in drop_list])

# Drop duplicates
print("Before duplication removal: ", df.count())
df = df.dropDuplicates(['reviewerID', 'productId'])
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

# calculating age of review
df = df.withColumn("age", 2023 - year(df['reviewTime']))

# review counts
review_counts1 = df.groupBy("ProductId").agg(
  countDistinct("reviewID").alias("reviews_per_product"),
  countDistinct("reviewerID").alias("reviewers_per_product"),
  min("reviewTime").alias("product_earliest_review"),
  max("reviewTime").alias("product_latest_review"))
df = df.join(review_counts1, on="ProductId", how="inner")
df = df.withColumn("product_review_interval", datediff(df['product_latest_review'], df['product_earliest_review']))

# review_order
from pyspark.sql.window import Window
window_spec = Window.partitionBy("ProductId").orderBy(desc("reviewTime"))
df = df.withColumn("reviewOrder", row_number().over(window_spec))

# combining review text & summary
df = df.withColumn("combined_text", concat_ws(" ", "reviewText", "summary"))

# Convert the "review" column to lowercase
from pyspark.sql.functions import lower
df = df.withColumn("combined_text", lower(df["combined_text"]))

# drop NA
df = df.na.drop(subset=["reviewText", "label"])

# Sentiment & Polarity Analysis features
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StringType, StructType, StructField
from textblob import TextBlob

def calculate_polarity(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
polarity_udf = udf(calculate_polarity, FloatType())
df = df.withColumn("polarity_score", polarity_udf(df["combined_text"]))

def calculate_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.subjectivity
sentiment_udf = udf(calculate_sentiment, FloatType())
df = df.withColumn("sentiment_score", sentiment_udf(df["combined_text"]))

# length of text
df = df.withColumn("combined_text_length", length(col("combined_text")))

# Removing characters using Regex
pattern_to_remove = "[-? ! "", .]+"
df = df.withColumn("combined_text", regexp_replace(col("combined_text"), pattern_to_remove, " "))

# remove numbers
df = df.withColumn("combined_text", regexp_replace(col("combined_text"), r'\d+', ''))

# removing white space
df = df.withColumn("combined_text", regexp_replace(col("combined_text"), r'[^\w\s]', ''))
'''
# POS tagging
# Define a function to perform part-of-speech tagging using NLTK
def pos_tagging(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return " ".join([f"{word}/{tag}" for word, tag in pos_tags])
pos_tagging_udf = udf(pos_tagging, StringType())

# Apply the UDF to the DataFrame
df = df.withColumn("combined_text", pos_tagging_udf(df["combined_text"]))

# spellcheck
def spell_check(text):
    word_list = set(words.words())
    tokens = text.split()
    corrected_tokens = [word if word.lower() in word_list else f'[{word}]' for word in tokens]
    return ' '.join(corrected_tokens)
spell_check_udf = udf(spell_check, StringType())
df = df.withColumn("combined_text", spell_check_udf(df["combined_text"]))'''


display(df)
print((df.count(), len(df.columns)))


# COMMAND ----------

# DBTITLE 1,Create a Data Transformation/ML Pipeline
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Stemmer, LemmatizerModel, SentimentDLModel

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, NGram, Word2Vec, StringIndexer, RegexTokenizer, StopWordsRemover
from synapse.ml.lightgbm import LightGBMClassifier

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier

from imblearn.over_sampling import SMOTE
import numpy as np

# convert text column to spark nlp document
#document_assembler = DocumentAssembler() \
#    .setInputCol("combined_text") \
#    .setOutputCol("document")

'''
# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
'''

# We'll tokenize the text using a simple RegexTokenizer
tokenizer = RegexTokenizer(inputCol="combined_text", outputCol="words", pattern="\\W")

# clean tokens 
#normalizer = Normalizer() \
#    .setInputCols(["token"]) \
#    .setOutputCol("normalized")

# remove stopwords
#stopwords_cleaner = StopWordsCleaner()\
#      .setInputCols("normalized")\
#      .setOutputCol("cleanTokens")\
#      .setCaseSensitive(False)

# stems tokens to bring it to root form
#stemmer = Stemmer() \
#    .setInputCols(["words"]) \
#    .setOutputCol("stem")

# Remove standard Stopwords
stopwords_cleaner = StopWordsRemover(inputCol="words", outputCol="filtered")

# lemmatizer
#lemmatizer = LemmatizerModel.pretrained() \
#.setInputCols(["cleanTokens"]) \
#.setOutputCol("lemma")

# Convert custom document structure to array of tokens.
#finisher = Finisher() \
#    .setInputCols(["lemma"]) \
#    .setOutputCols(["token_features"]) \
#    .setOutputAsArray(True) \
#    .setCleanAnnotations(False)

# Create a StandardScaler to scale numerical features
from pyspark.ml.feature import StandardScaler,MinMaxScaler

# Generate Term Frequency
#tf = CountVectorizer(inputCol="token_features", outputCol="rawFeatures", vocabSize=10000, minTF=1, minDF=50, maxDF=0.40)

# Generate Inverse Document Frequency weighting
#idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=5)

# scaler_id
#scaler_id = MinMaxScaler(inputCol="idfFeatures", outputCol="scaled_idfFeatures")

####n-gram features######

# Create n-grams from tokenized words
ngram = NGram(n=2,inputCol="combined_text",outputCol="ngrams")

# Generate Term Frequency for ngram
#tf_n = CountVectorizer(inputCol="ngrams", outputCol="rawFeatures_n", vocabSize=10000, minTF=1, minDF=50, maxDF=0.40)

# Generate Inverse Document Frequency weighting for ngram
#idf_n = IDF(inputCol="rawFeatures_n", outputCol="idfFeatures_n", minDocFreq=5)

# scaler_n
#scaler_n = MinMaxScaler(inputCol="idfFeatures_n", outputCol="scaled_idfFeatures_n")

# string indexer
#string_indexer_reviewerID = StringIndexer(inputCol="reviewerID", outputCol="indexed_reviewerID")
#string_indexer_asin = StringIndexer(inputCol="asin", outputCol="indexed_asin")

# Word2Vec
word2vec = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered", outputCol="word2vec_features")

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "year", "month", "day_of_week", "age", "reviews_per_product", "reviewers_per_product","product_review_interval","polarity_score","sentiment_score", "combined_text_length", "word2vec_features"
                                        ], outputCol="features_assembled")
# "indexed_reviewerID","indexed_asin","review_count","productId_count", "scaled_idfFeatures", "scaled_idfFeatures_n", "ngrams", "reviewOrder"

# scaler
scaler = MinMaxScaler(inputCol="features_assembled", outputCol="features")

# applying smote for imbalance
#smote = SMOTE(sampling_strategy=1.0, inputCol="features_scaled", outputCol="features")

# Machine Learning Algorithm
#ml_alg  = LogisticRegression(maxIter=20, regParam=0.1, elasticNetParam=0.0)
#ml_alg  = RandomForestClassifier()
#ml_alg = MultilayerPerceptronClassifier(labelCol="label",layers=[2,2,2],maxIter=10, blockSize=128)
# ml_alg = LightGBMClassifier(learningRate=0.3, numIterations=150, numLeaves=31)
#ml_alg = LightGBMClassifier(learningRate=0.1, numIterations=150, numLeaves=31,metric ="AUC", lambdaL1=0.3)
#ml_alg = LightGBMClassifier(learningRate=0.1, numIterations=150, numLeaves=31,metric ="AUC", lambdaL1=0.2,featureFraction=0.90)-0.87377

#ml_alg = LightGBMClassifier(learningRate=0.1, numIterations=150, numLeaves=31,metric ="AUC", lambdaL1=0.2,lambdaL2=0.1, featureFraction=0.85)

#ml_alg = LightGBMClassifier(learningRate=0.1, numIterations=150, numLeaves=31,metric ="AUC",parallelism ="voting_parallel")

# Parameter Grid for ml_alg
# for logistic regression
'''paramGrid = ParamGridBuilder() \
    .addGrid(ml_alg.maxIter, [10, 20, 30]) \
    .addGrid(ml_alg.regParam, [0.1, 0.01, 0.001]) \
    .addGrid(ml_alg.elasticNetParam, [0.0, 0.1, 0.2]) \
    .build()

crossval = CrossValidator(estimator=ml_alg,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(metricName="areaUnderROC"),
                          numFolds=3)'''

pipeline = Pipeline(
    stages=[#document_assembler, 
            tokenizer,
#            normalizer,
#            stemmer,
            stopwords_cleaner, 
#            lemmatizer, 
#            finisher,           
#            tf,
#            idf,
#            scaler_id,
#            ngram,
#            tf_n,
#            idf_n,
#            scaler_n,
#            string_indexer_reviewerID,
#            string_indexer_asin,
            word2vec,
#            assembler,
            scaler])
#            smote,
#            ml_alg])

# COMMAND ----------

# DBTITLE 1,Split into testing/training
# set seed for reproducibility
(trainingData, testData) = df.randomSplit([0.8, 0.2], seed = 47)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count:     " + str(testData.count()))

# COMMAND ----------

# DBTITLE 1,Transform Training Data & Tune
# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(trainingData)
trainingDataTransformed = pipelineFit.transform(trainingData)
trainingDataTransformed.show(5)

# COMMAND ----------

# initiate model
ml_alg = LightGBMClassifier(learningRate=0.05, numIterations=2000, numLeaves=50,metric ="AUC", lambdaL1=1,lambdaL2=1, featureFraction=0.85, featuresCol="features")

# COMMAND ----------

# train model
pipelineModel = ml_alg.fit(trainingDataTransformed)

# COMMAND ----------

# transforming test data
testingDataTransform = pipelineFit.transform(testData)
display(testingDataTransform)

# COMMAND ----------

# DBTITLE 1,Predict Testing Data
predictions = pipelineModel.transform(testingDataTransform)
predictions.groupBy("prediction").count().show()

# COMMAND ----------

display(predictions)

# COMMAND ----------

# DBTITLE 1,Performance Metrics on Testing Data
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

#acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
#pre_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
#rec_evaluator = MulticlassClassificationEvaluator(metricName="weightedRecall")
#pr_evaluator  = BinaryClassificationEvaluator(metricName="areaUnderPR")
auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

#print("Test Accuracy       = %g" % (acc_evaluator.evaluate(predictions)))
#print("Test Precision      = %g" % (pre_evaluator.evaluate(predictions)))
#print("Test Recall         = %g" % (rec_evaluator.evaluate(predictions)))
#print("Test areaUnderPR    = %g" % (pr_evaluator.evaluate(predictions)))
print("Test areaUnderROC   = %g" % (auc_evaluator.evaluate(predictions)))

# COMMAND ----------

# DBTITLE 1,Make Predictions on Kaggle Test Data
test_df = spark.sql("select * from default.reviews_test")


from pyspark.sql.functions import col

from pyspark.sql import functions as F


# Assuming you have a Spark DataFrame named 'df' with a boolean column 'verified'
# Convert 'verified' to an integer column
test_df = test_df.withColumn("verified", col("verified").cast("integer"))



# Group by "reviewID" and count the occurrences
review_counts_test = test_df.groupBy("reviewerID").agg(F.count("*").alias("review_count"))

# Left join sum_df to df by "reviewID" and select "sumbyreviewID" from sum_df
merged_df_test = test_df.join(review_counts_test.select("reviewerID","review_count"), on="reviewerID", how="left")


# Ensure the "label" column is cast to an integer type

merged_df_test = merged_df_test.withColumnRenamed("asin", "productId")

# Group by "productId" and count the occurrences
Product_counts_test = merged_df_test.groupBy("productId").agg(F.count("*").alias("productId_count"))

# Left join sum_df to df by "productId" and select "sumbyreviewID" from sum_df
merged_df_test2 = merged_df_test.join(Product_counts_test.select("productId","productId_count"), on="productId", how="left")

display(merged_df_test2)


print((merged_df_test2 .count(), len(merged_df_test2.columns)))

# COMMAND ----------

test_df = merged_df_test2.alias("test_df")

print((test_df.count(), len(test_df.columns)))

# COMMAND ----------

from pyspark.sql.functions import year, month, dayofweek
# For our intitial modeling efforts, we are not going to use the following features
#drop_list = ['summary','asin', 'reviewID', 'reviewerID', 'unixReviewTime','reviewTime', 'image', 'style', 'reviewerName']
#df = df.select([column for column in df.columns if column not in drop_list])

# Drop duplicates
#print("Before duplication removal: ", test_df.count())
#test_df = test_df.dropDuplicates(['reviewerID', 'asin'])
#print("After duplication removal: ", test_df.count())


# Convert Unix timestamp to readable date & create date columns
from pyspark.sql.functions import from_unixtime, to_date
from pyspark.sql.types import *
test_df = test_df.withColumn("reviewTime", to_date(from_unixtime(test_df.unixReviewTime))) \
                                                .drop("unixReviewTime")
test_df = test_df.withColumn('reviewTime', test_df['reviewTime'].cast('date'))

# Extract year, month, and day of the week into separate columns
test_df = test_df.withColumn('year', year(test_df['reviewTime']))
test_df = test_df.withColumn('month', month(test_df['reviewTime']))
test_df = test_df.withColumn('day_of_week', dayofweek(test_df['reviewTime']))

# age of review
test_df = test_df.withColumn("age", 2023 - year(test_df['reviewTime']))

# review counts
review_counts1 = test_df.groupBy("ProductId").agg(
  countDistinct("reviewID").alias("reviews_per_product"),
  countDistinct("reviewerID").alias("reviewers_per_product"),
  min("reviewTime").alias("product_earliest_review"),
  max("reviewTime").alias("product_latest_review"))
test_df = test_df.join(review_counts1, on="ProductId", how="inner")
test_df = test_df.withColumn("product_review_interval", datediff(test_df['product_latest_review'], test_df['product_earliest_review']))

# review_order
from pyspark.sql.window import Window
window_spec = Window.partitionBy("ProductId").orderBy(desc("reviewTime"))
test_df = test_df.withColumn("reviewOrder", row_number().over(window_spec))

# combining review text & summary
test_df = test_df.withColumn("combined_text", concat_ws(" ", "reviewText", "summary"))

# Convert the "review" column to lowercase
from pyspark.sql.functions import lower
test_df = test_df.withColumn("combined_text", lower(test_df["combined_text"]))

# drop NA
#test_df = test_df.na.drop(subset=["reviewText"])
#display(test_df)
#print((test_df.count(), len(test_df.columns)))

# Sentiment & Polarity Analysis features
test_polarity_udf = udf(calculate_polarity, FloatType())
test_df = test_df.withColumn("polarity_score", test_polarity_udf(test_df["combined_text"]))

test_sentiment_udf = udf(calculate_sentiment, FloatType())
test_df = test_df.withColumn("sentiment_score", test_sentiment_udf(test_df["combined_text"]))

# length of text
test_df = test_df.withColumn("combined_text_length", length(col("combined_text")))

# Removing characters using Regex
pattern_to_remove = "[-? ! "", .]+"
test_df = test_df.withColumn("combined_text", regexp_replace(col("combined_text"), pattern_to_remove, " "))

# remove numbers
test_df = test_df.withColumn("combined_text", regexp_replace(col("combined_text"), r'\d+', ''))

# removing white space
test_df = test_df.withColumn("combined_text", regexp_replace(col("combined_text"), r'[^\w\s]', ''))

'''# POS tagging
# Define a function to perform part-of-speech tagging using NLTK
def pos_tagging(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return " ".join([f"{word}/{tag}" for word, tag in pos_tags])

# Create a UDF to apply the POS tagging function
pos_tagging_udf = udf(pos_tagging, StringType())

# Apply the UDF to the DataFrame
test_df = test_df.withColumn("combined_text", pos_tagging_udf(test_df["combined_text"]))
'''

test_df_transformed = pipelineFit.transform(test_df)
kaggle_pred = pipelineModel.transform(test_df_transformed)
display(kaggle_pred)
kaggle_pred.groupBy("prediction").count().show()

# COMMAND ----------

display(test_df)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

probelement=udf(lambda v:float(v[1]),FloatType())
submission_data = kaggle_pred.select('reviewID', probelement('probability')).withColumnRenamed('<lambda>(probability)', 'label')

# COMMAND ----------

# Download this and submit to Kaggle!
display(submission_data.select(["reviewID", "label"]))

# COMMAND ----------

submission_data.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("dbfs:/FileStore/cl_test/trial11.9.csv")

#dbfs:/FileStore/cl_test/trial8.4.2.csv/part-00000-tid-6545238503219276197-5f40b6c2-d550-4e4b-a0c5-43f951d75f05-20222-1-c000.csv
#df_sub = spark.read.csv("/FileStore/cl_test/trial8.4.3.csv/part-00000-tid-6545238503219276197-5f40b6c2-d550-4e4b-a0c5-43f951d75f05-20222-1-c000.csv", header=True, inferSchema=True)
#display(df_sub.select(["reviewID", "label"]))

# Define the path to save the file (local path)
#local_path = "/Users/22lxv@queensu.ca/Competition/test_datalg15.csv"

# Write the DataFrame to a CSV file (local file)
#submission_data.write.option("header", "true").csv(local_path)

# COMMAND ----------


#df2 = spark.read.csv("dbfs:/Users/22lxv@queensu.ca/Competition/submission_data.csv/part-00000-tid-6478110848838769667-2fd9e73b-f7c3-4f9b-b65d-bce5e8df9953-1149619-1-c000.csv", header=True, inferSchema=True)

# COMMAND ----------

# Getting feature importance from the Random Forest model

feature_importance = pipelineModel.stages[-2].featureImportances
print(feature_importance)

# COMMAND ----------


import numpy as np
import pandas as pd

top20_indice = np.flip(np.argsort(feature_importance.toArray()))[:20].tolist()
top20_importance = []
for index in top20_indice:
    top20_importance.append(feature_importance[index])

top20_df = spark.createDataFrame(pd.DataFrame(list(zip(top20_indice, top20_importance)), columns =['index', 'importance']))

display(top20_df)
