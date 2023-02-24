from __future__ import print_function

import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from  pyspark.sql.functions import split
import pandas as pd 
from pyspark.ml.feature import StringIndexer # to convert class labels to numeric
from pyspark.sql.functions import col # to convert class labels to double to integer
from pyspark.sql import functions as F # to process reject expressions 
from pyspark.sql.functions import col, when
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml import Pipeline



def readFile(file_path):
    df_row = spark.read.csv(file_path)
    # print("original")
    # print(df_row.show(4,False))
    df= df_row.withColumn("text", split(df_row['_c0'], ';').getItem(0))\
            .withColumn('emotion', split(df_row['_c0'], ';').getItem(1))
    df1= df.select("text", "emotion")
    return df1

# to clean punctuations and none letters item 
def clean(df):
    # Remove html tags from text
    df = df.withColumn("text_c", F.regexp_replace(F.col("text"), r'<[^>]+>', ""))
    # Remove non-letters and remove punctuation 
    df = df.withColumn("text_c", F.regexp_replace("text_c", r"[^a-zA-Z ]", ""))
    # Remove words 1, 2 char
    df = df.withColumn("text_c", F.regexp_replace("text_c", r"\b\w{1,2}\b", ""))
    return df

# reads txt file and splits in to two columns 

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # file paths
    file_path_training = "../emotions-detector/data/train.txt"
    file_path_testing = "../emotions-detector/data/test.txt"
    file_path_val= "../emotions-detector/data/val.txt"
  
    # reading documents to df 
    df_training = readFile(file_path= file_path_training)
    df_testing = readFile(file_path= file_path_training)
    df_val = readFile(file_path= file_path_val)
    # print("sample")
    # print(df_val.show(4,False))

    #information about the data
    print("size of the data")
    print("training size ",df_training.toPandas().shape , "data points") 
    print("testing size ",df_testing.toPandas().shape, "data points")
    print("validation size ",df_val.toPandas().shape, "data points") 
    print("class count for each documents in training")
    print("training class count", df_training.groupBy("emotion").count().show())
    training_size = df_training.count()
    #converting class labels
    
    string_indexer = StringIndexer(inputCol="emotion",outputCol="class")

    # fit the StringIndexer to the DataFrame and transform the DataFrame to add the 'class_numeric' column
    df_training_class = string_indexer.fit(df_training).transform(df_training).withColumn('class', col('class').cast("integer"))
    df_testing_class = string_indexer.fit(df_testing).transform(df_testing).withColumn('class', col('class').cast("integer"))
    df_val_class = string_indexer.fit(df_val).transform(df_val).withColumn('class', col('class').cast("integer"))
    # print(df_training_class.show(2))

    # step 1 cleaning 
    df_training_clean = clean(df_training_class )
    df_testing_clean= clean(df_testing_class )
    df_val_clean =clean(df_val_class)

    # since we have unbalanced data we are adding weight
    w_joy = df_training_clean.filter('class == 0').count()/ training_size
    w_sadness = df_training_clean.filter('class == 1').count()/ training_size
    w_anger = df_training_clean.filter('class == 2').count()/ training_size
    w_fear = df_training_clean.filter('class == 3').count()/ training_size
    w_love = df_training_clean.filter('class == 4').count()/ training_size
    w_surprize = df_training_clean.filter('class == 5').count()/ training_size

    print(w_joy, w_sadness, w_anger, w_fear, w_love, w_surprize)

    df_training_weight = df_training_clean.withColumn("weight", when(F.col("class")==0,w_joy).
                                                  when(F.col("class")==1,w_sadness).
                                                  when(F.col("class")==2,w_anger).
                                                  when(F.col("class")==3,w_fear).
                                                  when(F.col("class")==4,w_love).
                                                  otherwise(w_surprize))

    df_training_weight.show(10)

    # data cleaning pipelines 
    # Tokenize the review text
    tokenizer = Tokenizer(inputCol="text_c", outputCol="words",)
    # Remove stop words
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
    # Create a count vectoriser
    countVectorizer = CountVectorizer(inputCol=remover.getOutputCol(), outputCol="rawFeatures", vocabSize=1000)
    # Calculate the TF-IDF
    idf = IDF(inputCol=countVectorizer.getOutputCol(), outputCol="featuresIDF")
    # Crate a preprocessing pipeline wiht 4 stages
    pipeline_p = Pipeline(stages=[tokenizer,remover, countVectorizer, idf])

    #data prep model
    data_model  = pipeline_p.fit(df_training_weight)
    print(dir(data_model ))

    # Transform
    transformed_training= data_model.transform(df_training_weight)
    transformed_test= data_model.transform(df_testing_clean)
    transformed_val= data_model.transform(df_val_clean)


