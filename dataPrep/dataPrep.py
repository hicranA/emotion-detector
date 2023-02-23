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

def readFile(file_path):
    df_row = spark.read.csv(file_path)
    # print("original")
    # print(df_row.show(4,False))
    df= df_row.withColumn("text", split(df_row['_c0'], ';').getItem(0))\
            .withColumn('emotion', split(df_row['_c0'], ';').getItem(1))
    df1= df.select("text", "emotion")
    return df1

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

    #converting class labels
    
    string_indexer = StringIndexer(inputCol="emotion",outputCol="class")

    # fit the StringIndexer to the DataFrame and transform the DataFrame to add the 'class_numeric' column
    df_training_class = string_indexer.fit(df_training).transform(df_training).withColumn('class', col('class').cast("integer"))
    df_testing_class = string_indexer.fit(df_testing).transform(df_testing).withColumn('class', col('class').cast("integer"))
    df_val_class = string_indexer.fit(df_val).transform(df_val).withColumn('class', col('class').cast("integer"))
    
    print(df_training_class.show(2))