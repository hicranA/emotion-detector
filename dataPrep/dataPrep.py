##### Table of content
## 1-   reading the files
## 2-   exploring the data
## 3-   converting class labels to numeric labels 
## 4-   data cleaning steps 
## 4.1  remove htlm tags, none letter words and word less than one or two character 
## 4.2  lemitizing  
## 5-   Unbalanced data - adding weight 
## 6-   Pipeline 1, model1 (tokenizer, stopwords remover, countvectorizer, idf) 
## 7-   Pipeline 2, model2 (chisquare, tokenizer,remover,countVectorizer)  
## 8-   Pipeline 3, model3 (ctokenizer3,countVectorizer3, idf3
## 9-   Pipeline 4, model4 (using un cleaned data, tokenizer, countvectorizer, idf) 
##10-   save the PySpark DataFrame to a Parquet file
############## libraries ####################
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
from pyspark.ml.feature import ChiSqSelector
import spacy
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import NGram

############## libraries ####################

###### reading the data ####################
def readFile(file_path):
    df_row = spark.read.csv(file_path)
    # print("original")
    # print(df_row.show(4,False))
    df= df_row.withColumn("text", split(df_row['_c0'], ';').getItem(0))\
            .withColumn('emotion', split(df_row['_c0'], ';').getItem(1))
    df1= df.select("text", "emotion")
    return df1
###### reading the data ####################

##############clean punctuations and none letters item  ####################
def clean(df):
    # Remove html tags from text
    df = df.withColumn("text_c", F.regexp_replace(F.col("text"), r'<[^>]+>', ""))
    # Remove non-letters and remove punctuation 
    df = df.withColumn("text_c", F.regexp_replace("text_c", r"[^a-zA-Z ]", ""))
    # Remove words 1, 2 char
    df = df.withColumn("text_c", F.regexp_replace("text_c", r"\b\w{1,2}\b", ""))
    return df
##############clean punctuations and none letters item  ####################

############### Defining a function to apply the lemmatizer to a text############
def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)
############### Defining a function to apply the lemmatizer to a text############


if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # file paths
    file_path_training = "../emotions-detector/data/train.txt"
    file_path_testing = "../emotions-detector/data/test.txt"
    file_path_val= "../emotions-detector/data/val.txt"
  
    ################### 1-  reading documents to df ################
    df_training = readFile(file_path= file_path_training)
    df_testing = readFile(file_path= file_path_testing)
    df_val = readFile(file_path= file_path_val)
    # print("sample")
    # print(df_val.show(4,False))
    ################### 1-  reading documents to df ################

    ################### 2-  exploring the data ################
    #information about the data
    print("size of the data")
    print("training size ",df_training.toPandas().shape , "data points") 
    print("testing size ",df_testing.toPandas().shape, "data points")
    print("validation size ",df_val.toPandas().shape, "data points") 
    print("class count for each documents in training")
    print("training class count", df_training.groupBy("emotion").count().show())
    training_size = df_training.count()
    ################### 2-  exploring the data ################
    
    ################ 3- converting class labels   ################
    # initiating string index object(maps a string column of labels to and ML column of label)
    string_indexer = StringIndexer(inputCol="emotion",outputCol="label")

    # fit the StringIndexer to the DataFrame and transform the DataFrame to add the 'class_numeric' column
    df_training_class = string_indexer.fit(df_training).transform(df_training).withColumn('label', col('label').cast("integer"))
    df_testing_class = string_indexer.fit(df_testing).transform(df_testing).withColumn('label', col('label').cast("integer"))
    df_val_class = string_indexer.fit(df_val).transform(df_val).withColumn('label', col('label').cast("integer"))
    # print(df_training_class.show(2))
    ################ 3- converting class labels   ################

    ################ 4- data cleaning steps    ################
    ##4.1  remove htlm tags, none letter words and word less than one or two character 
    df_training_clean = clean(df_training_class )
    df_testing_clean= clean(df_testing_class )
    df_val_clean =clean(df_val_class)

    ##4.2 2 lemitazing
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # Define a UDF to apply the lemmatizer to a column
    lemmatize_udf = udf(lemmatize_text, StringType())
    #applying lemitizing to the text column
    df_training_clean = df_training_clean.withColumn("text_c", lemmatize_udf(df_training_clean["text_c"]))
    df_testing_clean = df_testing_clean.withColumn("text_c", lemmatize_udf(df_testing_clean["text_c"]))
    df_val_clean = df_val_clean.withColumn("text_c", lemmatize_udf(df_val_clean["text_c"]))
    print("lemitizing is over")
    ################ 4- data cleaning steps    ################

    ######### 5 Unbalanced data - adding weight   ################
    # since we have unbalanced data we are adding weight
    w_joy = df_training_clean.filter('label == 0').count()/ training_size
    w_sadness = df_training_clean.filter('label == 1').count()/ training_size
    w_anger = df_training_clean.filter('label == 2').count()/ training_size
    w_fear = df_training_clean.filter('label == 3').count()/ training_size
    w_love = df_training_clean.filter('label == 4').count()/ training_size
    w_surprize = df_training_clean.filter('label == 5').count()/ training_size

    print(w_joy, w_sadness, w_anger, w_fear, w_love, w_surprize)

    df_training_weight = df_training_clean.withColumn("weight", when(F.col("label")==0,w_joy).
                                                  when(F.col("label")==1,w_sadness).
                                                  when(F.col("label")==2,w_anger).
                                                  when(F.col("label")==3,w_fear).
                                                  when(F.col("label")==4,w_love).
                                                  otherwise(w_surprize))

    df_training_weight.show(10)
    ######### 5- Unbalanced data - adding weight   ################

    ######### 6- Pipeline 1, model1 (tokenizer, stopwords remover, countvectorizer, idf)################
    # data cleaning pipelines 
    # Tokenize the review text
    tokenizer = Tokenizer(inputCol="text_c", outputCol="words",)
    # Remove stop words
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
    # Create a count vectoriser
    countVectorizer = CountVectorizer(inputCol=remover.getOutputCol(), outputCol="rawFeatures", vocabSize=1000)
    # Calculate the TF-IDF
    idf = IDF(inputCol=countVectorizer.getOutputCol(), outputCol="featuresIDF")
    
    # Crate a preprocessing pipeline with 4 stages
    pipeline_p = Pipeline(stages=[tokenizer,remover, countVectorizer, idf])

    #data prep model
    data_model  = pipeline_p.fit(df_training_weight)
    print(dir(data_model))

    # Transformation 1 
    transformed_training= data_model.transform(df_training_weight)
    transformed_test= data_model.transform(df_testing_clean)
    transformed_val= data_model.transform(df_val_clean)
    ######### 6- Pipeline 1, model1 (tokenizer, stopwords remover, countvectorizer, idf)################

    ######### 7-Pipeline 2, model2 (chisquare, tokenizer,remover,countVectorizer)   ################
    print("stage 2")
    #ngram = NGram(n=2, inputCol=remover.getOutputCol(),  outputCol="ngrams")
    #hashingTF = HashingTF(inputCol="ngrams", outputCol="rawFeatures", numFeatures=1000)
    # countVectorizer = CountVectorizer(inputCol=ngram.getOutputCol(), outputCol="rawFeatures", vocabSize=1000)
    # idf = IDF(inputCol=countVectorizer.getOutputCol(), outputCol="featuresIDF")
    selector = ChiSqSelector(numTopFeatures=300, featuresCol=idf.getOutputCol(), outputCol="features", labelCol="label")
    # Crate a preprocessing pipeline wiht 5 stages
    pipeline_2 = Pipeline(stages=[tokenizer,remover,countVectorizer, idf, selector])
    # Learn the data preprocessing model
    data_model_2 = pipeline_2.fit(df_training_weight)

    # Transform the data 2
    transformed_training_p2= data_model_2.transform(df_training_weight)
    transformed_test_p2 = data_model_2.transform(df_testing_clean)
    transformed_val_p2 = data_model_2.transform(df_val_clean)
    ######### 7-Pipeline 2, model2 (chisquare, tokenizer,remover,countVectorizer)##############

    ######### 8-   Pipeline 3, model3 (ctokenizer3,countVectorizer3, idf3) ################
    #data model 3 row text no cleaning only transforming to numbers 
    #Tokenize the text
    tokenizer3 = Tokenizer(inputCol="text", outputCol="words")
    # Create a count vectoriser
    countVectorizer3 = CountVectorizer(inputCol=tokenizer3.getOutputCol(), outputCol="rawFeatures", vocabSize=1000)
    idf3 = IDF(inputCol=countVectorizer3.getOutputCol(), outputCol="featuresIDF")
    # Create pipeline
    pipeline_3= Pipeline(stages=[tokenizer3,countVectorizer3, idf3])
    #create a data model 
    data_model_3  = pipeline_3.fit(df_training_weight)
    # Transform 3 
    transformed_training_p3= data_model_3.transform(df_training_weight)
    transformed_test_p3 = data_model_3.transform(df_testing_clean)
    transformed_val_p3 = data_model_3.transform(df_val_clean)
    ######### 8-   Pipeline 3, model3 (ctokenizer3,countVectorizer3, idf3) ################

    ########### 9- Pipeline 4, model4 (using un cleaned data, tokenizer, countvectorizer, idf)########### 
    #data model 4 row text no cleaning only transforming to numbers 
    # Tokenize the review text
    tokenizer_4 = Tokenizer(inputCol="text_c", outputCol="words",)
    # Remove stop words
    remover_4 = StopWordsRemover(inputCol=tokenizer_4.getOutputCol(), outputCol="filtered")
    ngram = NGram(n=2, inputCol=remover_4.getOutputCol(),  outputCol="ngrams")
    # Create a count vectoriser
    hashingTF = HashingTF(inputCol="ngrams", outputCol="rawFeatures", numFeatures=1000)
    idf2 = IDF(inputCol=hashingTF.getOutputCol(), outputCol="featuresIDF")
    # Create pipeline
    pipeline_4= Pipeline(stages=[tokenizer_4, remover_4,ngram,hashingTF, idf2])
    #create a data model 
    data_model_4  = pipeline_4.fit(df_training_weight)
     # Transform 4
    transformed_training_p4= data_model_4.transform(df_training_weight)
    transformed_test_p4 = data_model_4.transform(df_testing_clean)
    transformed_val_p4 = data_model_4.transform(df_val_clean)
    ########### 9- Pipeline 4, model4 (using un cleaned data, tokenizer, countvectorizer, idf)########### 

    ########### 10- save the PySpark DataFrame to a Parquet file ###########
    #data model 1
    transformed_training.write.parquet("data/transformed_training.parquet")
    transformed_test.write.parquet("data/transformed_test.parquet")
    transformed_val.write.parquet("data/transformed_val.parquet")
    #data model 2
    transformed_training_p2.write.parquet("data/transformed_training_p2.parquet")
    transformed_test_p2.write.parquet("data/transformed_test_p2.parquet")
    transformed_val_p2.write.parquet("data/transformed_val_p2.parquet")
    #data model 3
    transformed_training_p3.write.parquet("data/transformed_training_p3.parquet")
    transformed_test_p3.write.parquet("data/transformed_test_p3.parquet")
    transformed_val_p3.write.parquet("data/transformed_val_p3.parquet")
    #data model 4
    transformed_training_p4.write.parquet("data/transformed_training_p4.parquet")
    transformed_test_p4.write.parquet("data/transformed_test_p4.parquet")
    transformed_val_p4.write.parquet("data/transformed_val_p4.parquet")
    ########### 10- save the PySpark DataFrame to a Parquet file ###########


