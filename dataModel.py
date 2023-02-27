from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import lit,array
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import row_number
#evaluation
from pyspark.mllib.evaluation import MultilabelMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.window import Window
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import Imputer

import time
def m_metrics_l(ml_model,test_data):
    predictions = ml_model.transform(test_data).cache()
    predictionAndLabels = predictions.select("label","prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
    # Print some predictions vs labels
    # print(predictionAndLabels.take(10))
    metrics = MulticlassMetrics(predictionAndLabels)
    # Overall statistics
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    print(f"Precision = {precision:.4f} Recall = {recall:.4f} F1 Score = {f1Score:.4f}")
    print("Confusion matrix \n", metrics.confusionMatrix().toArray().astype(int))
    return precision, recall, f1Score, metrics

def liniearSVMMaker(df, df_val, dm):
    if dm ==0:
        classifier = LinearSVC(maxIter=10, regParam=0.1, featuresCol = "featuresIDF", weightCol="weight", labelCol="label")
        ovr = OneVsRest(classifier=classifier, labelCol="label", featuresCol="featuresIDF", weightCol="weight")
    elif dm==1:
        classifier = LinearSVC(maxIter=10, regParam=0.1, featuresCol = "features", weightCol="weight", labelCol="label")
        ovr = OneVsRest(classifier=classifier, labelCol="label", featuresCol="features", weightCol="weight")

    # Define OneVsRest strategy
    start = time.time()
    pipeline = Pipeline(stages=[ovr])
    model = pipeline.fit(df)
    training_time = time.time()-start
    precision_svm, recall_svm , f1Score_svm,  metrics = m_metrics_l(model,df_val)
    return precision_svm, recall_svm, f1Score_svm, metrics, training_time
def logRegmaker(df, df_val, dm):
    if dm ==0:
        classifier = LogisticRegression(maxIter=10, regParam=0.1, featuresCol = "featuresIDF", weightCol="weight")
    elif dm==1:
        classifier = LogisticRegression(maxIter=10, regParam=0.1, featuresCol = "features", weightCol="weight")
    start = time.time()
    pipeline = Pipeline(stages=[classifier])
    print(f"Training started.")
    model = pipeline.fit(df)
    training_time = time.time()-start
    precision, recall , f1Score,  metrics = m_metrics_l(model,df_val)
    return precision, recall, f1Score, metrics, training_time
def nBMaker(df, df_val, dm):
    if dm ==0:
        nb = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol = "featuresIDF", weightCol="weight")
    elif dm==1: 
        nb = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol = "features", weightCol="weight") 
    # nb_model = nb.fit(df_training_m1)
    start = time.time()
    pipeline = Pipeline(stages=[nb])
    print(f"Training started.")
    model = pipeline.fit(df)
    training_time = time.time()-start
    precision, recall , f1Score,  metrics = m_metrics_l(model,df_val)
    return precision, recall, f1Score, metrics, training_time
def mPMaker(df, df_val):
    # drop rows with missing values
    df_1 = df.na.drop()
    df_val_1 = df_val.na.drop()
    layers = [1000, 30, 2]
    mp = MultilayerPerceptronClassifier(maxIter=10,labelCol="label" ,layers=layers,featuresCol = "featuresIDF", blockSize=128, seed=1234)
    start = time.time()
    pipeline = Pipeline(stages=[mp])
    print(f"Training started.")
    model = pipeline.fit(df_1)
    training_time = time.time()-start
    precision, recall , f1Score,  metrics = m_metrics_l(model,df_val_1)
    return precision, recall, f1Score, metrics, training_time


    
# create a SparkSession object

if __name__ == "__main__":
    spark = SparkSession.builder.appName('Read DataFrame').getOrCreate()

    # read the PySpark DataFrame from the Parquet file
    df_training_m1 = spark.read.parquet("data/transformed_training.parquet")
    df_testing_m1 = spark.read.parquet("data/transformed_test.parquet")
    df_val_m1 = spark.read.parquet("data/transformed_val.parquet")
    df_training_m1.cache()
    df_testing_m1.cache()
    df_val_m1.cache()
    print(df_testing_m1.show(5))
  
    precision_svm, recall_svm, f1Score_svm, metrics, training_time_svm = liniearSVMMaker(df= df_training_m1, df_val=df_val_m1, dm=0)
    results_df_1= spark.createDataFrame([("countVectorizer+iDF",
                                     "svm", 
                                    round(training_time_svm,3),
                                    round(precision_svm,3), 
                                     round(recall_svm,3), 
                                     round(f1Score_svm,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])

    #test code , it is giving an error 
    # precision_mp, recall_mp, f1Score_mp, metrics_mp, training_time_mp = mPMaker(df= df_training_m1, df_val=df_val_m1)


    precision_lr, recall_lr , f1Score_lr , metrics_lr, training_time_lr = logRegmaker(df= df_training_m1, df_val=df_val_m1, dm=0)

    results_df_2 = spark.createDataFrame([("countVectorizer+iDF",
                                     "lr", 
                                    round(training_time_lr,3),
                                    round(precision_lr,3), 
                                     round(recall_lr,3), 
                                     round(f1Score_lr,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])
  
    precision_nb, recall_nb, f1Score_nb, metrics_nb, training_time_nb = nBMaker(df=df_training_m1, df_val=df_val_m1, dm=0)
    results_df_3 = spark.createDataFrame([("countVectorizer+iDF",
                                     "nb", 
                                    round(training_time_nb,3),
                                    round(precision_nb,3), 
                                     round(recall_nb,3), 
                                     round(f1Score_nb,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])
    
    # read the PySpark DataFrame from the Parquet file
    df_training_m2 = spark.read.parquet("data/transformed_training_p2.parquet")
    df_testing_m2 = spark.read.parquet("data/transformed_test_p2.parquet")
    df_val_m2 = spark.read.parquet("data/transformed_val_p2.parquet")
    df_training_m2.cache()
    df_testing_m2.cache()
    df_val_m2.cache()
    # print(df_testing_m2.schema)
    precision_svm_2, recall_svm_2, f1Score_svm_2, metrics_2, training_time_svm_2 = liniearSVMMaker(df= df_training_m2, df_val=df_val_m2, dm=1)
    results_df_4 = spark.createDataFrame([("countVectorizer+iDF+chisquare",
                                     "svm", 
                                    round(training_time_svm_2,3),
                                    round(precision_svm_2,3), 
                                     round(recall_svm_2,3), 
                                     round(f1Score_svm_2,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])

  
    
    precision_lr_2, recall_lr_2 , f1Score_lr_2, metrics_lr_2, training_time_lr_2 = logRegmaker(df= df_training_m2, df_val=df_val_m2, dm=1)
    results_df_5 = spark.createDataFrame([("countVectorizer+iDF+chisquare",
                                     "lr", 
                                    round(training_time_lr_2,3),
                                    round(precision_lr_2,3), 
                                     round(recall_lr_2,3), 
                                     round(f1Score_lr_2,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])

    
    
    precision_nb_2, recall_nb_2, f1Score_nb_2, metrics_nb_2, training_time_nb_2 = nBMaker(df=df_training_m2, df_val=df_val_m2, dm=1)
    results_df_6 = spark.createDataFrame([("countVectorizer+iDF+chisquare",
                                     "nb", 
                                    round(training_time_nb_2,3),
                                    round(precision_nb_2,3), 
                                     round(recall_nb_2,3), 
                                     round(f1Score_nb_2,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])

    
    
     # read the PySpark DataFrame from the Parquet file
    df_training_m3 = spark.read.parquet("data/transformed_training_p3.parquet")
    df_testing_m3 = spark.read.parquet("data/transformed_test_p3.parquet")
    df_val_m3 = spark.read.parquet("data/transformed_val_p3.parquet")
    df_training_m3.cache()
    df_testing_m3.cache()
    df_val_m3.cache()
    print(df_val_m3.show())

    # print(df_testing_m2.schema)
    precision_svm_3, recall_svm_3, f1Score_svm_3, metrics_3, training_time_svm_3 = liniearSVMMaker(df= df_training_m3, df_val=df_val_m3, dm=0)
    results_df_7 = spark.createDataFrame([("unCleanData+countVectorizer+iDF",
                                     "svm", 
                                    round(training_time_svm_3,3),
                                    round(precision_svm_3,3), 
                                     round(recall_svm_3,3), 
                                     round(f1Score_svm_3,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])

   
    
    precision_lr_3, recall_lr_3 , f1Score_lr_3, metrics_lr_3, training_time_lr_3 = logRegmaker(df= df_training_m3, df_val=df_val_m3, dm=0)
    results_df_8 = spark.createDataFrame([("unCleanData+countVectorizer+iDF",
                                     "lr", 
                                    round(training_time_lr_3,3),
                                    round(precision_lr_3,3), 
                                     round(recall_lr_3,3), 
                                     round(f1Score_lr_3,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])

    
    precision_nb_3, recall_nb_3, f1Score_nb_3, metrics_nb_3, training_time_nb_3 = nBMaker(df=df_training_m3, df_val=df_val_m3, dm=0)
    results_df_9= spark.createDataFrame([("unCleanData+countVectorizer+iDF",
                                     "nb", 
                                    round(training_time_nb_3,3),
                                    round(precision_nb_3,3), 
                                     round(recall_nb_3,3), 
                                     round(f1Score_nb_3,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])

   
    # read the PySpark DataFrame from the Parquet file
    df_training_m4 = spark.read.parquet("data/transformed_training_p4.parquet")
    df_testing_m4 = spark.read.parquet("data/transformed_test_p4.parquet")
    df_val_m4= spark.read.parquet("data/transformed_val_p4.parquet")
    df_training_m4.cache()
    df_testing_m4.cache()
    df_val_m4.cache()
    
    print(df_val_m4.show(5))
     # print(df_testing_m2.schema)
    precision_svm_4, recall_svm_4, f1Score_svm_4, metrics_4, training_time_svm_4= liniearSVMMaker(df= df_training_m4, df_val=df_val_m4, dm=0)
    results_df_10= spark.createDataFrame([("ngram+hashingtf+iDF",
                                     "svm", 
                                    round(training_time_svm_4,3),
                                    round(precision_svm_4,3), 
                                     round(recall_svm_4,3), 
                                     round(f1Score_svm_4,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])
  

    precision_lr_4, recall_lr_4 , f1Score_lr_4, metrics_lr_4, training_time_lr_4= logRegmaker(df= df_training_m4, df_val=df_val_m4, dm=0)
    results_df_11= spark.createDataFrame([("ngram+hashingtf+iDF",
                                     "lr", 
                                    round(training_time_lr_4,3),
                                    round(precision_lr_4,3), 
                                     round(recall_lr_4,3), 
                                     round(f1Score_lr_4,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])

    # df.loc[len(df.index)]= ["ngram+hashingtf+iDF",
    #                                  "lr", training_time_lr_4, precision_lr_4, recall_lr_4, f1Score_lr_4]

    precision_nb_4, recall_nb_4, f1Score_nb_4, metrics_nb_4, training_time_nb_4 = nBMaker(df=df_training_m4, df_val=df_val_m4, dm=0)
    results_df_12= spark.createDataFrame([("ngram+hashingtf+iDF",
                                     "nb", 
                                    round(training_time_nb_4,3),
                                    round(precision_nb_4,3), 
                                     round(recall_nb_4,3), 
                                     round(f1Score_nb_4,3))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])
    
    # Add a new column to each DataFrame with a constant value
    dataFrames = [results_df_1,results_df_2,results_df_3,results_df_4,results_df_5,
                    results_df_6,results_df_7,results_df_8,results_df_9,results_df_10,
                    results_df_11, results_df_12]

    # Union the three DataFrames together
    df_result_union = results_df_1.union(results_df_2).union(results_df_3).union(results_df_4).union(results_df_5)\
                            .union(results_df_6).union(results_df_7).union(results_df_8).union(results_df_9)\
                              .union(results_df_10).union(results_df_11).union(results_df_12)
 
    df_result_union.show()


