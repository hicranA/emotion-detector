from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml import Pipeline
#evaluation
from pyspark.mllib.evaluation import MultilabelMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes

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
    results_df = spark.createDataFrame([("countVectorizer+iDF",
                                     "svm", 
                                    round(training_time_svm,2),
                                    round(precision_svm,2), 
                                     round(recall_svm,2), 
                                     round(f1Score_svm,2))], 
                                     ["dataModel", "modelName", "trainingTime", "precision","recall", "f1Score"])

    results_df.show()
    precision_lr, recall_lr , f1Score_lr , metrics_lr, training_time_lr = logRegmaker(df= df_training_m1, df_val=df_val_m1, dm=0)
    df= results_df.toPandas()
    df.loc[len(df.index)]= ["countVectorizer+iDF",
                                     "lr", training_time_lr, precision_lr, recall_lr, f1Score_lr]
    precision_nb, recall_nb, f1Score_nb, metrics_nb, training_time_nb = nBMaker(df=df_training_m1, df_val=df_val_m1, dm=0)
    df.loc[len(df.index)]= ["countVectorizer+iDF",
                                     "nb", training_time_nb, precision_nb, recall_nb, f1Score_nb]
    # read the PySpark DataFrame from the Parquet file
    df_training_m2 = spark.read.parquet("data/transformed_training_p2.parquet")
    df_testing_m2 = spark.read.parquet("data/transformed_test_p2.parquet")
    df_val_m2 = spark.read.parquet("data/transformed_val_p2.parquet")
    df_training_m2.cache()
    df_testing_m2.cache()
    df_val_m2.cache()
    # print(df_testing_m2.schema)
    precision_svm_2, recall_svm_2, f1Score_svm_2, metrics_2, training_time_svm_2 = liniearSVMMaker(df= df_training_m2, df_val=df_val_m2, dm=1)
    df.loc[len(df.index)]= ["countVectorizer+iDF+chisquare",
                                     "svm", training_time_svm_2, precision_svm_2, recall_svm_2, f1Score_svm_2]
    precision_lr_2, recall_lr_2 , f1Score_lr_2, metrics_lr_2, training_time_lr_2 = logRegmaker(df= df_training_m2, df_val=df_val_m2, dm=1)
    df.loc[len(df.index)]= ["countVectorizer+iDF+chisquare",
                                     "lr", training_time_lr_2, precision_lr_2, recall_lr_2, f1Score_lr_2]
    precision_nb_2, recall_nb_2, f1Score_nb_2, metrics_nb_2, training_time_nb_2 = nBMaker(df=df_training_m2, df_val=df_val_m2, dm=1)
    df.loc[len(df.index)]= ["countVectorizer+iDF+chisquare",
                                     "nb", training_time_nb_2, precision_nb_2, recall_nb_2, f1Score_nb_2]
    
     # read the PySpark DataFrame from the Parquet file
    df_training_m3 = spark.read.parquet("data/transformed_training_p3.parquet")
    df_testing_m3 = spark.read.parquet("data/transformed_test_p3.parquet")
    df_val_m3 = spark.read.parquet("data/transformed_val_p3.parquet")
    df_training_m3.cache()
    df_testing_m3.cache()
    df_val_m3.cache()

    # print(df_testing_m2.schema)
    precision_svm_3, recall_svm_3, f1Score_svm_3, metrics_3, training_time_svm_3 = liniearSVMMaker(df= df_training_m3, df_val=df_val_m3, dm=0)
    df.loc[len(df.index)]= ["unCleanData+countVectorizer+iDF",
                                     "svm", training_time_svm_3, precision_svm_3, recall_svm_3, f1Score_svm_3]
    precision_lr_3, recall_lr_3 , f1Score_lr_3, metrics_lr_3, training_time_lr_3 = logRegmaker(df= df_training_m3, df_val=df_val_m3, dm=0)
    df.loc[len(df.index)]= ["unCleanData+countVectorizer+iDF",
                                     "lr", training_time_lr_3, precision_lr_3, recall_lr_3, f1Score_lr_3]
    precision_nb_3, recall_nb_3, f1Score_nb_3, metrics_nb_3, training_time_nb_3 = nBMaker(df=df_training_m3, df_val=df_val_m3, dm=0)
    df.loc[len(df.index)]= ["unCleanData+countVectorizer+iDF",
                                     "nb", training_time_nb_3, precision_nb_3, recall_nb_3, f1Score_nb_3]
    print(df)
    # print(df_training_m3.show(5))

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
    df.loc[len(df.index)]= ["ngram+hashingtf+iDF",
                                     "svm", training_time_svm_4, precision_svm_4, recall_svm_4, f1Score_svm_4]
    precision_lr_4, recall_lr_4 , f1Score_lr_4, metrics_lr_4, training_time_lr_4= logRegmaker(df= df_training_m3, df_val=df_val_m4, dm=0)
    df.loc[len(df.index)]= ["ngram+hashingtf+iDF",
                                     "lr", training_time_lr_4, precision_lr_4, recall_lr_4, f1Score_lr_4]
    precision_nb_4, recall_nb_4, f1Score_nb_4, metrics_nb_4, training_time_nb_4 = nBMaker(df=df_training_m3, df_val=df_val_m4, dm=0)
    df.loc[len(df.index)]= ["ngram+hashingtf+iDF",
                                     "nb", training_time_nb_4, precision_nb_4, recall_nb_4, f1Score_nb_4]
    print(type(df))


   


