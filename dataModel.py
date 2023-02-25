from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml import Pipeline
#evaluation
from pyspark.mllib.evaluation import MultilabelMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

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


# create a SparkSession object

if __name__ == "__main__":
    spark = SparkSession.builder.appName('Read DataFrame').getOrCreate()

    # read the PySpark DataFrame from the Parquet file
    df_training_m1 = spark.read.parquet("data/transformed_training.parquet")
    df_testing_m1 = spark.read.parquet("data/transformed_test.parquet")
    df_val_m1 = spark.read.parquet("data/transformed_val.parquet")
    print(df_testing_m1.show(5))

    import time
    classifier = LinearSVC(maxIter=10, regParam=0.1, featuresCol = "featuresIDF", weightCol="weight", labelCol="label")
    # Define OneVsRest strategy
    ovr = OneVsRest(classifier=classifier, labelCol="label", featuresCol="featuresIDF", weightCol="weight")
    pipeline = Pipeline(stages=[ovr])
    start = time.time()
    print(f"Training started.")
    model = pipeline.fit(df_training_m1)
    print(f"Model created in {time.time()-start:.2f}s.")
    m_metrics_l(model,df_val_m1)
    print(f"Total time {time.time()-start:.2f}s.")

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="f1")

    # Define the hyperparameter grid
    param_grid = ParamGridBuilder() \
     .addGrid(classifier.maxIter, [10, 50, 100]) \
        .build()
    
    # Define the cross-validator
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

    # Fit the cross-validator to the training set
    cv_model = cv.fit(df_training_m1)

    # Evaluate the best model on the validation set
    best_model = cv_model.bestModel
    # Evaluate the best model on the validation set
    best_model = cv_model.bestModel
    predictions = best_model.transform(df_val_m1)
    f1 = evaluator.evaluate(predictions)
    print("f1 on validation set for best model:", f1)



