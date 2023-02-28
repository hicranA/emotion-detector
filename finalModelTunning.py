from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
sc = SparkContext("local")
spark = SparkSession.builder.getOrCreate()

# read the PySpark DataFrame from the Parquet file
df_training_m3 = spark.read.parquet("data/transformed_training_p3.parquet")
df_testing_m3 = spark.read.parquet("data/transformed_test_p3.parquet")
df_val_m3 = spark.read.parquet("data/transformed_val_p3.parquet")
df_training_m3.cache()
df_testing_m3.cache()
df_val_m3.cache()
print(df_val_m3.show(3))

# import time
classifier = LinearSVC(featuresCol = "featuresIDF", weightCol="weight", labelCol="label")
# Define OneVsRest strategy
ovr = OneVsRest(classifier=classifier, labelCol="label", featuresCol="featuresIDF", weightCol="weight")
pipeline = Pipeline(stages=[ovr])
# model = pipeline.fit(df_training_m3)


evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="f1")

# Define the hyperparameter grid
param_grid = ParamGridBuilder() \
    .addGrid(classifier.maxIter, [10, 50, 100]) \
    .addGrid(classifier.regParam,  [0.1, 0.01, 0.001])\
    .addGrid(classifier.weightCol, ['weight']) \
    .build()
#0.01, 0.1, 1.0
# Define the cross-validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

# Fit the cross-validator to the training set
cv_model = cv.fit(df_training_m3)

# Evaluate the best model on the validation set
best_model = cv_model.bestModel
predictions = best_model.transform(df_val_m3)
f1 = evaluator.evaluate(predictions)
print("f1 on validation set for best model:", f1)

# Evaluate the best model on the validation set
best_model = cv_model.bestModel
predictions2 = best_model.transform(df_testing_m3)
f2 = evaluator.evaluate(predictions)
print("f1 on validation set for best model:", f2)

from pyspark.mllib.evaluation import MulticlassMetrics

def m_metrics_l(ml_model, test_data):
    predictions = ml_model.transform(test_data).cache()
    predictionAndLabels = predictions.select("label","prediction").rdd.map(lambda x: (float(x[1]), float(x[0]))).cache()
    
    # Calculate evaluation metrics
    metrics = MulticlassMetrics(predictionAndLabels)
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1Score = metrics.fMeasure(1.0)
    accuracy = metrics.accuracy
    
    # Print evaluation metrics
    print(f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1Score:.4f}, Accuracy = {accuracy:.4f}")
    
    # Print confusion matrix
    labels = predictionAndLabels.map(lambda x: x[1]).distinct().collect()
    confusion_matrix = metrics.confusionMatrix().toArray().astype(int)
    print("Confusion matrix:")
    print("Actual\Predicted\t", end="")
    for label in labels:
        print(f"{label}\t\t", end="")
    print("")
    for i in range(len(labels)):
        print(f"{labels[i]}\t\t", end="")
        for j in range(len(labels)):
            print(f"{confusion_matrix[i][j]:<8}\t\t", end="")
        print("")
    
    return precision, recall, f1Score, accuracy, metrics

m_metrics_l(ml_model=best_model, test_data=df_testing_m3)