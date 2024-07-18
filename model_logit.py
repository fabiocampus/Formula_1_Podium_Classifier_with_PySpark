from pyspark.sql import SparkSession
from scaler_encoder import RobustColumnScaler, OHencoder
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


spark = SparkSession.builder.appName("PySpark_logit").getOrCreate()

if __name__ == "__main__":

    # we import the dataset
    df = spark.read.csv("f1_cleaned.csv", header=True, inferSchema=True)

    # first, we create an instance of our class RobustColumnScaler and then we applied its method to scale
    scaler = RobustColumnScaler(df)
    scaled_df = scaler.scale()
    # we create an instance of our class OHencoder and then we applied its method to encode
    encoder = OHencoder(scaled_df)
    transformed_df = encoder.encode()

    # now, we develop the logistic regression
    # we start by assembling the features
    feature_columns = transformed_df.columns
    feature_columns.remove("top_3")
    # we convert the response variable to numerical values
    indexer = StringIndexer(inputCol="top_3", outputCol="label")
    indexed_df = indexer.fit(transformed_df).transform(transformed_df)
    target_column = "label"
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df_assembled = assembler.transform(indexed_df)
    # we are now able to split dataset into training set and test set
    train_data, test_data = df_assembled.randomSplit([0.7, 0.3], seed=1)

    # we define the logistic model
    logit = LogisticRegression(featuresCol="features", labelCol=target_column)
    # we define the evaluator based on accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction",
                                                  metricName="accuracy")
    # we define the grid
    param_grid = (ParamGridBuilder()
                  .addGrid(logit.regParam, [0.01, 0.1, 1.0])
                  .addGrid(logit.elasticNetParam, [0.0, 0.5, 1.0])
                  .build())
    # we create an instance of CrossValidator class
    crossval = CrossValidator(estimator=logit,
                              estimatorParamMaps=param_grid,
                              evaluator=evaluator,
                              numFolds=5)
    # we fit the model on the training set (so that we can test it on totally unknown data)
    logit_model = crossval.fit(train_data)

    # we get the parameters of the best model
    best_params = logit_model.bestModel.extractParamMap()
    for param, value in best_params.items():
        print(param.name, ":", value)
    # we evaluate the accuracy based on predictions on test set
    predictions = logit_model.transform(test_data) # predictions of the model on new test data
    test_accuracy = evaluator.evaluate(predictions)
    print("Accuracy:", test_accuracy)
    # we evaluate the precision
    evaluator_precision = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction",
                                                            metricName="weightedPrecision")
    precision = evaluator_precision.evaluate(predictions)
    print("Precision:", precision)
    # we evaluate the recall
    evaluator_recall = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction",
                                                         metricName="weightedRecall")
    recall = evaluator_recall.evaluate(predictions)
    print("Recall:", recall)
    # we evaluate the f1-score
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction",
                                                     metricName="f1")
    f1_score = evaluator_f1.evaluate(predictions)
    print("F1-score:", f1_score)
    # we evaluate the ROC-AUC
    binary_evaluator = BinaryClassificationEvaluator(labelCol=target_column)
    roc_auc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
    print("ROC-AUC:", roc_auc)

    # we get features' coefficients and their respective name in descendent order
    coefficients = logit_model.bestModel.coefficients.toArray()
    feature_coefficients = list(zip(feature_columns, coefficients))
    feature_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)
    for feature, coefficient in feature_coefficients:
        print(f"{feature}: {coefficient}")

spark.stop()