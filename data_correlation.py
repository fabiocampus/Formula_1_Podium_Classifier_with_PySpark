from pyspark.sql import SparkSession
from scaler_encoder import RobustColumnScaler, OHencoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from itertools import combinations


spark = SparkSession.builder.appName("PySpark_correlation").getOrCreate()

if __name__ == "__main__":

    # we import the dataset
    df = spark.read.csv("f1_cleaned.csv", header=True, inferSchema=True)

    # first, we create an instance of our class RobustColumnScaler and then we applied its method to scale
    scaler = RobustColumnScaler(df)
    scaled_df = scaler.scale()
    # we create an instance of our class OHencoder and then we applied its method to encode
    encoder = OHencoder(scaled_df)
    transformed_df = encoder.encode()

    # now we want to compute correlation between variables, in order to avoid multicollinearity
    feature_columns = transformed_df.columns
    feature_columns.remove("top_3")
    # we need to create a vector containing all the features except the target one
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_df = assembler.transform(transformed_df).select("features")
    # we compute correlation matrix
    correlation_matrix = Correlation.corr(assembled_df, "features").head()
    corr_matrix = correlation_matrix[0].toArray()
    # we want to visualize values coming from each combination of features
    correlation_list = []
    num_features = len(feature_columns)
    for i, j in combinations(range(num_features), 2):
        feature_corr = (feature_columns[i], feature_columns[j], corr_matrix[i][j])
        correlation_list.append(feature_corr)
    # we sort and show the first 10 absolute values in descending order
    correlation_list_desc = sorted(correlation_list, key=lambda x: abs(x[2]), reverse=True)
    print(correlation_list_desc[:10])
    # given the results, we can say there aren't any pairs with a value near to 1

spark.stop()