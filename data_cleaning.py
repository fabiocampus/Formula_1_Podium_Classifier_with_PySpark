import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, split


spark = SparkSession.builder.appName("PySpark_cleaning").getOrCreate()

# we import the dataset
df = spark.read.csv("f1_updated.csv", header=True, inferSchema=True)

# we remove the features we decideed to ignore for different reasons - read the report
df = df.drop("raceId", "round", "circuitId", "race_url", "fp1_date", "fp2_date", "fp3_date", "quali_date", "sprint_date",
             "circuit_name", "lat", "lng", "circuit_url", "resultId", "driverId", "constructorId", "statusId", "driver_url",
             "constructorRef", "constructor_url", "qualifyId", "quali_position", "code", "dob", "driverRef", "forename",
             "surname", "race_name", "circuitRef", "location", "constructor_name", "points", "status", "rank", "fastestLapSpeed",
             "fastestLapTime", "fastestLap")

# by looking at the data, we found out some features present "\N" values. Hence, we want to know how many
# results = {}
# def count_N(column_name):
#     return df.filter(col(column_name) == "\\N").count()
#
# for column in df.columns:
#     results[column] = count_N(column)
#
# for column, count in results.items():
#     print(f"Colonna '{column}': {count} valori '\\N'")

# thanks to the code above, we have information about how many "\N" for each feature.
# We remove all the features that present more than 30% of their data as "\N"
df = df.drop("fp1_time", "fp2_time", "fp3_time", "quali_time", "sprint_time", "driver_number", "q2", "q3")

# for the other features that present "\N", we replace such value with the median value.
# First we need to convert them in the correct type
# "alt" needs to be converted in int
df = df.withColumn("alt", df["alt"].cast("int"))

# we create a function that convert features about time (with the following format: "min:sec.millsec") in milliseconds
def convert_time(df, col_name):
    # we add a new column
    new_col_name = col_name + "_millsec"
    # we populate the new column by using the former one (then we drop it)
    df = df.withColumn(new_col_name,
                       (split(col(col_name), ":")[0].cast("int") * 60 * 1000) +
                       (split(split(col(col_name), ":")[1], "\.")[0].cast("int") * 1000) +
                       split(split(col(col_name), ":")[1], "\.")[1].cast("int"))
    df = df.drop(col_name)
    return df
# we use the function on q1
df = convert_time(df, "q1")

# now we are able to find median values for our features in order to replace missing values with them
# we define a function that compute median
def calc_median(df, column_name):
    median_value = df.approxQuantile(column_name, [0.5], 0.1)[0]
    return median_value
# we create a dictionary where the key is the feature and the value its respective median value
median_dict = {column: calc_median(df, column) for column in ["alt", "q1_millsec"]}
# for each column of our dictionary, we replace missing values with median values
for column, median_value in median_dict.items():
    df = df.withColumn(column, when(col(column).isNull(), median_value).otherwise(col(column)))

# we decide to keep only the month when the race took place
df = df.withColumn("race_month", (split(col("date"), "-")[1].cast("int")))
df = df.drop("date")

# lastly, we decide to convert the response variable from quantitative to qualitative (binary)
df = df.withColumn("top_3",
                       when((df["race_position_order"] >= 1) & (df["race_position_order"] <= 3), "yes")
                       .otherwise("no"))
df = df.drop("race_position_order")

# now we create the cleaned csv file
# df_pand = df.toPandas()
# df_pand.to_csv("f1_cleaned.csv", index=False)

spark.stop()