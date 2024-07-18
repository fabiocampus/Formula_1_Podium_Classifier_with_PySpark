import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("PySpark_visualization").getOrCreate()

# we import the dataset
df = spark.read.csv("f1_cleaned.csv", header=True, inferSchema=True)

# we want to show driver_nationalities' distribution
nationality_freq = df.groupBy("driver_nationality").count().orderBy("count", ascending=False).toPandas()
plt.bar(nationality_freq["driver_nationality"], nationality_freq["count"], color="red")
plt.title("Frequency of drivers' nationality")
plt.xticks(rotation=90, ha="right")
plt.subplots_adjust(bottom=0.3)
plt.show()

# we want to show constructor_nationalities' distribution
constructor_freq = df.groupBy("constructor_nationality").count().orderBy("count", ascending=False).toPandas()
plt.bar(constructor_freq["constructor_nationality"], constructor_freq["count"], color="blue")
plt.title("Frequency of constructors' nationality")
plt.xticks(rotation=90, ha="right")
plt.subplots_adjust(bottom=0.3)
plt.show()

# we want to show circuit_countries' distribution
country_freq = df.groupBy("country").count().orderBy("count", ascending=False).toPandas()
plt.bar(country_freq["country"], country_freq["count"], color="red")
plt.title("Frequency of host countries")
plt.xticks(rotation=90, ha="right")
plt.subplots_adjust(bottom=0.3)
plt.show()

# we want to show the frequency of top 3 positions per constructor_nationality
constructor_first_freq = df.filter(df["top_3"] == "yes").groupBy("constructor_nationality").count()\
    .orderBy("count", ascending=False).toPandas()
plt.bar(constructor_first_freq["constructor_nationality"], constructor_first_freq["count"], color="blue")
plt.title("Frequency of constructors' nationality finished on podium")
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.show()

# we want to show the frequency of top 3 positions per driver_nationality
nationality_first_freq = df.filter(df["top_3"] == "yes")\
    .groupBy("driver_nationality").count().orderBy("count", ascending=False).toPandas()
plt.bar(nationality_first_freq["driver_nationality"], nationality_first_freq["count"], color="red")
plt.title("Frequency of drivers' nationality finished on podium")
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.show()

spark.stop()