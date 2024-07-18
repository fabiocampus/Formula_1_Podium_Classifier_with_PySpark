from pyspark.sql import SparkSession
from pyspark.ml.feature import RobustScaler, VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline


spark = SparkSession.builder.appName("PySpark_transformation").getOrCreate()

# we create a class that performs RobustScaler for our concerned columns
class RobustColumnScaler:
    def __init__(self, df):
        self.df = df
        self.assembled_df = None
        self.scaled_df = None
        self.columns_to_scale = []

    # first, we find our quantitative columns
    def find_quantitative_columns(self):
        self.columns_to_scale = [col_name for col_name, col_type in self.df.dtypes if
                                 col_type != "string" and col_name != "top_3"]

    # we assemble the features
    def assemble_features(self):
        self.assembled_df = self.df
        for input_col in self.columns_to_scale:
            assembler = VectorAssembler(inputCols=[input_col], outputCol=input_col + "_features")
            self.assembled_df = assembler.transform(self.assembled_df)

    # we scale each feature
    def scale_columns(self):
        self.scaled_df = self.assembled_df
        for input_col in self.columns_to_scale:
            scaler = RobustScaler(inputCol=input_col + "_features", outputCol=input_col + "_scaled")
            scaler_model = scaler.fit(self.scaled_df)
            self.scaled_df = scaler_model.transform(self.scaled_df)

    # we select all original features but scaled ones
    def select_columns(self):
        columns_to_select = [col_name for col_name in self.df.columns if col_name not in self.columns_to_scale]
        columns_to_select += [input_col + "_scaled" for input_col in self.columns_to_scale]
        self.scaled_df = self.scaled_df.select(columns_to_select)

    # we call all the previous methods in succession
    def scale(self):
        self.find_quantitative_columns()
        self.assemble_features()
        self.scale_columns()
        self.select_columns()
        return self.scaled_df

# we create a class that performs OneHotEncoder for our concerned columns
class OHencoder:
    def __init__(self, df):
        self.df = df
        self.encoded_df = None
        self.string_columns = []

    # first, we find our string columns
    def find_string_columns(self):
        self.string_columns = [col_name for col_name, col_type in self.df.dtypes if
                               col_type == "string" and col_name != "top_3"]

    # we index and encode each feature
    def encode_columns(self):
        self.encoded_df = self.df
        for input_col in self.string_columns:
            indexer = StringIndexer(inputCol=input_col, outputCol=input_col + "_index", handleInvalid="skip")
            encoder = OneHotEncoder(inputCol=input_col + "_index", outputCol=input_col + "_encoded")
            pipeline = Pipeline(stages=[indexer, encoder])
            self.encoded_df = pipeline.fit(self.encoded_df).transform(self.encoded_df)

    # we select all original features but encoded ones
    def select_columns_to_keep(self):
        encoded_columns = [col_name for col_name in self.encoded_df.columns if "_encoded" in col_name]
        non_encoded_columns = [col_name for col_name in self.encoded_df.columns if "_scaled" in col_name or "top_3" == col_name]
        return self.encoded_df.select(encoded_columns + non_encoded_columns)

    # we call all the previous methods in succession
    def encode(self):
        self.find_string_columns()
        self.encode_columns()
        self.encoded_df = self.select_columns_to_keep()
        return self.encoded_df

spark.stop()