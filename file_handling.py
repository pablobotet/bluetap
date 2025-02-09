from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from pyspark.sql.types import *  
import os
import findspark

def create_gcs_session():
    spark = SparkSession.builder \
    .appName("SparkGCS") \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("spark.hadoop.fs.gs.auth.service.account.enable", "true") \
    .config("spark.hadoop.fs.gs.auth.service.account.json.keyfile", os.environ["GOOGLE_APPLICATION_CREDENTIALS"]) \
    .getOrCreate()
    return spark

def read_file_from_gcs(spark, bucket_name, file_path, file_format="csv", schema=None, **kwargs):
    try:
        gcs_path = f"gs://{bucket_name}/{file_path}"
        if file_format == "csv":
            reader = spark.read.csv(gcs_path, **kwargs)
            if schema:
                reader = reader.schema(schema)  
            df = reader.load()
        elif file_format == "parquet":
            reader = spark.read.parquet(gcs_path, **kwargs)
            if schema:
                reader = reader.schema(schema)  
            df = reader.load()
        else:
            print(f"Unsupported file format: {file_format}")
            return None
        return df
    except AnalysisException as e:
        print(f"Error reading file: {e}")
        return None
    
def write_file_to_gcs(df, bucket_name, file_path, file_format="csv", partition_cols=None, **kwargs):
    try:
        gcs_path = f"gs://{bucket_name}/{file_path}"
        writer = df.write.format(file_format)

        if partition_cols:
            writer = writer.partitionBy(*partition_cols)  # Partition by specified columns

        writer = writer.options(**kwargs)  # Apply additional options
        writer.save(gcs_path)  # Save to GCS
        return True
    except AnalysisException as e:
        print(f"Error writing file: {e}")
        return False

def setup_environment(java_home="/usr/lib/jvm/java-1.11.0-openjdk-amd64", gcs_credentials="/content/rosy-cogency-448715-a0-0a1a24b232e2.json"):
    """Sets up the environment variables for Spark and Google Cloud Storage."""
    findspark.init()
    os.environ["JAVA_HOME"] =java_home
    os.environ["SPARK_HOME"] = findspark.find()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcs_credentials
    print("Environment variables set.")
    return True