import datasets
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
import os
import requests
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType, TimestampType
)
from pyspark.sql.functions import col, first, mean, pandas_udf, explode
import yfinance as yf
from datetime import datetime
import pandas as pd
from file_handling import create_gcs_session,read_file_from_gcs, write_file_to_gcs

@udf(returnType=StructType([
    StructField("input_ids", ArrayType(IntegerType())),
    StructField("attention_mask", ArrayType(IntegerType()))
]))
def tokenize_text(text_1,text_2,tokenizer):
    output = tokenizer(
        text_1,
        text_2,
        padding='max_length',  
        truncation=True,       
        max_length=128         
    )
    return output["input_ids"],output["attention_mask"]


# Creamos la sesión de Spark
spark=create_gcs_session()

#Leemos APIs y construimos las tablas
url = "https://financialmodelingprep.com/api/v3/stock/list?apikey=IPhZSSdvD5qWRFlZ6RnhieRIsYYlVDuN"
response = requests.get(url)
data = response.json()

initial_schema = StructType([
    StructField(field_name, StringType(), True) for field_name in data[0]
])

df = spark.createDataFrame(data, schema=initial_schema)
companies_df = df.withColumn("price", col("price").cast(DoubleType()))

exchange_df = df.groupBy("exchangeShortName") \
    .agg(
        first("exchange").alias("exchange"),
        mean("price").alias("avg_price")
    )

schema_stock = "array<struct<Open:double, High:double, Low:double, Close:double, Volume:long, Dividends:double, Stock_Splits:double, Date:timestamp>>"

@pandas_udf(schema_stock)
def stock_info(ticker_series: pd.Series) -> pd.Series:
    def get_stock_data(ticker):
        try:
            data = yf.Ticker(ticker)
            stock_data = data.history(period="4y").reset_index()
            stock_data = stock_data.rename(columns={"Stock Splits": "Stock_Splits"})
            return stock_data.to_dict(orient="records")
        except Exception:
            return [{"Open": -1.0, "High": -1.0, "Low": -1.0, "Close": -1.0,
                     "Volume": -1, "Dividends": -1.0, "Stock_Splits": -1.0,
                     "Date": datetime(1900, 1, 1)}]

    return ticker_series.apply(get_stock_data)

schema_etf = "array<struct<Open:double, High:double, Low:double, Close:double, Volume:long, Dividends:double, Stock_Splits:double, Capital_Gains:double, Date:timestamp>>"

@pandas_udf(schema_etf)
def etf_info(ticker_series: pd.Series) -> pd.Series:
    def get_etf_data(ticker):
        try:
            data = yf.Ticker(ticker)
            etf_data = data.history(period="4y").reset_index()
            etf_data = etf_data.rename(columns={"Stock Splits": "Stock_Splits", "Capital Gains": "Capital_Gains"})
            return etf_data.to_dict(orient="records")
        except Exception:
            return [{"Open": -1.0, "High": -1.0, "Low": -1.0, "Close": -1.0,
                     "Volume": -1, "Dividends": -1.0, "Stock_Splits": -1.0, "Capital_Gains": -1.0,
                     "Date": datetime(1900, 1, 1)}]

    return ticker_series.apply(get_etf_data)

stock_df = companies_df.filter(companies_df['type'] == 'stock') \
    .withColumn("stock_data", stock_info("symbol")) \
    .select("symbol", explode("stock_data").alias("exploded_data")) \
    .select(
        "symbol",
        col("exploded_data.Date"),
        col("exploded_data.Open"),
        col("exploded_data.High"),
        col("exploded_data.Low"),
        col("exploded_data.Close"),
        col("exploded_data.Volume"),
        col("exploded_data.Dividends"),
        col("exploded_data.Stock_Splits")
    )

etf_df = companies_df.filter(companies_df['type'] == 'etf') \
    .withColumn("etf_data", etf_info("symbol")) \
    .select("symbol", explode("etf_data").alias("exploded_etf_data")) \
    .select(
        "symbol",
        col("exploded_etf_data.Date"),
        col("exploded_etf_data.Open"),
        col("exploded_etf_data.High"),
        col("exploded_etf_data.Low"),
        col("exploded_etf_data.Close"),
        col("exploded_etf_data.Volume"),
        col("exploded_etf_data.Dividends"),
        col("exploded_etf_data.Stock_Splits"),
        col("exploded_etf_data.Capital_Gains")
    )


write_file_to_gcs(df=companies_df, bucket_name='financial_data_bluetap',file_path='/Tables/companies_2',file_format='parquet',)
write_file_to_gcs(df=exchange_df, bucket_name='financial_data_bluetap',file_path='/Tables/exchange_2',file_format='parquet',)
write_file_to_gcs(df=stock_df, bucket_name='financial_data_bluetap',file_path='/Tables/stock_2',file_format='parquet',partition_bypartition_cols='symbol')
write_file_to_gcs(df=etf_df, bucket_name='financial_data_bluetap',file_path='/Tables/etf_2',file_format='parquet',partition_bypartition_cols='symbol')
