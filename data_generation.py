from file_handling import create_gcs_session, read_file_from_gcs
import datasets
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
from pyspark.sql.functions import struct, lit

spark = create_gcs_session()

#Creamos y leemos los datos cualitativos
schema_qualitative = StructType([
    StructField("symbol", StringType(), True),
    StructField("Company Name", StringType(), True),
    StructField("CEO", StringType(), True),
    StructField("Country", StringType(), True),
    StructField("Sector", StringType(), True),
    StructField("Industry", StringType(), True),
    StructField("Founded Year", LongType(), True),
    StructField("Headquarters", StringType(), True),

])

qualitative_df=read_file_from_gcs(spark=spark, bucket_name='financial_data_bluetap',file_path='Data-Science Trial/company_qualitative.csv',schema=schema_qualitative)
   

company_schema = StructType([
    StructField("symbol", StringType(), True),
    StructField("name", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("exchange", StringType(), True),
    StructField("exchangeShortName", StringType(), True),
    StructField("type", StringType(), True)
])
qualitative_df=read_file_from_gcs(spark=spark, bucket_name='financial_data_bluetap',file_path='companies/part-00000-ea013bf2-122f-4fbe-ae59-7e1dafe1bfec-c000.snappy.parquet',format='parquet',schema=company_schema)


#Manipulamos las tablas para generar nuestro dataset
quali_df = quali_df.withColumn('text_1',row_to_text_udf(struct(*quali_df.columns)))
company_df = company_df.withColumn('text_2',row_to_text_udf(struct(*company_df.columns)))

df_match= quali_df.join(company_df, on='symbol',how='inner').select('text_1','text_2').withColumn('label',lit(1))
fraction = float(df_match.count()) / quali_df.crossJoin(company_df).count()

df_cross = quali_df.crossJoin(company_df).select('text_1','text_2').withColumn('label',lit(0)).sample(fraction=fraction)

df_union=df_match.union(df_cross)
labels = df_union.select("label").rdd.flatMap(lambda x: x).collect()
df_union=df_union.drop('label')
df_union=df_union.withColumn('tokenizer_output',tokenize_text(df_union['text_1'],df_union['text_2']))
df_union=df_union.drop('text_1','text_2')
tokenizer_output = df_union.select("tokenizer_output").rdd.flatMap(lambda x: x).collect()

input_ids = [row[0] for row in tokenizer_output]
attention_mask = [row[1] for row in tokenizer_output]
dataset = datasets.Dataset.from_dict({
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels
})

dataset.save_to_disk("model/data/dataset")  # Replace with your desired path
