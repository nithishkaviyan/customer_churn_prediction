"""
    This script contains code to create modelling dataset
"""

# Load required packages
from pathlib import Path
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import time
import numpy as np


start_time = time.time()

# Create spark session
spark = SparkSession.builder.appName("Dataset_Prep").getOrCreate()

data_folder = Path.cwd().parents[0].joinpath("data")

# Read members data
members_df = spark.read.format("csv") \
             .option("inferSchema", "False") \
             .option("header", "True") \
             .option("sep", ",") \
             .load(str(data_folder.joinpath("members_v3.csv")))

# Read transactions data
transactions_df = spark.read.format("csv") \
                  .option("inferSchema", "False") \
                  .option("header", "True") \
                  .option("sep", ",") \
                  .load(str(data_folder.joinpath("transactions.csv")))

# Read user logs data
user_logs_df = spark.read.format("csv") \
               .option("inferSchema", "False") \
               .option("header", "True") \
               .option("sep", ",") \
               .load(str(data_folder.joinpath("user_logs.csv")))

# Read train data with ground truth
train_df = spark.read.format("csv") \
           .option("inferSchema", "False") \
           .option("header", "True") \
           .option("sep", ",") \
           .load(str(data_folder.joinpath("train.csv")))


# Modelling dataset preparation

# In transaction data, select only customers in train data
# Step 1: Keep only customer msno present in train data
transactions_df = transactions_df.join(train_df.select("msno"), on="msno", how="inner")

# Step 2: For each msno, keep only the latest transaction
transactions_df = transactions_df.withColumn("transaction_date", to_date(transactions_df["transaction_date"],
                                                                         format="yyyyMMdd"))
transactions_df = transactions_df.sort(col("transaction_date").desc())
transactions_df = transactions_df.dropDuplicates(["msno"])

print("Processed transactions data")


# In user logs data, select only customers in train data
# Step 1: Keep only customer msno present in train data
user_logs_df = user_logs_df.join(train_df.select("msno"), on="msno", how="inner")

# Step 2: Cast date column to DateType
user_logs_df = user_logs_df.withColumn("date", to_date(user_logs_df["date"], format="yyyyMMdd"))

# Step 3: Since train set contains customer membership date which expires in the month of February,
# to use last one month's filter df starting from January 30
user_logs_df = user_logs_df.filter((user_logs_df["date"] >= (lit("2017-01-30").cast(DateType()))) &
                                   (user_logs_df["date"] <= (lit("2017-02-28").cast(DateType()))))

print("Processed user logs data")


# For members data, select only records which are present in train set
members_df = members_df.join(train_df.select("msno"), on="msno", how="inner")

print("Processed members data")


# For each user, compute last 30 day average of user logs
# Convert required columns to numeric/float type
int_col = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq"]
float_col = ["total_secs"]

for col in int_col:
    user_logs_df = user_logs_df.withColumn(col, user_logs_df[col].cast(IntegerType()))

for col in float_col:
    user_logs_df = user_logs_df.withColumn(col, user_logs_df[col].cast(FloatType()))

# Aggregate data to compute last 30 day statistics
user_logs_df = user_logs_df.groupBy("msno") \
               .agg(mean("num_25").alias("avg_num_25"),
                    mean("num_50").alias("avg_num_50"),
                    mean("num_75").alias("avg_num_75"),
                    mean("num_985").alias("avg_num_985"),
                    mean("num_100").alias("avg_num_100"),
                    mean("num_unq").alias("avg_num_unq"),
                    mean("total_secs").alias("avg_total_secs"),
                    count("num_25").alias("number_of_days_used"))

print("Processed last 30 day user logs")


# Join all processed dfs with training data
model_dataset = train_df.join(members_df, on="msno", how="left") \
                .join(transactions_df, on="msno", how="left") \
                .join(user_logs_df, on="msno", how="left")


# Save each of the processed df to csv
data_folder.joinpath("processed_data").mkdir(parents=True, exist_ok=True)

transactions_df.toPandas().to_csv(data_folder.joinpath("processed_data", "transactions_processed.csv"), index=False)
members_df.toPandas().to_csv(data_folder.joinpath("processed_data", "members_processed.csv"), index=False)
user_logs_df.toPandas().to_csv(data_folder.joinpath("processed_data", "user_logs_processed.csv"), index=False)
model_dataset.toPandas().to_csv(data_folder.joinpath("processed_data", "model_dataset.csv"), index=False)

print("All data saved sucessfully")
print(f"Total runtime: {np.round(time.time() - start_time, 4)}")
