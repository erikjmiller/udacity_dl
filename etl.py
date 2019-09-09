import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pandas as pd
pd.set_option('max_colwidth', 800)
from pyspark.sql.functions import udf, monotonically_increasing_id
from datetime import datetime


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config.get('AWS', 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = config.get('AWS', 'AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """
    Function to create a new spark session object to pull down data and perform etl processes on.

    :return: spark: The spark session object
    """
    SparkContext.setSystemProperty('spark.executor.memory', '2g')  #increase memory usage for better performance
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Function to parse all song and and artist data

    :param spark: The spark session object
    :param input_data: The root path to the input data
    :param output_data: The root path to the output data
    :return:
    """
    # get filepath to song data file
    song_data = input_data + "/song_data/*/*/*/*.json"
    # use a subset to test smaller set in AWS
    # song_data = input_data + "/song_data/A/A/A/*.json"

    # read song data file
    df = spark.read.json(song_data, multiLine=True)
    df.printSchema()
    df.createOrReplaceTempView("song_data")

    # extract columns to create songs table
    songs_table = spark.sql("""
        SELECT DISTINCT song_id, title, artist_id, year, duration
        FROM song_data
    """)

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode('overwrite').partitionBy("year","artist_id").parquet(output_data + "/songs.parquet")

    # extract columns to create artists table
    artists_table = spark.sql("""
        SELECT DISTINCT artist_id, artist_name, artist_location, artist_latitude, artist_longitude
        FROM song_data
    """)

    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(output_data + "/artists.parquet")


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = input_data + "/log_data/*/*/*.json"
    # use a subset to test smaller set in AWS
    # log_data = input_data + "/log_data/2018/11/*.json"

    # read log data file
    df = spark.read.json(log_data)
    df.printSchema()

    # filter by actions for song plays
    df_next_song = df.where(df.page == 'NextSong')

    # extract columns for users table
    df_next_song.createOrReplaceTempView("users")
    users_table = spark.sql("""
    SELECT DISTINCT
        userId AS user_id,
        firstName AS first_name,
        lastName AS last_name,
        gender,
        level
    FROM users
    """)

    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(output_data + "/users.parquet")

    # create timestamp column from original timestamp column
    from pyspark.sql.types import TimestampType
    @udf(TimestampType())
    def get_timestamp(ts):
        return datetime.fromtimestamp(ts/1000)
    df_next_song = df_next_song.withColumn('date_time', get_timestamp('ts'))

    # extract columns to create time table
    df_next_song.createOrReplaceTempView("song_plays")
    time_table = spark.sql("""
    SELECT DISTINCT ts AS start_time,
        hour(date_time)    AS hour,
        dayofmonth(date_time)    AS day,
        weekofyear(date_time)   AS week,
        month(date_time)   AS month,
        year(date_time)   AS year,
        dayofweek(date_time)   AS weekday
    FROM song_plays
    """)
    time_table.createOrReplaceTempView('time')

    # write time table to parquet files partitioned by year and month
    time_table.write.mode('overwrite').partitionBy("year", "month").parquet(output_data + "/time.parquet")

    # read in song data to use for songplays table
    song_df = spark.read.parquet(output_data + "/songs.parquet")
    artist_df = spark.read.parquet(output_data + "/artists.parquet")
    song_df.createOrReplaceTempView("songs")
    artist_df.createOrReplaceTempView("artists")

    # extract columns from joined song and log datasets to create songplays table
    # songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent
    songplays_table = spark.sql("""
        SELECT  monotonically_increasing_id() AS songplay_id,
                sp.ts         AS start_time,
                sp.userId     AS user_id,
                sp.level      AS level,
                s.song_id     AS song_id,
                a.artist_id   AS artist_id,
                sp.sessionId  AS session_id,
                sp.location   AS location,
                sp.userAgent  AS user_agent,
                t.year        AS year,
                t.month       AS month
        FROM song_plays sp
        INNER JOIN time t ON t.start_time = sp.ts
        LEFT OUTER JOIN songs s ON s.title = sp.song 
        LEFT OUTER JOIN artists a ON a.artist_name = sp.artist 
    """)

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode('overwrite').partitionBy("year","month").parquet(output_data + "/songplays.parquet")


def main():
    spark = create_spark_session()

    # For local testing
    # input_data = config.get('DEV', 'INPUT_DIR')
    # output_data = config.get('DEV', 'OUTPUT_DIR')

    input_data = config.get('AWS', 'INPUT_DIR')
    output_data = config.get('AWS', 'OUTPUT_DIR')

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
