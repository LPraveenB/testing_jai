import argparse
import time
import transform as ts
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType
from pyspark.sql import functions as F

SKU_COLUMN_NAME = "SKU"
LOCATION_COLUMN_NAME = "LOCATION"
DATE_COLUMN_NAME = "DATE"
DATA_SPLIT_COLUMN_NAME = "data_split"
LOCATION_GROUP_FOLDER_PREFIX = "LOCATION_GROUP="

target_columns_list = None

# os.environ['PYSPARK_PYTHON'] = '/opt/conda/default/bin/python'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'


def get_spark_session(run_mode, max_records_per_batch):
    spark = SparkSession.builder.appName("data_split_util").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
    if run_mode == "I":
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", max_records_per_batch)
    return spark


def add_data_split_column_historical(input_df):
    return ts.add_split_column(input_df, train_test_frac=0.9, train_validation_frac=0.9, date_col_name=DATE_COLUMN_NAME,
                               min_df_len_to_process=270, target=target_columns_list)


def add_data_split_column_incremental(iterator):
    for df in iterator:
        result = ts.add_split_column(df, train_test_frac=0.9, train_validation_frac=0.9, date_col_name=DATE_COLUMN_NAME,
                                     min_df_len_to_process=10, target=target_columns_list)
        yield result


def prepare_schema(df_to_split):
    df_schema = StructType.fromJson(df_to_split.schema.jsonValue())
    df_schema.add(DATA_SPLIT_COLUMN_NAME, StringType())
    return df_schema


def generate_results(df, run_mode):
    result = df.mapInPandas(eval("add_data_split_column_incremental"), schema=prepare_schema(df))
    '''
    if run_mode == "I":
        result = df.mapInPandas(eval("add_data_split_column_incremental"), schema=prepare_schema(df))
    else:
        result = df.groupBy(SKU_COLUMN_NAME, LOCATION_COLUMN_NAME).applyInPandas(
            eval("add_data_split_column_historical"),
            schema=prepare_schema(df))
    '''
    return result


def execute(feature_store_path_input, target_store_path_input, data_split_out_path, location_group_list,
            max_records_per_batch, run_mode, current_date_str, train_start_date, last_train_date):
    for location_group in location_group_list:
        feature_store_path = feature_store_path_input + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group
        target_store_path = target_store_path_input + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group
        spark = get_spark_session(run_mode, max_records_per_batch)
        feature_store_df = spark.read.parquet(feature_store_path).drop('LOCATION_GROUP', 'LOAD_DATE')

        if train_start_date is not None:
            feature_store_df = feature_store_df.filter(F.col('DATE') >= train_start_date)
        '''
        print('feature_store_df.count()', feature_store_df.count())
        fs_group = feature_store_df.groupBy('SKU', 'LOCATION').agg(
            F.sum('neg_cc_flag').alias('FILTER_CC_COUNT'), F.sum('sales_flag').alias('FILTER_SALES_COUNT'),
            F.sum('transfer_flag').alias('FILTER_INTRANSIT_COUNT'), F.min('DATE').alias('MIN_DATE'))
        print('fs_group.count()', fs_group.count())
        fs_group = fs_group.filter((fs_group.MIN_DATE <= train_start_date) & (fs_group.FILTER_CC_COUNT > 0) & (fs_group.FILTER_SALES_COUNT > 0) &  (fs_group.FILTER_INTRANSIT_COUNT > 0))
        fs_group = fs_group.select('SKU','LOCATION')
        print('fs_group.count()', fs_group.count())
        feature_store_df = feature_store_df.join(fs_group, ['SKU','LOCATION'], 'inner')
        print('feature_store_df.count()', feature_store_df.count())
        '''
        target_store_df = spark.read.parquet(target_store_path)
        if last_train_date is not None:
            target_store_df = target_store_df.filter(F.col('LOAD_DATE') > last_train_date)
        if train_start_date is not None:
            target_store_df = target_store_df.filter(F.col('DATE') >= train_start_date)
        target_store_df = target_store_df.drop('LOCATION_GROUP', 'LOAD_DATE', 'YEAR_AUDIT_DAY_FLAG')
        input_df = target_store_df.join(feature_store_df, ['SKU', 'LOCATION', 'DATE'], 'inner')
        # input_df = input_df.filter(F.col('audit_day_flag') == 1)
        result = generate_results(input_df, run_mode)
        if current_date_str is not None:
            result = result.filter(F.col('DATE') <= current_date_str)
        out_path = data_split_out_path + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group
        if current_date_str is not None:
            out_path = out_path + "/LOAD_DATE=" + current_date_str
        (result.write.partitionBy(DATA_SPLIT_COLUMN_NAME).option("compression", "gzip").mode("overwrite").
         parquet(out_path))
        print("Successfully processed location group ", location_group)


def main(args=None):
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Running data_split")
    parser.add_argument(
        '--feature_store_path',
        dest='feature_store_path',
        type=str,
        required=True,
        help='Base path of input dataframe to split')
    parser.add_argument(
        '--target_store_path',
        dest='target_store_path',
        type=str,
        required=True,
        help='Base path of input dataframe to split')
    parser.add_argument(
        '--data_split_out_path',
        dest='data_split_out_path',
        type=str,
        required=True,
        help='data_split_out_path')
    parser.add_argument(
        '--location_groups',
        dest='location_groups',
        type=str,
        required=True,
        help='Comma separated location groups to split')
    parser.add_argument(
        '--target_column_list',
        dest='target_column_list',
        type=str,
        required=True,
        help='Comma separated target columns')
    parser.add_argument(
        '--run_mode',
        dest='run_mode',
        type=str,
        choices={'H', 'I'},
        required=True,
        help='"H" for historical mode and "I" for incremental mode')
    parser.add_argument(
        '--max_records_per_batch',
        dest='max_records_per_batch',
        type=str,
        required=False,
        default="10000",
        help='max_records_per_batch')

    parser.add_argument(
        '--current_date_str',
        dest='current_date_str',
        type=str,
        required=True,
        help='The string value  of the current date')

    parser.add_argument(
        '--train_start_date',
        dest='train_start_date',
        type=str,
        required=True,
        help='The string value  of the train start date')

    parser.add_argument(
        '--last_train_date',
        dest='last_train_date',
        type=str,
        required=False,
        help='The string value  of the last train date')

    args = parser.parse_args(args)
    print("args:")
    print(args)

    global target_columns_list
    target_columns_list = args.target_column_list.split(",")

    execute(args.feature_store_path, args.target_store_path, args.data_split_out_path, args.location_groups.split(","),
            args.max_records_per_batch, args.run_mode, args.current_date_str, args.train_start_date,
            args.last_train_date)
    print("<-----------Data Split Component Successful----------->")
    print('Total Time Taken', time.time() - start_time, 'Seconds')


if __name__ == '__main__':
    main()
