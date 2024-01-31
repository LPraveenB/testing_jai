import dask
import os
import xgboost as xgb
import numpy as np
import pyarrow as pa

from dask.distributed import Client
from dask import dataframe as dd
from distributed import LocalCluster
from urllib.parse import urlparse
from constant import *


def get_local_dask_cluster(num_workers_local_cluster, num_threads_per_worker, memory_limit_local_worker):
    dask.config.set({"dataframe.shuffle.method": "tasks"})
    dask_cluster = LocalCluster(n_workers=num_workers_local_cluster, threads_per_worker=num_threads_per_worker,
                                memory_limit=memory_limit_local_worker)
    dask_client = Client(dask_cluster)
    dask.config.set({"dataframe.shuffle.method": "tasks"})
    return dask_client


def get_remote_dask_client(dask_address, dask_connection_timeout):
    dask.config.set({"dataframe.shuffle.method": "tasks"})
    dask_client = Client(dask_address, timeout=dask_connection_timeout)
    dask.config.set({"dataframe.shuffle.method": "tasks"})
    return dask_client


def get_dask_client(local_dask_flag, num_workers_local_cluster, num_threads_per_worker, memory_limit_local_worker,
                    dask_address, dask_connection_timeout):
    if local_dask_flag == Y:
        dask_client = get_local_dask_cluster(num_workers_local_cluster, num_threads_per_worker,
                                             memory_limit_local_worker)
    else:
        dask_client = get_remote_dask_client(dask_address, dask_connection_timeout)
    return dask_client


def download_model(client, prod_model_dir, local_dir, local_file_name):
    gcs_path = urlparse(prod_model_dir, allow_fragments=False)
    bucket_name = gcs_path.netloc
    path = gcs_path.path.lstrip('/')
    path = path + "/model.json"
    bucket = client.bucket(bucket_name=bucket_name)
    blob = bucket.blob(path)
    local_full_path = os.path.join(local_dir, local_file_name)
    blob.download_to_filename(local_full_path)


def download_and_load_model(gcs_client, step_name, source_path_model):
    print(f'Loading model: {step_name}')
    model_local_path = "."
    model_local_filename = f'model_{step_name}.json'
    download_model(gcs_client, source_path_model, model_local_path, model_local_filename)
    loaded_model = None
    if step_name in [STEP_1, STEP_2]:
        loaded_model = xgb.dask.DaskXGBClassifier()
    elif step_name in [STEP_3, STEP_4]:
        loaded_model = xgb.dask.DaskXGBRegressor()
    loaded_model.load_model(model_local_filename)
    feature_names = loaded_model.get_booster().feature_names
    return loaded_model, feature_names


def read_input_data(dd_df, input_path, dask_client):
    print('Reading from', input_path)
    input_data_dd = dd_df.read_parquet(input_path, engine='pyarrow', calculate_divisions=False)
    input_data_dd = dask_client.persist(input_data_dd)
    return input_data_dd


def prepare_for_inference(input_data_dd, feature_names):
    """
    This method checks the data type each feature column in the input dataframe
    and if it finds it to be categorical, that column is cast as an int column
    :param input_data_dd: input data frame
    :param feature_names: feature name list of the model
    :return: input data frame with no categorical feature columns
    """
    for feature in feature_names:
        if input_data_dd[feature].dtype == 'O':
            input_data_dd[feature] = input_data_dd[feature].astype(int)
    return input_data_dd


def generate_schema(step):
    schema = {SKU: pa.string(), LOCATION: pa.string(), DATE: pa.string(), LOCATION_GROUP: pa.string()}
    if step == STEP_1:
        schema[BINARY_PRED] = pa.float64()
        schema[TARGET_CLASSIFICATION_BINARY] = pa.int64()
        schema[IP_QTY_EOP_SOH] = pa.int64()
        schema[CURDAY_IP_QTY_EOP_SOH] = pa.int64()
        schema[REV_IP_QTY_EOP_SOH] = pa.int64()
        schema[TOTAL_RETAIL] = pa.int64()
    elif step == STEP_2:
        schema[HL_ORIG_PRED] = pa.float64()
        schema[TARGET_CLASSIFICATION_BINARY_HL] = pa.int64()
    elif step == STEP_3 or step == STEP_4:
        schema[ORIG_PRED] = pa.float64()
        schema[ORIG_PRED_LOW] = pa.float64()
        schema[ORIG_PRED_HIGH] = pa.float64()
        schema[TARGET_REGRESSION] = pa.float64()
    elif step == FINAL_PREDICTION:
        schema[IP_QTY_EOP_SOH] = pa.int64()
        schema[CURDAY_IP_QTY_EOP_SOH] = pa.int64()
        schema[REV_IP_QTY_EOP_SOH] = pa.int64()
        schema[TOTAL_RETAIL] = pa.int64()
        schema[OOS] = pa.int64()
        schema[ORIG_PRED_LOW] = pa.float64()
        schema[ORIG_PRED_HIGH] = pa.float64()
        schema[TARGET_CLASSIFICATION_BINARY] = pa.int64()
    else:
        raise ValueError("Invalid step provided")
    print("Returning the following schema for the dataframe to be persisted:")
    print(schema)
    return schema


def run_predict(input_data_dd, dask_client, loaded_model_s1, feature_names_s1, loaded_model_s2, feature_names_s2,
                loaded_model_s3, feature_names_s3, loaded_model_s4, feature_names_s4, decision_threshold_step_1,
                decision_threshold_step_2):
    print('Using Binary OOS classifier: step 1')
    input_data_dd = prepare_for_inference(input_data_dd, feature_names_s1)
    input_data_dd[BINARY_PRED] = xgb.dask.predict(dask_client, loaded_model_s1.get_booster(),
                                                  input_data_dd[feature_names_s1])
    input_data_dd_all_oos = input_data_dd[(input_data_dd[BINARY_PRED] > decision_threshold_step_1)]
    step_1_dd = dask_client.persist(input_data_dd_all_oos)
    input_data_dd_all_oos[ORIG_PRED_LOW] = 0
    input_data_dd_all_oos[ORIG_PRED_HIGH] = 0
    input_data_dd_all_oos[ORIG_PRED] = 0

    print('Using Binary HL classifier: step 2')
    input_data_dd_all_rest = input_data_dd[(input_data_dd[BINARY_PRED] <= decision_threshold_step_1)]
    input_data_dd_all_rest = prepare_for_inference(input_data_dd_all_rest, feature_names_s2)
    input_data_dd_all_rest[HL_ORIG_PRED] = xgb.dask.predict(dask_client, loaded_model_s2.get_booster(),
                                                            input_data_dd_all_rest[feature_names_s2])

    print('Using LS Regression Model: step 3')
    input_data_dd_all_rest_ls = input_data_dd_all_rest[
        input_data_dd_all_rest[HL_ORIG_PRED] > decision_threshold_step_2]
    input_data_dd_all_rest_ls = prepare_for_inference(input_data_dd_all_rest_ls, feature_names_s3)
    input_data_dd_all_rest_ls[ORIG_PRED] = xgb.dask.predict(dask_client, loaded_model_s3.get_booster(),
                                                            input_data_dd_all_rest_ls[feature_names_s3])
    input_data_dd_all_rest_ls[ORIG_PRED] = (input_data_dd_all_rest_ls[ORIG_PRED] *
                                            input_data_dd_all_rest_ls[CURDAY_IP_QTY_EOP_SOH])
    input_data_dd_all_rest_ls = input_data_dd_all_rest_ls.reset_index(drop=True)
    input_data_dd_all_rest_ls = input_data_dd_all_rest_ls.map_partitions(set_low_high_range)

    print('Using HS Regression model: step 4')
    input_data_dd_all_rest_hs = input_data_dd_all_rest[
        input_data_dd_all_rest[HL_ORIG_PRED] <= decision_threshold_step_2]
    input_data_dd_all_rest_hs = prepare_for_inference(input_data_dd_all_rest_hs, feature_names_s4)
    input_data_dd_all_rest_hs[ORIG_PRED] = xgb.dask.predict(
        dask_client, loaded_model_s4.get_booster(), input_data_dd_all_rest_hs[feature_names_s4])
    input_data_dd_all_rest_hs = input_data_dd_all_rest_hs.reset_index(drop=True)
    input_data_dd_all_rest_hs = input_data_dd_all_rest_hs.map_partitions(set_low_high_range)

    print("Concatenate results")
    input_data_dd_all_day_rev = dd.concat([input_data_dd_all_oos, input_data_dd_all_rest_ls, input_data_dd_all_rest_hs])
    # Prediction post-processing
    input_data_dd_all_day_rev[OOS] = 1
    input_data_dd_all_day_rev[OOS] = input_data_dd_all_day_rev[OOS].where(
        (input_data_dd_all_day_rev[ORIG_PRED_LOW] == 0), 0)

    return (input_data_dd_all_day_rev, step_1_dd, input_data_dd_all_rest, input_data_dd_all_rest_ls,
            input_data_dd_all_rest_hs)


def dd_to_parquet(df, output_path, step):
    print(" writing as parquet with infer schema", len(df))
    # schema = generate_schema(step)
    df.to_parquet(output_path, schema="infer")


def dd_to_csv(df, output_path, step):
    print(" writing as csv", len(df))
    df.to_csv(output_path + "/part-*.csv", index=False)


def set_low_high_range(df):
    df[ORIG_PRED_TEMP] = round(df[ORIG_PRED], 2)
    df[ORIG_PRED_LOW] = round(df[ORIG_PRED_TEMP] - (df[ORIG_PRED_TEMP] * 0.15))
    df[ORIG_PRED_HIGH] = round(df[ORIG_PRED_TEMP] + (df[ORIG_PRED_TEMP] * 0.15))
    df[ORIG_PRED_LOW] = df[ORIG_PRED_LOW].apply(np.floor)
    df[ORIG_PRED_HIGH] = df[ORIG_PRED_HIGH].apply(np.ceil)

    df.loc[(df[ORIG_PRED_LOW] == df[ORIG_PRED_HIGH]), ORIG_PRED_LOW] = df[ORIG_PRED_LOW] - 2
    df.loc[(df[ORIG_PRED_HIGH] - df[ORIG_PRED_LOW]) == 1, ORIG_PRED_LOW] = df[ORIG_PRED_LOW] - 1
    df.loc[df[ORIG_PRED_LOW] < 0, ORIG_PRED_HIGH] = df[ORIG_PRED_HIGH] + 1
    df.loc[df[ORIG_PRED_LOW] < 0, ORIG_PRED_LOW] = 0
    return df


def add_dask_cluster_arguments(parser):
    parser.add_argument(
        '--local_dask_flag',
        dest='local_dask_flag',
        type=str,
        choices={Y, N},
        required=True,
        help='Flag to determine whether dask is local or not')
    parser.add_argument(
        '--dask_address',
        dest='dask_address',
        type=str,
        default=None,
        required=False,
        help='Address of the remote dask cluster')
    parser.add_argument(
        '--dask_connection_timeout',
        dest='dask_connection_timeout',
        type=int,
        default=-1,
        required=False,
        help='Remote dask connection timeout in seconds')
    parser.add_argument(
        '--num_workers_local_cluster',
        dest='num_workers_local_cluster',
        type=int,
        default=0,
        required=False,
        help='Number of workers for the local dask cluster')
    parser.add_argument(
        '--num_threads_per_worker',
        dest='num_threads_per_worker',
        type=int,
        default=0,
        required=False,
        help='Number of threads per local dask cluster worker')
    parser.add_argument(
        '--memory_limit_local_worker',
        dest='memory_limit_local_worker',
        type=str,
        default=None,
        required=False,
        help='Memory limit per worker in the local dask cluster')

    return parser


def add_common_arguments(parser):
    parser.add_argument(
        '--source_path_model_s1',
        dest='source_path_model_s1',
        type=str,
        required=True,
        help='Step1 model directory')
    parser.add_argument(
        '--source_path_model_s2',
        dest='source_path_model_s2',
        type=str,
        required=True,
        help='Step2 model directory')
    parser.add_argument(
        '--source_path_model_s3',
        dest='source_path_model_s3',
        type=str,
        required=True,
        help='Step3 model directory')
    parser.add_argument(
        '--source_path_model_s4',
        dest='source_path_model_s4',
        type=str,
        required=True,
        help='Step4 model directory')
    parser.add_argument(
        '--input_base_path',
        dest='input_base_path',
        type=str,
        required=True,
        help='Input base path containing all the location groups')
    parser.add_argument(
        '--output_base_path',
        dest='output_base_path',
        type=str,
        required=True,
        help='Directory to write the predictions')

    parser = add_dask_cluster_arguments(parser)

    return parser


def add_common_test_component_arguments(parser):
    parser = add_common_arguments(parser)
    parser.add_argument(
        '--location_group_list',
        dest='location_group_list',
        type=str,
        required=True,
        help='Space separated location groups to split')

    return parser


def validate_dask_cluster_arguments(args):
    if args.local_dask_flag == Y:
        if (args.num_workers_local_cluster == 0) or (args.num_threads_per_worker == 0) or \
                (args.memory_limit_local_worker is None):
            raise ValueError("num_workers_local_cluster, num_threads_per_worker & memory_limit_local_worker need to "
                             "have valid values for a local dask cluster")
    else:
        if (args.dask_address is None) or (args.dask_connection_timeout == -1):
            raise ValueError("dask_address & dask_connection_timeout need to have valid values for remote dask cluster")