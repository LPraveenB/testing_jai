import argparse
import time
import dask
import json
import os

from dask.distributed import Client
from distributed import LocalCluster
from dask import dataframe as dd
from google.cloud import storage
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


def validate_dask_cluster_arguments(args):
    if args.local_dask_flag == Y:
        if (args.num_workers_local_cluster == 0) or (args.num_threads_per_worker == 0) or \
                (args.memory_limit_local_worker is None):
            raise ValueError("num_workers_local_cluster, num_threads_per_worker & memory_limit_local_worker need to "
                             "have valid values for a local dask cluster")
    else:
        if (args.dask_address is None) or (args.dask_connection_timeout == -1):
            raise ValueError("dask_address & dask_connection_timeout need to have valid values for remote dask cluster")


def read_input_data(dd_df, input_path, dask_client):
    print('Reading from', input_path)
    input_data_dd = dd_df.read_parquet(input_path, engine='pyarrow', calculate_divisions=False)
    input_data_dd = dask_client.persist(input_data_dd)
    return input_data_dd


def upload_metrics_to_gcs(metrics, metrics_output_file_path):
    local_metrics_filename = 'confidence_metrics.json'
    print(f'Writing {local_metrics_filename} to local')
    with open(local_metrics_filename, 'w') as f:
        json.dump(metrics, f)
    local_dir = '.'
    local_full_path = os.path.join(local_dir, local_metrics_filename)

    gcs_client = storage.Client()
    gcs_path = urlparse(metrics_output_file_path)
    bucket_name = gcs_path.netloc
    gcs_upload_dir = gcs_path.path.lstrip('/') + '/' + local_metrics_filename
    bucket = gcs_client.bucket(bucket_name=bucket_name)
    blob = bucket.blob(gcs_upload_dir)
    blob.upload_from_filename(local_full_path)
    print("Successfully uploaded metrics to GCS")


def create_tp_bins(threshold):
    if threshold < 0 or threshold > 1:
        raise ValueError("Unexpected threshold value")
    bin_list = [threshold]
    incrementer = threshold
    while incrementer + 0.1 <= 1:
        incrementer += 0.1
        bin_list.append(round(incrementer, 1))
    print("Returning Bin List : ", bin_list)
    return bin_list


def create_tn_bins(threshold):
    if threshold < 0 or threshold > 1:
        raise ValueError("Unexpected threshold value")
    bin_list = [threshold]
    incrementer = threshold
    while round(incrementer, 1) > 0:
        incrementer -= 0.1
        bin_list.append(abs(round(incrementer, 1)))
    bin_list.sort()
    print("Returning Bin List : ", bin_list)
    return bin_list


def calculate_tp_bins_percentage(tp_df, tp_bins, tp_df_len, prediction_field):
    result = {}
    iteration_range = len(tp_bins) - 1
    for i in range(iteration_range):
        cond_1 = (tp_df[prediction_field] >= tp_bins[i])
        cond_2 = (tp_df[prediction_field] < tp_bins[i+1])
        if i == iteration_range:
            cond_2 = tp_df[prediction_field] <= tp_bins[i+1]
        target_df = tp_df[cond_1 & cond_2]
        result_percentage = 0.0
        if tp_df_len > 0:
            result_percentage = round(len(target_df)/tp_df_len*100, 2)
        result[tp_bins[i]] = result_percentage
    return result


def calculate_tn_bins_percentage(tn_df, tn_bins, tn_df_len, prediction_field):
    result = {}
    for i in range(len(tn_bins) - 1):
        start = tn_bins[i]
        end = round(start + 0.1, 1)
        target_df = tn_df[(tn_df[prediction_field] >= start) & (tn_df[prediction_field] < end)]
        result_percentage = 0.0
        if tn_df_len > 0:
            result_percentage = round(len(target_df)/tn_df_len * 100, 2)
        result[start] = result_percentage
    return result


def calculate_metrics(df, oos_threshold, hl_threshold):
    metrics = {}
    record_count = len(df)

    oos_tp_df = df[(df[BINARY_PRED] >= oos_threshold) & (df[TARGET_CLASSIFICATION_BINARY] == 1)]
    oos_tp_df_len = len(oos_tp_df)
    oos_tp_perc = 0.0
    if record_count > 0:
        oos_tp_perc = round(oos_tp_df_len/record_count*100, 2)
    oos_tp_mean_probability = 0.0
    if oos_tp_df_len > 0:
        oos_tp_mean_probability = round(oos_tp_df[BINARY_PRED].sum().compute()/oos_tp_df_len, 2)
    oos_tn_df = df[(df[BINARY_PRED] < oos_threshold) & (df[TARGET_CLASSIFICATION_BINARY] == 0)]
    oos_tn_df_len = len(oos_tn_df)
    oos_tn_perc = 0.0
    if record_count > 0:
        oos_tn_perc = round(oos_tn_df_len/record_count * 100, 2)
    oos_tn_mean_probability = 0.0
    if oos_tn_df_len > 0:
        oos_tn_mean_probability = round(oos_tn_df[BINARY_PRED].sum().compute()/oos_tn_df_len, 2)
    oos_tp_bins = create_tp_bins(oos_threshold)
    oos_tp_bin_perc = calculate_tp_bins_percentage(oos_tp_df, oos_tp_bins, oos_tp_df_len, BINARY_PRED)
    oos_tn_bins = create_tn_bins(oos_threshold)
    oos_tn_bin_perc = calculate_tn_bins_percentage(oos_tn_df, oos_tn_bins, oos_tn_df_len, BINARY_PRED)
    oos_metrics = {
        OOS_BINARY_TP_PERCENTAGE: oos_tp_perc,
        OOS_BINARY_OUT_OF_STOCK_MEAN_PROBABILITY: oos_tp_mean_probability,
        OOS_BINARY_TN_PERCENTAGE: oos_tn_perc,
        OOS_BINARY_IN_STOCK_MEAN_PROBABILITY: oos_tn_mean_probability,
        OOS_BINARY_OUT_OF_STOCK_BINS_PERCENTAGE: oos_tp_bin_perc,
        OOS_BINARY_IN_STOCK_BINS_PERCENTAGE: oos_tn_bin_perc
    }
    metrics[OOS_BINARY] = oos_metrics

    hl_df = df[df[BINARY_PRED] < oos_threshold]
    hl_df_len = len(hl_df)

    hl_tp_df = hl_df[(hl_df[ORIG_PRED_LOW] == 0) & (hl_df[TARGET_CLASSIFICATION_BINARY] == 1)]
    hl_tp_df_len = len(hl_tp_df)
    hl_tp_perc = 0.0
    if hl_df_len > 0:
        hl_tp_perc = round(hl_tp_df_len/hl_df_len*100, 2)
    hl_tp_mean_probability = 0.0
    if hl_tp_df_len > 0:
        hl_tp_mean_probability = round(hl_tp_df[HL_ORIG_PRED].sum().compute()/hl_tp_df_len, 2)
    hl_tn_df = hl_df[(hl_df[ORIG_PRED_LOW] > 0) & (hl_df[TARGET_CLASSIFICATION_BINARY] == 0)]
    hl_tn_df_len = len(hl_tn_df)
    hl_tn_perc = 0.0
    if hl_df_len > 0:
        hl_tn_perc = round(hl_tn_df_len/hl_df_len*100, 2)
    hl_tn_mean_probability = 0.0
    if hl_tn_df_len > 0:
        hl_tn_mean_probability = round(hl_tn_df[HL_ORIG_PRED].sum().compute()/hl_tn_df_len, 2)
    hl_tp_bins = create_tp_bins(hl_threshold)
    hl_tp_bin_perc = calculate_tp_bins_percentage(hl_tp_df, hl_tp_bins, hl_tp_df_len, HL_ORIG_PRED)
    hl_tn_bins = create_tn_bins(hl_threshold)
    hl_tn_bin_perc = calculate_tn_bins_percentage(hl_tn_df, hl_tn_bins, hl_tn_df_len, HL_ORIG_PRED)
    hl_metrics = {
        HL_BINARY_TP_PERCENTAGE: hl_tp_perc,
        HL_BINARY_OUT_OF_STOCK_MEAN_PROBABILITY: hl_tp_mean_probability,
        HL_BINARY_TN_PERCENTAGE: hl_tn_perc,
        HL_BINARY_IN_STOCK_MEAN_PROBABILITY: hl_tn_mean_probability,
        HL_BINARY_OUT_OF_STOCK_BINS_PERCENTAGE: hl_tp_bin_perc,
        HL_BINARY_IN_STOCK_BINS_PERCENTAGE: hl_tn_bin_perc
    }
    metrics[HL_BINARY] = hl_metrics

    lt_df = hl_df[hl_df[HL_ORIG_PRED] >= hl_threshold]
    lt_df_len = len(lt_df)
    lt_tp_df = lt_df[(lt_df[ORIG_PRED_LOW] == 0) & (lt_df[TARGET_CLASSIFICATION_BINARY] == 1)]
    lt_tn_df = lt_df[(lt_df[ORIG_PRED_LOW] > 0) & (lt_df[TARGET_CLASSIFICATION_BINARY] == 0)]
    lt_tp_perc = 0.0
    lt_tn_perc = 0.0
    if lt_df_len > 0:
        lt_tp_perc = round(len(lt_tp_df)/lt_df_len*100, 2)
        lt_tn_perc = round(len(lt_tn_df)/lt_df_len*100, 2)
    lt_metrics = {
        LT_TP_PERCENTAGE: lt_tp_perc,
        LT_TN_PERCENTAGE: lt_tn_perc
    }
    metrics[LT] = lt_metrics

    hs_df = hl_df[hl_df[HL_ORIG_PRED] < hl_threshold]
    hs_df_len = len(hs_df)
    hs_tp_df = hs_df[(hs_df[ORIG_PRED_LOW] == 0) & (hs_df[TARGET_CLASSIFICATION_BINARY] == 1)]
    hs_tn_df = hs_df[(hs_df[ORIG_PRED_LOW] > 0) & (hs_df[TARGET_CLASSIFICATION_BINARY] == 0)]
    hs_tp_perc = 0.0
    hs_tn_perc = 0.0
    if hs_df_len > 0:
        hs_tp_perc = round(len(hs_tp_df)/hs_df_len*100, 2)
        hs_tn_perc = round(len(hs_tn_df)/hs_df_len*100, 2)
    hs_metrics = {
        HS_TP_PERCENTAGE: hs_tp_perc,
        HS_TN_PERCENTAGE: hs_tn_perc
    }
    metrics[HS] = hs_metrics

    return metrics


def execute(prediction_base_path, load_date, oos_threshold, hl_threshold, output_base_path, local_dask_flag, dask_address,
            dask_connection_timeout, num_workers_local_cluster, num_threads_per_worker, memory_limit_local_worker):
    dask_client = get_dask_client(local_dask_flag, dask_address, dask_connection_timeout, num_workers_local_cluster,
                                  num_threads_per_worker, memory_limit_local_worker)
    path_to_read = (prediction_base_path + "/" + LOAD_DATE_FOLDER_PREFIX + load_date + "/" + FINAL_PREDICTION + "/*" +
                    PARQUET_FILE_EXTENSION)
    input_dd = read_input_data(dd, path_to_read, dask_client)
    metrics = calculate_metrics(input_dd, oos_threshold, hl_threshold)
    output_path = output_base_path + "/" + LOAD_DATE_FOLDER_PREFIX + load_date
    upload_metrics_to_gcs(metrics, output_path)


def main(args=None):
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Running Confidence Metrics Component")

    parser.add_argument(
        '--prediction_base_path',
        dest='prediction_base_path',
        type=str,
        required=True,
        help='Prediction base path')
    parser.add_argument(
        '--load_date',
        dest='load_date',
        type=str,
        required=True,
        help='UTC load date in ISO format')
    parser.add_argument(
        '--decision_threshold_step_1',
        dest='oos_threshold',
        type=float,
        default=0.4,
        required=False,
        help='Decision threshold for Binary OOS prediction')
    parser.add_argument(
        '--decision_threshold_step_2',
        dest='hl_threshold',
        type=float,
        default=0.4,
        required=False,
        help='Decision threshold for Binary H/L prediction')
    parser.add_argument(
        '--output_base_path',
        dest='output_base_path',
        type=str,
        required=True,
        help='Base Directory to write output metrics')
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

    args = parser.parse_args(args)
    print("args:")
    print(args)

    validate_dask_cluster_arguments(args)

    execute(args.prediction_base_path, args.load_date, args.oos_threshold, args.hl_threshold, args.output_base_path,
            args.local_dask_flag, args.dask_address, args.dask_connection_timeout, args.num_workers_local_cluster,
            args.num_threads_per_worker, args.memory_limit_local_worker)

    print("<-----------Confidence Metrics Component Successful----------->")
    print('Total Time Taken', time.time() - start_time, 'Seconds')


if __name__ == '__main__':
    main()