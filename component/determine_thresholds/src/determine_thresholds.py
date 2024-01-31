import argparse
import time
import dask
import json
import os

from distributed import LocalCluster
from dask.distributed import Client
from dask import dataframe as dd
from google.cloud import storage
from urllib.parse import urlparse
from constant import *

"""
This component determines the best OOS threshold & HL threshold values which give the best overall OOS prediction. 
It needs individual predictions of OOS Binary classifier, HL Binary Classifier, LS Regression & HS Regression models.
Also, it needs the possible values of OOS threshold & HL threshold to choose from.
"""


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


def str_to_float_list(s: str):
    out_list = [float(i) for i in s.split(',')]
    return out_list


def combine_lists(list1, list2):
    combined_list = []
    for i in range(len(list1)):
        for j in range(len(list2)):
            combined_list.append((list1[i], list2[j]))
    return combined_list


def read_dataframe(dd_df, input_path, dask_client):
    print('Reading from', input_path)
    input_data_dd = dd_df.read_csv(input_path)
    input_data_dd = dask_client.persist(input_data_dd)
    return input_data_dd


def read_predictions_into_dataframe(dd_df, predictions_base_path, dask_client):
    oos_dd = read_dataframe(dd_df, predictions_base_path + "/*/" + STEP_1 + '/*' + CSV_FILE_EXTENSION, dask_client)
    hl_dd = read_dataframe(dd_df, predictions_base_path + "/*/" + STEP_2 + '/*' + CSV_FILE_EXTENSION, dask_client)
    ls_dd = read_dataframe(dd_df, predictions_base_path + "/*/" + STEP_3 + '/*' + CSV_FILE_EXTENSION, dask_client)
    hs_dd = read_dataframe(dd_df, predictions_base_path + "/*/" + STEP_4 + '/*' + CSV_FILE_EXTENSION, dask_client)
    return oos_dd, hl_dd, ls_dd, hs_dd


def generate_final_prediction(oos_predictions, hl_predictions, ls_predictions, hs_predictions, oos_threshold,
                              hl_threshold):
    print("Generating the final prediction using the following thresholds values")
    print("oos_threshold: ", oos_threshold)
    print("hl_threshold: ", hl_threshold)

    oos_dd = oos_predictions[oos_predictions[BINARY_PRED] > oos_threshold]
    oos_dd[ORIG_PRED_LOW] = 0
    oos_dd[ORIG_PRED_HIGH] = 0
    oos_dd = oos_dd.drop(columns=[BINARY_PRED])
    hl_dd = oos_predictions[oos_predictions[BINARY_PRED] <= oos_threshold].merge(hl_predictions,
                                                                                 on=[SKU, LOCATION, DATE], how='inner')
    ls_dd = hl_dd[hl_dd[HL_ORIG_PRED] > hl_threshold].merge(ls_predictions, on=[SKU, LOCATION, DATE], how='inner')
    ls_dd = ls_dd.drop(columns=[BINARY_PRED, HL_ORIG_PRED, TARGET_CLASSIFICATION_BINARY_HL, ORIG_PRED,
                                TARGET_REGRESSION])
    hs_dd = hl_dd[hl_dd[HL_ORIG_PRED] <= hl_threshold].merge(hs_predictions, on=[SKU, LOCATION, DATE], how='inner')
    hs_dd = hs_dd.drop(columns=[BINARY_PRED, HL_ORIG_PRED, TARGET_CLASSIFICATION_BINARY_HL, ORIG_PRED,
                                TARGET_REGRESSION_H])
    result = dd.concat([oos_dd, ls_dd, hs_dd])
    result[OOS] = 1
    result[OOS] = result[OOS].where((result[ORIG_PRED_LOW] <= 0), 0)
    return result


def calculate_tp_count(df):
    print("Calculating ", TP_COUNT)
    return len(df[(df[REV_IP_QTY_EOP_SOH] <= 0) & (df[ORIG_PRED_LOW] <= 0)])


def get_tp_count(df, metric_cache):
    result = metric_cache.get(TP_COUNT)
    if result is None:
        result = calculate_tp_count(df)
    metric_cache[TP_COUNT] = result
    return result


def calculate_fn_count(df):
    print("Calculating ", FN_COUNT)
    return len(df[(df[REV_IP_QTY_EOP_SOH] <= 0) & (df[ORIG_PRED_LOW] > 0)])


def get_fn_count(df, metric_cache):
    result = metric_cache.get(FN_COUNT)
    if result is None:
        result = calculate_fn_count(df)
    metric_cache[FN_COUNT] = result
    return result


def calculate_tn_count(df):
    print("Calculating ", TN_COUNT)
    return len(df[(df[REV_IP_QTY_EOP_SOH] > 0) & (df[ORIG_PRED_LOW] > 0)])


def get_tn_count(df, metric_cache):
    result = metric_cache.get(TN_COUNT)
    if result is None:
        result = calculate_tn_count(df)
    metric_cache[TN_COUNT] = result
    return result


def calculate_fp_count(df):
    print("Calculating ", FP_COUNT)
    return len(df[(df[REV_IP_QTY_EOP_SOH] > 0) & (df[ORIG_PRED_LOW] <= 0)])


def get_fp_count(df, metric_cache):
    result = metric_cache.get(FP_COUNT)
    if result is None:
        result = calculate_fp_count(df)
    metric_cache[FP_COUNT] = result
    return result


def calculate_tp_and_fn_based_percent(df, metric_name, metric_cache):
    print("Calculating ", metric_name)
    tp = get_tp_count(df, metric_cache)
    fn = get_fn_count(df, metric_cache)
    result = 0
    if (tp + fn) > 0:
        if metric_name == FN_PERCENT:
            result = round((fn / (tp + fn)) * 100, 2)
        else:
            result = round((tp / (tp + fn)) * 100, 2)
    return result


def get_tp_and_fn_based_percent(df, metric_name, metric_cache):
    result = metric_cache.get(metric_name)
    if result is None:
        result = calculate_tp_and_fn_based_percent(df, metric_name, metric_cache)
    if metric_name == TP_PERCENT or metric_name == RECALL_PERCENT:
        metric_cache[TP_PERCENT] = result
        metric_cache[RECALL_PERCENT] = result
    else:
        metric_cache[FN_PERCENT] = result
    return result


def calculate_tn_and_fp_based_percent(df, metric_name, metric_cache):
    print("Calculating ", metric_name)
    tn = get_tn_count(df, metric_cache)
    fp = get_fp_count(df, metric_cache)
    result = 0
    if (tn + fp) > 0:
        if metric_name == TN_PERCENT:
            result = round((tn / (tn + fp)) * 100, 2)
        else:
            result = round((fp / (tn + fp)) * 100, 2)
    return result


def get_tn_and_fp_based_percent(df, metric_name, metric_cache):
    result = metric_cache.get(metric_name)
    if result is None:
        result = calculate_tn_and_fp_based_percent(df, metric_name, metric_cache)
    metric_cache[metric_name] = result
    return result


def calculate_precision_percent(df, metric_cache):
    print("Calculating ", PRECISION_PERCENT)
    tp = get_tp_count(df, metric_cache)
    fp = get_fp_count(df, metric_cache)
    result = 0
    if (tp + fp) > 0:
        result = round((tp / (tp + fp)) * 100, 2)
    return result


def get_precision_percent(df, metric_name, metric_cache):
    result = metric_cache.get(metric_name)
    if result is None:
        result = calculate_precision_percent(df, metric_cache)
    metric_cache[metric_name] = result
    return result


def calculate_f1_score(df, metric_cache):
    print("Calculating ", F1_SCORE)
    precision_percent = get_precision_percent(df, PRECISION_PERCENT, metric_cache)
    recall_percent = get_tp_and_fn_based_percent(df, RECALL_PERCENT, metric_cache)
    result = 0
    if (precision_percent + recall_percent) > 0:
        result = round(2 * ((precision_percent * recall_percent) / (precision_percent + recall_percent)), 2)
    return result


def get_f1_score(df, metric_name, metric_cache):
    result = metric_cache.get(metric_name)
    if result is None:
        result = calculate_f1_score(df, metric_cache)
    metric_cache[metric_name] = result
    return result


def calculate_accuracy(df, metric_cache):
    print("Calculating ", ACCURACY)
    tp = get_tp_count(df, metric_cache)
    fp = get_fp_count(df, metric_cache)
    fn = get_fn_count(df, metric_cache)
    tn = get_tn_count(df, metric_cache)
    result = 0
    if (tp + tn + fp + fn) > 0:
        result = round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)
    return result


def get_accuracy(df, metric_name, metric_cache):
    result = metric_cache.get(metric_name)
    if result is None:
        result = calculate_accuracy(df, metric_cache)
    metric_cache[metric_name] = result
    return result


def calculate_metric(df, metric_name, metric_cache):
    print("Going to calculate ", metric_name)
    print("Metric Cache before calculating the metric:")
    print(metric_cache)
    if metric_name == TP_COUNT:
        calculated_metric = get_tp_count(df, metric_cache)
    elif metric_name == FN_COUNT:
        calculated_metric = get_fn_count(df, metric_cache)
    elif metric_name == TN_COUNT:
        calculated_metric = get_tn_count(df, metric_cache)
    elif metric_name == FP_COUNT:
        calculated_metric = get_fp_count(df, metric_cache)
    elif metric_name == TP_PERCENT:
        calculated_metric = get_tp_and_fn_based_percent(df, metric_name, metric_cache)
    elif metric_name == RECALL_PERCENT:
        calculated_metric = get_tp_and_fn_based_percent(df, metric_name, metric_cache)
    elif metric_name == FN_PERCENT:
        calculated_metric = get_tp_and_fn_based_percent(df, metric_name, metric_cache)
    elif metric_name == TN_PERCENT:
        calculated_metric = get_tn_and_fp_based_percent(df, metric_name, metric_cache)
    elif metric_name == FP_PERCENT:
        calculated_metric = get_tn_and_fp_based_percent(df, metric_name, metric_cache)
    elif metric_name == PRECISION_PERCENT:
        calculated_metric = get_precision_percent(df, metric_name, metric_cache)
    elif metric_name == F1_SCORE:
        calculated_metric = get_f1_score(df, metric_name, metric_cache)
    elif metric_name == ACCURACY:
        calculated_metric = get_accuracy(df, metric_name, metric_cache)
    else:
        raise ValueError("Invalid metric name")
    print("Metric Cache after calculating the metric:")
    print(metric_cache)
    print(metric_name + " = ", calculated_metric)
    return calculated_metric


def score_final_prediction(df, metrics_config):
    result = 0
    metric_cache = {}
    for metric in metrics_config['metrics_list']:
        metric_weight = metric['metric_weight']
        metric = calculate_metric(df, metric['metric_name'], metric_cache)
        result += (metric_weight*metric)

    return result


def load_metric_config(gcs_client, metrics_config_file_path):
    local_dir = "."
    local_file_name = 'metrics_config.json'
    gcs_path = urlparse(metrics_config_file_path, allow_fragments=False)
    bucket_name = gcs_path.netloc
    path = gcs_path.path.lstrip('/')
    bucket = gcs_client.bucket(bucket_name=bucket_name)
    blob = bucket.blob(path)
    local_full_path = os.path.join(local_dir, local_file_name)
    blob.download_to_filename(local_full_path)
    with open(local_full_path) as f:
        json_config = json.load(f)
    return json_config


def evaluate_thresholds(dd_df, dask_client, gcs_client, predictions_base_path, thresholds_list,
                        metrics_config_file_path):
    oos_dd, hl_dd, ls_dd, hs_dd = read_predictions_into_dataframe(dd_df, predictions_base_path, dask_client)
    best_oos_threshold = 0.0
    best_hl_threshold = 0.0
    best_score = 0.0
    best_prediction_dd = None
    metrics_config = load_metric_config(gcs_client, metrics_config_file_path)
    print("Got the following metrics config:")
    print(metrics_config)
    for item in thresholds_list:
        oos_threshold = item[0]
        hl_threshold = item[1]
        final_prediction_dd = generate_final_prediction(oos_dd, hl_dd, ls_dd, hs_dd, oos_threshold, hl_threshold)

        prediction_score = score_final_prediction(final_prediction_dd, metrics_config)
        print("Final prediction score = ", prediction_score)
        if prediction_score > best_score:
            print("Updating best score from ", best_score, " to ", prediction_score)
            best_score = prediction_score
            best_oos_threshold = oos_threshold
            best_hl_threshold = hl_threshold
            best_prediction_dd = final_prediction_dd

    print("Best Metric Score = ", best_score)

    return best_oos_threshold, best_hl_threshold, best_prediction_dd


def upload_thresholds_to_gcs(gcs_client, local_filename, thresholds, metrics_output_file_path):
    print(f'Writing {local_filename} to local')
    with open(local_filename, 'w') as f:
        json.dump(thresholds, f)
    local_dir = '.'
    local_full_path = os.path.join(local_dir, local_filename)

    gcs_path = urlparse(metrics_output_file_path)
    bucket_name = gcs_path.netloc
    gcs_upload_dir = gcs_path.path.lstrip('/')
    bucket = gcs_client.bucket(bucket_name=bucket_name)
    blob = bucket.blob(gcs_upload_dir)
    blob.upload_from_filename(local_full_path)
    print("Successfully uploaded metrics to GCS")


def execute(predictions_base_path: str, possible_oos_threshold_values: str, possible_hl_threshold_values: str,
            output_path: str, metrics_config_file_path: str, local_dask_flag: str, dask_address: str,
            dask_connection_timeout: int, num_workers_local_cluster: int, num_threads_per_worker: int,
            memory_limit_local_worker: str):
    if local_dask_flag == Y:
        dask_client = get_local_dask_cluster(num_workers_local_cluster, num_threads_per_worker,
                                             memory_limit_local_worker)
    else:
        dask_client = get_remote_dask_client(dask_address, dask_connection_timeout)

    oos_threshold_list = str_to_float_list(possible_oos_threshold_values)
    hl_threshold_list = str_to_float_list(possible_hl_threshold_values)
    thresholds_list = combine_lists(oos_threshold_list, hl_threshold_list)
    print("Thresholds list:")
    print(thresholds_list)

    gcs_client = storage.Client()

    oos_threshold, hl_threshold, final_prediction_dd = evaluate_thresholds(dd, dask_client, gcs_client,
                                                                           predictions_base_path, thresholds_list,
                                                                           metrics_config_file_path)
    final_prediction_path = predictions_base_path + "/" + FINAL_PREDICTION
    final_prediction_dd.to_csv(final_prediction_path + "/part-*.csv")
    print("Persisted the best prediction dataframe to ", final_prediction_path)
    result = {
        "oos_threshold": oos_threshold,
        "hl_threshold": hl_threshold
    }
    print("Best thresholds:")
    print(result)
    json_filename = "best_thresholds.json"
    upload_thresholds_to_gcs(gcs_client, json_filename, result, output_path + "/" + json_filename)


def main(args=None):
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Running Determine Inference thresholds Component")
    parser.add_argument(
        '--predictions_base_path',
        dest='predictions_base_path',
        type=str,
        required=True,
        help='Base path for new models individual predictions')
    parser.add_argument(
        '--possible_oos_threshold_values',
        dest='possible_oos_threshold_values',
        type=str,
        required=True,
        help='Comma-separated possible values of OOS model threshold')
    parser.add_argument(
        '--possible_hl_threshold_values',
        dest='possible_hl_threshold_values',
        type=str,
        required=True,
        help='Comma-separated possible values of HL model threshold')
    parser.add_argument(
        '--output_path',
        dest='output_path',
        type=str,
        required=True,
        help='Output directory under which thresholds JSON would be generated')
    parser.add_argument(
        '--metrics_config_file_path',
        dest='metrics_config_file_path',
        type=str,
        required=True,
        help='Full path to the metrics configuration JSON file')
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

    if args.local_dask_flag == Y:
        if (args.num_workers_local_cluster == 0) or (args.num_threads_per_worker == 0) or \
                (args.memory_limit_local_worker is None):
            raise ValueError("num_workers_local_cluster, num_threads_per_worker & memory_limit_local_worker need to "
                             "have valid values for a local dask cluster")
    else:
        if (args.dask_address is None) or (args.dask_connection_timeout == -1):
            raise ValueError("dask_address & dask_connection_timeout need to have valid values for remote dask cluster")

    execute(args.predictions_base_path, args.possible_oos_threshold_values, args.possible_hl_threshold_values,
            args.output_path, args.metrics_config_file_path, args.local_dask_flag, args.dask_address,
            args.dask_connection_timeout, args.num_workers_local_cluster, args.num_threads_per_worker,
            args.memory_limit_local_worker)

    print("<-----------Determine Inference thresholds Component Successful----------->")
    print('Total Time Taken', time.time() - start_time, 'Seconds')


if __name__ == '__main__':
    main()
