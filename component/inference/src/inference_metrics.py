import argparse
import time
import json
import os
import dask

from dask.distributed import Client
from dask import dataframe as dd
from distributed import LocalCluster
from google.cloud import storage
from urllib.parse import urlparse
from constant import *


"""
This component generates the data to calculate the model performance metrics for OOS predictions
It needs an audit record corresponding to each prediction. The audit records needs a minimum of SKU, LOCATION, DATE,
IP_QTY_EOP_SOH, REV_IP_QTY_EOP_SOH, CURDAY_IP_QTY_EOP_SOH, TOTAL_RETAIL and the actual OOS outcome flag. For expected
column names in each audit record, please refer to the relevant column names in constant.py
 
It compares the audit records for each prediction to compute the metrics
The computed metrics JSON is uploaded to the output GCS location
Also, a model performance report CSV is generated which contains each audit record and it's corresponding OOS prediction

Optionally, this program can also combine predictions across all the location groups for a particular run_id and
generate a single predictions CSV. However, since we have a large number of inference records each run_id, this option
should be chosen wisely because we loose the dask parallelism while generating the single CSV and the component 
execution time would significantly increase
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


def read_data(file_path, dask_client):
    data_dd = dd.read_csv(file_path)
    data_dd = dask_client.persist(data_dd)
    return data_dd


def calculate_metrics(df):
    tp = len(df[(df[ACTUAL_OOS_COLUMN_NAME] == 1) & (df[OOS] == 1)])
    fn = len(df[(df[ACTUAL_OOS_COLUMN_NAME] == 1) & (df[OOS] == 0)])
    tn = len(df[(df[ACTUAL_OOS_COLUMN_NAME] == 0) & (df[OOS] == 0)])
    fp = len(df[(df[ACTUAL_OOS_COLUMN_NAME] == 0) & (df[OOS] == 1)])

    print("True Positive Count: ", tp)
    print("False Negative Count: ", fn)
    print("True Negative Count: ", tn)
    print("False Positive Count: ", fp)

    tp_perc = 0
    fn_perc = 0
    recall_perc = 0

    if (tp + fn) > 0:
        tp_perc = round((tp / (tp + fn)) * 100, 2)
        fn_perc = round((fn / (tp + fn)) * 100, 2)
        recall_perc = round((tp / (tp + fn)) * 100, 2)

    print('TP %', tp_perc)
    print('FN %', fn_perc)
    print('Recall %', recall_perc)

    tn_perc = 0
    fp_perc = 0

    if (tn + fp) > 0:
        tn_perc = round((tn / (tn + fp)) * 100, 2)
        fp_perc = round((fp / (tn + fp)) * 100, 2)

    print('TN %', tn_perc)
    print('FP %', fp_perc)

    precision_perc = 0

    if (tp + fp) > 0:
        precision_perc = round((tp / (tp + fp)) * 100, 2)

    print('Precision %', precision_perc)

    if (precision_perc + recall_perc) > 0:
        f1_score = round(2 * ((precision_perc * recall_perc) / (precision_perc + recall_perc)), 2)
    else:
        f1_score = 0

    if (tp + tn + fp + fn) > 0:
        accuracy_perc = round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)
    else:
        accuracy_perc = 0

    print('F1 Score', f1_score)
    print('Accuracy %', accuracy_perc)

    df_low = df[df[REV_IP_QTY_EOP_SOH] <= df[CURDAY_IP_QTY_EOP_SOH]]
    df_low_len = len(df_low)
    print("df_low_len: ", df_low_len)

    under_stock_acc = len(df_low[(df_low[ORIG_PRED_LOW] <= df_low[REV_IP_QTY_EOP_SOH]) &
                                 (df_low[ORIG_PRED_HIGH] >= df_low[REV_IP_QTY_EOP_SOH])])
    under_stock_acc_perf = 0.0
    if df_low_len > 0:
        under_stock_acc_perf = round((under_stock_acc / df_low_len) * 100, 2)
    print("Actual Less Than Book Accurate Performance %", under_stock_acc_perf)

    under_stock_overall = len(df_low[(df_low[ORIG_PRED_LOW] <= df_low[CURDAY_IP_QTY_EOP_SOH])])
    under_stock_overall_perf = 0.0
    if df_low_len > 0:
        under_stock_overall_perf = round((under_stock_overall / df_low_len) * 100, 2)
    print("Actual Less Than Book Overall  Performance %", under_stock_overall_perf)

    df_high = df[df[REV_IP_QTY_EOP_SOH] > df[CURDAY_IP_QTY_EOP_SOH]]
    df_high_len = len(df_high)
    print("df_high_len: ", df_high_len)

    over_stock_acc = len(df_high[(df_high[ORIG_PRED_LOW] <= df_high[REV_IP_QTY_EOP_SOH]) &
                                 (df_high[ORIG_PRED_HIGH] >= df_high[REV_IP_QTY_EOP_SOH])])
    over_stock_acc_perf = 0.0
    if df_high_len > 0:
        over_stock_acc_perf = round((over_stock_acc / df_high_len) * 100, 2)
    print('Actual Greater Than Book Accurate  Performance %', over_stock_acc_perf)

    over_stock_overall = len(df_high[(df_high[ORIG_PRED_HIGH] > df_high[CURDAY_IP_QTY_EOP_SOH])])
    over_stock_overall_perf = 0.0
    if df_high_len > 0:
        over_stock_overall_perf = round((over_stock_overall / df_high_len) * 100, 2)
    print('Actual Greater Than Book Overall  Performance %', over_stock_overall_perf)

    df_len = len(df)

    br_pr = len(df[(df[ORIG_PRED_LOW] <= 0) & (df[IP_QTY_EOP_SOH] <= 0) & (df[ACTUAL_OOS_COLUMN_NAME] == 1)])
    br_pr_perc = round((br_pr / df_len) * 100, 2)
    print("BR_PR %: ", br_pr_perc)
    br_pr_value = df[(df[ORIG_PRED_LOW] <= 0) & (df[IP_QTY_EOP_SOH] <= 0) &
                     (df[ACTUAL_OOS_COLUMN_NAME] == 1)][TOTAL_RETAIL].sum().compute()
    print("BR_PR $ Value: ", br_pr_value)

    br_pw = len(df[(df[ORIG_PRED_LOW] > 0) & (df[IP_QTY_EOP_SOH] <= 0) & (df[ACTUAL_OOS_COLUMN_NAME] == 1)])
    br_pw_perc = round((br_pw / df_len) * 100, 2)
    print("BR_PW %: ", br_pw_perc)

    bw_pr = len(df[(df[ORIG_PRED_LOW] <= 0) & (df[IP_QTY_EOP_SOH] > 0) & (df[ACTUAL_OOS_COLUMN_NAME] == 1)])
    bw_pr_perc = round((bw_pr / df_len) * 100, 2)
    print("BW_PR %: ", bw_pr_perc)

    revenue_recovery_due_to_model = df[(df[ORIG_PRED_LOW] <= 0) & (df[IP_QTY_EOP_SOH] > 0) &
                                       (df[ACTUAL_OOS_COLUMN_NAME] == 1)][TOTAL_RETAIL].sum().compute()
    print("Revenue Recovery Due to Model: ", revenue_recovery_due_to_model)
    loss_due_to_model = df[(df[ORIG_PRED_LOW] > 0) & (df[ACTUAL_OOS_COLUMN_NAME] == 1)][TOTAL_RETAIL].sum().compute()
    print("Loss Due to Model: ", loss_due_to_model)
    book_recovery_of_loss_due_to_model = df[(df[ORIG_PRED_LOW] > 0) & (df[ACTUAL_OOS_COLUMN_NAME] == 1) &
                                            (df[IP_QTY_EOP_SOH] <= 0)][TOTAL_RETAIL].sum().compute()
    print("Part of Loss Due to Model that Book Can Recover: ", book_recovery_of_loss_due_to_model)

    metrics = {
        'True Positive Count': tp,
        'False Negative Count': fn,
        'True Negative Count': tn,
        'False Positive Count': fp,
        'Recall %': recall_perc,
        'Precision %': precision_perc,
        'F1 Score': f1_score,
        'Accuracy %': accuracy_perc,
        'Actual Less Than Book Accurate Performance %': under_stock_acc_perf,
        'Actual Less Than Book Overall  Performance %': under_stock_overall_perf,
        'Actual Greater Than Book Accurate  Performance %': over_stock_acc_perf,
        'Actual Greater Than Book Overall  Performance %': over_stock_overall_perf,
        'BR_PR %': br_pr_perc,
        "BR_PW %": br_pw_perc,
        "BW_PR %": bw_pr_perc,
        "BR_PR $ Value": int(br_pr_value),
        "Revenue Recovery Due to Model": int(revenue_recovery_due_to_model),
        "Loss Due to Model": int(loss_due_to_model),
        "Part of Loss Due to Model that Book Can Recover": int(book_recovery_of_loss_due_to_model)
    }

    return metrics


def upload_metrics_to_gcs(metrics, metrics_output_file_path):
    local_metrics_filename = 'output_metrics.json'
    print(f'Writing {local_metrics_filename} to local')
    with open(local_metrics_filename, 'w') as f:
        json.dump(metrics, f)
    local_dir = '.'
    local_full_path = os.path.join(local_dir, local_metrics_filename)

    gcs_client = storage.Client()
    gcs_path = urlparse(metrics_output_file_path)
    bucket_name = gcs_path.netloc
    gcs_upload_dir = gcs_path.path.lstrip('/') + '/output_metrics.json'
    bucket = gcs_client.bucket(bucket_name=bucket_name)
    blob = bucket.blob(gcs_upload_dir)
    blob.upload_from_filename(local_full_path)
    print("Successfully uploaded metrics to GCS")


def execute(actual_file_path: str, inference_output_file_path: str, create_single_prediction_csv: str,
            single_prediction_csv_output_path: str, metrics_output_file_path: str,
            model_performance_report_output_file_path: str, local_dask_flag: str, dask_address: str,
            dask_connection_timeout: int, num_workers_local_cluster: int, num_threads_per_worker: int,
            memory_limit_local_worker: str):
    if local_dask_flag == 'Y':
        dask_client = get_local_dask_cluster(num_workers_local_cluster, num_threads_per_worker,
                                             memory_limit_local_worker)
    else:
        dask_client = get_remote_dask_client(dask_address, dask_connection_timeout)

    actual_dd = read_data(actual_file_path, dask_client)
    actual_dd[DATE] = dd.to_datetime(actual_dd[DATE], '%Y-%m-%d', utc=True)
    full_to_read_predictions = (inference_output_file_path + "/" + TEMP_FOLDER_NAME + "/*/" + CONCATENATED_FOLDER_NAME +
                                "/*" + PART_EXTENSION)
    inference_dd = read_data(full_to_read_predictions, dask_client)
    inference_dd[DATE] = dd.to_datetime(inference_dd[DATE], '%Y-%m-%d', utc=True)
    merged_dd = actual_dd.merge(inference_dd, on=[SKU, LOCATION, DATE], how='inner')
    metrics = calculate_metrics(merged_dd)
    upload_metrics_to_gcs(metrics, metrics_output_file_path)
    merged_dd.repartition(1).to_csv(model_performance_report_output_file_path + '/model-performance-report-*.csv',
                                    index=False)
    if create_single_prediction_csv == 'Y':
        inference_dd.repartition(1).to_csv(single_prediction_csv_output_path + '/prediction-*.csv', index=False)


def main(args=None):
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Running Inference Metrics")
    parser.add_argument(
        '--actual_file_path',
        dest='actual_file_path',
        type=str,
        required=True,
        help='File where actual OOS outcomes are recorded')
    parser.add_argument(
        '--inference_output_file_path',
        dest='inference_output_file_path',
        type=str,
        required=True,
        help='OOS inference output file location')
    parser.add_argument(
        '--create_single_prediction_csv',
        dest='create_single_prediction_csv',
        type=str,
        choices={'Y', 'N'},
        required=True,
        help='Whether to create a single CSV file which includes predictions for all the location groups')
    parser.add_argument(
        '--single_prediction_csv_output_path',
        dest='single_prediction_csv_output_path',
        type=str,
        default=None,
        required=False,
        help='Location to generate the single predictions CSV file')
    parser.add_argument(
        '--metrics_output_file_path',
        dest='metrics_output_file_path',
        type=str,
        required=True,
        help='Metrics output file path')
    parser.add_argument(
        '--model_performance_report_output_file_path',
        dest='model_performance_report_output_file_path',
        type=str,
        required=True,
        help='Model performance report output file location')
    parser.add_argument(
        '--local_dask_flag',
        dest='local_dask_flag',
        type=str,
        choices={'Y', 'N'},
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

    if args.local_dask_flag == 'Y':
        if (args.num_workers_local_cluster == 0) or (args.num_threads_per_worker == 0) or \
                (args.memory_limit_local_worker is None):
            raise ValueError("num_workers_local_cluster, num_threads_per_worker & memory_limit_local_worker need to "
                             "have valid values for a local dask cluster")
    else:
        if (args.dask_address is None) or (args.dask_connection_timeout == -1):
            raise ValueError("dask_address & dask_connection_timeout need to have valid values for remote dask cluster")

    if args.create_single_prediction_csv == 'Y' and args.single_prediction_csv_output_path is None:
        raise ValueError("single_prediction_csv_output_path is a required param if create_single_prediction_csv=Y")

    execute(args.actual_file_path, args.inference_output_file_path, args.create_single_prediction_csv,
            args.single_prediction_csv_output_path, args.metrics_output_file_path,
            args.model_performance_report_output_file_path, args.local_dask_flag, args.dask_address,
            args.dask_connection_timeout, args.num_workers_local_cluster, args.num_threads_per_worker,
            args.memory_limit_local_worker)
    print("<-----------Inference Metrics Component Successful----------->")
    print('Total Time Taken', time.time() - start_time, 'Seconds')


if __name__ == '__main__':
    main()
