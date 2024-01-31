import collections
import json
import os
import pprint
from urllib.parse import urlparse

import dask
from dask.distributed import Client
from distributed import LocalCluster
from google.cloud import storage
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.metrics import roc_auc_score, confusion_matrix

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


def path_exists(input_path):
    gcs_path = urlparse(input_path, allow_fragments=False)
    bucket = gcs_path.netloc
    path = gcs_path.path.lstrip('/')
    return storage.Client().bucket(bucket_name=bucket).blob(blob_name=path).exists()


def read_dataframe(dd_df, input_path, dask_client):
    print('Reading from', input_path)
    input_data_dd = dd_df.read_csv(input_path)
    input_data_dd = dask_client.persist(input_data_dd)
    return input_data_dd


def upload_metrics_to_gcs(local_metrics_filename, metrics, metrics_output_file_path):
    print(f'Writing {local_metrics_filename} to local')
    with open(local_metrics_filename, 'w') as f:
        json.dump(metrics, f)
    local_dir = '.'
    local_full_path = os.path.join(local_dir, local_metrics_filename)

    gcs_client = storage.Client()
    gcs_path = urlparse(metrics_output_file_path)
    bucket_name = gcs_path.netloc
    gcs_upload_dir = gcs_path.path.lstrip('/')
    bucket = gcs_client.bucket(bucket_name=bucket_name)
    blob = bucket.blob(gcs_upload_dir)
    blob.upload_from_filename(local_full_path)
    print("Successfully uploaded metrics to GCS")


def get_basic_binary_metrics(metrics_name, tp, fn, tn, fp):
    zero_to_one_ratio = 0.0
    if (tp + fn) > 0:
        zero_to_one_ratio = round((tn + fp) / (tp + fn), 3)
    count_metrics = collections.OrderedDict({
        f'{metrics_name}_0s': tn + fp,
        f'{metrics_name}_1s': tp + fn,
        f'{metrics_name}_0_to_1_ratio': zero_to_one_ratio,
        f'{metrics_name}_tp': tp,
        f'{metrics_name}_fn': fn,
        f'{metrics_name}_tn': tn,
        f'{metrics_name}_fp': fp,
    })
    print(f'Basic metrics for: {metrics_name}')
    pprint.pprint(count_metrics)
    print(f'Basic metrics for: {metrics_name}. DONE')

    tp_perc = 0
    fn_perc = 0
    recall = 0
    if (tp + fn) > 0:
        tp_perc = round((tp / (tp + fn)) * 100, 2)
        fn_perc = round((fn / (tp + fn)) * 100, 2)
        recall = round((tp / (tp + fn)) * 100, 2)

    tn_perc = 0
    fp_perc = 0
    if (tn + fp) > 0:
        tn_perc = round((tn / (tn + fp)) * 100, 2)
        fp_perc = round((fp / (tn + fp)) * 100, 2)

    precision = 0
    if (tp + fp) > 0:
        precision = round((tp / (tp + fp)) * 100, 2)

    f1_score = 0
    if (precision + recall) > 0:
        f1_score = round(
            2 * ((precision * recall) / (precision + recall)), 2)

    accuracy = 0
    if (tp + tn + fp + fn) > 0:
        accuracy = round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)

    result = {
        TP_PERCENT: tp_perc,
        FN_PERCENT: fn_perc,
        TN_PERCENT: tn_perc,
        FP_PERCENT: fp_perc,
        RECALL: recall,
        PRECISION: precision,
        F1_SCORE: f1_score,
        ACCURACY: accuracy,
    }
    return result


def get_basic_regression_distribution(s):
    return s.min(), s.mean(), s.max(), s.std()


def calculate_technical_metrics(step_name, dd_df, target, prediction_column, decision_threshold=0.4, curday_soh=None):
    round_digit = 3
    print(f'Getting metrics for: {step_name}')
    y_true = dd_df[target].compute()
    y_pred_score = dd_df[prediction_column].compute()

    result = None
    if step_name in [STEP_1, STEP_2]:
        y_pred = y_pred_score.apply(lambda x: 1 if x > decision_threshold else 0)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]).ravel()
        basic_binary_metrics = get_basic_binary_metrics(step_name, tp, fn, tn, fp)

        roc_auc = (
            round(roc_auc_score(y_true, y_pred_score), round_digit)
            if len(y_true.unique()) > 1 else 0.0)

        result = collections.OrderedDict({
            f'{step_name}_tp_%': basic_binary_metrics[TP_PERCENT],
            f'{step_name}_fn_%': basic_binary_metrics[FN_PERCENT],
            f'{step_name}_tn_%': basic_binary_metrics[TN_PERCENT],
            f'{step_name}_fp_%': basic_binary_metrics[FP_PERCENT],
            f'{step_name}_accuracy': basic_binary_metrics[ACCURACY],
            f'{step_name}_roc_auc': roc_auc,
            f'{step_name}_precision': basic_binary_metrics[PRECISION],
            f'{step_name}_recall': basic_binary_metrics[RECALL],
            f'{step_name}_f1_score': basic_binary_metrics[F1_SCORE],
        })

    elif step_name in [STEP_3, STEP_4]:
        if curday_soh is not None:
            print('Getting original target values for regression')
            y_true = y_true * curday_soh
        print('{0} TARGET distribution (min, mean, max, std): {1}'.format(step_name,
                                                                          get_basic_regression_distribution(y_true)))
        print('{0} PREDICTION distribution (min, mean, max, std): {1}'.format(step_name,
                                                                              get_basic_regression_distribution(y_pred_score)))

        mean_squared_err = round(
            mean_squared_error(y_true, y_pred_score), round_digit)
        mean_absolute_err = round(
            mean_absolute_error(y_true, y_pred_score), round_digit)
        median_absolute_err = round(
            median_absolute_error(y_true, y_pred_score), round_digit)

        r2 = r2_score(y_true, y_pred_score)
        r2 = round(r2, round_digit)
        result = {
            f'{step_name}_r2': r2,
            f'{step_name}_mean_squared_error': mean_squared_err,
            f'{step_name}_mean_absolute_error': mean_absolute_err,
            f'{step_name}_median_absolute_error': median_absolute_err,
        }
    pprint.pprint(result)
    return result


def calculate_overall_metrics(df):
    stock_threshold = 0
    tp = len(df[(df[REV_IP_QTY_EOP_SOH] <= stock_threshold) & (df[ORIG_PRED_LOW] <= stock_threshold)])
    fn = len(df[(df[REV_IP_QTY_EOP_SOH] <= stock_threshold) & (df[ORIG_PRED_LOW] > stock_threshold)])
    tn = len(df[(df[REV_IP_QTY_EOP_SOH] > stock_threshold) & (df[ORIG_PRED_LOW] > stock_threshold)])
    fp = len(df[(df[REV_IP_QTY_EOP_SOH] > stock_threshold) & (df[ORIG_PRED_LOW] <= stock_threshold)])
    print("TP Count: ", tp)
    print("TN Count: ", tn)
    print("FP Count: ", fp)
    print("FN Count: ", fn)

    basic_binary_metrics = get_basic_binary_metrics(FINAL_PREDICTION, tp, fn, tn, fp)

    df_low = df[df[REV_IP_QTY_EOP_SOH] <= df[CURDAY_IP_QTY_EOP_SOH]]
    df_low_len = len(df_low)
    print("df_low_len: ", df_low_len)

    under_stock_acc = len(df_low[(df_low[ORIG_PRED_LOW] <= df_low[REV_IP_QTY_EOP_SOH]) &
                                 (df_low[ORIG_PRED_HIGH] >= df_low[REV_IP_QTY_EOP_SOH])])
    under_stock_acc_perf = 100
    if df_low_len > 0:
        under_stock_acc_perf = round((under_stock_acc / df_low_len) * 100, 2)
    print("Actual Less Than Book Accurate Performance %", under_stock_acc_perf)

    under_stock_overall = len(df_low[(df_low[ORIG_PRED_LOW] <= df_low[CURDAY_IP_QTY_EOP_SOH])])
    under_stock_overall_perf = 100
    if df_low_len > 0:
        under_stock_overall_perf = round((under_stock_overall / df_low_len) * 100, 2)
    print("Actual Less Than Book Overall  Performance %", under_stock_overall_perf)

    df_high = df[df[REV_IP_QTY_EOP_SOH] > df[CURDAY_IP_QTY_EOP_SOH]]
    df_high_len = len(df_high)
    print("df_high_len: ", df_high_len)

    over_stock_acc = len(df_high[(df_high[ORIG_PRED_LOW] <= df_high[REV_IP_QTY_EOP_SOH]) &
                                 (df_high[ORIG_PRED_HIGH] >= df_high[REV_IP_QTY_EOP_SOH])])
    over_stock_acc_perf = 100
    if df_high_len > 0:
        over_stock_acc_perf = round((over_stock_acc / df_high_len) * 100, 2)
    print('Actual Greater Than Book Accurate  Performance %', over_stock_acc_perf)

    over_stock_overall = len(df_high[(df_high[ORIG_PRED_HIGH] > df_high[CURDAY_IP_QTY_EOP_SOH])])
    over_stock_overall_perf = 100
    if df_high_len > 0:
        over_stock_overall_perf = round((over_stock_overall / df_high_len) * 100, 2)
    print('Actual Greater Than Book Overall  Performance %', over_stock_overall_perf)

    df_len = len(df)

    br_pr = len(df[(df[ORIG_PRED_LOW] <= stock_threshold) & (df[IP_QTY_EOP_SOH] <= stock_threshold) &
                   (df[TARGET_CLASSIFICATION_BINARY] == 1)])
    br_pr_perc = round((br_pr / df_len) * 100, 2)
    print("BR_PR %: ", br_pr_perc)
    br_pr_value = df[(df[ORIG_PRED_LOW] <= stock_threshold) & (df[IP_QTY_EOP_SOH] <= stock_threshold) &
                     (df[TARGET_CLASSIFICATION_BINARY] == 1)][TOTAL_RETAIL].sum().compute()
    print("BR_PR $ Value: ", br_pr_value)

    br_pw = len(df[(df[ORIG_PRED_LOW] > stock_threshold) & (df[IP_QTY_EOP_SOH] <= stock_threshold) &
                   (df[TARGET_CLASSIFICATION_BINARY] == 1)])
    br_pw_perc = round((br_pw / df_len) * 100, 2)
    print("BR_PW %: ", br_pw_perc)

    bw_pr = len(df[(df[ORIG_PRED_LOW] <= stock_threshold) & (df[IP_QTY_EOP_SOH] > stock_threshold) &
                   (df[TARGET_CLASSIFICATION_BINARY] == 1)])
    bw_pr_perc = round((bw_pr / df_len) * 100, 2)
    print("BW_PR %: ", bw_pr_perc)

    revenue_recovery_due_to_model = df[(df[ORIG_PRED_LOW] <= stock_threshold) & (df[IP_QTY_EOP_SOH] > stock_threshold) &
                                       (df[TARGET_CLASSIFICATION_BINARY] == 1)][TOTAL_RETAIL].sum().compute()
    print("Revenue Recovery Due to Model: ", revenue_recovery_due_to_model)
    loss_due_to_model = df[(df[ORIG_PRED_LOW] > stock_threshold) &
                           (df[TARGET_CLASSIFICATION_BINARY] == 1)][TOTAL_RETAIL].sum().compute()
    print("Loss Due to Model: ", loss_due_to_model)
    book_recovery_of_loss_due_to_model = df[(df[ORIG_PRED_LOW] > stock_threshold) &
                                            (df[TARGET_CLASSIFICATION_BINARY] == 1) &
                                            (df[IP_QTY_EOP_SOH] <= stock_threshold)][TOTAL_RETAIL].sum().compute()
    print("Part of Loss Due to Model that Book Can Recover: ", book_recovery_of_loss_due_to_model)

    result = {
        "TP Count": tp,
        "TN Count": tn,
        "FP Count": fp,
        "FN Count": fn,
        "TP %": basic_binary_metrics[TP_PERCENT],
        "TN %": basic_binary_metrics[TN_PERCENT],
        "FP %": basic_binary_metrics[FP_PERCENT],
        "FN %": basic_binary_metrics[FN_PERCENT],
        "Precision %": basic_binary_metrics[PRECISION],
        "Recall %": basic_binary_metrics[RECALL],
        "F1 Score": basic_binary_metrics[F1_SCORE],
        "Accuracy %": basic_binary_metrics[ACCURACY],
        "Actual Less Than Book Accurate Performance %": under_stock_acc_perf,
        "Actual Less Than Book Overall  Performance %": under_stock_overall_perf,
        "Actual Greater Than Book Accurate  Performance %": over_stock_acc_perf,
        "Actual Greater Than Book Overall  Performance %": over_stock_overall_perf,
        "BR_PR %": br_pr_perc,
        "BR_PW %": br_pw_perc,
        "BW_PR %": bw_pr_perc,
        "BR_PR $ Value": int(br_pr_value),
        "Revenue Recovery Due to Model": int(revenue_recovery_due_to_model),
        "Loss Due to Model": int(loss_due_to_model),
        "Part of Loss Due to Model that Book Can Recover": int(book_recovery_of_loss_due_to_model)
    }
    return result


def add_scoring_component_args(parser):
    parser.add_argument(
        '--predictions_base_path',
        dest='predictions_base_path',
        type=str,
        required=True,
        help='Directory to read the predictions made by the models')
    parser.add_argument(
        '--score_base_path',
        dest='score_base_path',
        type=str,
        required=True,
        help='Base path to the model scores')
    parser.add_argument(
        '--decision_threshold_step_1',
        dest='decision_threshold_step_1',
        type=float,
        required=False,
        default=0.4,
        help='Decision threshold for Binary OOS prediction for the model')
    parser.add_argument(
        '--decision_threshold_step_2',
        dest='decision_threshold_step_2',
        type=float,
        required=False,
        default=0.4,
        help='Decision threshold for Binary H/L prediction for the model')
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
