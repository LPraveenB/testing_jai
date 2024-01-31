import argparse
import time

from dask import dataframe as dd
from google.cloud import storage

import utils
from constant import *


def execute(source_path_model_s1, source_path_model_s2, source_path_model_s3, source_path_model_s4,
            decision_threshold_step_1, decision_threshold_step_2, input_base_path, output_base_path, load_date,
            local_dask_flag, dask_address, dask_connection_timeout, num_workers_local_cluster, num_threads_per_worker,
            memory_limit_local_worker):
    dask_client = utils.get_dask_client(local_dask_flag=local_dask_flag,
                                        num_workers_local_cluster=num_workers_local_cluster,
                                        num_threads_per_worker=num_threads_per_worker,
                                        memory_limit_local_worker=memory_limit_local_worker,
                                        dask_address=dask_address, dask_connection_timeout=dask_connection_timeout)

    gcs_client = storage.Client()
    loaded_model_s1, feature_names_s1 = utils.download_and_load_model(gcs_client, STEP_1, source_path_model_s1)
    loaded_model_s2, feature_names_s2 = utils.download_and_load_model(gcs_client, STEP_2, source_path_model_s2)
    loaded_model_s3, feature_names_s3 = utils.download_and_load_model(gcs_client, STEP_3, source_path_model_s3)
    loaded_model_s4, feature_names_s4 = utils.download_and_load_model(gcs_client, STEP_4, source_path_model_s4)

    input_path = input_base_path + "/" + LOAD_DATE_FOLDER_PREFIX + load_date + "/*" + PARQUET_FILE_EXTENSION
    input_dd = utils.read_input_data(dd, input_path, dask_client)

    final_prediction_dd, step_1_dd, step_2_dd, step_3_dd, step_4_dd = utils.run_predict(input_dd, dask_client,
                                                                                        loaded_model_s1,
                                                                                        feature_names_s1,
                                                                                        loaded_model_s2,
                                                                                        feature_names_s2,
                                                                                        loaded_model_s3,
                                                                                        feature_names_s3,
                                                                                        loaded_model_s4,
                                                                                        feature_names_s4,
                                                                                        decision_threshold_step_1,
                                                                                        decision_threshold_step_2)

    output_path = output_base_path + "/" + LOAD_DATE_FOLDER_PREFIX + load_date + "/" + FINAL_PREDICTION
    final_prediction_dd_out = final_prediction_dd[REQ_COLUMNS_FINAL_PREDICTION]

    utils.dd_to_parquet(final_prediction_dd_out, output_path, FINAL_PREDICTION)


def main(args=None):
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Running Predict for Audit Data Component")
    parser = utils.add_common_arguments(parser)
    parser.add_argument(
        '--decision_threshold_step_1',
        dest='decision_threshold_step_1',
        type=float,
        required=False,
        default=0.4,
        help='Decision threshold for Binary OOS prediction')
    parser.add_argument(
        '--decision_threshold_step_2',
        dest='decision_threshold_step_2',
        type=float,
        required=False,
        default=0.4,
        help='Decision threshold for Binary H/L prediction')
    parser.add_argument(
        '--load_date',
        dest='load_date',
        type=str,
        required=True,
        help='UTC load date in ISO format')

    args = parser.parse_args(args)
    print("args:")
    print(args)

    utils.validate_dask_cluster_arguments(args)

    execute(args.source_path_model_s1, args.source_path_model_s2, args.source_path_model_s3, args.source_path_model_s4,
            args.decision_threshold_step_1, args.decision_threshold_step_2, args.input_base_path, args.output_base_path,
            args.load_date, args.local_dask_flag, args.dask_address, args.dask_connection_timeout,
            args.num_workers_local_cluster, args.num_threads_per_worker, args.memory_limit_local_worker)
    print("<-----------Test Production Models  Component Successful----------->")
    print('Total Time Taken', time.time() - start_time, 'Seconds')


if __name__ == '__main__':
    main()