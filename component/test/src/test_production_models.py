import argparse
import time
import utils

from dask import dataframe as dd
from google.cloud import storage
from constant import *

"""
This component generates the predictions using the 4 release-01 models
Also, final OOS predictions are generated combining the above predictions

It makes use of 4 models for the same in the following manner:
Model 1 -> Binary classifier -> Predicts OOS
Model 2 -> Binary classifier -> Predicts whether inventory lower than the book
Model 3 -> Regression -> Predicts the inventory range for items whose inventory is lesser than book
Model 4 -> Regression -> Predicts inventory range for items whose inventory is greater than book

The output is generated in parquet format at the output location
It can accept a list of location groups as input and processes each location group sequentially
It can run on a local dask cluster as well as a remote dask cluster
"""


def execute(source_path_model_s1: str, source_path_model_s2: str, source_path_model_s3: str, source_path_model_s4: str,
            decision_threshold_step_1: float, decision_threshold_step_2: float, input_base_path: str, output_path: str,
            location_group_list: list, local_dask_flag: str, dask_address: str, dask_connection_timeout: int,
            num_workers_local_cluster: int, num_threads_per_worker: int, memory_limit_local_worker: str):
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

    for location_group in location_group_list:
        print("Processing LOCATION_GROUP=" + location_group)
        input_path = (input_base_path + '/' + LOCATION_GROUP_FOLDER_PREFIX + location_group + '/'
                      + LOAD_DATE_FOLDER_PREFIX + '*/' + DATA_SPLIT_TEST_DATA_FOLDER + '/*' + PARQUET_FILE_EXTENSION)
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
        output_path_step_1 = output_path + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group + "/" + STEP_1
        output_path_step_2 = output_path + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group + "/" + STEP_2
        output_path_step_3 = output_path + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group + "/" + STEP_3
        output_path_step_4 = output_path + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group + "/" + STEP_4
        output_path_final_prediction = (output_path + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group + "/" +
                                        FINAL_PREDICTION)
        step_1_dd_out = step_1_dd[REQ_COLUMNS_OOS_CLASSIFIER]
        step_2_dd_out = step_2_dd[REQ_COLUMNS_HL_CLASSIFIER]
        step_3_dd_out = step_3_dd[REQ_COLUMNS_REGRESSION_LB]
        step_4_dd_out = step_4_dd[REQ_COLUMNS_REGRESSION_HB]
        final_prediction_dd_out = final_prediction_dd[REQ_COLUMNS_FINAL_PREDICTION]

        utils.dd_to_csv(step_1_dd_out, output_path_step_1, STEP_1)
        utils.dd_to_csv(step_2_dd_out, output_path_step_2, STEP_2)
        utils.dd_to_csv(step_3_dd_out, output_path_step_3, STEP_3)
        utils.dd_to_csv(step_4_dd_out, output_path_step_4, STEP_4)
        utils.dd_to_csv(final_prediction_dd_out, output_path_final_prediction, FINAL_PREDICTION)

        print("Finished processing LOCATION_GROUP=" + location_group)


def main(args=None):
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Running Test Production Models Component")
    parser = utils.add_common_test_component_arguments(parser)

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

    args = parser.parse_args(args)
    print("args:")
    print(args)

    location_groups = args.location_group_list.split(',')
    utils.validate_dask_cluster_arguments(args)

    execute(args.source_path_model_s1, args.source_path_model_s2, args.source_path_model_s3, args.source_path_model_s4,
            args.decision_threshold_step_1, args.decision_threshold_step_2, args.input_base_path, args.output_base_path,
            location_groups, args.local_dask_flag, args.dask_address, args.dask_connection_timeout,
            args.num_workers_local_cluster, args.num_threads_per_worker, args.memory_limit_local_worker)
    print("<-----------Test Production Models  Component Successful----------->")
    print('Total Time Taken', time.time() - start_time, 'Seconds')


if __name__ == '__main__':
    main()