import argparse
import time
import utils
import xgboost as xgb

from dask import dataframe as dd
from google.cloud import storage
from constant import *


def predict_individual_model(dask_client, input_dd, loaded_model, feature_names, prediction_field):
    result_dd = input_dd.copy()
    result_dd = utils.prepare_for_inference(result_dd, feature_names)
    result_dd[prediction_field] = xgb.dask.predict(dask_client, loaded_model.get_booster(), result_dd[feature_names])
    return result_dd


def run_predict(input_data_dd, dask_client, loaded_model_oos, feature_names_oos, loaded_model_hl, feature_names_hl,
                loaded_model_ls, feature_names_ls, loaded_model_hs, feature_names_hs):
    print('Using Binary OOS classifier')
    oos_dd = predict_individual_model(dask_client=dask_client, input_dd=input_data_dd, loaded_model=loaded_model_oos,
                                      feature_names=feature_names_oos, prediction_field=BINARY_PRED)

    print('Using Binary HL classifier')
    hl_dd = predict_individual_model(dask_client=dask_client, input_dd=input_data_dd, loaded_model=loaded_model_hl,
                                     feature_names=feature_names_hl, prediction_field=HL_ORIG_PRED)

    print('Using Regression LS model')
    ls_dd = predict_individual_model(dask_client=dask_client, input_dd=input_data_dd, loaded_model=loaded_model_ls,
                                     feature_names=feature_names_ls, prediction_field=ORIG_PRED)
    ls_dd[ORIG_PRED] = ls_dd[ORIG_PRED] * ls_dd[CURDAY_IP_QTY_EOP_SOH]
    ls_dd = ls_dd.reset_index(drop=True)
    ls_dd = ls_dd.map_partitions(utils.set_low_high_range)

    print('Using Regression LS model')
    hs_dd = predict_individual_model(dask_client=dask_client, input_dd=input_data_dd, loaded_model=loaded_model_hs,
                                     feature_names=feature_names_hs, prediction_field=ORIG_PRED)
    hs_dd = hs_dd.reset_index(drop=True)
    hs_dd = hs_dd.map_partitions(utils.set_low_high_range)

    return oos_dd, hl_dd, ls_dd, hs_dd


def execute(source_path_model_s1: str, source_path_model_s2: str, source_path_model_s3: str, source_path_model_s4: str,
            input_base_path: str, output_path: str, location_group_list: list, local_dask_flag: str, dask_address: str,
            dask_connection_timeout: int, num_workers_local_cluster: int, num_threads_per_worker: int,
            memory_limit_local_worker: str):
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
        
        oos_dd, hl_dd, ls_dd, hs_dd = run_predict(input_dd, dask_client, loaded_model_s1,
                                                  feature_names_s1, loaded_model_s2, feature_names_s2, loaded_model_s3,
                                                  feature_names_s3, loaded_model_s4, feature_names_s4)

        output_path_oos = output_path + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group + "/" + STEP_1
        output_path_hl = output_path + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group + "/" + STEP_2
        output_path_ls = output_path + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group + "/" + STEP_3
        output_path_hs = output_path + "/" + LOCATION_GROUP_FOLDER_PREFIX + location_group + "/" + STEP_4

        oos_dd_out = oos_dd[REQ_COLUMNS_OOS_CLASSIFIER]
        hl_dd_out = hl_dd[REQ_COLUMNS_HL_CLASSIFIER]
        ls_dd_out = ls_dd[REQ_COLUMNS_REGRESSION_LB]
        hs_dd_out = hs_dd[REQ_COLUMNS_REGRESSION_HB]

        utils.dd_to_csv(oos_dd_out, output_path_oos, STEP_1)
        utils.dd_to_csv(hl_dd_out, output_path_hl, STEP_2)
        utils.dd_to_csv(ls_dd_out, output_path_ls, STEP_3)
        utils.dd_to_csv(hs_dd_out, output_path_hs, STEP_4)

        print("Finished processing LOCATION_GROUP=" + location_group)


def main(args=None):
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Running New Production Models Component")
    parser = utils.add_common_test_component_arguments(parser)

    args = parser.parse_args(args)
    print("args:")
    print(args)

    utils.validate_dask_cluster_arguments(args)
    location_groups = args.location_group_list.split(',')

    execute(args.source_path_model_s1, args.source_path_model_s2, args.source_path_model_s3, args.source_path_model_s4,
            args.input_base_path, args.output_base_path, location_groups, args.local_dask_flag,
            args.dask_address, args.dask_connection_timeout, args.num_workers_local_cluster,
            args.num_threads_per_worker, args.memory_limit_local_worker)
    print("<-----------Test New Models  Component Successful----------->")
    print('Total Time Taken', time.time() - start_time, 'Seconds')


if __name__ == '__main__':
    main()
