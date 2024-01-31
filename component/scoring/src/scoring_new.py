import time
import argparse
import utils

from dask import dataframe as dd
from constant import *


def read_predictions_into_dataframe(dd_df, predictions_base_path, dask_client):
    s1_dd = utils.read_dataframe(dd_df, predictions_base_path + "/*/" + STEP_1 + '/*' +
                                 CSV_FILE_EXTENSION, dask_client)
    s2_dd = utils.read_dataframe(dd_df, predictions_base_path + "/*/" + STEP_2 + '/*' +
                                 CSV_FILE_EXTENSION, dask_client)
    s3_dd = utils.read_dataframe(dd_df, predictions_base_path + "/*/" + STEP_3 + '/*' +
                                 CSV_FILE_EXTENSION, dask_client)
    s4_dd = utils.read_dataframe(dd_df, predictions_base_path + "/*/" + STEP_4 + '/*' +
                                 CSV_FILE_EXTENSION, dask_client)
    return s1_dd, s2_dd, s3_dd, s4_dd


def execute(predictions_base_path: str, score_base_path: str, decision_threshold_step_1: float,
            decision_threshold_step_2: float, local_dask_flag: str, dask_address: str, dask_connection_timeout: int,
            num_workers_local_cluster: int, num_threads_per_worker: int, memory_limit_local_worker: str):
    dask_client = utils.get_dask_client(local_dask_flag=local_dask_flag, dask_address=dask_address,
                                        dask_connection_timeout=dask_connection_timeout,
                                        num_workers_local_cluster=num_workers_local_cluster,
                                        num_threads_per_worker=num_threads_per_worker,
                                        memory_limit_local_worker=memory_limit_local_worker)

    s1_dd, s2_dd, s3_dd, s4_dd = read_predictions_into_dataframe(dd, predictions_base_path, dask_client)

    oos_dd = s1_dd[s1_dd[BINARY_PRED] > decision_threshold_step_1]
    hl_dd = s1_dd[s1_dd[BINARY_PRED] <= decision_threshold_step_1].merge(s2_dd, on=[SKU, LOCATION, DATE], how='inner')
    ls_dd = hl_dd[hl_dd[HL_ORIG_PRED] > decision_threshold_step_2].merge(s3_dd, on=[SKU, LOCATION, DATE], how='inner')
    hs_dd = hl_dd[hl_dd[HL_ORIG_PRED] <= decision_threshold_step_2].merge(s4_dd, on=[SKU, LOCATION, DATE], how='inner')

    tech_metrics = {}

    oos_score = utils.calculate_technical_metrics(STEP_1, oos_dd, TARGET_CLASSIFICATION_BINARY,
                                                  BINARY_PRED, decision_threshold_step_1)
    tech_metrics.update(oos_score)
    binary_hl_score = utils.calculate_technical_metrics(STEP_2, hl_dd,
                                                        TARGET_CLASSIFICATION_BINARY_HL, HL_ORIG_PRED,
                                                        decision_threshold_step_2)
    tech_metrics.update(binary_hl_score)
    regression_ls_score = utils.calculate_technical_metrics(STEP_3, ls_dd, TARGET_REGRESSION, ORIG_PRED)
    tech_metrics.update(regression_ls_score)
    regression_hs_score = utils.calculate_technical_metrics(STEP_4, hs_dd, TARGET_REGRESSION_H, ORIG_PRED)
    tech_metrics.update(regression_hs_score)

    utils.upload_metrics_to_gcs(local_metrics_filename="individual_tech_metrics_new.json", metrics=str(tech_metrics),
                                metrics_output_file_path=score_base_path + "/individual_tech_metrics_new.json")

    final_predictions_dd = utils.read_dataframe(dd, predictions_base_path + "/" + FINAL_PREDICTION + '/*' +
                                                CSV_FILE_EXTENSION, dask_client)
    overall_metrics = utils.calculate_overall_metrics(final_predictions_dd)
    utils.upload_metrics_to_gcs(local_metrics_filename="overall_metrics_new.json", metrics=overall_metrics,
                                metrics_output_file_path=score_base_path + "/overall_metrics_new.json")


def main(args=None):
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Running Scoring New Component")
    parser = utils.add_scoring_component_args(parser)

    args = parser.parse_args(args)
    print("args:")
    print(args)

    utils.validate_dask_cluster_arguments(args)

    execute(args.predictions_base_path, args.score_base_path, args.decision_threshold_step_1,
            args.decision_threshold_step_2, args.local_dask_flag, args.dask_address, args.dask_connection_timeout,
            args.num_workers_local_cluster, args.num_threads_per_worker, args.memory_limit_local_worker)

    print("<-----------Scoring New Component Successful----------->")
    print('Total Time Taken', time.time() - start_time, 'Seconds')


if __name__ == '__main__':
    main()
