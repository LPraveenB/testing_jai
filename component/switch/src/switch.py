import argparse
import time

import pandas as pd
from google.cloud import storage


OVERALL_SCORE_COLUMN_NAME = "OVERALL_SCORE"


def switch_models(client, current_model_bucket, current_model_path, prod_model_bucket, prod_model_path, run_id):
    current_bucket = client.bucket(current_model_bucket)
    prod_bucket = client.bucket(prod_model_bucket)
    current_blob = current_bucket.get_blob(current_model_path)
    current_bucket.copy_blob(current_blob, prod_bucket, prod_model_path)
    print("Switched models, current production model run_id = ", run_id)


def read_score(model_score_bucket, model_score_path):
    score_path = "gs://" + model_score_bucket + "/" + model_score_path
    score_df = pd.read_csv(score_path)
    overall_score = score_df[OVERALL_SCORE_COLUMN_NAME].iloc[0]
    print('OVERALL_SCORE Achieved', overall_score)
    return overall_score


def execute(comparison_needed, current_model_score_bucket, current_model_score_path, prod_model_score_bucket,
            prod_model_score_path, current_model_bucket, current_model_path, prod_model_bucket, prod_model_path, run_id):
    client = storage.Client()
    if comparison_needed == 'Y':
        print("Performing score comparison between current and prod models")
        print("Reading Current Model's score:")
        current_model_score = read_score(current_model_score_bucket, current_model_score_path)
        print("Reading Prod Model's score:")
        prod_model_score = read_score(prod_model_score_bucket, prod_model_score_path)
        if current_model_score > prod_model_score:
            switch_models(client, current_model_bucket, current_model_path, prod_model_bucket, prod_model_path, run_id)
    else:
        switch_models(client, current_model_bucket, current_model_path, prod_model_bucket, prod_model_path, run_id)


def main(args=None):
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Running Switch")
    parser.add_argument(
        '--comparison_needed',
        dest='comparison_needed',
        type=str,
        choices={'Y', 'N'},
        required=False,
        default='Y',
        help='comparison_needed')
    parser.add_argument(
        '--current_model_score_bucket',
        dest='current_model_score_bucket',
        type=str,
        required=True,
        help='current_model_score_bucket')
    parser.add_argument(
        '--current_model_score_path',
        dest='current_model_score_path',
        type=str,
        required=True,
        help='current_model_score_path')
    parser.add_argument(
        '--prod_model_score_bucket',
        dest='prod_model_score_bucket',
        type=str,
        required=True,
        help='prod_model_score_bucket')
    parser.add_argument(
        '--prod_model_score_path',
        dest='prod_model_score_path',
        type=str,
        required=True,
        help='prod_model_score_path')
    parser.add_argument(
        '--current_model_bucket',
        dest='current_model_bucket',
        type=str,
        required=True,
        help='current_model_bucket')
    parser.add_argument(
        '--current_model_path',
        dest='current_model_path',
        type=str,
        required=True,
        help='current_model_path')
    parser.add_argument(
        '--prod_model_bucket',
        dest='prod_model_bucket',
        type=str,
        required=True,
        help='prod_model_bucket')
    parser.add_argument(
        '--prod_model_path',
        dest='prod_model_path',
        type=str,
        required=True,
        help='prod_model_path')
    parser.add_argument(
        '--run_id',
        dest='run_id',
        type=str,
        required=True,
        help='run_id')

    args = parser.parse_args(args)
    print("args:")
    print(args)
    
    execute(args.comparison_needed, args.current_model_score_bucket, args.current_model_score_path,
            args.prod_model_score_bucket, args.prod_model_score_path, args.current_model_bucket,
            args.current_model_path, args.prod_model_bucket, args.prod_model_path, args.run_id)
    print('Total Time Taken', time.time() - start_time, 'Seconds')
    

if __name__ == '__main__':
    main()
