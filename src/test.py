import argparse
import json
import pickle
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json
from notebooks.utilities import (dict_distance_key,
                                 key_based_accuracy,
                                 ensure_folder_exists,
                                 generate_uuid_prefix)


def test(args):
    pretrained_model = DonutModel.from_pretrained(
        args.pretrained_model_name_or_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    result_list = []
    pred_list = []

    # evaluator = JSONParseEvaluator()
    dataset = load_dataset(args.dataset_name_or_path, split=args.split)

    for _, sample in tqdm(enumerate(dataset), total=len(dataset)):

        file_path = sample['image'].filename
        file_name = f"{file_path[file_path.rfind('/')+1: ]}"

        ground_truth = json.loads(sample["ground_truth"])['gt_parse']

        output = pretrained_model.inference(image=sample["image"], prompt="<s_>")[
            "predictions"][0]
        output['image_name'] = file_name

        dict_ = dict_distance_key(ground_truth, output, file_name)
        dict_['predictions'] = output
        dict_['ground_truth'] = ground_truth

        (dict_['total_accuracy'],
         dict_['date_accuracy'],
         dict_['city_accuracy'],
         dict_['state_accuracy'],
         dict_['zip_accuracy']) = key_based_accuracy(dict_['DATE'], dict_['CITY'], dict_['STATE'], dict_['ZIP'],
                                                     dict_['DATE_LEN'], dict_['CITY_LEN'], dict_['STATE_LEN'], dict_['ZIP_LEN'])

        result_list.append(dict_)
        pred_list.append(output)

    if args.save_path:

        df_result = pd.DataFrame(result_list)
        df_pred = pd.DataFrame(pred_list)

        fla = df_result[df_result.total_accuracy ==
                        100].shape[0]/df_result.shape[0]

        result_dict = {}
        result_dict['split'] = args.split
        result_dict['data'] = args.dataset_name_or_path
        result_dict['model_name'] = args.pretrained_model_name_or_path
        result_dict['fla'] = np.round(fla, 4)*100
        result_dict['CBA TOTAL FULL DATASET'] = np.round(
            df_result['total_accuracy'].mean(), 4)
        result_dict['CBA DATE FULL DATASET'] = np.round(
            df_result['date_accuracy'].mean(), 4)
        result_dict['CBA CITY FULL DATASET'] = np.round(
            df_result['city_accuracy'].mean(), 4)
        result_dict['CBA STATE FULL DATASET'] = np.round(
            df_result['state_accuracy'].mean(), 4)
        result_dict['CBA ZIP FULL DATASET'] = np.round(
            df_result['zip_accuracy'].mean(), 4)

        with open(f'{args.save_path}/{args.split}_summary.json', 'w') as f:
            json.dump(result_dict, f, indent=4)

        df_result.to_csv(
            f'{args.save_path}/{args.split}_result.csv', index=False)
        df_pred.to_csv(
            f'{args.save_path}/{args.split}_pred.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dataset_name_or_path", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args, left_argv = parser.parse_known_args()

    if args.save_path:
        prefix = generate_uuid_prefix()
        args.save_path = f'{args.save_path}/{prefix}'
        ensure_folder_exists(args.save_path)

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)
        print(
            f'this is the basename: {os.path.basename(args.dataset_name_or_path)}')
        print(f'this is the task_name: {args.task_name}')
        print('*'*100)

    predictions = test(args)
