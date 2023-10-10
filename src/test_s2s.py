"""
Sequence to Sequence
Copyright Sohrab Redjai Sani
MIT License
"""
import argparse
import os
from sconf import Config
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset

from peft import (LoraConfig,
                  get_peft_model,
                  TaskType,
                  PeftModel,
                  PeftConfig)

from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          GenerationConfig,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback,
                          IntervalStrategy)

from notebooks.utilities import (print_number_of_trainable_model_parameters,
                                 init_json,
                                 get_accuracy_dict,
                                 key_based_accuracy,
                                 load_data_to_df,
                                 fix_string_v1,
                                 fix_string_v2,
                                 fix_string_v3,
                                 fix_string_v4,
                                 )


def test(df1,
         df2,
         dataset,
         fm_model_name,
         peft_model_path,
         gt_keys_path,
         result_file_name='result',
         split_name='test',):

    if result_file_name is None:
        result_file_name = 'result'

    ground_truth_list = []
    result_list = []
    error_list = []

    model_name = peft_model_path[peft_model_path.rindex('/')+1:]
    result_save_path = f"dataset/result/nist_form/{model_name}/"

    if peft_model_path:
        os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

    print('Loading ground truth...')
    dataset_ = dataset[split_name]

    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(fm_model_name,
                                                            device_map="auto",
                                                            torch_dtype=torch.bfloat16)

    peft_model = PeftModel.from_pretrained(peft_model_base,
                                           peft_model_path,
                                           torch_dtype=torch.bfloat16,
                                           is_trainable=False)

    tokenizer = AutoTokenizer.from_pretrained(fm_model_name,
                                              device_map="auto",)

    print(print_number_of_trainable_model_parameters(peft_model))

    for _, sample in tqdm(enumerate(dataset_), total=len(dataset_)):
        file_path = sample['image'].filename
        file_name = f"{file_path[file_path.rfind('/')+1: ]}"

        ground_truth = json.loads(sample["ground_truth"])['gt_parse']
        ground_truth['image_name'] = file_name
        ground_truth_list.append(ground_truth)

        df_1 = df1[df1.image_name == file_name]
        df_2 = df2[df2.image_name == file_name]

        # make sure they both JSONS have all the keys
        json1 = init_json(df_1)
        json2 = init_json(df_2)

        prompt = f"""convert the two input JSONs delimited by triple backticks into one OUTPUT_JSON.

            INPUT_JSONs: ```
            JSON1 = {json1}, 
            JSON2 = {json2}```
            
            OUTPUT_JSON = 
            """

        input_ids = tokenizer(
            prompt, return_tensors="pt").input_ids.to('cuda')

        instruct_model_outputs = peft_model.generate(input_ids=input_ids,
                                                     generation_config=GenerationConfig(max_new_tokens=200,
                                                                                        num_beams=2))
        data_str = tokenizer.decode(instruct_model_outputs[0],
                                    skip_special_tokens=True)

        try:
            data_dict = fix_string_v1(data_str)
            data_dict['image_name'] = file_name
            result_list.append(data_dict)

        except:
            try:
                data_dict = fix_string_v2(data_str)
                data_dict['image_name'] = file_name
                result_list.append(data_dict)
            except:
                try:
                    data_dict = fix_string_v3(data_str)
                    data_dict['image_name'] = file_name
                    result_list.append(data_dict)
                except:
                    try:
                        data_dict = fix_string_v4(data_str)
                        data_dict['image_name'] = file_name
                        result_list.append(data_dict)
                    except:
                        error_list.append(file_path)

    df_result = pd.DataFrame(result_list)
    print(df_result.shape)

    df_gt = pd.DataFrame(ground_truth_list)
    print(df_gt.shape)

    dict_list, error_list = get_accuracy_dict(df_result, gt_keys_path)

    df_ = pd.DataFrame(dict_list)
    print(f"dict_list size: {df_.shape}")

    df_['total_accuracy'], df_['date_accuracy'], df_['city_accuracy'], df_['state_accuracy'], df_['zip_accuracy'] = zip(*df_.apply(lambda x: key_based_accuracy(x["DATE"],

                                                                                                                                                                x["CITY"],
                                                                                                                                                                x["STATE"],
                                                                                                                                                                x["ZIP"],

                                                                                                                                                                x["DATE_LEN"],
                                                                                                                                                                x["CITY_LEN"],
                                                                                                                                                                x["STATE_LEN"],
                                                                                                                                                                x["ZIP_LEN"],


                                                                                                                                                                ), axis=1))
    df_result_key = df_.describe()

    fla = df_[df_.total_accuracy == 100].shape[0]/df_.shape[0]
    result_dict = {}
    result_dict['fla'] = np.round(fla, 4)*100

    result_dict['CBA TOTAL DATASET'] = np.round(
        df_result_key['total_accuracy']['mean'], 4)
    result_dict['CBA DATE DATASET'] = np.round(
        df_result_key['date_accuracy']['mean'], 4)
    result_dict['CBA CITY DATASET'] = np.round(
        df_result_key['city_accuracy']['mean'], 4)
    result_dict['CBA STATE DATASET'] = np.round(
        df_result_key['state_accuracy']['mean'], 4)
    result_dict['CBA ZIP DATASET'] = np.round(
        df_result_key['zip_accuracy']['mean'], 4)

    with open(f"{result_save_path}/{result_file_name}__{split_name}.json", "w") as outfile:
        json.dump(result_dict, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args, left_argv = parser.parse_known_args()
    print(args, left_argv)
    config = Config(args.config)
    config.argv_update(left_argv)

    df1 = load_data_to_df(config.df1_path)
    df2 = load_data_to_df(config.df2_path)

    dataset_gt = load_dataset(config.ground_truth_dataset_path)

    print('Running Test...')
    test(df1=df1,
         df2=df2,
         dataset=dataset_gt,
         fm_model_name=config.fm_model_name,
         peft_model_path=config.trained_mode_path,
         gt_keys_path=config.gt_keys_path,
         result_file_name=config.result_file_name,
         split_name=config.split_name)
