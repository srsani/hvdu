"""
Sequence to Sequence
Copyright Sohrab Redjai Sani
MIT License
"""
import argparse
from sconf import Config

from datasets import load_dataset
import torch
import time
import evaluate
import pandas as pd
import numpy as np
import ast


from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          GenerationConfig,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback,
                          IntervalStrategy)

from notebooks.utilities import (print_number_of_trainable_model_parameters,
                                 )

from peft import (LoraConfig,
                  get_peft_model,
                  TaskType,
                  PeftModel,
                  PeftConfig)


def load_data(df_path):

    df = pd.read_csv(df_path)
    df.fillna('None', inplace=True)
    print(f'{df_path} size: {df.shape}')

    return df


def train(config):

    df1 = load_data(config.df1_path)
    df2 = load_data(config.df2_path)

    print('Loading ground truth...')
    dataset = load_dataset(config.dataset_name_or_path)

    print('Load model and tokenizer...')
    original_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name,
                                                           device_map="auto",
                                                           torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name,
                                              device_map="auto",)
    number_of_param = print_number_of_trainable_model_parameters(
        original_model)
    print(number_of_param)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)
    train(config)
