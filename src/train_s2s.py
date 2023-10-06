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

from peft import (LoraConfig,
                  get_peft_model,
                  TaskType,
                  PeftModel,
                  PeftConfig)


def train(config):
    df1 = pd.read_csv(config.df1_path)
    print(config.df1_path)
    df1.fillna('None', inplace=True)
    print(df1.shape)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)
    train(config)
