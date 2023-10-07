"""
Sequence to Sequence
Copyright Sohrab Redjai Sani
MIT License
"""
import argparse
from sconf import Config
import json

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
                                 init_json,
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


def tokenize_function(batch, df1, df2, tokenizer):

    # Create containers for input_ids and labels for the entire batch
    input_ids_list = []
    labels_list = []

    # Iterate over each example in the batch
    for idx in range(len(batch["ground_truth"])):

        ground_truth = str(json.loads(batch["ground_truth"][idx])['gt_parse'])
        file_path = batch['image'][idx].filename
        input_image_name = f"{file_path[file_path.rfind('/')+1: ]}"

        df_1 = df1[df1.image_name == input_image_name]
        df_2 = df2[df2.image_name == input_image_name]

        # make sure they both JSONS have all the keys
        json1 = init_json(df_1)
        json2 = init_json(df_2)

        prompt = f"""convert the two input JSONs delimited by triple backticks into one OUTPUT_JSON.\n\n

         INPUT_JSONs: ```
         JSON1 = {json1}, 
         JSON2 = {json2}``` 
         
         \n\nOUTPUT_JSON:  
        """

        # Tokenize and append to the batch lists
        input_ids_list.append(
            tokenizer(prompt, padding="max_length", truncation=True).input_ids)
        labels_list.append(
            tokenizer(ground_truth, padding="max_length", truncation=True).input_ids)

    batch['input_ids'] = input_ids_list
    batch['labels'] = labels_list

    # Convert lists to tensors
    return batch


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

    print('Tokenize dataset...')
    tokenized_datasets = dataset.map(tokenize_function,
                                     fn_kwargs={'df1': df1,
                                                'df2': df2,
                                                'tokenizer': tokenizer},
                                     batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ['ground_truth', 'image'])

    lora_config = LoraConfig(r=config.r_rank,  # Rank
                             lora_alpha=config.lora_alpha,
                             target_modules=["q", "v"],
                             lora_dropout=0.025,
                             bias="none",
                             task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T5)
                             )

    peft_model = get_peft_model(original_model,
                                lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))

    print('Initialize Trainer...')
    output_dir = f'../result/t5-lora-{str(int(time.time()))}'
    print(f'model will the saved in: {output_dir}')

    peft_training_args = TrainingArguments(output_dir=output_dir,
                                           auto_find_batch_size=True,
                                           learning_rate=1e-3,
                                           #     weight_decay=0.01,
                                           num_train_epochs=config.num_train_epochs,
                                           logging_steps=1,
                                           #     max_steps=500,
                                           load_best_model_at_end=True,
                                           eval_steps=50,
                                           save_total_limit=5,
                                           evaluation_strategy=IntervalStrategy.STEPS,
                                           )

    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    if config.test:
        print('Testing...')
        peft_trainer.evaluate(tokenized_datasets["test"])
        return True

    peft_trainer.train()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args, left_argv = parser.parse_known_args()
    print(args, left_argv)
    config = Config(args.config)
    config.argv_update(left_argv)
    train(config)
