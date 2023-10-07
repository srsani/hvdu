from donut import DonutModel, JSONParseEvaluator, load_json, save_json
import numpy as np
import pandas as pd

import uuid
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import pathlib
import os
import shutil
import pickle

import shutil
import pickle
import time
import re
import glob
import json

import asyncio
import random
import nest_asyncio
import concurrent.futures

from PIL import Image, ImageDraw, ImageFont
from nltk import edit_distance
import plotly.express as px

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

import nest_asyncio
import openai

nest_asyncio.apply()
openai.api_key = os.environ['OPENAI_API_KEY']

prefix_prompt = f"""Use the following five examples of two INPUT_JSONs(JSON1 and JSON2)and the corresponding singular OUTPUT_JSON to convert the two input JSONs delimited by triple backticks at the end of this prompt into one OUTPUT_JSON."""
postfix_prompt = """Consider example 6: In the STATE field, if you got two words and the second one was an abbreviation of a state name in the United States of America, the first word should be part of the  CITY field. ru 
Make sure that the CITY is in the correct STATE. Make sure that the STATE is in the correct CITY. STATE value cannot and should not contain any numbers or digits. Don't change the STATE value from abbreviation to complete format. ZIP is a zip code and contains only numbers. If you have a ZIP with a punctuation, remove the punctuation. ZIP is zip code and does not contain /, and / must be changed to 1.
YOU MUST FOLLOW THE NEXT 8 RULES: RULES 1. DO NOT CHANGE THE LETTER CASING OF THE INPUT. RULES 2. If the STATE field had a sequence of numbers, they must be moved to the ZIP field. RULES 3. If the STATE field appears as the full version of the state name and not the abbreviation, keep the full version. RULES 4. If the STATE field letters appear capitalised, DO NOT convert the letters to small letters. RULES 5. If the CITY field letters are capitalised, DO NOT convert them to small letters; they MUST be capitalised. RULES 6. DO NOT convert the letters from upper case to lower case. DO NOT change lowercase to uppercase.  RULES 7. Check the validity of values in the STATE and CITY fields. For example, if the value for the CITY was "ROMEOUILLE" and the value for the STATE field was "ILL",  then the correct spelling for the  CITY field should be "ROMEOVILLE". In such cases, correct the spelling of the CITY field. RULES 8. Check the validity of values in the STATE and CITY fields. For example, if the value for the CITY was "New York City" and the STATE field was "NU" filed the spelling for the STATE field and update it to "NY".
OUTPUT_JSON format:
Format your response as a JSON object with all four `keys'.
OUTPUT_JSON reply MUST be a single JSON with the following KEYS: DATE - CITY - STATE- ZIP
REPLY ONLY A SINGLE JSON WITH: DATE, CITY, STATE, ZIP OR I WILL KILL YOU. IF YOU CANT MAKE IT TO ONE JSON, JUST RETURN THE JSON1
Return a single json and don not assign a name to it like `OUTPUT_JSON =`. 
Return a single json that it can be loaded using json.loads(YOUR_RESULT) in python.
"""
example_dict = {
    'example1':
    [str({'DATE': '8-10-89', 'CITY': 'None', 'STATE': 'Sacramento Ca', 'ZIP': '95841'}),
     str({'DATE': '8-10-89', 'CITY': 'Sacrament',
          'STATE': 'None', 'ZIP': 'Ca95841'}),
     str({'DATE': 'Sacramento', 'CITY': 'Sacramento', 'STATE': 'Ca', 'ZIP': '95841'})],

    'example2':
    [str({'DATE': '9-21-89', 'CITY': 'Sonora, Texas', 'STATE': '76950', 'ZIP': 'None'}),
     str({'DATE': '9-21-89', 'CITY': 'Sonora,',
          'STATE': 'Texas 76950', 'ZIP': 'None'}),
     str({'DATE': '9-21-89', 'CITY': 'Sonora', 'STATE': 'Texas', 'ZIP': '76950'})],

    'example3':
    [str({'DATE': '9/27/89', 'CITY': 'Austin', 'STATE': 'MN', 'ZIP': '55912'}),
     str({'DATE': '9/27/89', 'CITY': 'Austin',
          'STATE': 'MN', 'ZIP': '55412'}),
     str({'DATE': '9/27/89', 'CITY': 'Austin', 'STATE': 'MN', 'ZIP': '55912'})],

    'example4':
    [str({'DATE': '09-17-89', 'CITY': 'Brownsville', 'STATE': 'TX. 78521', 'ZIP': 'None'}),
     str({'DATE': '09-17-89', 'CITY': 'BROWnsville',
          'STATE': 'None', 'ZIP': 'None'}),
     str({'DATE': '09-17-89', 'CITY': 'BROWnsville', 'STATE': 'TX.', 'ZIP': '78521'})],

    'example5':
    [str({'DATE': 'Aug. 4,1989', 'CITY': 'Marietta, Ohio', 'STATE': '45754', 'ZIP': 'None'}),
     str({'DATE': 'Aug. 4,1989', 'CITY': 'Marietta, Ohio',
          'STATE': 'None', 'ZIP': '45754'}),
     str({'DATE': 'Aug. 4,1989', 'CITY': 'Marietta', 'STATE': 'Ohio', 'ZIP': '45754'})],
    'example6':
    [str({'DATE': '9/11/89', 'CITY': 'Richland', 'STATE': 'Center Wi 53581', 'ZIP': 'None'}),
     str({'DATE': '9/4189', 'CITY': 'Rich land',
          'STATE': 'Center Wi', 'ZIP': '53581'}),
     str({'DATE': '9/21/89', 'CITY': 'Richland Center', 'STATE': 'Wi', 'ZIP': '53581'})]
}

evaluator = JSONParseEvaluator()


def draw_(image_path: str,
          dict_: dict) -> Image:
    """
    Overlay specific values from a JSON-like dictionary onto an image at designated coordinates.

    This function takes an image path and a dictionary containing fields like DATE, CITY, STATE, 
    and ZIP. It overlays the values of these fields onto the image at specified coordinates, 
    enhancing the original image with additional textual information.

    Args:
        image_path (str): The path to the .png image which will be overlaid with text.
        dict_ (dict): A dictionary resembling a JSON structure, expected to contain the following keys:
            - DATE
            - CITY
            - STATE
            - ZIP
          The corresponding values for these keys will be overlaid on the image.

    Returns:
        Image: The original image overlaid with text from the provided dictionary.

    Note:
        - The function expects the dictionary to contain all the mentioned keys for successful execution.
        - The font "Aaargh.ttf" should be present in the working directory or accessible path for correct font rendering.
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Aaargh.ttf", 45,)

    txt = str(dict_['DATE'])
    draw.text((800, 90), txt, font=font, size=10000,)

    txt = str(dict_['CITY'])
    draw.text((1200, 91), txt, font=font, size=10000)

    txt = str(dict_['STATE'])
    draw.text((1500, 90), txt, font=font, size=10000)

    txt = str(dict_['ZIP'])
    draw.text((1700, 90), txt, font=font, size=10000)

    return img


def save_json_2nd(image_name: str,
                  todo_list_: list,
                  input_dict: dict) -> None:
    """
    Save the provided data as a JSON file during the second round of image labeling.

    This function takes an image name, a to-do list of images, and an input dictionary. It saves 
    the dictionary data as a JSON file, naming it after the provided image (but with a .json extension). 
    The function then attempts to remove the image name from the to-do list, signifying that it's been processed.

    Args:
        image_name (str): Name of the image, including the .png extension. Used to name the JSON file.
        todo_list_ (list): List of image names. This function will attempt to remove the processed image name from this list.
        input_dict (dict): Dictionary containing the data to be saved as a JSON. It typically includes annotations or labels related to the image.

    Note:
        - This function will print a message if an attempt is made to remove an image name that's not in the to-do list.
        - It assumes the existence of the path "../dataset/raw/nist/keys_2nd/".
    """
    outfile = input_dict
    with open(f"../dataset/raw/nist/keys_2nd/{image_name[:-4]}.json", "w") as outfile:
        json.dump(input_dict, outfile, indent=4)

    try:
        todo_list_.remove(image_name)
    except:
        print(f"{image_name} has already removed")


def dict_distance(dict1: dict,
                  dict2: dict) -> float:
    """
    Calculate the edit distance between the values of two dictionaries 
    for the same keys and returns the accuracy of similarity.

    This function calculates the Levenshtein distance (or edit distance) 
    between the values of two dictionaries for matching keys using the
    `edit_distance` function from the NLTK library. The edit distance is the 
    number of characters that need to be substituted, inserted, or deleted, 
    to transform one string into another. For instance, transforming "rain" 
    to "shine" requires at least three steps.

    The accuracy of similarity is computed as:
    accuracy = 100 - (sum of edit distances / sum of maximum character lengths) * 100

    Args:
    - dict1 (dict): The first dictionary to compare.
    - dict2 (dict): The second dictionary to compare.

    Note: 
    - dict1 and dict2 should have the same keys for a meaningful comparison.

    Returns:
    - float: Accuracy of similarity between values of the two dictionaries.

    Example:
    If dict1 = {'a': 'apple', 'b': 'banana'} and 
       dict2 = {'a': 'apples', 'b': 'banane'}, 
    the function would return: 83.33
    """
    if set(dict1.keys()) != set(dict2.keys()):
        raise ValueError("dict1 and dict2 must have the same keys.")

    dict_distance = []
    dict_car_len = []
    for key, _ in dict1.items():

        dict_distance.append(edit_distance(dict1[key].strip(),
                                           dict2[key].strip()))

        if len(dict1[key]) > len(dict2[key]):
            dict_car_len.append((len(dict1[key])))

        else:
            # if len(dict2[key]) >= len(dict1[key]):
            dict_car_len.append((len(dict2[key])))
    accuracy = 100 - (sum(dict_distance)/(sum(dict_car_len)))*100

    return accuracy


def ensure_folder_exists(folder_path):
    path = Path(folder_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Folder '{folder_path}' created!")


def compare_and_add_missing_keys(dict1, dict2):
    # Iterate over keys in dict1
    for key in dict1:
        # If the key is missing in dict2, add it with value 'None'
        if key not in dict2:
            dict2[key] = 'None'

    return dict2


def dict_distance_key(dict1: dict, dict2: dict, file_name: str) -> dict:
    try:
        dict2 = compare_and_add_missing_keys(dict1, dict2)
        dict_ = {}
        for key, _ in dict1.items():
            dict_['image_name'] = file_name
            dict_[key] = edit_distance(dict1[key].strip(),
                                       dict2[key].strip())

            if len(dict1[key]) > len(dict2[key]):
                dict_[f'{key}_LEN'] = len(dict1[key])

            else:
                dict_[f'{key}_LEN'] = len(dict2[key])
    except:
        print(f'dict1: {dict1}, \n dict2: {dict2}, \n {file_name}')

    return dict_


def key_based_accuracy(DATE,
                       CITY,
                       STATE,
                       ZIP,

                       DATE_LEN,
                       CITY_LEN,
                       STATE_LEN,
                       ZIP_LEN,
                       ):
    # total accuracy
    sum_distance = DATE + CITY + STATE + ZIP
    sum_len = DATE_LEN + CITY_LEN + STATE_LEN + ZIP_LEN

    total_accuracy_ = 100 - (sum_distance/sum_len)*100

    # DATE accuracy
    date_accuracy = 100 - (DATE/DATE_LEN)*100

    # CITY accuracy
    city_accuracy = 100 - (CITY/CITY_LEN)*100

    # STATE accuracy
    state_accuracy = 100 - (STATE/STATE_LEN)*100

    # ZIP accuracy
    zip_accuracy = 100 - (ZIP/ZIP_LEN)*100

    return total_accuracy_, date_accuracy, city_accuracy, state_accuracy, zip_accuracy


def get_accuracy_dict(df, keys_path):

    dict_list = []
    error_list = []

    for file_name in df.image_name:
        try:
            # get filename and the corresponding json
            json_name = file_name[file_name.rfind('/')+1: file_name.rfind('.')]

            # covert azure data to a dict
            dict1 = df[df.image_name == file_name][[
                'DATE', 'CITY', 'STATE', 'ZIP']].iloc[0].to_dict()

            # open the manually cleaned corresponding json
            with open(f"{keys_path}/{json_name}.json") as f:
                dict2 = json.load(f)

            dict_list.append(dict_distance_key(dict1, dict2, file_name))

        except:
            error_list.append(file_name)
    len(error_list)

    return dict_list, error_list


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


def generate_uuid_prefix(use_time=True, length=8):

    if use_time:
        # Returns a prefix based on the current time in milliseconds
        return datetime.now().strftime('%Y%m%d%H%M%S')
    else:
        # Returns the first 'length' characters of a UUID
        return str(uuid.uuid4())[:length]


def get_first_value_or_default(df_, column_name, default='None'):
    try:
        return df_[column_name].iloc[0]
    except:
        return default


def init_json(df_):
    return {
        'DATE': get_first_value_or_default(df_, 'DATE'),
        'CITY': get_first_value_or_default(df_, 'CITY'),
        'STATE': get_first_value_or_default(df_, 'STATE'),
        'ZIP': get_first_value_or_default(df_, 'ZIP')
    }


def create_prompt(json1, json2, prefix_prompt, postfix_prompt, example_dict):

    for key, value in example_dict.items():
        prompt = f"{prefix_prompt} {key}"
        prompt = f"{prompt} JSON1 = {value[0]}"
        prompt = f"{prompt} JSON2 = {value[1]}"
        prompt = f"{prompt} OUTPUT_JSON = {value[2]}"
    prompt = f"{prompt} {postfix_prompt}"
    prompt = f"{prompt}  INPUT_JSONs: ```JSON1 = {json1}, JSON2 = {json2}```"
    return prompt


def print_number_of_trainable_model_parameters(model):
    """
    This function calculates and returns a formatted string containing the count of trainable
    and total parameters in a given PyTorch model, along with the percentage of trainable parameters.

    Args:
    model (torch.nn.Module): The PyTorch model for which to calculate parameter counts.

    Returns:
    str: A formatted string containing the counts of trainable and total parameters,
         and the percentage of trainable parameters.
    """

    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"""trainable model parameters: {trainable_model_params}
                \nall model parameters: {all_model_params}
                \npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"""


