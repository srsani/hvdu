# Handwritten Visual Document Understanding

Handwritten Visual Document Understanding

## Create virtual environment

`conda env create -f environment.yml`
`conda activate ocr37`
`python -m ipykernel install --user --name ocr37`

## Model fine-tuning

### train a new model

- `cd src`
- `python train.py --config config/train_nist.yaml`

Update the `train_nist`


### testing:

- `cd src`

- `python test.py \
--dataset_name_or_path  ./dataset/training/nist_form \
--pretrained_model_name_or_path ./result/train_nist/20230905_212956 \
--save_path ./dataset/result/nist_form \
--split test`

Update the following accordingly: 

1. where the test dataset is `dataset_name_or_path` 
2. where the model is stored `pretrained_model_name_or_path`
3. folder path to save the results `save_path`

## Reference

We have used Donut pre-train script and model in this work

```
@inproceedings{kim2022donut,
  title     = {OCR-Free Document Understanding Transformer},
  author    = {Kim, Geewook and Hong, Teakgyu and Yim, Moonbin and Nam, JeongYeon and Park, Jinyoung and Yim, Jinyeong and Hwang, Wonseok and Yun, Sangdoo and Han, Dongyoon and Park, Seunghyun},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```

