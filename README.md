# EE-559: Deep Learning, Spring 2024
This ReadMe contains all the informations required to run our code related to the EE-559 course project. 

Author: - Albias Havolli, MA4
        - Théodore Maradan, MA4
        - Armance Nouvel, MA2

## Environment 

The code has been run using python 3.10.4 and torch 2.0.1+cu121. The entire list of dependencies can be found in the requirements.txt file. Note that for reasonable computation time, the code requires to be run with a GPU available. 

## Files organization 

- `processing_data.ipynb` : this notebook contains the code required to process our data so that it can be used for experiments.
- `utils.py` : this file contains all the functions required to run the other files. 
- `train_eval_model.py` : this file contains the code for training our models. 
- `eval.py` this file contains the code for evaluating our models. 

## Required datasets 
To train/evaluate our code, one will need to download the following datasets and put them in the same directory as the project folder :

- [Sexism dataset](https://github.com/ellamguest/online-misogyny-eacl2021)
- [Misogyny dataset](https://github.com/rewire-online/edos)

Then, the interested reader needs to run `processing_data.ipynb` to process the data and make it ready for training/evaluation. 

Actually, the processed data is available as presented below : 

```
├── /datasets
    ├── sexism-dataset
    │   └── sexism_data.jsonl
    └── misogyny-dataset
        ├── original_data.jsonl
        └── augmented_data.jsonl

```

## Models 

For this project, we used [BERT](https://huggingface.co/google-bert/bert-base-uncased) and [RoBERTa](https://huggingface.co/FacebookAI/roberta-base) as models. Both models are available in Hugging Face library.


## Training 

To pre-train on the sexism dataset using BERT, uncomment line 138, comment line 139 of `utils.py` and run:

```Shell
python train_eval_model.py --model bert --experiment_name pretraining_bert --file_path datasets/sexism_dataset/sexism_data.jsonl
```

To train on the original misogyny dataset using BERT with pretrained weights (here denominated as "model_epoch_i.pth"), comment line 138, uncomment line 139 of `utils.py` and run:

```Shell
python train_eval_model.py --model bert --restore_ckpt pretraining_bert/model_epoch_i.pth --experiment_name training_bert --file_path datasets/misogyny_dataset/original_data.jsonl
```

## Evaluation

To evaluate on the original misogyny dataset using the BERT trained model (here denominated as "model_epoch_i.pth"), run : 

```Shell
python eval.py --model bert --restore_ckpt training_bert/model_epoch_i.pth --file_path datasets/misogyny_dataset/original_data.jsonl
```

The same can be applied using RoBERTa, replacing the references to BERT by references to RoBERTa. 
