import argparse
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from torch.utils.data import DataLoader
import torch
import os
import numpy as np

from utils import TextDataset, calculate_weights  # Assuming this is defined in your utils.py
from utils import train, evaluate  

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate NLP model for sequence classification.")

    parser.add_argument("--experiment_name", type=str, required=True, help="Name for the experiment directory.")
    parser.add_argument("--model", type=str,  default='bert', choices=["bert", "roberta"], help="Which model to use")
    parser.add_argument("--file_path", type=str, required=True, help="File path to the dataset.")
    parser.add_argument("--max_len", type=int, default=216, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for BERT model.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--use_weights", action="store_true", help="Use weighted loss for imbalanced data.")
    parser.add_argument('--restore_ckpt', default=None, help="load the weights from a specific checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on.")
    
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    #create experiment directory
    experiment_dir = os.path.join(args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    if args.model=='bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.hidden_dropout_prob = args.dropout_rate  
        config.attention_probs_dropout_prob = args.dropout_rate
        config.num_labels = 2   
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    
    elif args.model=='roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        config = BertConfig.from_pretrained('roberta-base')
        config.hidden_dropout_prob = args.dropout_rate 
        config.attention_probs_dropout_prob = args.dropout_rate  
        config.num_labels = 2 
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)

    # Train and evaluate (on validation set)
    train(model, args.restore_ckpt, args.file_path, tokenizer, args.max_len, args.batch_size, args.epochs, args.lr, args.weight_decay, args.use_weights, args.device, experiment_dir)
    
    print("Training and evaluation complete.")

if __name__ == "__main__":
    main()
