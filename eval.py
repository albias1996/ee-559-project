import argparse
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch
import os
import numpy as np

from utils import TextDataset, evaluate_test_set  

def main():
    parser = argparse.ArgumentParser(description="Evaluate NLP model for sequence classification on the test set.")

    parser.add_argument("--model", type=str,  default='bert', choices=["bert", "roberta"], help="Which model to use")
    parser.add_argument("--restore_ckpt", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--file_path", type=str, required=True, help="File path to the dataset.")
    parser.add_argument("--max_len", type=int, default=216, help="Maximum sequence length for tokens.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to evaluate on.")

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if args.model=='bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_labels = 2   
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    
    elif args.model=='roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        config = BertConfig.from_pretrained('roberta-base')
        config.num_labels = 2 
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)

    model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate the model
    avg_loss, precision, recall, f1, accuracy, conf_matrix = evaluate_test_set(
        model=model,
        checkpoint_path=args.restore_ckpt,
        file_path=args.file_path,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        criterion=criterion,
        device=args.device
    )

    print(f"Test Set Evaluation:\nLoss: {avg_loss:.4f}, Precision: {precision}, Recall: {recall}, F1 Score: {f1},  Accuracy: {accuracy:.4f}")
    print(f"Test Set confusion matrix: {conf_matrix}")

if __name__ == "__main__":
    main()
