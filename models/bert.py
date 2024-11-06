import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import os
import time
import logging
from collections import Counter
import re
import configuration
from configuration import parameters as conf

# Assuming you have a configuration file or object named 'conf'
# If not, you'll need to define these parameters

tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
model1 = BertModel.from_pretrained('bert-base-uncased')

class Bert_model(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Bert_model, self).__init__()
        self.bert = model1
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, 2)  # 2 for start/end

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_multiple_answers(predictions, ground_truths):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    best_f1 = 0
    for ground_truth in ground_truths:
        for prediction in predictions:
            f1 = f1_score(prediction, ground_truth)
            if f1 > best_f1:
                best_f1 = f1
    return best_f1

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs, device):
    model.to(device)
    best_val_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            optimizer.zero_grad()
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average train loss: {avg_train_loss}")

        # Validation
        val_f1 = evaluate(model, val_dataloader, device)
        print(f"Validation F1: {val_f1}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"Best validation F1: {best_val_f1}")

def evaluate(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            for i in range(input_ids.size(0)):
                start_logits_i = start_logits[i].cpu().numpy()
                end_logits_i = end_logits[i].cpu().numpy()

                # Get top k answer spans
                start_indexes = start_logits_i.argsort()[-conf.topk:][::-1]
                end_indexes = end_logits_i.argsort()[-conf.topk:][::-1]

                print(f"\nExample {i+1}:")
                print("Top start position scores:")
                for idx, start_idx in enumerate(start_indexes):
                    print(f"  Position {start_idx}: {start_logits_i[start_idx]:.4f}")

                print("Top end position scores:")
                for idx, end_idx in enumerate(end_indexes):
                    print(f"  Position {end_idx}: {end_logits_i[end_idx]:.4f}")

                predictions = []
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if start_index <= end_index:
                            prediction = tokenizer1.decode(input_ids[i][start_index:end_index+1])
                            score = start_logits_i[start_index] + end_logits_i[end_index]
                            predictions.append((prediction, score))

                print("Predictions with scores:")
                for idx, (pred, score) in enumerate(sorted(predictions, key=lambda x: x[1], reverse=True)):
                    print(f"  {idx+1}. {pred} (Score: {score:.4f})")

                all_predictions.append([pred for pred, _ in predictions])

                if 'answers' in batch:
                    ground_truth = batch['answers'][i]
                    print(f"Ground Truth: {ground_truth}")
                    all_ground_truths.append(ground_truth)
                else:
                    print("Ground Truth not available in this batch")

                print("-" * 50)

    if all_ground_truths:
        f1_scores = [evaluate_multiple_answers(preds, truths)
                     for preds, truths in zip(all_predictions, all_ground_truths)]
        average_f1 = sum(f1_scores) / len(f1_scores)
        print(f"Average F1 Score: {average_f1:.4f}")
        return average_f1
    else:
        print("No ground truth available for evaluation")
        return None

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

     # Add console handler
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    # Add file handler
    file_handler = logging.FileHandler('output.log')
    logger.addHandler(file_handler)

    try:
        #Create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)

        # Initialize model
        bertmodel = Bert_model(dropout_rate=conf.dropout_rate)

        # Set up optimizer and scheduler
        optimizer = torch.optim.AdamW(bertmodel.parameters(), lr=conf.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * conf.num_epochs)

        # Train the model
        train(bertmodel, train_dataloader, val_dataloader, optimizer, scheduler, conf.num_epochs, conf.device)

        # Load best model for testing
        bertmodel.load_state_dict(torch.load('best_model.pth'))

        # Evaluate on test set
        test_f1 = evaluate(bertmodel, test_dataloader, conf.device)
        print(f"Test F1: {test_f1}")  # Print statement for immediate console output
        logger.info(f"Test F1: {test_f1}")  # Logging statement
        logger.handlers[0].flush()  # Flush the logger

    except Exception as e:
        logger.exception("An error occurred during execution")

if __name__ == "__main__":
    main()