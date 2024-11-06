import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time
from collections import Counter
import string
import re

def train_distilbert_qa(train_dataset, model, training_args):
    # Set default values for training arguments
    default_args = {
        'num_epochs': 10,
        'batch_size': 8,
        'accumulation_steps': 4,
        'learning_rate': 5e-5,
        'warmup_steps': 0,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Update default arguments with provided arguments
    default_args.update(training_args)

    # Set up training parameters
    device = torch.device(default_args['device'])
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=default_args['learning_rate'], weight_decay=default_args['weight_decay'])

    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=default_args['batch_size'], shuffle=True)
    # Calculate total training steps
    total_steps = len(train_dataloader) * default_args['num_epochs'] // default_args['accumulation_steps']
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=default_args['warmup_steps'],
                                                num_training_steps=total_steps)

    # Initialize the GradScaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    model.train()
    for epoch in range(default_args['num_epochs']):
        print(f"\nEpoch {epoch+1}/{default_args['num_epochs']}")
        print("-" * 40)

        epoch_loss = 0.0
        epoch_start_time = time.time()

        progress_bar = tqdm(train_dataloader, desc=f"Training", leave=False)
        for step, batch in enumerate(progress_bar):
            # Move tensors to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            # Mixed precision training
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss / default_args['accumulation_steps']

            # Backward pass
            scaler.scale(loss).backward()

            if (step + 1) % default_args['accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), default_args['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * default_args['accumulation_steps']

            # Update progress bar
            progress_bar.set_postfix({'loss': f"{epoch_loss / (step + 1):.4f}"})

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = epoch_loss / len(train_dataloader)

        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

    print("\nTraining completed!")
    return model

# # Usage example:
# training_args = {
#     'num_epochs': 20,
#     'batch_size': 16,
#     'learning_rate': 3e-5,
#     'warmup_steps': 500
# }
# trained_model = train_distilbert_qa(train_dataset, model2, training_args)

def clean_answer(answer):
    # Remove spaces between digits and commas
    answer = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', answer)
    # Remove spaces between digits and decimal points
    answer = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', answer)
    # Remove leading/trailing spaces and symbols
    answer = answer.strip(' $,.')
    return answer

def lower(text):
    return text.lower()

def get_tokens(s):
    if not s: return []
    s = lower(s)
    return s.split()

def compute_exact(a_gold, a_pred):
    return int((a_gold) == (a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate(model, eval_dataset, tokenizer, device, batch_size=8):
    model.eval()
    exact_scores = []
    f1_scores = []

    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i:i+batch_size]
        questions = [item['question'] for item in batch]
        contexts = [lower(item['context']) for item in batch]
        gold_answers = [lower(item['answer']) for item in batch]

        inputs = tokenizer(
            questions,
            contexts,
            max_length=512,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

        for j, (start, end) in enumerate(zip(start_logits, end_logits)):
            start_idx = torch.argmax(start)
            end_idx = torch.argmax(end)

            pred_answer = tokenizer.decode(input_ids[j][start_idx:end_idx+1])
            pred_answer = clean_answer(pred_answer)
            gold_answer = clean_answer(gold_answers[j])

            print(f"Question: {questions[j]}")
            print(f"Context: {contexts[j]}")
            print(f"Gold Answer: {gold_answer}")
            print(f"Predicted Answer: {pred_answer}\n")

            exact_scores.append(compute_exact(gold_answer, pred_answer))
            f1_scores.append(compute_f1(gold_answer, pred_answer))

    exact_match = sum(exact_scores) / len(exact_scores)
    f1 = sum(f1_scores) / len(f1_scores)

    return {
        'exact_match': exact_match,
        'f1': f1
    }

# # Usage example:
# eval_results = evaluate(trained_model, test_dict_list, tokenizer2, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# print(f"Exact Match: {eval_results['exact_match']:.4f}")
# print(f"F1 Score: {eval_results['f1']:.4f}")