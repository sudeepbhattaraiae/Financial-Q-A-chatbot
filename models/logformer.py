from transformers import LongformerTokenizer,LongformerForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader,Dataset
from torch.cuda.amp import autocast, GradScaler
import torch
import json
import re
from tqdm import tqdm


tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model3 = LongformerForQuestionAnswering.from_pretrained('allenai/longformer-base-4096')

def train_longformer_qa(train_dataset, num_epochs=10, batch_size=1, accumulation_steps=4, learning_rate=2e-5):
    # Initialize the model
    model = LongformerForQuestionAnswering.from_pretrained('allenai/longformer-base-4096')

    # Set up training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Calculate total training steps
    total_steps = len(train_dataloader) * num_epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Initialize the GradScaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            # Move tensors to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch['global_attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            # Mixed precision training
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss / accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{num_epochs} completed")

    return model

# Usage example:
# trained_model = train_longformer_qa(train_dataset)

# 1. Custom Dataset
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        context = item['context']
        answer = item['answer']

        encoding = self.tokenizer3.encode_plus(
            question,
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Create global attention mask
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[0] = 1  # global attention to [CLS] token

        # Find answer start and end positions
        answer_start = context.find(answer)
        answer_end = answer_start + len(answer)

        start_positions = encoding.char_to_token(answer_start)
        end_positions = encoding.char_to_token(answer_end - 1)

        # If the answer is not fully within the context, use the CLS token index
        if start_positions is None or end_positions is None:
            start_positions = end_positions = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'global_attention_mask': global_attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions,
            'answer': answer
        }

# 3. Training function
def train(model, train_dataloader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch['global_attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader):.4f}")

# 4. Evaluation function
def evaluate(model, eval_dataloader, device):
    model.eval()
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch['global_attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_pred = torch.argmax(start_logits, dim=1)
            end_pred = torch.argmax(end_logits, dim=1)

            correct = ((start_pred == start_positions) & (end_pred == end_positions)).sum().item()
            total_correct += correct
            total_examples += input_ids.size(0)

    accuracy = total_correct / total_examples
    return accuracy

# 5. Inference function
def answer_question(model, tokenizer, question, context, device):
    model.eval()
    encoding = tokenizer.encode_plus(
        question,
        context,
        max_length=4096,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1  # global attention to [CLS] token

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)

    answer_tokens = input_ids[0][start_idx:end_idx+1]
    answer = tokenizer.decode(answer_tokens)

    return answer

# 6. Arithmetic operations
def perform_arithmetic(question, answer):
    numbers = re.findall(r'\d+(?:\.\d+)?', answer)
    numbers = [float(num) for num in numbers]

    if 'sum' in question.lower() or 'total' in question.lower():
        return sum(numbers)
    elif 'difference' in question.lower():
        return numbers[0] - numbers[1] if len(numbers) >= 2 else None
    elif 'average' in question.lower() or 'mean' in question.lower():
        return sum(numbers) / len(numbers) if numbers else None
    elif 'rate' in question.lower() or 'change' in question.lower():
        return (numbers[-1] - numbers[0]) / numbers[0] if len(numbers) >= 2 else None
    else:
        return None

# 7. Main execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    # train_data = load_data('train.json')
    # test_data = load_data('test.json')

    # Initialize tokenizer and model
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerForQuestionAnswering.from_pretrained('allenai/longformer-base-4096')
    model3.to(device)

    # Create datasets and dataloaders
    # train_dataset = QADataset(traindataset, tokenizer, max_length=4096)
    # test_dataset = QADataset(test_data, tokenizer, max_length=4096)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4)

    # Training
    optimizer = AdamW(model3.parameters(), lr=2e-5)
    num_epochs = 3
    train(model3, train_dataloader, optimizer, device, num_epochs)

    # Evaluation
    accuracy = evaluate(model3, test_dataloader, device)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save the model
    torch.save(model3.state_dict(), 'longformer_qa_model.pth')

    # Inference example
    question = "What is the sum of the numbers in the passage?"
    context = "The first number is 10, the second is 20, and the third is 30."

    answer = answer_question(model3, tokenizer, question, context, device)
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Extracted Answer: {answer}")

    # Perform arithmetic if needed
    arithmetic_result = perform_arithmetic(question, answer)
    if arithmetic_result is not None:
        print(f"Arithmetic Result: {arithmetic_result}")
    else:
        print("No arithmetic operation required.")

if __name__ == "__main__":
    main()




