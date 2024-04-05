import torch
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch.nn as nn
import torch.optim as optim

class MultiTaskBERT(nn.Module):
    def __init__(self, num_labels_task1, num_labels_task2):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Task-specific layers
        self.dropout = nn.Dropout(0.1)
        self.classifier_task1 = nn.Linear(self.bert.config.hidden_size, num_labels_task1)
        self.classifier_task2 = nn.Linear(self.bert.config.hidden_size, num_labels_task2)
        
    def forward(self, input_ids, attention_mask, task):
        """
        Forward pass for multitask learning.
        
        Parameters:
        - input_ids: Tensor of input IDs
        - attention_mask: Tensor for attention mask
        - task: Integer specifying the task (1 for task1, 2 for task2)
        
        Returns:
        - logits: Task-specific logits
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Determine which task is being asked for and use the appropriate classifier
        if task == 1:
            logits = self.classifier_task1(pooled_output)
        elif task == 2:
            logits = self.classifier_task2(pooled_output)
        else:
            raise ValueError("Invalid task identifier.")
        
        return logits




# Assume tokenizer is already initialized
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Data for Task 1: Sentiment Analysis
sentences_task1 = ["This is a great movie", "I love this new phone", "That was a boring lecture"]
sentiment_labels = [1, 1, 0]  # 0: Negative, 1: Positive

# Data for Task 2: Topic Classification
sentences_task2 = ["The match was a thrilling experience", "Politics is changing", "Technology advances rapidly"]
topic_labels = [1, 2, 3]  # 1: Sports, 2: Politics, 3: Technology

# Tokenize and prepare datasets separately for each task
inputs_task1 = tokenizer(sentences_task1, padding=True, truncation=True, return_tensors="pt")
inputs_task2 = tokenizer(sentences_task2, padding=True, truncation=True, return_tensors="pt")

dataset_task1 = TensorDataset(inputs_task1.input_ids, inputs_task1.attention_mask, torch.tensor(sentiment_labels))
dataset_task2 = TensorDataset(inputs_task2.input_ids, inputs_task2.attention_mask, torch.tensor(topic_labels))

# DataLoaders for each task
batch_size = 2
dataloader_task1 = DataLoader(dataset_task1, batch_size=batch_size, shuffle=True)
dataloader_task2 = DataLoader(dataset_task2, batch_size=batch_size, shuffle=True)

# Assume the MultiTaskBERT model is already defined and initialized
model = MultiTaskBERT(num_labels_task1=2, num_labels_task2=4)  # Adjust num_labels_task2 as needed
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training Loop Handling Different Sentences for Each Task
model.train()
for epoch in range(3):  # Example: 3 epochs
    for (batch1, batch2) in zip(dataloader_task1, dataloader_task2):
        # Handle Task 1
        input_ids, attention_mask, labels = [item.to('cpu') for item in batch1]
        optimizer.zero_grad()
        logits_task1 = model(input_ids, attention_mask, task=1)
        loss_task1 = nn.CrossEntropyLoss()(logits_task1, labels)
        loss_task1.backward()  # Accumulate gradients

        # Handle Task 2
        input_ids, attention_mask, labels = [item.to('cpu') for item in batch2]
        logits_task2 = model(input_ids, attention_mask, task=2)
        loss_task2 = nn.CrossEntropyLoss()(logits_task2, labels)
        loss_task2.backward()  # Accumulate gradients further

        optimizer.step()  # Perform optimization step for both tasks

    print(f"Epoch {epoch+1} completed.")
    
model.eval()

# Synthetic Test Data for Task 1 (Sentiment Analysis) and Task 2 (Topic Classification)
test_sentences_task1 = ["This movie was a great disappointment", "The new product launch was a massive success", "I've seen better days", "This is a masterpiece"]
test_labels_task1 = [0, 1, 0, 1]  # 0: Negative, 1: Positive

test_sentences_task2 = ["He scored a winning goal", "The election results were surprising", "Innovations in technology continue to amaze us", "Historical novels offer insights into the past"]
test_labels_task2 = [1, 2, 3, 4]  # 1: Sports, 2: Politics, 3: Technology, 4: Education

# Tokenizing test data for both tasks
test_inputs_task1 = tokenizer(test_sentences_task1, return_tensors='pt', padding=True, truncation=True, max_length=512)
test_labels_task1 = torch.tensor(test_labels_task1)

test_inputs_task2 = tokenizer(test_sentences_task2, return_tensors='pt', padding=True, truncation=True, max_length=512)
test_labels_task2 = torch.tensor(test_labels_task2)

def evaluate(model, inputs, labels, task):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'], task)
        predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        accuracy = (predictions == labels).float().mean()
    return accuracy.item()
# Evaluate Task 1 (Sentiment Analysis)
accuracy_task1 = evaluate(model, test_inputs_task1, test_labels_task1, task=1)
print(f"Task 1 (Sentiment Analysis) Accuracy: {accuracy_task1:.4f}")

# Evaluate Task 2 (Topic Classification)
accuracy_task2 = evaluate(model, test_inputs_task2, test_labels_task2, task=2)
print(f"Task 2 (Topic Classification) Accuracy: {accuracy_task2:.4f}")
