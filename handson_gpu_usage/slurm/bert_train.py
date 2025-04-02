from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import wandb

print("Starting the script...")
# Load the IMDb dataset
dataset = load_dataset('imdb')

wandb.init(project="imdb_classification", entity="vinays")
print(dataset)
train_dataset = dataset['train'].shuffle(seed=42)
print(train_dataset.shape)
test_dataset = dataset['test']
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 2 labels for binary classification

# Preprocess the data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True).select(range(1000))  # Select first 100 samples for training
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True).select(range(100)) 

print(tokenized_train_dataset[0])

# Compute metrics function for evaluation
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="wandb",  # Enable logging to wandb
)


# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,  # Pass the compute_metrics function
)

# Train the model
trainer.train()

# Save the model
model_path = "./bert-imdb"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
# Optionally, you can finish the wandb run when training is done
wandb.finish()
