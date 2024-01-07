import json
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.metrics import classification_report

def load_and_process_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    utterances, intents = [], []

    for dialogue in data:
        for turn in dialogue['turns']:
            if 'frames' in turn:
                for frame in turn['frames']:
                    if 'state' in frame and 'active_intent' in frame['state']:
                        if frame['state']['active_intent'] != "NONE":
                            utterances.append(turn['utterance'])
                            intents.append(frame['state']['active_intent'])

    return utterances, intents

def encode_data(tokenizer, utterances, intents, max_length):
    encoded_data = tokenizer(utterances, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(intents)
    return encoded_data, labels, label_encoder

def create_dataloader(encoded_data, labels, batch_size, shuffle=True):
    dataset = TensorDataset(encoded_data['input_ids'], encoded_data['attention_mask'], torch.tensor(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def train_model(model, train_dataloader, optimizer, device):
    model.to(device)
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_dataloader)
    return average_loss

def evaluate_model(model, test_dataloader, device, label_encoder):
    model.to(device)
    model.eval()
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Save LabelEncoder
    label_encoder_path = 'label_encoder.pkl'
    joblib.dump(label_encoder, label_encoder_path)

    number_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}
    predicted_labels_text = [number_to_label[i] for i in predicted_labels]
    true_labels_text = [number_to_label[i] for i in true_labels]

    report = classification_report(true_labels_text, predicted_labels_text)
    print(report)

# Load and process data
train_utterances, train_intents = load_and_process_data('dialogues_001.json')
test_utterances, test_intents = load_and_process_data('dialogues_002.json')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
max_length = 128

# Encode data
encoded_train_data, y_train, label_encoder = encode_data(tokenizer, train_utterances, train_intents, max_length)
encoded_test_data, y_test, _ = encode_data(tokenizer, test_utterances, test_intents, max_length)

# Create dataloaders
batch_size = 1
train_dataloader = create_dataloader(encoded_train_data, y_train, batch_size, shuffle=True)
test_dataloader = create_dataloader(encoded_test_data, y_test, batch_size, shuffle=False)

# Model
model = BertForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=len(label_encoder.classes_))

# Hyperparameters
learning_rate = 1e-5

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5

for epoch in range(num_epochs):
    average_loss = train_model(model, train_dataloader, optimizer, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

# Evaluation
evaluate_model(model, test_dataloader, device, label_encoder)

# Save the model
model_save_path = 'model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
