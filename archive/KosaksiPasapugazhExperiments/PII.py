import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm

# Sample data creation (for demonstration purposes)
data = {
    'text': [
        "My name is John Doe", "Contact me at john@example.com", "Call me at 123-456-7890",
        "Jane Doe is my friend", "Email her at jane@example.com", "My phone number is 987-654-3210",
        "Bob Smith's email is bob@example.com", "Reach out at alice@example.com",
        "Let's meet at 10 AM tomorrow", "Call 555-1234", "My name is Alice Johnson", "You can reach me at 321-654-0987",
        "My phone is 111-222-3333", "I prefer emails to texts", "This is a generic text message.",
        "A different person here, my number is 222-333-4444."
    ],
    'label': [
        0, 1, 2, 0, 1, 2, 1, 1, 0, 2, 0, 2, 2, 1, 0, 2
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
)


# Define a custom Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Hyperparameters
BATCH_SIZE = 4
EPOCHS = 5
MAX_LENGTH = 32
LEARNING_RATE = 2e-5

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create DataLoader
train_dataset = TextDataset(train_texts.reset_index(drop=True), train_labels.reset_index(drop=True), tokenizer,
                            MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

# Evaluation
model.eval()

# Creating a test dataset
test_dataset = TextDataset(test_texts.reset_index(drop=True), test_labels.reset_index(drop=True), tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Collect predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

# Calculate F1 score
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f'F1 Score: {f1:.4f}')

# Classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# Optional: Visualize Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Name', 'Email', 'Phone'],
            yticklabels=['Name', 'Email', 'Phone'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
