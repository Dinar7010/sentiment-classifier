from torch.optim import AdamW
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from pathlib import Path

df=pd.read_csv("C:\\Users\\User\PyCharmMiscProject\sentiment-project\sentiment_dataset.csv")
dfs=df.sample(n=1000,random_state=42)

ID2LABEL={0: "нейтральный", 1: "позитивный", 2: "негативный"}
LABEL2ID={v: k for k, v in ID2LABEL.items()}

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased",num_labels=3,id2label=LABEL2ID,label2id=LABEL2ID)
model = AutoModelForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=3)

texts=dfs["text"].tolist()
labels=dfs["label"].values

texts_temp, texts_test, y_temp, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
texts_train, texts_val, y_train, y_val = train_test_split(texts_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

res_train=tokenizer(texts_train,padding=True,truncation=True,max_length=128,return_tensors="pt")
res_val=tokenizer(texts_val,padding=True,truncation=True,max_length=128,return_tensors="pt")
res_test=tokenizer(texts_test,padding=True,truncation=True,max_length=128,return_tensors="pt")

y_train_t=torch.tensor(y_train,dtype=torch.long)
y_val_t=torch.tensor(y_val,dtype=torch.long)
y_test_t=torch.tensor(y_test,dtype=torch.long)

train_data=TensorDataset(res_train["input_ids"],res_train["attention_mask"],y_train_t)
train_loader = DataLoader(train_data,batch_size=32,shuffle=True)

optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None
for epoch in range(8):
    model.train()
    for batch_input_ids, batch_attention_mask, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss=outputs.loss
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_output = model(input_ids=res_val["input_ids"], attention_mask=res_val["attention_mask"], labels=y_val_t)
        val_loss = val_output.loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state=copy.deepcopy(model.state_dict())
        else:
            patience_counter+=1
        if patience_counter>=5:
            print(f"Epoch:{epoch}")
            break
        prediction = torch.argmax(val_output.logits,dim=1)
        accuracy=(prediction==y_val_t).float().mean()
    model.train()
    scheduler.step(val_loss.item())
    current_lr=optimizer.param_groups[0]['lr']
    print(f"Epoch:{epoch}, Accuracy: {accuracy.item():.4f},Val Loss:{val_loss.item():.4f},LR: {current_lr:.8f}")
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    test_output = model(input_ids=res_test["input_ids"], attention_mask=res_test["attention_mask"], labels=y_test_t)
    test_loss = test_output.loss
    test_prediction = torch.argmax(test_output.logits, dim=1)
    test_accuracy = (test_prediction == y_test_t).float().mean()
print(f"Final Accuracy: {test_accuracy.item():.4f},Test Loss:{test_loss.item():.4f}")
SAVE_DIR = Path(__file__).parent / "saved_model"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Модель сохранена в {SAVE_DIR}")