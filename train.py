import torch
import pandas as pd
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

batch_size = 48
epoch_count = 1
lr = 1e-4
margin = 0.3
device = 'cuda'
train_size = 0.9
T = 2

def to_device(tokens):
    return {key: val.to(device) for key, val in tokens.items()}

def CalcLoss(en, ru):
    matrix = torch.matmul(en, torch.transpose(ru, 0, 1))
    matrix -= margin * torch.eye(matrix.size(0)).to(device)
    fwd_loss = torch.log_softmax(matrix, dim=0)
    bwd_loss = torch.log_softmax(matrix, dim=1)
    return -torch.mean(fwd_loss.diag() + bwd_loss.diag())

def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for en_tokens, ru_tokens in tqdm(dataloader):
        optimizer.zero_grad()
        
        en_emb = model(to_device(en_tokens))['sentence_embedding']
        ru_emb = model(to_device(ru_tokens))['sentence_embedding']
        
        loss = CalcLoss(en_emb, ru_emb)

        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    print(f"Train loss: {total_loss / len(dataloader):.3f}")

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    for en_tokens, ru_tokens in tqdm(dataloader):
        en_emb = model(to_device(en_tokens))['sentence_embedding']
        ru_emb = model(to_device(ru_tokens))['sentence_embedding']
        loss = CalcLoss(en_emb, ru_emb)

        total_loss += loss.item()
        
    print(f"Eval loss: {total_loss / len(dataloader):.3f}")

def train(model, train_data, eval_data):
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epoch_count):
        print(f"Epoch {epoch}")
        train_epoch(model, train_data, optimizer)
        evaluate(model, eval_data)

class EnRuDataset(Dataset):
    def __init__(self, en, ru, tok):
        self.en = en
        self.ru = ru
        self.tok = tok
    
    def __getitem__(self, index):
        return (self.en[index], self.ru[index])
    
    def __len__(self):
        return len(self.en)
    
    def collate_fn(self, batch):
        en = list(map(lambda x: x[0], batch))
        ru = list(map(lambda x: x[1], batch))
        return self.tok(en), self.tok(ru)
    
train_df = pd.read_csv('data/en-ru/train.csv')
train_df.head(5)

model = SentenceTransformer("sentence-transformers/LaBSE")
config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['query', 'key', 'value', 'dense'],
    lora_dropout=0.1,
)

lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()


train_en, val_en, train_ru, val_ru = train_test_split(
    train_df['en'].to_list(),
    train_df['ru'].to_list(),
    train_size=train_size
)
train_ds = EnRuDataset(train_en, train_ru, lora_model.tokenize)
val_ds = EnRuDataset(val_en, val_ru, lora_model.tokenize)


train_dataloader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=train_ds.collate_fn
)

val_dataloader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=val_ds.collate_fn
)

evaluate(lora_model, val_dataloader)
train(lora_model, train_dataloader, val_dataloader)
torch.save(lora_model, 'lora_labse2')