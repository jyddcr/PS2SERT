import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import numpy as np
from tqdm import tqdm
import json
import utils
import pandas as pd
import random
import argparse

seed = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(seed)

class RegressionHead(nn.Module):
    def __init__(self, d_model: int, num_outputs: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, d_model]
        return self.regression_head(x)


class PeptideTokenizer:
    def __init__(self, vocab_list, max_seq_len=50):
        self.vocab_list = vocab_list
        self.max_seq_len = max_seq_len

        self.token_to_id = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        self.pad_token_id = self.token_to_id['<PAD>']
        for idx, token in enumerate(vocab_list, start=2):
            self.token_to_id[token] = idx

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.vocab_size = len(self.token_to_id)

    def encode(self, sequence, padding=True, truncation=True):
        input_ids = []
        for token in sequence:
            input_ids.append(self.token_to_id.get(
                token, self.token_to_id['<UNK>']))

        if truncation and len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]

        attention_mask = [1] * len(input_ids)
        if padding:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids += [self.token_to_id['<PAD>']] * pad_len
            attention_mask += [0] * pad_len

        return input_ids, attention_mask

    def decode(self, input_ids):
        sequence = []
        for idx in input_ids:
            token = self.id_to_token.get(idx, '<UNK>')
            if token != '<PAD>':
                sequence.append(token)
        return sequence

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            pe = self.pe[:x.size(1)].squeeze(1)
            x = x + pe.unsqueeze(0).to(x.device)
        else:
            pe = self.pe[:x.size(0)].squeeze(1)
            x = x + pe.unsqueeze(1).to(x.device)
        return self.dropout(x)


class PhysicoChemicalVectorFusion(nn.Module):
    def __init__(self, d_model=128, device=None):
        super().__init__()
        self.vec_dims = [1, 1, 1, 3, 1, 273]
        self.projection = []
        for dim in self.vec_dims:
            self.projection.append(nn.Linear(dim, d_model, device=device))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, seq_embed, weight, net_charge, charge_density, structural, pI, ctd):
        out = []
        extVecList = [weight, net_charge, charge_density, structural, pI, ctd]
        for idx, _ in enumerate(self.vec_dims):
            tmp_vec_proj = self.projection[idx](extVecList[idx])
            tmp_vec_proj = tmp_vec_proj.unsqueeze(1)
            out.append(tmp_vec_proj)
        out.append(seq_embed)
        fused = torch.cat(out, dim=1)
        return self.dropout(self.norm(fused))

    def get_ext_len(self):
        return len(self.vec_dims)

class TransformerRegressor(nn.Module):
    def __init__(self,
                 tokenizer: PeptideTokenizer,
                 vector_fusion: PhysicoChemicalVectorFusion,
                 d_model=128,
                 num_encoder_layers=2,
                 nhead=4,
                 dim_feedforward=256,
                 max_seq_len=250,
                 dropout=0.1
                 ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(
            tokenizer.vocab_size, d_model, padding_idx=tokenizer.pad_token_id)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1)
        )
       
        self.additional_input_projection = vector_fusion

        self.pos_encoder = PositionalEncoding(
            d_model, max_seq_len+self.additional_input_projection.get_ext_len(), dropout)
        

        self.attention_weights = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.normal_(self.regression_head[2].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.regression_head[2].bias, 0.0)
        nn.init.normal_(self.regression_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.regression_head[-1].bias, 0.0)
        for idx, _ in enumerate(self.additional_input_projection.projection):
            nn.init.xavier_uniform_(
                self.additional_input_projection.projection[idx].weight)
            nn.init.constant_(
                self.additional_input_projection.projection[idx].bias, 0.0)

    def forward(self, letter_ids, weight, net_charge, charge_density, structural, pI, ctd, src_key_padding_mask=None):
        embedded_letters = self.embedding(letter_ids)
        combined_src = self.additional_input_projection(
            embedded_letters, weight, net_charge, charge_density, structural, pI, ctd)
        if src_key_padding_mask is not None:
            additional_mask = torch.zeros(
                src_key_padding_mask.size(0),
                self.additional_input_projection.get_ext_len(),
                dtype=src_key_padding_mask.dtype,
                device=src_key_padding_mask.device
            )
            combined_mask = torch.cat(
                [additional_mask, src_key_padding_mask], dim=1)
        else:
            combined_mask = None
        combined_src = self.pos_encoder(combined_src)
        memory = self.transformer_encoder(
            combined_src, src_key_padding_mask=combined_mask)
        if combined_mask is not None:
            mask = ~combined_mask.unsqueeze(-1)
            weights = self.attention_weights(memory)
            weights = weights * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            aggregated = (memory * weights).sum(dim=1)
        else:
            weights = self.attention_weights(memory)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            aggregated = (memory * weights).sum(dim=1)
        output = self.regression_head(aggregated)
        return output

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer: PeptideTokenizer):
        '''
        weight, net_charge, charge_density, structural, pI, ctd
        '''
        self.tokenizer = tokenizer
        self.letter_sequences = []
        self.weight = []
        self.net_charge = []
        self.charge_density = []
        self.structural = []
        self.pI = []
        self.ctd = []
        self.mic = []
        with open(data_path, "r") as dr:
            lines = dr.readlines()
            for line in lines:
                tmp = line.split("\t")
                if len(tmp) < 2:
                    continue
                if len(tmp[0]) < 3:
                    continue
                self.letter_sequences.append(str(tmp[0]).upper())
                ext_infos = utils.get_seq_ext_info(tmp[0])
                exts = []
                for idx, ext_info in enumerate(ext_infos):
                    tmp_ext = None
                    if not isinstance(ext_info, list):
                        tmp_ext = [ext_info]
                    else:
                        tmp_ext = ext_info
                    exts.append(torch.tensor(tmp_ext,dtype=torch.float))
                self.weight.append(exts[0])
                self.net_charge.append(exts[1])
                self.charge_density.append(exts[2])
                self.structural.append(exts[3])
                self.pI.append(exts[4])
                self.ctd.append(exts[5])
                self.mic.append(torch.tensor(float(tmp[1]), dtype=torch.float))

        # ----------------------------------------

    def __len__(self):
        return len(self.mic)

    def __getitem__(self, idx):
        encoded_letters, _ = self.tokenizer.encode(self.letter_sequences[idx])
        return self.letter_sequences[idx],torch.tensor(encoded_letters), self.weight[idx],  self.net_charge[idx], self.charge_density[idx], self.structural[idx], self.pI[idx], self.ctd[idx], self.mic[idx], len(self.letter_sequences[idx])

def collate_fn(batch):
    batch.sort(key=lambda x: x[8], reverse=True)
    letter_sequences,encoded_letters_list,  weight, net_charge, charge_density, structural, pI, ctd, mic, seq_lens = zip(
        *batch)
    padded_letter_ids = nn.utils.rnn.pad_sequence(
        encoded_letters_list, batch_first=True, padding_value=0)
    stacked_weight = torch.stack(weight)
    stacked_net_charge = torch.stack(net_charge)
    stacked_charge_density = torch.stack(charge_density)
    stacked_structural = torch.stack(structural)
    stacked_pI = torch.stack(pI)
    stacked_ctd = torch.stack(ctd)
    stacked_mic = torch.stack(mic)

    batch_size, max_seq_len = padded_letter_ids.shape
    src_key_padding_mask = torch.zeros(
        batch_size, max_seq_len, dtype=torch.bool)
    for i, seq_len in enumerate(seq_lens):
        if seq_len < max_seq_len:
            src_key_padding_mask[i, seq_len:] = True

    return letter_sequences,padded_letter_ids, stacked_weight, stacked_net_charge, stacked_charge_density, stacked_structural, stacked_pI, stacked_ctd, stacked_mic, src_key_padding_mask

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    batch_num = 0
    for batch_idx, (_,letter_ids, weight, net_charge, charge_density, structural, pI, ctd, mic, mask) in enumerate(tqdm(train_loader, desc="Training")):
        letter_ids = letter_ids.to(device)
        weight = weight.to(device)
        net_charge = net_charge.to(device)
        charge_density = charge_density.to(device)
        structural = structural.to(device)
        pI = pI.to(device)
        ctd = ctd.to(device)
        mic = mic.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        output = model(letter_ids, weight, net_charge, charge_density, structural, pI, ctd,
                       src_key_padding_mask=mask)
        mic_log = torch.log10(mic.unsqueeze(1) + eps)
        loss = criterion(output, mic_log)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batch_num = batch_idx + 1
    avg_loss = total_loss / batch_num

    return avg_loss


def evaluate(model, val_loader, criterion, epoch,device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_preds_log = [] 
    all_targets = []
    all_targets_log = []
    all_seqs = []
    batch_num = 0
    with torch.no_grad():
        for batch_idx,(letter_seqs,letter_ids, weight, net_charge, charge_density, structural, pI, ctd, mic, mask) in enumerate(tqdm(val_loader, desc="Evaluating")):
            letter_ids = letter_ids.to(device)
            weight = weight.to(device)
            net_charge = net_charge.to(device)
            charge_density = charge_density.to(device)
            structural = structural.to(device)
            pI = pI.to(device)
            ctd = ctd.to(device)
            mic = mic.to(device)
            mask = mask.to(device)
            output = model(letter_ids, weight, net_charge, charge_density, structural, pI, ctd,
                       src_key_padding_mask=mask)
            mic_log = torch.log10(mic.unsqueeze(1) + eps)
            loss = criterion(output, mic_log)
            total_loss += loss.item()
            all_preds.extend((torch.pow(10,output) - eps).squeeze().cpu().numpy())
            all_preds_log.extend(output.squeeze().cpu().numpy())
            all_targets.extend(mic.cpu().numpy())
            all_targets_log.extend(mic_log.squeeze().cpu().numpy())
            all_seqs.extend(letter_seqs)
            batch_num = batch_idx + 1
    out = {
        "seq":all_seqs,
        "preds":all_preds,
        "mic":all_targets
    }
    df = pd.DataFrame(out)
    df.to_csv(f"./output/{train_idx}_out_{epoch}.tsv",index=False,sep="\t")
    avg_loss = total_loss / batch_num
    mae = np.mean(np.abs(np.array(all_preds_log) - np.array(all_targets_log)))
    preds_array = np.array(all_preds_log)
    targets_array = np.array(all_targets_log)
    errors = preds_array - targets_array
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    info_dict = {
        'MAE': mae,
        'MSE': mse, 
        'RMSE': rmse,
    }
    return avg_loss, mae, info_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_parameters_detailed(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen, trainable + frozen

def main():
    check_hyperparameters = [
        'd_model',
        'num_encoder_layers',
        'nhead',
        'dim_feedforward',
        'max_seq_len',
        'dropout',
        'batch_size',
        'lr',
        'weight_decay',
        'epochs',
        'patience',
    ]
    hpf=open(args.hyperparameters,'r')
    hyperparameters = json.load(hpf)
    hpf.close()
    for check_hp in check_hyperparameters:
        if hyperparameters.get(check_hp) is None:
            raise ValueError(check_hp)
    hyperparameters['eps'] = eps
    hyperparameters['seed'] = seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    tokenizer = PeptideTokenizer(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L',
                                 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'], hyperparameters['max_seq_len'])
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    train_dataset = CustomDataset(
        args.train, tokenizer)
    val_dataset = CustomDataset(
        args.val, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=hyperparameters['batch_size'], shuffle=False, collate_fn=collate_fn)

    model = TransformerRegressor(
        tokenizer=tokenizer,
        vector_fusion=PhysicoChemicalVectorFusion(d_model=hyperparameters['d_model'],device=device),
        d_model=hyperparameters['d_model'], num_encoder_layers=hyperparameters['num_encoder_layers'],
        nhead=hyperparameters['nhead'], dim_feedforward=hyperparameters['dim_feedforward'],
        max_seq_len=hyperparameters['max_seq_len'], dropout=hyperparameters['dropout']
    ).to(device)
   
    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=hyperparameters['lr'], 
        weight_decay=hyperparameters['weight_decay']
        )
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', patience=hyperparameters['patience'], factor=0.5)

    torch.save(tokenizer, 'PeptideTokenizer.pth')
    print("Tokenizer saved to ' PeptideTokenizer.pth'")

    with open(f"./output/{train_idx}_model_conf.json","w") as mcf:
        mcf.write(json.dumps(hyperparameters,indent=4))

    for epoch in range(1, hyperparameters['epochs'] + 1):
        print(f"\nEpoch {epoch}/{hyperparameters['epochs']}")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae,val_info = evaluate(
            model, val_loader, criterion, epoch,device)

        scheduler.step(val_loss)
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
        writer.add_scalars('Loss Comparison', {
            'Train Loss': train_loss,
            'Validation Loss': val_loss,
        }, epoch)
        writer.add_scalar('Val MAE', val_mae, epoch)
        for i in val_info:
            writer.add_scalar(f"Val {i}",val_info[i],epoch)
        model_path = f"./output/{train_idx}_model_epoch_{epoch}.pth"
        torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss, 'hyperparameters': hyperparameters
            }, model_path)
        print(f"model saved to {model_path}")
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PS2SERT")

    parser.add_argument("--train", type=str, default="train.tsv", help="train")
    parser.add_argument("--val", type=str, default="val.tsv", help="train")
    parser.add_argument("--hyperparameters", type=str, default="conf.json", help="config")
    parser.add_argument("--name", type=str, default="MSE_Log_EC", help="name")
    args = parser.parse_args()
    eps = 1e-6
    train_idx = f"{args.name}_seed_{seed}"
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'./runs/{train_idx}')
    main()
