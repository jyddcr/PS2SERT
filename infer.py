import torch
from torch import nn
import numpy as np
import json
from train import TransformerRegressor, PeptideTokenizer, PhysicoChemicalVectorFusion
import sys
from torch.utils.data import DataLoader, Dataset
import utils
from tqdm import tqdm
import random
import numpy as np
import torch

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


class CustomDatasetInfer(Dataset):
    def __init__(self, seq_list, tokenizer: PeptideTokenizer):
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
        for seq in seq_list:
            if len(seq) < 3:
                continue
            self.letter_sequences.append(str(seq).upper())
            ext_infos = utils.get_seq_ext_info(str(seq).upper())
            exts = []
            for idx, ext_info in enumerate(ext_infos):
                tmp_ext = None
                if not isinstance(ext_info, list):
                    tmp_ext = [ext_info]
                else:
                    tmp_ext = ext_info
                exts.append(torch.tensor(tmp_ext, dtype=torch.float))
            self.weight.append(exts[0])
            self.net_charge.append(exts[1])
            self.charge_density.append(exts[2])
            self.structural.append(exts[3])
            self.pI.append(exts[4])
            self.ctd.append(exts[5])

        # ----------------------------------------

    def __len__(self):
        return len(self.letter_sequences)

    def __getitem__(self, idx):
        encoded_letters, _ = self.tokenizer.encode(self.letter_sequences[idx])
        return self.letter_sequences[idx], torch.tensor(encoded_letters), self.weight[idx],  self.net_charge[idx], self.charge_density[idx], self.structural[idx], self.pI[idx], self.ctd[idx], len(self.letter_sequences[idx])


def collate_fn(batch):
    batch.sort(key=lambda x: x[8], reverse=True)
    letter_sequences, encoded_letters_list,  weight, net_charge, charge_density, structural, pI, ctd, seq_lens = zip(
        *batch)

    # Padding
    padded_letter_ids = nn.utils.rnn.pad_sequence(
        encoded_letters_list, batch_first=True, padding_value=0)
    stacked_weight = torch.stack(weight)
    stacked_net_charge = torch.stack(net_charge)
    stacked_charge_density = torch.stack(charge_density)
    stacked_structural = torch.stack(structural)
    stacked_pI = torch.stack(pI)
    stacked_ctd = torch.stack(ctd)

    batch_size, max_seq_len = padded_letter_ids.shape
    src_key_padding_mask = torch.zeros(
        batch_size, max_seq_len, dtype=torch.bool)
    for i, seq_len in enumerate(seq_lens):
        if seq_len < max_seq_len:
            src_key_padding_mask[i, seq_len:] = True
    return letter_sequences, padded_letter_ids, stacked_weight, stacked_net_charge, stacked_charge_density, stacked_structural, stacked_pI, stacked_ctd, src_key_padding_mask


def load_model_and_tokenizer(model_path, tokenizer_path, device):
    tokenizer = torch.load(
        tokenizer_path, map_location=device, weights_only=False)
    checkpoint = torch.load(model_path, map_location=device)
    hyperparams = checkpoint['hyperparameters']

    model = TransformerRegressor(
        tokenizer=tokenizer,
        vector_fusion=PhysicoChemicalVectorFusion(
            d_model=hyperparams['d_model'], device=device),
        d_model=hyperparams['d_model'], num_encoder_layers=hyperparams['num_encoder_layers'],
        nhead=hyperparams['nhead'], dim_feedforward=hyperparams['dim_feedforward'],
        max_seq_len=hyperparams['max_seq_len'], dropout=hyperparams['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer, hyperparams


def create_inference_mask(input_ids_list, device):
    seq_lens = [len(ids) for ids in input_ids_list]
    max_seq_len = max(seq_lens)
    batch_size = len(input_ids_list)
    src_key_padding_mask = torch.zeros(
        batch_size, max_seq_len, dtype=torch.bool, device=device)
    for i, seq_len in enumerate(seq_lens):
        if seq_len < max_seq_len:
            src_key_padding_mask[i, seq_len:] = True
    return src_key_padding_mask


def predict(model, tokenizer, text_list, hyperparams, batch_size=1, device=None):
    val_dataset = CustomDatasetInfer(
        text_list, tokenizer)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_preds = []
    all_seqs = []
    output = []
    model.eval()
    with torch.no_grad():
        for _, (letter_seqs, letter_ids, weight, net_charge, charge_density, structural, pI, ctd, mask) in enumerate(val_loader):
            letter_ids = letter_ids.to(device)
            weight = weight.to(device)
            net_charge = net_charge.to(device)
            charge_density = charge_density.to(device)
            structural = structural.to(device)
            pI = pI.to(device)
            ctd = ctd.to(device)
            mask = mask.to(device)

            output = model(letter_ids, weight, net_charge, charge_density, structural, pI, ctd,
                           src_key_padding_mask=mask)
            if batch_size == 1:
                all_preds.extend(
                    (torch.pow(10, output) - hyperparams['eps']).cpu().numpy().squeeze(0))
            else:
                all_preds.extend(
                    (torch.pow(10, output) - hyperparams['eps']).squeeze().cpu().numpy())

            all_seqs.extend(letter_seqs)
    out = {
        "seq": all_seqs,
        "preds": all_preds,
    }
    return out


def main():
    output_path = "./predict_result/"
    if len(sys.argv) != 4:
        print("Usage: python infer.py <path_to_best_model.pth> <path_to_tokenizer.pth> <predict_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    tokenizer_path = sys.argv[2]
    file_name = sys.argv[3]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}...")
    print(f"Loading tokenizer from {tokenizer_path}...")
    model, tokenizer, hyperparams = load_model_and_tokenizer(
        model_path, tokenizer_path, device)
    print("Model and tokenizer loaded successfully.")
    batch_size = 1024
    peptides = []
    result_filename = f"{output_path}/{file_name}_result"
    result_f = open(result_filename, "w")
    with open(file_name, 'r', encoding='utf-8') as file:
        for linex in tqdm(file):
            tmp = linex.strip().split("\t")
            line = tmp[0]
            if len(peptides) == batch_size:
                tmpout = []
                out = predict(model, tokenizer, peptides, hyperparams,
                              device=device, batch_size=batch_size)
                for idx, seq in enumerate(out['seq']):
                    pmic = float(out['preds'][idx])
                    tmpout.append(f"{seq}\t{pmic}")
                result_f.write("\n".join(tmpout))
                result_f.write("\n")
                result_f.flush()
                peptides = []
            peptides.append(line)
        if len(peptides) > 0:
            tmpout = []
            out = predict(model, tokenizer, peptides, hyperparams,
                          device=device, batch_size=batch_size)
            for idx, seq in enumerate(out['seq']):
                pmic = float(out['preds'][idx])
                tmpout.append(f"{seq}\t{pmic}")
            result_f.write("\n".join(tmpout))
            result_f.write("\n")
            result_f.flush()
        result_f.close()


if __name__ == '__main__':
    main()
