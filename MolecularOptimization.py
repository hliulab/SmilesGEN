import argparse
import os
import torch
import pandas as pd
from torch import nn
from utils import *
from dataset import pre_load_smiles_data, load_smiles_data, load_test_gene_data
from model import create_smiles_model, GeneVAE
from tokenizer import vocabulary
from utils import get_device

epoch_lst = [100, 200]
parser = argparse.ArgumentParser(description='SmilesGEN parse')

parser.add_argument("--result_file", type=str, default="123",
                    help="Result file name")
i = "mcf7"

cell_name = "MCF7"

protein_list = ["AKT1", "AKT2", "AURKB", "CTSK",
                "EGFR", "HDAC1", "MTOR", "PIK3CA",
                "SMAD3", "TP53",
                "TNBC", "RC", "LUAD",
                "A1NYA", "A1NYB", "A1QB",
                "A2NYA", "A2NYB", "A2QB"]
a = protein_list[0]


lr_lst = [5e-4, 1e-4, 1e-4]
epoch_lst = [100, 200]
parser = argparse.ArgumentParser(description='SmilesGEN parse')

parser.add_argument("--result_file", type=str, default="123",
                    help="Result file name")

file_path = os.path.join(f"results/{cell_name}/", str(i))
parser.add_argument("--use_seed", action="store_true",
                    help="Apply seed for reproduce experimental results")
parser.add_argument("--use_all", action="store_true",
                    help="Apply seed for reproduce experimental results")
parser.add_argument("--just_cal", action="store_true",
                    help="Apply seed for reproduce experimental results")
parser.add_argument("--cell_name", type=str, default="MCF7",
                    help="Cell name of LINCS files, e.g., mcf7")
parser.add_argument("--protein_name", type=str, default=f"{a}",
                    help="10 proteins are AKT1, AKT2, AURKB, CTSK, EGFR, HDAC1, MTOR, PIK3CA, SMAD3, and TP53")
parser.add_argument("--model", type=str, default="RNN",
                    help="Model type, e.g., RNN, Transformer, GRU, LSTM")
parser.add_argument("--smiles_dropout", type=float, default=0.1,
                    help="Dropout rate for SmilesGEN")
# ============================================
# pre-train parameters
parser.add_argument("--pre_train_smiles_vae", action="store_true",
                    help="Pre-train SmilesVAE")
parser.add_argument("--test_smiles_vae", action="store_true",
                    help="Test SmilesVAE")
parser.add_argument("--smiles_epochs", type=int, default=epoch_lst[0],
                    help="Number of training epochs for SmilesVAE")
parser.add_argument("--emb_size", type=int, default=128,
                    help="Embedding size for SmilesVAE")
parser.add_argument("--hidden_size", type=int, default=256,
                    help="Hidden layer size for SmilesVAE")  # 192
parser.add_argument("--num_layers", type=int, default=3,
                    help="Number of training layers for SmilesVAE")
parser.add_argument('--latent_size', type=int, default=64,
                    help='Latent vector dimension of SmilesVAE', )
parser.add_argument("--pre_train_smiles_lr", type=float, default=lr_lst[0],
                    help="Learning rate for Pre-Train SmilesVAE")
parser.add_argument('--bidirectional', type=bool, default='True',
                    help='Apply bidirectional RNN')
parser.add_argument('--temperature', type=float, default=1,
                    help='Temperature of the SMILES VAE')
parser.add_argument('--train_rate', type=float, default=0.9,
                    help='Split training and validating subsets by training rate')
parser.add_argument('--max_len', type=int, default=100,
                    help='Maximum length of SMILES strings')
parser.add_argument('--saved_pre_smiles_vae', type=str, default=f'{file_path}/saved_model/saved_pre_smiles_vae',
                    help='Save the pre-trained SmilesVAE')
parser.add_argument('--pre_train_valid_smiles_file', type=str,
                    default=f'{file_path}/pre_train/pre_train_predicted_valid_smiles',
                    help='Save the valid SMILES into file', )
parser.add_argument('--pre_train_final_smiles_file', type=str,
                    default=f'{file_path}/pre_train/pre_train_final_smiles',
                    help='Save the valid SMILES into file', )
parser.add_argument('--smiles_vae_pre_train_results', type=str,
                    default=f'{file_path}/pre_train/smiles_vae_pre_train_results',
                    help='Path to save the results of pre-trained SmilesVAE')
parser.add_argument('--variant', action='store_true',
                    help='Apply variant smiles')
# ===========================
# train SmilesGEN
parser.add_argument('--train', action='store_true',
                    help='Train GeneVAE')
parser.add_argument('--test_gene_vae', action='store_true',
                    help='Validate GeneVAE')
parser.add_argument('--generation', action='store_true',
                    help='Validate GeneVAE')

parser.add_argument('--train_epochs', type=int, default=epoch_lst[1],
                    help='GeneVAE training epochs')
parser.add_argument('--gene_num', type=int, default=978,
                    help='Number of gene values')
parser.add_argument('--gene_hidden_sizes', type=int, default=[512, 256, 192],
                    help='Hidden layer sizes of GeneVAE')  
parser.add_argument('--gene_lr', type=float, default=lr_lst[2],
                    help='Learning rate of GeneVAE')  # MCF7: 1e-4
parser.add_argument('--gene_batch_size', type=int, default=64,
                    help='Batch size for training GeneVAE')  # 64
parser.add_argument('--gene_dropout', type=float, default=0.2,
                    help='Dropout probability')
parser.add_argument('--gene_expression_file_path', type=str, default=f'datasets/LINCS/{cell_name}/',
                    help='Path of the training gene expression profile dataset for the VAE')
parser.add_argument('--test_gene_data', type=str, default='datasets/test_protein/',
                    help='Path of the gene expression profile dataset for test proteins or test disease')
parser.add_argument('--saved_gene_vae', type=str, default=f'{file_path}/saved_model/saved_gene_vae',
                    help='Save the trained GeneVAE')
parser.add_argument('--gene_vae_train_results', type=str,
                    default=f'{file_path}/train_results/gene_vae_train_results.csv',
                    help='Path to save the results of trained GeneVAE')
parser.add_argument('--one_gene_density_figure', type=str,
                    default=f'{file_path}/evaluate/one_gene_density_figure.pdf',
                    help='Path to save the density figures of gene data')
parser.add_argument('--all_gene_density_figure', type=str,
                    default=f'{file_path}/evaluate/all_gene_density_figure.pdf',
                    help='Path to save the density figures of gene data')
parser.add_argument('--smiles_lr', type=float, default=lr_lst[1],
                    help='Learning rate of Train SmilesVAE')
parser.add_argument('--smiles_vae_train_results', type=str,
                    default=f'{file_path}/train_results/smiles_vae_train_results',
                    help='Path to save the results of trained SmilesVAE')
parser.add_argument('--valid_smiles_file', type=str, default=f'{file_path}/train_results/predicted_valid_smiles',
                    help='Save the valid SMILES into file')
parser.add_argument('--final_smiles_file', type=str, default=f'{file_path}/train_results/final_smiles',
                    help='Save the valid SMILES into file')
parser.add_argument('--saved_smiles_vae', type=str, default=f'{file_path}/saved_model/saved_smiles_vae',
                    help='Save the trained SmilesVAE', )

# ===========================
# Molecule selection with similar ligands
parser.add_argument('--calculate_tanimoto', action='store_true',
                    help='Calculate tanimoto similarity for the source ligand and generated SMILES')
# Add --calculate_tanimoto to calculate Tanimoto similarity
parser.add_argument('--candidate_num', type=int, default=50,
                    help='Number of candidate SMILES strings')
parser.add_argument('--gene_type', type=str, default='gene_symbol',
                    help='Gene types')
parser.add_argument('--source_path', type=str, default='datasets/ligands/source_',
                    help='Load the source SMILES strings of known ligands')
parser.add_argument('--gen_path', type=str, default=f'{file_path}/generation/',
                    help='Save the generated SMILES strings')

parser.add_argument('--mol_figure_path', type=str, default=f'{file_path}/evaluate/mol_img/',
                    help='Save the image of model')

args = parser.parse_args()
rng = set_seed(42)
tokenizer = vocabulary(args)
tokenizer.build_vocab()

device = get_device()

smiles_vae = create_smiles_model(
    args.model, args.emb_size,
    args.hidden_size, args.num_layers,
    args.latent_size, args.bidirectional,
    tokenizer, device, dropout=args.smiles_dropout)

gene_vae = GeneVAE(
    args.gene_num, args.gene_hidden_sizes,
    args.latent_size, args.gene_num,
    nn.ReLU(), args.gene_dropout).to(device)

l = ["AKT1", "AKT2", "AURKB", "CTSK",
     "EGFR", "HDAC1", "MTOR", "PIK3CA",
     "SMAD3", "TP53", ]
hhh = 1
args.protein_name = l[hhh - 1]
smiles = "C1=CC=CC=C1"

candidate_num = 50
max_len = 60
smiles_vae.load_model(
    r"SmilesGEN\results\MCF7\saved_model\saved_smiles_vae_RNN.pkl")
gene_vae.load_model(
    r"SmilesGEN\results\MCF7\saved_model\saved_gene_vae_MCF7_RNN.pkl")

test_gene_loader = load_test_gene_data(
    args.test_gene_data, args.cell_name, args.protein_name, args.gene_type, args.gene_batch_size)

smiles_vae.eval()
gene_vae.eval()
res_smiles = []
for _, genes in enumerate(test_gene_loader):
    genes = genes.to(device)

    gene_latent_vectors = gene_vae.encode(genes)
    encoded_smi = tokenizer.encode(smiles)
    encoded_smi = torch.tensor(encoded_smi).unsqueeze(0).to(get_device())
    rand_z, _, _ = smiles_vae.encode(encoded_smi)
    rand_z = rand_z.repeat(candidate_num, 1)

    gene_latent_vectors = gene_latent_vectors.repeat(candidate_num, 1)

    dec_sampled_char = smiles_vae.generation(rand_z, max_len, tokenizer, gene_latent_vectors)

    output_smiles = ["".join(tokenizer.decode(
        dec_sampled_char[i].squeeze().detach().cpu().numpy())).strip("^$ ").split("$")[0]
                     for i in range(dec_sampled_char.size(0))]
    res_smiles.append(output_smiles)
test_data = pd.DataFrame(columns=['SMILES'], data=res_smiles[0])


test_data.to_csv(file_path + f'res-{args.protein_name}.csv', index=False)
