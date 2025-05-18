import json
import os

import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from utils import mean_similarity

from rdkit import rdBase
import matplotlib.pyplot as plt

rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')


class Trainer:
    def __init__(self, model: str, smiles_vae, gene_vae,
                 smile_vae_optimizer, gene_vae_optimizer: torch.optim.Optimizer, device):
        self.model = model
        self.smiles_vae = smiles_vae
        self.gene_vae = gene_vae
        self.smile_vae_optimizer = smile_vae_optimizer
        self.gene_vae_optimizer = gene_vae_optimizer
        self.device = device

    def pre_train_smiles_vae(
            self,
            train_smile_dataloader, valid_smile_dataloader, tokenizer,
            smiles_vae_pre_train_results,
            smiles_epochs, temperature, max_len,
            pre_train_valid_smiles_file, pre_train_final_smiles_file, saved_pre_smiles_vae, ):
        temperature = 1.0
        pre_train_file_dir = os.path.dirname(pre_train_valid_smiles_file)
        os.makedirs(pre_train_file_dir, exist_ok=True)

        saved_model_file_dir = os.path.dirname(saved_pre_smiles_vae)
        os.makedirs(saved_model_file_dir, exist_ok=True)

        with open(smiles_vae_pre_train_results + "_" + self.model + ".csv", 'a+') as wf:
            # wf.truncate(0)
            wf.write("================================================\n")
            wf.write(
                '{},{},{},{},{},{},{},{},{}\n'.format(
                    'Epoch',
                    'Joint_loss',
                    'Rec_loss',
                    'KLD_loss',
                    'Total',
                    'Valid(Valid_rate)',
                    'Unique(Unique_rate)',
                    'Novelty(Novel_rate)',
                    'Diversity',
                )
            )
        print('\n')
        print('Pre Training Information:')

        for epoch in range(smiles_epochs):
            total_joint_loss = 0
            total_rec_loss = 0
            total_kld_loss = 0
            self.smiles_vae.train()

            for _, smiles in tqdm(enumerate(train_smile_dataloader), total=len(train_smile_dataloader),
                                  desc='Epoch {:d} / {:d}'.format(epoch + 1, smiles_epochs)):
                smiles = smiles.to(self.device)

                _, decoded = self.smiles_vae(smiles, None, temperature)
                alphas = (torch.cat([torch.linspace(0.99, 0.5, int(smiles_epochs * 0.8)),
                                     0.5 * torch.ones(smiles_epochs - int(smiles_epochs * 0.8)), ]).double().to(
                    self.device))

                joint_loss, rec_loss, kld_loss = self.smiles_vae.joint_loss(
                    decoded, targets=smiles, alpha=alphas[epoch], beta=1.0)

                self.smile_vae_optimizer.zero_grad()
                joint_loss.backward()
                self.smile_vae_optimizer.step()

                total_joint_loss += joint_loss.item()
                total_rec_loss += rec_loss.item()
                total_kld_loss += kld_loss.item()

            mean_joint_loss = total_joint_loss / smiles.size(0)
            mean_rec_loss = total_rec_loss / (smiles.size(0))
            mean_kld_loss = total_kld_loss / (smiles.size(0))

            self.smiles_vae.eval()
            valid_smiles = []
            label_smiles = []
            total_num_data = len(valid_smile_dataloader.dataset)

            for _, smiles in enumerate(valid_smile_dataloader):

                smiles = smiles.to(self.device)

                latent_smile, _, _ = self.smiles_vae.encode(smiles)

                dec_sampled_char = self.smiles_vae.generation(latent_smile, max_len, tokenizer)

                output_smiles = ["".join(tokenizer.decode(
                    dec_sampled_char[i].squeeze().detach().cpu().numpy())).strip("^$ ").split("$")[0]
                                 for i in range(dec_sampled_char.size(0))]
                for i in range(len(output_smiles)):
                    mol = Chem.MolFromSmiles(output_smiles[i])
                    if (mol is not None
                            and mol.GetNumAtoms() > 1
                            and Chem.MolToSmiles(mol) != ' '):
                        valid_smiles.extend([output_smiles[i]])
                        label_smiles.extend(
                            ["".join(tokenizer.decode(
                                smiles[i].squeeze().detach().cpu().numpy())).strip("^$ ")])

            unique_smiles = list(set(valid_smiles))
            novel_smiles = [smi for smi in unique_smiles if smi not in label_smiles]

            pd.DataFrame(valid_smiles).to_csv(pre_train_valid_smiles_file + "_" + self.model + ".csv", index=False)

            valid_num = len(valid_smiles)
            valid_rate = 100 * len(valid_smiles) / total_num_data
            unique_num = len(unique_smiles)
            novel_num = len(novel_smiles)

            if valid_num != 0:
                unique_rate = 100 * unique_num / valid_num
                diversity = mean_similarity(valid_smiles, label_smiles)
            else:
                unique_rate = 100 * unique_num / (valid_num + 1)
                diversity = 1

            if unique_num != 0:
                novel_rate = 100 * novel_num / unique_num
            else:
                novel_rate = 100 * novel_num / (unique_num + 1)

            print(
                f'Epoch: {epoch + 1:d} / {smiles_epochs:d}, joint_loss: {mean_joint_loss:.3f}, rec_loss: {mean_rec_loss:.3f}, kld_loss: {mean_kld_loss:.3f}, Total: {total_num_data:d}, valid: {valid_num:d} ({valid_rate:.2f}), unique: {unique_num:d} ({unique_rate:.2f}), novel: {novel_num:d} ({novel_rate:.2f}), diversity: {diversity:.3f}'
            )

            with open(smiles_vae_pre_train_results + "_" + self.model + ".csv", 'a+') as wf:
                wf.write(
                    f'{epoch + 1},{mean_joint_loss:.3f},{mean_rec_loss:.3f},{mean_kld_loss:.3f},{total_num_data},{valid_num}({valid_rate:.2f}),{unique_num}({unique_rate:.2f}),{novel_num}({novel_rate:.2f}),{diversity:.3f}\n'
                )
            final_smiles = {'predict': valid_smiles, 'label': label_smiles}
            final_smiles = pd.DataFrame(final_smiles)

            final_smiles.to_csv(pre_train_final_smiles_file + "_" + self.model + ".csv", index=False)

        print('=' * 50)
        self.smiles_vae.save_model(saved_pre_smiles_vae + "_" + self.model + ".pkl")
        print(f'Pre-Trained SmilesVAE is saved in {saved_pre_smiles_vae + "_" + self.model + ".pkl"}')

        return self.smiles_vae

    def train(self, train_smile_dataloader, valid_smile_dataloader, tokenizer,
              train_epochs, temperature,
              latent_size, emb_size, max_len,
              valid_smiles_file, smiles_vae_train_results,
              saved_smiles_vae, saved_gene_vae, cell_name):

        self.gene_vae.train()
        self.smiles_vae.train()
        temperature = 1
        saved_model_file_dir = os.path.dirname(saved_smiles_vae)
        os.makedirs(saved_model_file_dir, exist_ok=True)
        train_result_file_dir = os.path.dirname(smiles_vae_train_results)
        os.makedirs(train_result_file_dir, exist_ok=True)
        with open(smiles_vae_train_results + "_" + self.model + ".csv", 'a+') as wf:
            # wf.truncate(0)
            wf.write("================================================\n")
            wf.write(
                '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                    'Epoch',
                    'gene_joint_loss',
                    'gene_rec_loss',
                    'gene_kld_loss',
                    'pert_gene_joint_loss',
                    'pert_gene_rec_loss',
                    'pert_gene_kld_loss',
                    'smile_joint_loss',
                    'smile_rec_loss',
                    'smile_kld_loss',
                    'Total',
                    'Valid_num(Valid_rate)',
                    'Unique_num(Unique_rate)',
                    'Novelty_num(Novel_rate)',
                    'Diversity',
                )
            )
        print('\n')
        print('Training Information:')

        with open(f'datasets/LINCS/{cell_name}/landmark_dict.json', 'r') as f:
            files = json.load(f)

        dfs = {}
        for cell_type, file_path in files.items():
            dfs[cell_type] = pd.read_csv(file_path, sep=',').iloc[:, 1:].values

        def row_wise_scale(df):
            row_mins = df.min(axis=1)
            row_maxs = df.max(axis=1)
            scaled_df = (df - row_mins[:, np.newaxis]) / (row_maxs[:, np.newaxis] - row_mins[:, np.newaxis])
            return scaled_df

        # 应用标准化
        for cell_type, df in dfs.items():
            # dfs[cell_type] = row_wise_scale(df)
            dfs[cell_type] = df

        gene_label = {}
        for i, j in files.items():
            gene_label[i] = dfs[i].mean(axis=0)

        for epoch in range(train_epochs):
            total_gene_joint_loss = 0
            total_gene_rec_loss = 0
            total_gene_kld_loss = 0

            total_pert_gene_joint_loss = 0
            total_pert_gene_rec_loss = 0
            total_pert_gene_kld_loss = 0
            total_smile_joint_loss = 0
            total_smile_rec_loss = 0
            total_smile_kld_loss = 0
            count = 0
            self.smiles_vae.train()

            l1 = []
            l2 = []
            l3 = []

            for _, (smiles, genes, cell) in tqdm(enumerate(train_smile_dataloader), total=len(train_smile_dataloader),
                                                 desc='Epoch {:d} / {:d}'.format(epoch + 1, train_epochs)):

                smiles, genes = smiles.to(self.device), genes.to(self.device)

                ctr_genes = torch.zeros((genes.shape[0], genes.shape[1]), dtype=torch.float32).to(self.device)

                for i, label in enumerate(cell):
                    ctr_genes[i] = torch.tensor(gene_label[label], dtype=torch.float32).to(self.device)

                smile_latent_vectors, smiles_mu, smiles_logvar = self.smiles_vae.encode(smiles)

                pert_gene_latent_vectors = self.gene_vae.encode(genes)

                decoded_gene_pert = self.gene_vae.decode(pert_gene_latent_vectors)

                gene_latent_vectors = self.gene_vae.encode(genes, smiles_mu, smiles_logvar)

                decoded_gene = self.gene_vae.decode(gene_latent_vectors)

                decoded = self.smiles_vae.decode(smiles, smile_latent_vectors, pert_gene_latent_vectors, temperature)

                alphas = (torch.cat([torch.linspace(0.99, 0.5, int(train_epochs * 0.5)),
                                     0.5 * torch.ones(train_epochs - int(train_epochs * 0.5)), ]).double().to(
                    self.device))

                smile_joint_loss, smile_rec_loss, smile_kld_loss = (
                    self.smiles_vae.joint_loss(decoded, targets=smiles, alpha=0.99, beta=1.0))

                gene_joint_loss, gene_rec_loss, gene_kld_loss = self.gene_vae.joint_loss(
                    decoded_gene, targets=ctr_genes, isCtl=1, alpha=alphas[epoch], beta=1.0)

                pert_gene_joint_loss, pert_gene_rec_loss, pert_gene_kld_loss = self.gene_vae.joint_loss(
                    decoded_gene_pert, targets=genes, isCtl=0, alpha=alphas[epoch], beta=1.0)

                loss = gene_joint_loss + pert_gene_joint_loss + smile_rec_loss
                l1.append(smile_rec_loss.item())
                l2.append(gene_joint_loss.item())
                l3.append(pert_gene_joint_loss.item())

                self.smile_vae_optimizer.zero_grad()
                self.gene_vae_optimizer.zero_grad()

                loss.backward()

                self.smile_vae_optimizer.step()
                self.gene_vae_optimizer.step()

                total_gene_joint_loss += gene_joint_loss.item()
                total_gene_rec_loss += gene_rec_loss.item()
                total_gene_kld_loss += gene_kld_loss.item()

                total_smile_joint_loss += smile_joint_loss.item()
                total_smile_rec_loss += smile_rec_loss.item()
                total_smile_kld_loss += smile_kld_loss.item()

                total_pert_gene_joint_loss += pert_gene_joint_loss.item()
                total_pert_gene_rec_loss += pert_gene_rec_loss.item()
                total_pert_gene_kld_loss += pert_gene_kld_loss.item()

            mean_gene_joint_loss = total_gene_joint_loss / genes.size(0)
            mean_gene_rec_loss = total_gene_rec_loss / (genes.size(0) * 978)
            mean_gene_kld_loss = total_gene_kld_loss / (genes.size(0) * 64)

            mean_pert_gene_joint_loss = total_pert_gene_joint_loss / genes.size(0)
            mean_pert_gene_rec_loss = total_pert_gene_rec_loss / (genes.size(0) * 978)
            mean_pert_gene_kld_loss = total_pert_gene_kld_loss / (genes.size(0) * 64)

            mean_smile_joint_loss = total_smile_joint_loss / smiles.size(0)
            mean_smile_rec_loss = total_smile_rec_loss / (smiles.size(0))
            mean_smile_kld_loss = total_smile_kld_loss / (smiles.size(0))

            self.gene_vae.eval()
            self.smiles_vae.eval()

            valid_smiles = []
            label_smiles = []
            total_num_data = len(valid_smile_dataloader.dataset)

            for _, (smiles, genes, _) in enumerate(valid_smile_dataloader):

                smiles, genes = smiles.to(self.device), genes.to(self.device)

                gene_info = self.gene_vae.encode(genes)
                rand_smile_latent = torch.randn(genes.size(0), latent_size).to(self.device)

                dec_sampled_char = self.smiles_vae.generation(rand_smile_latent, max_len, tokenizer,
                                                              gene_info)
                output_smiles = ["".join(tokenizer.decode(
                    dec_sampled_char[i].squeeze().detach().cpu().numpy())).strip("^$ ").split("$")[0]
                                 for i in range(dec_sampled_char.size(0))]

                for i in range(len(output_smiles)):
                    mol = Chem.MolFromSmiles(output_smiles[i])
                    if (mol is not None
                            and mol.GetNumAtoms() > 1
                            and Chem.MolToSmiles(mol) != ' '):
                        valid_smiles.extend([output_smiles[i]])
                        label_smiles.extend(["".join(tokenizer.decode(
                            smiles[i].squeeze().detach().cpu().numpy())).strip("^$ ")])
            unique_smiles = list(set(valid_smiles))
            novel_smiles = [smi for smi in unique_smiles if smi not in label_smiles]
            pd.DataFrame(valid_smiles).to_csv(valid_smiles_file + "_" + self.model + ".csv", index=False)

            valid_num = len(valid_smiles)
            valid_rate = 100 * len(valid_smiles) / total_num_data
            unique_num = len(unique_smiles)
            novel_num = len(novel_smiles)

            if valid_num != 0:
                unique_rate = 100 * unique_num / valid_num
                diversity = mean_similarity(valid_smiles, label_smiles)
            else:
                unique_rate = 100 * unique_num / (valid_num + 1)
                diversity = 1

            if unique_num != 0:
                novel_rate = 100 * novel_num / unique_num
            else:
                novel_rate = 100 * novel_num / (unique_num + 1)

            print(
                f'Epoch: {epoch + 1:d} / {train_epochs:d}, gene_joint_loss: {mean_gene_joint_loss:.3f}, gene_rec_loss: {mean_gene_rec_loss:.3f},  gene_kld_loss: {mean_gene_kld_loss:.3f},pert_gene_joint_loss: {mean_pert_gene_joint_loss:.3f}, pert_gene_rec_loss: {mean_pert_gene_rec_loss:.3f},pert_gene_kld_loss: {mean_pert_gene_kld_loss:.3f}, smile_joint_loss: {mean_smile_joint_loss:.3f}, smile_rec_loss: {mean_smile_rec_loss:.3f}, smile_kld_loss: {mean_smile_kld_loss:.3f}, Total: {total_num_data:d}, valid: {valid_num:d} ({valid_rate:.2f}), unique: {unique_num:d} ({unique_rate:.2f}), novel: {novel_num:d} ({novel_rate:.2f}), diversity: {diversity:.3f}'
            )

            # Save trained results to file
            with open(smiles_vae_train_results + "_" + self.model + ".csv", 'a+') as wf:
                wf.write(
                    f'{epoch + 1},{mean_gene_joint_loss:.3f},{mean_gene_rec_loss:.3f},{mean_gene_kld_loss:.3f}, {mean_pert_gene_joint_loss:.3f} ,{mean_pert_gene_rec_loss:.3f},{mean_pert_gene_kld_loss:.3f}, {mean_smile_joint_loss:.3f},{mean_smile_rec_loss:.3f},{mean_smile_kld_loss:.3f},{total_num_data},{valid_num}({valid_rate:.2f}),{unique_num}({unique_rate:.2f}),{novel_num}({novel_rate:.2f}),{diversity:.3f}\n'
                )
            final_smiles = {'predict': valid_smiles, 'label': label_smiles}
            final_smiles = pd.DataFrame(final_smiles)
            final_smiles.to_csv(valid_smiles_file + "_" + self.model + ".csv", index=False)
        print('=' * 50)
        self.smiles_vae.save_model(saved_smiles_vae + "_" + self.model + ".pkl")
        print(f'Trained SmilesVAE is saved in {saved_smiles_vae + "_" + self.model + ".pkl"}')

        self.gene_vae.save_model(saved_gene_vae + '_' + cell_name + "_" + self.model + '.pkl')
        print('Trained GeneVAE is saved in {}'.format(saved_gene_vae + '_' + cell_name + "_" + self.model + '.pkl'))

        return self.smiles_vae
