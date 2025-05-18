import os
import torch
import pandas as pd


def generation(model: str,
               smiles_vae, gene_vae,
               test_gene_loader,
               latent_size, candidate_num, max_len,
               gen_path, protein_name,
               tokenizer, device):
    smiles_vae.eval()
    gene_vae.eval()
    res_smiles = []
    for _, genes in enumerate(test_gene_loader):
        genes = genes.to(device)
        if protein_name not in ["AKT1", "AKT2", "AURKB", "CTSK", "EGFR", "HDAC1", "MTOR", "PIK3CA", "SMAD3", "TP53"]:
            genes = genes * -1
        gene_latent_vectors = gene_vae.encode(genes)



        if genes.size(0) != 1:
            rand_z = torch.randn(genes.size(0), latent_size).to(device)
        else:
            rand_z = torch.randn(candidate_num, latent_size).to(device)
            gene_latent_vectors = gene_latent_vectors.repeat(candidate_num, 1)


        dec_sampled_char = smiles_vae.generation(rand_z, max_len, tokenizer, gene_latent_vectors)

        output_smiles = ["".join(tokenizer.decode(
            dec_sampled_char[i].squeeze().detach().cpu().numpy())).strip("^$ ").split("$")[0]
            for i in range(dec_sampled_char.size(0))]
        res_smiles.append(output_smiles)
    test_data = pd.DataFrame(columns=['SMILES'], data=res_smiles[0])

    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    test_data.to_csv(gen_path + f'res-{protein_name}-{model}.csv', index=False)
