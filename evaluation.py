import os
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from matplotlib.image import imread
from rdkit import rdBase
import matplotlib.pyplot as plt

rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')


def evaluation(
        model: str, gene_expression_file_path, cell_name,
        gene_num,
        source_path, protein_name,
        gen_path, candidate_num,
        mol_figure_path):
    os.makedirs(mol_figure_path, exist_ok=True)
    train_data = pd.read_csv(gene_expression_file_path + cell_name + '_smiles.csv', sep=',', )

    train_data = train_data['smiles']

    train_data = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in train_data]

    source_path = source_path + protein_name + '.csv'

    source_data = pd.read_csv(source_path)

    canonical_source_data = []
    for smi in source_data['SMILES']:
        try:
            cano_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        except Exception:
            cano_smi = smi
        if cano_smi not in train_data:
            canonical_source_data.append(cano_smi)


    if not os.path.exists(gen_path):
        print(f'generated path {gen_path} does not exist!')
    else:
        gen_path = gen_path + 'res-' + protein_name + "-" + model + f'.csv'

        gen_data = pd.read_csv(gen_path, names=['SMILES'])
        gen_data = gen_data.dropna()
        gen_data = gen_data.reset_index(drop=True)

        tanimoto = []
        valid_smiles = []
        a = []
        a2 = []
        b = []
        b2 = []
        # print(gen_data)
        for i in range(len(gen_data['SMILES'])):

            m1 = Chem.MolFromSmiles(gen_data['SMILES'][i])
            if m1 != None and m1.GetNumAtoms() > 1 and Chem.MolToSmiles(m1) != ' ':
                valid_smiles.append(gen_data['SMILES'][i])
                try:
                    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
                    a.append(fp1)
                    a2.append(gen_data['SMILES'][i])
                except Exception:
                    pass

        for j in range(len(canonical_source_data)):
            try:
                m2 = Chem.MolFromSmiles(canonical_source_data[j])
                fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
                b.append(fp2)
                b2.append(canonical_source_data[j])
            except Exception:
                pass
        for i in range(len(a)):
            for j in range(len(b)):
                tanimoto.append(
                    [DataStructs.BulkTanimotoSimilarity(a[i], [b[j]])[0], b2[j],
                     a2[i]])


        res = pd.DataFrame(tanimoto)
        if len(res) != 0:

            max_res = res.iloc[res[0].idxmax()]
            print('protein name:', protein_name)
            print('Source ligand:', max_res[1])
            print('Best generation:', max_res[2])
            print(f'Tanimoto similarity: {max_res[0]:.2f}')
            df_sorted = res.sort_values(by=0, ascending=True)
            file_name = os.path.dirname(gen_path)
            df_sorted.to_csv(file_name + f'/sorted_{protein_name}_df.csv', index=False)



            if len(valid_smiles) != 0:
                valid_rate = 100 * len(valid_smiles) / candidate_num

                unique_smiles = list(set(valid_smiles))
                unique_rate = 100 * len(unique_smiles) / len(valid_smiles)

                novel_smiles = [smi for smi in unique_smiles if smi not in train_data]
                novel_rate = 100 * len(novel_smiles) / len(unique_smiles)
            else:
                unique_smiles = []
                novel_smiles = []
                valid_rate = 0
                unique_rate = 0
                novel_rate = 0
            print('\n')
            print('Valid generation:', valid_rate)
            print('Unique generation:', unique_rate)
            print('Novel generation:', novel_rate)

            with open(file_name + f"/{protein_name}_output.txt", "w", encoding="utf-8") as f:
                f.write(f'protein name: {protein_name}\n')
                f.write(f'Source ligand: {max_res[1]}\n')
                f.write(f'Best generation: {max_res[2]}\n')
                f.write(f'Tanimoto similarity: {max_res[0]:.2f}\n')
                f.write(f'Valid generation: {len(valid_smiles)}\n')
                f.write(f'Unique generation: {len(unique_smiles)}\n')
                f.write(f'Novel generation: {len(novel_smiles)}\n')

            source_smile = max_res[1]
            best_smile = max_res[2]
            mol1 = Chem.MolFromSmiles(source_smile)
            mol2 = Chem.MolFromSmiles(best_smile)

            Draw.MolToFile(mol1, f'{mol_figure_path}/{protein_name}_mol1.png')
            Draw.MolToFile(mol2, f'{mol_figure_path}/{protein_name}_mol2.png')

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            axes[0].imshow(imread(f'{mol_figure_path}/{protein_name}_mol1.png'))
            axes[0].axis('off')
            axes[0].set_title('Molecule 1')

            axes[1].imshow(imread(f'{mol_figure_path}/{protein_name}_mol2.png'))
            axes[1].axis('off')
            axes[1].set_title('Molecule 2')

            plt.tight_layout()
            plt.savefig(f'{mol_figure_path}/{protein_name}_mol_comparison.png')

            plt.close()
