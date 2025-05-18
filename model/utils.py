import torch
from .RNN import EncoderRNN, DecoderRNN, RNNSmilesVAE



def create_smiles_model(model,
                        emb_size, hidden_size, num_layers, latent_size,
                        bidirectional,
                        tokenizer, device, dropout=0.1, dtype=torch.float32, ):
    if model == "RNN":
        encoder = EncoderRNN(emb_size, hidden_size, num_layers, latent_size, bidirectional, tokenizer, dtype)
        decoder = DecoderRNN(emb_size, hidden_size, num_layers, latent_size, tokenizer, dtype)
        smiles_vae = RNNSmilesVAE(encoder, decoder).to(device)
    return smiles_vae


def create_optimizer(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return optimizer
