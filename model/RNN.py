import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Define EncoderRNN: encode a batch of SMILES to Z
class EncoderRNN(nn.Module):
    def __init__(
            self,
            emb_size,
            hidden_size,
            num_layers,
            latent_size,
            bidirectional,
            tokenizer,
            dtype=torch.float32,
    ):

        super().__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.tokenizer = tokenizer
        self.pad = self.tokenizer.pad
        self.vocab_size = tokenizer.n_tokens

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            self.vocab_size,
            self.emb_size,
            padding_idx=self.tokenizer.char_to_int[self.pad],
            dtype=dtype,
        )

        self.gru = nn.GRU(
            self.emb_size,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.1,
            bidirectional=self.bidirectional,
            batch_first=True,
            dtype=dtype,
        )

        self.latent_mean = nn.Linear(self.hidden_size, self.latent_size, dtype=dtype)
        self.latent_logvar = nn.Linear(self.hidden_size, self.latent_size, dtype=dtype)

    def forward(self, inputs):


        embed = self.embedding(inputs)

        output, hidden = self.gru(
            embed, None
        )

        output = output[:, -1, :].squeeze(1)

        if self.bidirectional:
            output = (
                    output[:, : self.hidden_size] + output[:, self.hidden_size:]
            )
        else:
            output = output[:, : self.hidden_size]


        mu = self.latent_mean(output)
        logvar = self.latent_logvar(output)

        return mu, logvar


# =============================================
#  Define DecoderRNN: decode Z to SMILES
class DecoderRNN(nn.Module):

    def __init__(
            self,
            emb_size,
            hidden_size,
            num_layers,
            latent_size,
            tokenizer,
            dtype=torch.float32,
    ):

        super().__init__()

        self.tokenizer = tokenizer
        self.start = self.tokenizer.start
        self.vocab_size = tokenizer.n_tokens

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_size = latent_size * 2

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size, dtype=dtype)
        self.gru = nn.GRU(
            self.emb_size + self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.i2h = nn.Linear(self.input_size, self.hidden_size, dtype=dtype)
        self.out = nn.Linear(
            self.hidden_size + self.input_size, self.vocab_size, dtype=dtype
        )

    def forward(self, inputs: torch.Tensor, z, condition=None, temperature=1.0):

        model_random_state = np.random.RandomState(1988)
        batch_size, n_steps = inputs.size()

        outputs = torch.zeros(batch_size, n_steps, self.vocab_size).to(inputs.device)

        input = (
                torch.ones([batch_size, 1], dtype=torch.int32)
                * self.tokenizer.char_to_int[self.start]
        )

        input = input.to(inputs.device)

        if condition is not None:

            decode_embed = torch.cat([z, condition], dim=1)
        else:


            decode_embed = torch.cat([z, z], dim=1)


        hidden = (
            self.i2h(decode_embed).unsqueeze(0).repeat(self.num_layers, 1, 1)
        )
        for i in range(n_steps):
            output, hidden = self.step(
                decode_embed, input, hidden
            )
            outputs[:, i] = output
            use_teacher_forcing = model_random_state.rand() < temperature

            if use_teacher_forcing:
                input = inputs[:, i]
            else:
                input = torch.multinomial(torch.exp(output), 1)

            if input.dim() == 0:
                input = input.unsqueeze(0)

        outputs = outputs.squeeze(1)

        return outputs

    def step(self, decode_embed, input, hidden):

        input = self.embedding(input).squeeze()  # [batch_size, emb_size]

        input = torch.cat(
            (input, decode_embed), 1
        )
        input = input.unsqueeze(1)

        output, hidden = self.gru(
            input, hidden
        )
        output = output.squeeze(1)

        output = torch.cat(
            (output, decode_embed), 1
        )
        output = self.out(output)

        return output, hidden


# =============================================
# Define SmilesVAE
class RNNSmilesVAE(nn.Module):

    def __init__(self, encoder, decoder):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def encode(self, inputs):


        self.mu, self.logvar = self.encoder(inputs)

        z = self.reparameterize(self.mu, self.logvar)
        return z, self.mu, self.logvar

    def decode(self, inputs, latent_smile, latent_gene, temperature):

        decoded = self.decoder(inputs, latent_smile, latent_gene, temperature)
        return decoded

    def forward(self, inputs, condition, temperature):

        latent_smile, _, _ = self.encode(inputs)

        decoded = self.decode(inputs, latent_smile, condition, temperature)

        return latent_smile, decoded

    def joint_loss(self, decoded, targets, alpha=0.5, beta=1):


        decoded = decoded.permute(0, 2, 1)


        rec_loss = self.criterion(decoded, targets)

        kld_loss = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())

        joint_loss = alpha * rec_loss + (1 - alpha) * beta * kld_loss

        return joint_loss, rec_loss, kld_loss

    def generation(self, latent_smile: torch.Tensor, max_len, tokenizer, condition=None):

        batch_size = latent_smile.size(0)

        generated_smiles_tokens = torch.zeros(batch_size, max_len).to(latent_smile.device)

        inputs = (
                torch.ones([batch_size, 1], dtype=torch.int32)
                * tokenizer.char_to_int[tokenizer.start]
        )
        inputs = inputs.to(latent_smile.device)

        if condition is not None:

            decode_embed = torch.cat([latent_smile, condition], 1)
        else:

            rand_z = torch.randn(latent_smile.size(0), latent_smile.size(1)).to(latent_smile.device)
            decode_embed = torch.cat([rand_z, latent_smile], 1)

        hidden = (
            self.decoder.i2h(decode_embed)
            .unsqueeze(0)
            .repeat(self.decoder.num_layers, 1, 1)
        )

        for i in range(max_len):
            output, hidden = self.decoder.step(
                decode_embed, inputs, hidden
            )
            output = torch.exp(output)

            inputs = torch.multinomial(output, 1)
            generated_smiles_tokens[:, i] = inputs.squeeze(1)

        return generated_smiles_tokens

    def load_model(self, path):
        weights = torch.load(path, map_location=torch.device('cuda:0'))
        self.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
