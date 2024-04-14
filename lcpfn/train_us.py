import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np

import lcpfn
from positional_encoding_us import PositionalEncoding
from transformer_us import Transformer
from generate_data import generate_random_data, batchify_data

# Generate data
train_data = generate_random_data(9000)
val_data = generate_random_data(3000)

train_dataloader = batchify_data(train_data)
val_dataloader = batchify_data(val_data)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    num_tokens=4,
    dim_model=8,
    num_heads=2,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dropout_p=0.1,
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


def train_loop(model, opt, loss_fn):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    model.train()
    total_loss = 0
    for batch in range(1, 100):
        # for batch in dataloader:
        # X, y = batch[:, 0], batch[:, 1]
        # X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)
        # print(y)
        get_batch_func = lcpfn.create_get_batch_func(prior=lcpfn.sample_from_prior)
        X, Y, Y_noisy = get_batch_func(batch_size=100, seq_len=100, num_features=1)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = Y[:, :-1]
        y_expected = Y[:, 1:]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(Y, y_input, tgt_mask)
        # pred = model(X, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()
        # print(total_loss)

    return total_loss / len(100)


# train_loop(model=model, opt=opt, loss_fn=loss_fn, dataloader=train_dataloader)

# def validation_loop(model, loss_fn, dataloader):
#     """
#     Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
#     Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
#     """

#     model.eval()
#     total_loss = 0

#     with torch.no_grad():
#         for batch in dataloader:
#             X, y = batch[:, 0], batch[:, 1]
#             X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(
#                 y, dtype=torch.long, device=device
#             )

#             # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
#             y_input = y[:, :-1]
#             y_expected = y[:, 1:]

#             # Get mask to mask out the next words
#             sequence_length = y_input.size(1)
#             tgt_mask = model.get_tgt_mask(sequence_length).to(device)

#             # Standard training except we pass in y_input and src_mask
#             pred = model(X, y_input, tgt_mask)

#             # Permute pred to have batch size first again
#             pred = pred.permute(1, 2, 0)
#             loss = loss_fn(pred, y_expected)
#             total_loss += loss.detach().item()

#     return total_loss / len(dataloader)


# def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
#     """
#     Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
#     Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
#     """

#     # Used for plotting later on
#     train_loss_list, validation_loss_list = [], []

#     print("Training and validating model")
#     for epoch in range(epochs):
#         print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

#         train_loss = train_loop(model, opt, loss_fn, train_dataloader)
#         train_loss_list += [train_loss]

#         validation_loss = validation_loop(model, loss_fn, val_dataloader)
#         validation_loss_list += [validation_loss]

#         print(f"Training loss: {train_loss:.4f}")
#         print(f"Validation loss: {validation_loss:.4f}")
#         print()

#     return train_loss_list, validation_loss_list


# train_loss_list, validation_loss_list = fit(
#     model, opt, loss_fn, train_dataloader, val_dataloader, 10
# )
