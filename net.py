import torch
import pytorch_lightning as pl
import torchvision.utils as tvutils

import os
from typing import List


class VAE(torch.nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 latent_size: int = 16):
        super(VAE, self).__init__()

        self.in_channels = in_channels
        self.latent_size = latent_size

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU()
        )

        self.mean_layer = torch.nn.Linear(64, latent_size)
        self.std_layer = torch.nn.Linear(64, latent_size)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_size, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Tanh()
        )

    def forward(self, inputs):
        mean, std = self.encode(inputs)
        latent = self.reparametrization_trick(mean, std)
        reconstruction = self.decode(latent)
        return reconstruction, mean, std

    def loss_function(self, inputs, reconstructions, mean, std, kld_weight):
        reconstruction_loss = torch.nn.functional.mse_loss(reconstructions, inputs)
        kld = torch.mean(-0.5 * torch.sum(1 + torch.log(std ** 2) - (mean ** 2) - (std ** 2), dim=1), dim=0)
        return {"loss": reconstruction_loss + kld_weight * kld, "rec_loss": reconstruction_loss, "kld_loss": kld}

    def encode(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        result = torch.flatten(inputs, start_dim=1)
        result = self.encoder(result)
        mean = self.mean_layer(result)
        std = self.std_layer(result)
        return [mean, std]

    def decode(self, inputs: torch.Tensor):
        result = self.decoder(inputs)
        result = result.view(-1, 1, 28, 28)
        return result

    def reparametrization_trick(self, mean, std):
        return mean + torch.randn_like(std) * std

    def sample(self, samples_num, device):
        samples = torch.randn(samples_num, self.latent_size)
        samples = samples.to(device)
        return self.decode(samples)


class LightningVAE(pl.LightningModule):

    def __init__(self, params: dict):
        super(LightningVAE, self).__init__()
        self.params = params
        self.model = VAE(**params["model"])

        self.curr_device = None

    def forward(self, inputs):
        return self.model(inputs)[0]

    def training_step(self, batch, batch_idx):
        X, y = batch
        self.curr_device = X.device
        X_hat, mean, std = self.model(X)
        losses = self.model.loss_function(X, X_hat, mean, std, 0.00025)
        self.log_dict({key: value for key, value in losses.items()}, on_epoch=True)

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        X, y = batch
        self.curr_device = X.device
        X_hat, mean, std = self.model(X)
        losses = self.model.loss_function(X, X_hat, mean, std, kld_weight=1)
        self.log_dict({"val_" + key: value for key, value in losses.items()}, on_epoch=True)

    def on_validation_end(self):
        self.sample_images()

    def sample_images(self):
        test_dataset, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_dataset = test_dataset.to(self.curr_device)
        reconstructions = self.model(test_dataset)[0]
        print(reconstructions.shape)
        tvutils.save_image(reconstructions.data,
                           os.path.join(self.logger.save_dir, "VAE", f"version_{self.logger.version}",
                                        "Reconstructions",
                                        f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                           normalize=True,
                           nrow=8)

        samples = self.model.sample(64, self.curr_device)
        tvutils.save_image(samples.data,
                           os.path.join(self.logger.save_dir, "VAE", f"version_{self.logger.version}",
                                        "Samples",
                                        f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                           normalize=True,
                           nrow=8)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params["optimizer"]["lr"])

