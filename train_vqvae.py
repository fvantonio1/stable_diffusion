from models.vqvae import VQVAE
from models.discriminator import Discriminator
from dataset.mnist import MnistDataLoader
from dataset.celeba import CelebDataLoader
import torch
import yaml
from argparse import ArgumentParser
import numpy as np
import random
import os
import torchvision
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # read config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    # set seed values
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # create model and dataset
    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    
    data_loader = {
        'mnist' : MnistDataLoader,
        'celebhq' : CelebDataLoader
    }[dataset_config['name']](
        batch_size=train_config['autoencoder_batch_size'],
        split='train',
        im_path=dataset_config['im_path']
    )

    os.makedirs(train_config['task_name'], exist_ok=True)

    # L1/L2 loss for reconstruction
    recon_criterion = torch.nn.MSELoss()
    # discriminator and disc loss
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    disc_criterion = torch.nn.MSELoss()

    # TODO: lpips model

    optimizer_g = torch.optim.Adam(model.parameters(),
                                   lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(),
                                   lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()
    
    disc_step_start = train_config['disc_start']

    # step parameters
    num_epochs = train_config['autoencoder_epochs']
    image_save_steps = train_config['autoencoder_img_save_steps']

    max_steps = num_epochs * len(data_loader)

    # directory to save images reconstruction
    save_samples_dir = os.path.join(train_config['task_name'],'vqvae_autoencoder_samples')
    os.makedirs(save_samples_dir, exist_ok=True)

    # directory to save autoenconder checkpoints
    autoencoder_ckpt_dir = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_dir'])
    os.makedirs(autoencoder_ckpt_dir, exist_ok=True)

    # log file
    log_file = os.path.join(train_config['task_name'], "log_autoencoder.txt")

    # create log file
    with open(log_file, 'w') as f:
        pass

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        model.train()

        imgs, _ = data_loader.next_batch()
        imgs = imgs.to(device)

        # reconstruct images
        output_imgs, z, quantize_losses = model(imgs)

        # once in a while save images
        if step % image_save_steps == 0 or last_step:
            sample_size = min(8, imgs.shape[0])
            save_output = torch.clamp(output_imgs[:sample_size], -1., 1.).detach().cpu()
            save_output = ((save_output + 1) / 2)
            save_input = ((imgs[:sample_size] + 1) / 2).detach().cpu()

            grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
            img = torchvision.transforms.ToPILImage()(grid)
            
            save_path = os.path.join(save_samples_dir, f"autoencoder_sample_{step}.png")

            img.save(save_path)
            img.close()

        # L2 loss
        recon_loss = recon_criterion(output_imgs, imgs)
        g_loss = (
            recon_loss +
            train_config['codebook_weight'] * quantize_losses['codebook_loss'] +
            train_config['commitment_beta'] * quantize_losses['commitment_loss']
        )

        # disc loss
        if step > disc_step_start:
            disc_fake_pred = discriminator(output_imgs)

            disc_fake_loss = disc_criterion(disc_fake_pred,
                                            torch.ones(disc_fake_pred.shape,
                                                       device=disc_fake_pred.device))
            
            g_loss += train_config['disc_weight'] * disc_fake_loss
            
        g_loss.backward()
        optimizer_g.step()
        optimizer_g.zero_grad()

        with open(log_file, "a") as f:
            f.write(f"{step} train {g_loss.item():.6f}\n")

        ##### Optimize discriminator ######
        if step > disc_step_start:
            fake = output_imgs
            disc_fake_pred = discriminator(fake.detach())
            disc_real_pred = discriminator(imgs)

            disc_fake_loss = disc_criterion(disc_fake_pred,
                                            torch.ones(disc_fake_pred.shape,
                                                       device=disc_fake_pred.device))
            disc_real_loss = disc_criterion(disc_real_pred,
                                            torch.ones(disc_real_pred.shape,
                                                       device=disc_real_pred.device))
            
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            disc_loss.backward()
            optimizer_d.step()
            optimizer_d.zero_grad()

            with open(log_file, "a") as f:
                f.write(f"{step} disc_loss {disc_loss.item():.6f}\n")

            print(f"step {step:4d} | loss = {g_loss.item():.6f} | disc_loss = {disc_loss.item():.6f}")

        else:
            print(f"step {step:4d} | loss = {g_loss.item():.6f}")

        # evaluate and save model for each epoch
        if (step>0 and step % len(data_loader) == 0) or last_step:
            # model.eval()
            # with torch.no_grad():
            #     val_loss_accum = 0.0
            #     val_loss_steps = len(val_data_loader)
            #     for _ in range(val_loss_steps):
            #         x, _ = val_data_loader.next_batch()
            #         x = x.to(device)

            #         x_recon, z, val_quantize_losses = model(x)
            #         val_recon_loss = recon_criterion(x_recon, x)
            #         val_g_loss = (
            #             val_recon_loss +
            #             train_config['codebook_weight'] * val_quantize_losses['codebook_loss'] +
            #             train_config['commitment_beta'] * val_quantize_losses['commitment_loss']
            #         )

            #         val_loss = val_g_loss / val_loss_steps
            #         val_loss_accum += val_loss.detach()
                
            #     print(f"validation loss: {val_loss_accum.item():.4f}")

            #     with open(log_file, 'a') as f:
            #         f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            checkpoint_path = os.path.join(autoencoder_ckpt_dir, f'autoenconder_ckpt_{step}.pth')
            checkpoint = {
                'model' : model.state_dict(),
                'config' : config,
                'step' : step,
            #    'val_loss' : val_loss_accum.item()
            }

            torch.save(checkpoint, checkpoint_path)



if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)