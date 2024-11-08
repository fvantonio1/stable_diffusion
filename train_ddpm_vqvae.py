from models.unet import Unet
from models.vqvae import VQVAE
from linear_noise_scheduler import LinearNoiseScheduler
from dataset.mnist import MnistDataLoader
from dataset.celeba import CelebDataLoader
from utils import sample

import torch
import os
import yaml
import argparse
import re


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(config):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    # data loader
    data_loader = {
        'mnist' : MnistDataLoader,
        'celebhq' : CelebDataLoader
    }[dataset_config['name']](
        batch_size=train_config['autoencoder_batch_size'],
        split='train',
        im_path=dataset_config['im_path']
    )

    # create model and set to train
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.train()

    #########################
    # load VAE
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    
    ckpt_folder = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_dir'])
    ckpt = sorted(os.listdir(ckpt_folder), key=lambda i: int(re.findall('[0-9]',i)[0]))[-1] # 
    ckpt = os.path.join(ckpt_folder, ckpt)
    ckpt = torch.load(ckpt)

    vae.load_state_dict(ckpt['model'])
    print(f"Loaded VQVAE checkpoint")# with {ckpt['val_loss']} val loss")

    # freeze vae parameters
    for param in vae.parameters():
        param.requires_grad = False

    ########################
    # training parameters
    max_steps = len(data_loader) * train_config['ldm_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    ########################
    # log file
    log_file = os.path.join(train_config['task_name'], "log_ddpm.txt")

    # create log file
    with open(log_file, 'w') as f:
        pass

    # sample dir
    sample_dir = os.path.join(train_config['task_name'], 'ddpm_samples')
    os.makedirs(sample_dir, exist_ok=True)

    # checkpoints dir
    ckpt_dir = os.path.join(train_config['task_name'], train_config['ldm_ckpt_dir'])
    os.makedirs(ckpt_dir, exist_ok=True)

    # Run training
    print(f"Training by {max_steps} steps")
    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        optimizer.zero_grad()
        model.train()

        # load batch of images and encode them
        imgs, _ = data_loader.next_batch()
        imgs = imgs.to(device)
        zs, _ = vae.encode(imgs)

        # sample random noise
        noise = torch.randn_like(zs).to(device)

        # sample timesteps
        t = torch.randint(0, diffusion_config['num_timesteps'], (zs.shape[0],)).to(device)

        # add noise to iamges according to each timestep
        noisy_zs = scheduler.add_noise(zs, noise, t)

        # predict noise for noisy_imgs
        noise_pred = model(noisy_zs, t)

        loss = criterion(noise_pred, noise)
        loss.backward()
        optimizer.step()

        print(f"step {step:6d} | loss = {loss.item():.6f}")

        with open(log_file, "a") as f:
            f.write(f"{step} train {loss.item():.6f}\n")

        # once in a while sample images from model
        if (step != 0 and step % 1000 == 0) or last_step:
            model.eval()
            
            down_sample = 2 ** sum(autoencoder_model_config['down_sample'])

            print("Sampling examples from model.....")
            with torch.no_grad():
                sample(
                    model, scheduler, im_size=dataset_config['im_size'],
                    down_sample=down_sample, device=device, num_samples=train_config['num_samples'],
                    num_grid_rows=train_config['num_grid_rows'], z_channels=autoencoder_model_config['z_channels'],
                    num_timesteps=diffusion_config['num_timesteps'], vae=vae, save_dir=sample_dir, step=step,
                    save_steps=False,
                )

            # save model
            if step % 5000 == 0:
                checkpoint_path = os.path.join(ckpt_dir, f'ddpm_ckpt_{step}.pth')
                checkpoint = {
                    'model' : model.state_dict(),
                    'config' : config,
                    'step' : step,
                    'loss' : loss.item()
                }
                
                torch.save(checkpoint, checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', default='config/mnist.yaml', type=str)
    args = parser.parse_args()

    train(args)