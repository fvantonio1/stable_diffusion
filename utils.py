import torch
from tqdm import tqdm
from torchvision.utils import make_grid
import torchvision
import os

def sample(model, scheduler, im_size, down_sample, device, num_samples, z_channels,
           num_grid_rows, num_timesteps, vae, save_dir='samples/', step=None, save_steps=False):
    
    def save_images(imgs):
        imgs = torch.clamp(imgs, -1., 1.).detach().cpu()
        imgs = (imgs + 1) / 2 # convert to [0, 1]
        grid = make_grid(imgs, nrow=num_grid_rows)

        img = torchvision.transforms.ToPILImage()(grid)

        os.makedirs(save_dir, exist_ok=True)

        if step is not None:
            img.save(os.path.join(save_dir, f'samples_step_{step}_x0_{i}.png'))
        else:
            img.save(os.path.join(save_dir, f'samples_x0_{i}.png'))

        img.close()


    im_size = im_size // down_sample

    xt = torch.randn((num_samples,
                      z_channels,
                      im_size, im_size)).to(device)
    
    save_count = 0
    for i in tqdm(reversed(range(num_timesteps))):
        # get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))


        if i == 0:
            # decode only final image to save
            imgs = vae.decode(xt)
            save_images(imgs)
        elif save_steps:
            imgs = xt
            save_images(imgs)