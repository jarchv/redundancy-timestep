import argparse
import time
import os
import torch
import itertools
import logging
import numpy as np
import utils
import ddpm
import autoencoder
import torchvision.transforms.functional as F

from glob import glob
from time import time
from torchsummary import summary
from diffusers import AutoencoderKL, VQModel

def create_logger(directory):
    logging.basicConfig(
        filename=directory,
        format='%(asctime)s %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S', 
        level=logging.INFO)
    logger = logging.getLogger()
    return logger

def save_model(model, opt, args, logger, checkpoints_dir, epoch):
    checkpoint = {
        "model": model.state_dict(),
        "opt"  : opt.state_dict(),
        "args" : args
    }

    checkpoint_path = f"{checkpoints_dir}/epoch-{epoch:03d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def create_experiment_dir(args):
    os.makedirs("results", exist_ok=True)  
    if args.state_num != 0:
        experiment_dir = f"results/state-{args.state_num:03d}"
        return experiment_dir
    experiment_index = len(glob("results/*"))
    experiment_dir = f"results/state-{experiment_index+1:03d}" 
    return experiment_dir
 
def create_checkpoint_dir(experiment_dir):
    """
    Create a checkpoint directory for saving model checkpoints.
    """
    checkpoint_dir = f"{experiment_dir}/checkpoints" 
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return checkpoint_dir

def load_model(args, model, opt):
    checkpoints_dir = os.path.join(
                'results',
                f"state-{args.state_num:03d}",
                'checkpoints')

    print('\nLoading "model@epoch[{:d}]"...'.format(args.load_epoch), end='')
    file_model = 'epoch-{:03d}.pt'.format(args.load_epoch)

    load_path  = os.path.join(checkpoints_dir, file_model)
    checkpoint = torch.load(load_path)

    model.load_state_dict(checkpoint['model'])
    opt.load_state_dict(checkpoint['opt'])
    #args = checkpoint['args']

    print("Done.")
    return checkpoints_dir, args

def main(args):
    torch.set_float32_matmul_precision('high')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(7)
    
    # Data loading
    train_loader, train_size = utils.get_celeb_data(args)
     
    # Logging
    logger = create_logger(args.log_path)
    logger.info("Data loaded successfully")
    logger.info(f"Training for {args.epochs} epochs...")

    # Create Model
    model = ddpm.GaussianDiffusion(args, device).to(device)
    opt   = torch.optim.Adam(itertools.chain(model.parameters()), lr=args.model_lr, betas=(args.beta1, 0.999))
    model = torch.compile(model)
    # VAE   
    vae = autoencoder.from_pretrained(f"vae-models/checkpoints/epoch-120.pt").to(device)

    #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    #vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    vae = vae.to(device)
    vae = torch.compile(vae)
    vae.eval()
    
    
    # Checkpoint directory & Loading model
    iterations = args.samples // args.batch_size
    load_epoch = args.load_epoch
    experiment_dir = create_experiment_dir(args)
    if args.load_epoch == 0:
        checkpoints_dir =  create_checkpoint_dir(experiment_dir)
    else:
        checkpoints_dir, args = load_model(args, model, opt)

    # Before training
    model.eval()
    os.makedirs(f"eval/state-{args.state_num}", exist_ok=True)  
    for i in range(iterations):
        logger.info(f"Iteration {i+1}/{iterations}...")

        x_hat = model.sample(args.batch_size)
        x_hat = x_hat * 25.0
        with torch.no_grad():
            x_hat = vae.decode(x_hat)
            x_hat = torch.clamp(x_hat, -1, 1)
            x_hat = x_hat * 0.5 + 0.5
        list_images = list(x_hat.permute(0, 2, 3, 1).detach().cpu().numpy())
        for j, image in enumerate(list_images):
            F.to_pil_image(image).save(f"eval/state-{args.state_num}/sample-{i}-{j}.png")
    
if __name__ == '__main__':
#   Dataset
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument('--root_path', default='./', help='root path for cheackpoints')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--in_resolution', type=int, default=16, help='image size')                      
    parser.add_argument('--in_channels', type=int, default=4, help='image channels')                    
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    parser.add_argument('--img_size', type=int, default=128)
#   Experiments
    parser.add_argument('--ckpt_every', type=int, default=10, help='save after every "ckpt_every" epoch')
    parser.add_argument('--load_epoch', type=int, default=0,help='load at "load_epoch" epoch')
    parser.add_argument('--log_path', default='train.log', help='log path')
    parser.add_argument('--log_every', type=int, default=100, help='log after every "log_every" steps')
    parser.add_argument('--state_num', type=int, default=0, help='state number')

#   Hyperparameters
    parser.add_argument('--samples', default=100, type=int, help="number of samples")
    parser.add_argument('--timesteps', type=int, default=1000, help='timesteps')
    parser.add_argument('--sampling_timesteps', type=int, default=100, help='sample steps')
    parser.add_argument('--batch_size', type=int, default=50 , help='batch size')                  
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--model_lr', type=float, default=1e-4, help='learning rate in rec_loss.') 
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')       
    parser.add_argument('--mlp_ratio', type=float, default=4., help='mlp ratio')
    parser.add_argument('--eta', type=float, default=1, help='eta')
    parser.add_argument('--hid_channels', type=int, default=64, help='hidden channels')

    args = parser.parse_args()
    main(args)