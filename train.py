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
import torch.nn.functional as F

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
    args = checkpoint['args']

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
    load_epoch = args.load_epoch
    experiment_dir = create_experiment_dir(args)
    if args.load_epoch == 0:
        checkpoints_dir =  create_checkpoint_dir(experiment_dir)
    else:
        checkpoints_dir, args = load_model(args, model, opt)

    # Before training
    model.train()
    total_steps = train_size // args.batch_size
    start_time = time()
    for epoch in range(load_epoch + 1, args.epochs + 1):
        train_steps = 0
        log_steps   = 0
        running_loss = 0
        logger.info(f"Epoch {epoch}, Iterations: {total_steps}...")
        print(f"Epoch {epoch}...")

        for image_batch, _ in train_loader:
            x = image_batch.to(device)
            with torch.no_grad():    
                #x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = vae.encode(x).detach() / 25.0
            loss = model(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                model.eval()
                x_hat = model.sample(16)
                x_hat = x_hat * 25.0
                with torch.no_grad():
                    #x_hat = vae.decode(x_hat*(1/0.18215)).sample
                    x_hat = vae.decode(x_hat)
                    x_hat = torch.clamp(x_hat, -1, 1)
                os.makedirs(f"{experiment_dir}/samples/", exist_ok=True)  
                utils.save_pil_image(x_hat, dir=f"{experiment_dir}/samples/sample.png")
                model.train()
                                
                print(f"\t[{train_steps:06d}/{total_steps:06d}] Train Loss: {running_loss / log_steps:.4f}")
                end_time = time()

                avg_loss = running_loss / log_steps
                logger.info(
                    f"""\t[{train_steps:06d}/{total_steps:06d}] Train Loss: {avg_loss:.4f}, Time Elapsed: {end_time - start_time:.2f}s""")
                
                running_loss = 0
                log_steps = 0
                start_time = time()
                
        if (epoch) % args.ckpt_every == 0:
            save_model(model, opt, args, logger, checkpoints_dir, epoch)

        print(f"Epoch {epoch} done.")
        logger.info(f"Epoch {epoch} done.")

    
if __name__ == '__main__':
#   Dataset
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument('--root_path', default='./', help='root path for cheackpoints')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--in_resolution', type=int, default=16, help='image size')                      
    parser.add_argument('--in_channels', type=int, default=4, help='image channels')                    
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--imgs_path', type=str, default="../coco_train2017")
    parser.add_argument('--caps_path', type=str, default="../coco_ann2017/captions_train2017.json")
    parser.add_argument('--img_size', type=int, default=128)
#   Experiments
    parser.add_argument('--ckpt_every', type=int, default=10, help='save after every "ckpt_every" epoch')
    parser.add_argument('--load_epoch', type=int, default=0,help='load at "load_epoch" epoch')
    parser.add_argument('--try_num', default=1, type=int, help="try number")
    parser.add_argument('--log_path', default='train.log', help='log path')
    parser.add_argument('--log_every', type=int, default=10, help='log after every "log_every" steps')
    parser.add_argument('--state_num', type=int, default=0, help='state number')

#   Hyperparameters
    parser.add_argument('--channels_mult', default=[1, 2, 4, 8], type=list, help="channels multiplier")
    parser.add_argument('--num_res_layers', type=int, default=2, help='number of residual layers')
    parser.add_argument('--timesteps', type=int, default=1000, help='timesteps')
    parser.add_argument('--eta', type=float, default=1, help='eta')
    parser.add_argument('--sampling_timesteps', type=int, default=100, help='sample steps')
    parser.add_argument('--hid_channels', type=int, default=64, help='hidden channels')
    parser.add_argument('--batch_size', type=int, default=64 , help='batch size')                  
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--model_lr', type=float, default=1e-4, help='learning rate in rec_loss.') 
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')       
    parser.add_argument('--mlp_ratio', type=float, default=4., help='mlp ratio')
    args = parser.parse_args()
    main(args)