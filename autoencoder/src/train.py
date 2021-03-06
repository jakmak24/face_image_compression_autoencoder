import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logger
from data_loader import ImageFolder720p
from utils import get_config, get_args, dump_cfg
from utils import save_imgs

# models
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models"))


def weighted_loss_function(y,x,w):
    suma = torch.sum(w * (y - x) ** 2).cuda()
    wagi = torch.sum(w).cuda()
    
    return torch.div(suma,wagi).cuda()


def prologue(cfg: Namespace, *varargs) -> SummaryWriter:
    # sanity checks
    assert cfg.device == "cpu" or (cfg.device == "cuda" and torch.cuda.is_available())

    # dirs
    base_dir = f"../experiments/{cfg.exp_name}"

    os.makedirs(f"{base_dir}/out", exist_ok=True)
    os.makedirs(f"{base_dir}/chkpt", exist_ok=True)
    os.makedirs(f"{base_dir}/logs", exist_ok=True)
    os.makedirs(f"{base_dir}/trainlogs", exist_ok=True)
    logger.open_log(f"{base_dir}/trainlogs")

    dump_cfg(f"{base_dir}/train_config.txt", vars(cfg))

    # tb writer
    writer = SummaryWriter(f"{base_dir}/logs")

    return writer


def epilogue(cfg: Namespace, *varargs) -> None:
    logger.close_log()
    writer = varargs[0]
    writer.close()


def train(cfg: Namespace) -> None:
    
    # initial setup
    writer = prologue(cfg)
    logger.info("=== Training ===")

    if cfg.alg == "32":
        from autoencoder2bpp import AutoEncoder
    elif cfg.alg == "16":
        from autoencoder025bpp import AutoEncoder
    elif cfg.alg == "8":
        from autoencoder006bpp import AutoEncoder


    # train-related code
    if cfg.device == "cuda":
        model = AutoEncoder(cudaD=True)
    else:
        model = AutoEncoder(cudaD=False)
    
    
    if cfg.chkpt:
        model.load_state_dict(torch.load(cfg.chkpt))
    model.train()
    if cfg.device == "cuda":
        model.cuda()

    logger.debug("Model loaded")

    dataset = ImageFolder720p(cfg.dataset_path, cfg.weigh_path)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers
    )

    logger.debug("Data loaded")

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    loss_criterion = nn.MSELoss()
    if cfg.weigh_path is not None:
        loss_criterion = weighted_loss_function

    avg_loss, epoch_avg = 0.0, 0.0
    ts = 0

    # train-loop
    for epoch_idx in range(cfg.start_epoch, cfg.num_epochs + 1):

        # scheduler.step()

        for batch_idx, data in enumerate(dataloader, start=1):
            img, patches, _ , weights = data

            if cfg.device == "cuda":
                patches = patches.cuda()
                

            avg_loss_per_image = 0.0
            
            optimizer.zero_grad()

            x = Variable(patches)
            y = model(x)
            
            if cfg.weigh_path:
                w = Variable(weights)
                w = w[:,None,:,:]
                w = torch.cat((w,w,w),dim=1)
                if cfg.device == "cuda":
                    w=w.cuda()
                loss = loss_criterion(y, x, w)
            else:
                loss = loss_criterion(y, x)

            avg_loss_per_image += loss.item()

            loss.backward()
            optimizer.step()

            avg_loss += avg_loss_per_image
            epoch_avg += avg_loss_per_image

            if batch_idx % cfg.batch_every == 0:
                writer.add_scalar("train/avg_loss", avg_loss / cfg.batch_every, ts)

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, ts)

                logger.debug(
                    '[%3d/%3d][%5d/%5d] avg_loss: %.8f' %
                    (epoch_idx, cfg.num_epochs, batch_idx, len(dataloader), avg_loss / cfg.batch_every)
                )
                avg_loss = 0.0
                ts += 1

            if batch_idx % cfg.save_every == 0:
                out = torch.zeros(3, 256, 256)
                
                
                if cfg.device =="cuda":
                    x = Variable(patches[0, :, :, :].unsqueeze(0)).cuda()
                else:
                    x = Variable(patches[0, :, :, :].unsqueeze(0)).cpu()
                    
                    
                out = model(x).cpu().data               
                out = np.reshape(out, (3, 256, 256))
                
                y = torch.cat((img[0], out), dim=2).unsqueeze(0)
                save_imgs(imgs=y, to_size=(3, 256, 2 * 256), name=f"../experiments/{cfg.exp_name}/out/out_{epoch_idx}_{batch_idx}.png")

        # -- batch-loop

        if epoch_idx % cfg.epoch_every == 0:
            epoch_avg /= (len(dataloader) * cfg.epoch_every)

            writer.add_scalar("train/epoch_avg_loss", avg_loss / cfg.batch_every, epoch_idx // cfg.epoch_every)

            logger.info("Epoch avg = %.8f" % epoch_avg)
            epoch_avg = 0.0
            torch.save(model.state_dict(), f"../experiments/{cfg.exp_name}/chkpt/model_{epoch_idx}.pth")

    # -- train-loop

    # save final model
    torch.save(model.state_dict(), f"../experiments/{cfg.exp_name}/model_final.pth")

    # final setup
    epilogue(cfg, writer)


if __name__ == '__main__':
    args = get_args()
    config = get_config(args)

    train(config)
