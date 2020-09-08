import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logger
from data_loader import ImageFolder720p
from utils import get_config, get_args, dump_cfg, save_imgs

# models
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models"))


def weighted_loss_function(y,x,w):
    suma = torch.sum(w * (y - x) ** 2).cpu()
    wagi = torch.sum(w).cpu()
    
    return torch.div(suma,wagi).cpu()


def prologue(cfg: Namespace, *varargs) -> None:
    # sanity checks
    assert cfg.chkpt not in [None, ""]
    assert cfg.device == "cpu" or (cfg.device == "cuda" and torch.cuda.is_available())

    # dirs
    base_dir = f"../experiments/{cfg.exp_name}"

    os.makedirs(f"{base_dir}/out_test", exist_ok=True)
    logger.open(f"{base_dir}/out_test")
    dump_cfg(f"{base_dir}/test_config.txt", vars(cfg))


def epilogue(cfg: Namespace, *varargs) -> None:
    logger.close()


def test(cfg: Namespace) -> None:
    logger.info("=== Testing ===")
    
    if cfg.alg == "32":
        from autoencoder2bpp import AutoEncoder
    elif cfg.alg == "16":
        from autoencoder025bpp import AutoEncoder
    elif cfg.alg == "8":
        from autoencoder006bpp import AutoEncoder

    # initial setup
    prologue(cfg)

    model = CAE()
    model.load_state_dict(torch.load(cfg.chkpt))
    model.eval()
    if cfg.device == "cuda":
        model.cuda()

    logger.info("Loaded model")

    dataset = ImageFolder720p(cfg.dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=cfg.shuffle)

    logger.info("Loaded data")

    loss_criterion = nn.MSELoss()
    if cfg.weigh_path is not None:
        loss_criterion = weighted_loss_function

    for batch_idx, data in enumerate(dataloader, start=1):
        img, patches, _ , weights = data
        if cfg.device == 'cuda':
            patches = patches.cuda()

        out = torch.zeros(3,256,256)
        avg_loss = 0

        x = Variable(patches[:, :, :, :]).cpu()
        y = model(x).cpu()
        out = y.data

        if cfg.weigh_path:
            w = Variable(weights)
            w = w[:,None,:,:]
            w = torch.cat((w,w,w),dim=1)
            if cfg.device == "cuda":
                w=w.cuda()
            loss = loss_criterion(y, x, w)
        else:
            loss = loss_criterion(y, x)
        avg_loss +=  loss.item()

        logger.debug('[%5d/%5d] avg_loss: %f' % (batch_idx, len(dataloader), avg_loss))

        # save output
        out = np.reshape(out, (3, 256, 256))

        #print(model.encoded)
        concat = torch.cat((img[0], out), dim=2).unsqueeze(0)
        save_imgs(imgs=concat, to_size=(3, 256, 2 * 256), name=f"../experiments/{cfg.exp_name}/out_test/test_{batch_idx}.png")


    # final setup
    epilogue(cfg)


if __name__ == '__main__':
    args = get_args()
    config = get_config(args)

    test(config)
