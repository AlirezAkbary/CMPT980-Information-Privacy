#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MNIST training with differential privacy.

"""

import argparse

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
from Model import *
import random



# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch, extra_sample=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    #print("found it")
    if extra_sample is not None:
        (extra_sample_x, extra_sample_target) = extra_sample
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        if extra_sample is not None:
            data = torch.cat([data, torch.unsqueeze(extra_sample_x, dim=0)], dim=0)
            target = torch.cat((target, torch.unsqueeze(torch.tensor(extra_sample_target), dim=0)))

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
            delta=args.delta
        )
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
        "--extra-sample",
        type=parse_boolean,
        default=True,
        help="Trains with extra sample",
    )
    parser.add_argument(
        "--init-mult",
        type=float,
        default=1.0,
        metavar="IM",
        help="Initialization multiplier",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="se",
        help="Seed for randomness",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainset = datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        )


    extra_sample_index = random.randint(0, len(trainset)-1)
    trainset_D_indices = list(range(0, extra_sample_index)) + list(range(extra_sample_index+1, len(trainset)))

    extra_sample = trainset[extra_sample_index]
    print('extra index:', extra_sample_index)

    trainset_D = torch.utils.data.Subset(trainset, trainset_D_indices)

    train_loader = torch.utils.data.DataLoader(trainset_D,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )
    print(len(train_loader))
    print(len(train_loader.dataset))
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    print(len(test_loader.dataset))
    run_results = []
    for _ in range(args.n_runs):
        #saved_seed = torch.get_rng_state()
        # torch.set_rng_state(saved_seed)

        # for name, W in model.named_parameters():
        #     if 'weight' in name:
        #         print(torch.norm(W))
        #
        # for name, W in model.named_parameters():
        #     if 'weight' in name:
        #         W.data *= args.init_mult

        model = SampleConvNet().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )

        for epoch in range(1, args.epochs + 1):
            if args.extra_sample:
                train(args, model, device, train_loader, optimizer, privacy_engine, epoch, extra_sample)
            else:
                train(args, model, device, train_loader, optimizer, privacy_engine, epoch, None)
            #print("salam")
        run_results.append(test(model, device, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )
    if args.extra_sample:
        repro_str = (
            f"mnist_{args.seed}_{args.extra_sample}_{extra_sample_index}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
        )
    else:
        repro_str = (
            f"mnist_{args.seed}_{args.extra_sample}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
        )
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model._module.state_dict(), f"mnist_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()