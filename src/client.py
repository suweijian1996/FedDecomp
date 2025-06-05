import copy
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
np.set_printoptions(edgeitems=30)
torch.set_printoptions(edgeitems=30)
import loralib
import time
import os
import cv2
import skimage.measure
import torch.nn as nn

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image1, image2 = self.dataset[self.idxs[item]]
        return image1.clone(), image2.clone()

class Sobel(nn.Module):
    def __init__(self, channels=1):
        super(Sobel, self).__init__()
        self.channels = channels
        kernel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
        kernel_x= torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_x = np.repeat(kernel_x, self.channels, axis=0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        kernel_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        kernel_y = np.repeat(kernel_y, self.channels, axis=0)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def __call__(self, x):
        x_x = F.conv2d(x, self.weight_x, padding=1, groups=self.channels)
        x_x = torch.abs(x_x)
        x_y = F.conv2d(x, self.weight_y, padding=1, groups=self.channels)
        x_y = torch.abs(x_y)
        x = torch.add(0.5*x_x,0.5*x_y)
        return x

class DatasetSplit_ME(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image1, image2, image3 = self.dataset[self.idxs[item]]
        return image1.clone(), image2.clone(), image3.clone()

class Client:
    def __init__(self, device, local_model, train_dataset, test_dataset, train_idxs, test_idxs, args, index, logger=None, task_flag = 'IV'):
        self.device = device
        self.args = args
        self.index = index
        self.local_model = local_model
        self.logger = logger
        self.task_flag = task_flag

        self.trainingLoss = None
        self.testingLoss = None
        self.testingAcc = None
        self.sobel = Sobel(1).to(self.device)

        if train_dataset is not None:
            self.trainloader, self.validloader, self.testloader, self.trainloader_full = self.train_val_test(
                train_dataset, list(train_idxs), test_dataset, list(test_idxs))
        else:
            print(list(test_idxs))
            self.testloader = self.test_data(
                test_dataset, list(test_idxs))


        # define Loss function
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.MSE = torch.nn.MSELoss().to(self.device)
        self.l1_loss = torch.nn.L1Loss().to(self.device)

    def train_val_test(self, train_dataset, train_idxs, test_dataset, test_idxs):
        """
        Returcns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        if self.task_flag == 'IV' or self.task_flag == 'Med':
            trainloader = DataLoader(DatasetSplit(train_dataset, train_idxs),
                                     batch_size=self.args.local_bs, shuffle=False)
            validloader = None
            testloader = DataLoader(DatasetSplit(test_dataset, test_idxs),
                                    batch_size=1, shuffle=False)
            trainloader_full = DataLoader(DatasetSplit(train_dataset, train_idxs), batch_size=len(train_idxs),
                                          shuffle=False)
        else:
            trainloader = DataLoader(DatasetSplit_ME(train_dataset, train_idxs),
                                     batch_size=self.args.local_bs, shuffle=False)
            validloader = None
            testloader = DataLoader(DatasetSplit_ME(test_dataset, test_idxs),
                                    batch_size=1, shuffle=False)
            trainloader_full = DataLoader(DatasetSplit_ME(train_dataset, train_idxs), batch_size=len(train_idxs),
                                          shuffle=False)

        return trainloader, validloader, testloader, trainloader_full

    def test_data(self, test_dataset, test_idxs):
        """
        Returcns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        if self.task_flag == 'IV' or self.task_flag == 'Med':
            testloader = DataLoader(DatasetSplit(test_dataset, test_idxs),
                                    batch_size=1, shuffle=False)
        else:
            testloader = DataLoader(DatasetSplit_ME(test_dataset, test_idxs),
                                    batch_size=1, shuffle=False)

        return  testloader
    def train(self, task_flag = 'IV'):
        self.local_model.to(self.device)
        self.local_model.train()
        epoch_loss = []

        # define optimizer
        if self.args.optimizer == 'sgd':
            weights = dict(self.local_model.named_parameters())
            lora_weights, non_lora_weights = [], []
            for k in weights.keys():
                if 'lora_A' in k or 'lora_B' in k:
                    lora_weights.append(weights[k])
                else:
                    non_lora_weights.append(weights[k])
            self.optimizer_lora = torch.optim.SGD(lora_weights, lr=self.args.lr, momentum=self.args.momentum)
            self.optimizer_nonlora = torch.optim.SGD(non_lora_weights, lr=self.args.lr, momentum=self.args.momentum)
        else:
            raise NotImplementedError

        start_time = time.time()
        # train personalized part
        loralib.mark_only_lora_as_trainable(self.local_model)
        for iter in range(self.args.local_p_ep):
            batch_loss = []
            for batch_idx, data in enumerate(self.trainloader):
                if self.task_flag == 'IV' or self.task_flag == 'Med':
                    (img1, img2) = data
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    input = torch.cat((img1, img2), dim=1)
                    output = self.local_model(input)
                    loss = self.MSE(torch.max(img1, img2), output) + self.MSE(torch.max(self.sobel(img1), self.sobel(img2)), self.sobel(output))
                    self.optimizer_lora.zero_grad()
                    loss.backward()

                    self.optimizer_lora.step()
                    batch_loss.append(loss.item())
                else:
                    (img1, img2, gt) = data
                    img1, img2, gt = img1.to(self.device), img2.to(self.device), gt.to(self.device)
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    input = torch.cat((img1, img2), dim=1)
                    output = self.local_model(input)
                    loss = self.MSE(gt, output)
                    self.optimizer_lora.zero_grad()
                    loss.backward()

                    self.optimizer_lora.step()
                    batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # train shared part
        loralib.mark_only_weight_as_trainable(self.local_model)
        for iter in range(self.args.local_ep - self.args.local_p_ep):
            batch_loss = []
            for batch_idx, data in enumerate(self.trainloader):
                if self.task_flag == 'IV' or self.task_flag == 'Med':
                    (img1, img2) = data
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    input = torch.cat((img1, img2), dim=1)
                    output = self.local_model(input)
                    loss = self.MSE(torch.max(img1, img2), output) + self.MSE(torch.max(self.sobel(img1), self.sobel(img2)), self.sobel(output))
                    self.optimizer_nonlora.zero_grad()
                    loss.backward()
                    self.optimizer_nonlora.step()

                    batch_loss.append(loss.item())
                else:
                    (img1, img2, gt) = data
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    input = torch.cat((img1, img2), dim=1)
                    output = self.local_model(input)
                    loss = self.MSE(gt, output)
                    self.optimizer_nonlora.zero_grad()
                    loss.backward()
                    self.optimizer_nonlora.step()

                    batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        self.trainingLoss = sum(epoch_loss) / len(epoch_loss)
        end_time = time.time()
        self.local_model.to('cpu')
        return sum(epoch_loss) / len(epoch_loss), end_time - start_time

    def savenp(self, tensor, path):
        y = tensor
        h = y.shape[2]
        w = y.shape[3]
        y = y * 255.0
        img_copy = y.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, 255)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        cv2.imwrite(path, img_copy)
        sd = np.std(img_copy)
        en = skimage.measure.shannon_entropy(img_copy)
        return sd, en

    def inference(self, epoch, mode='all'):
        self.local_model.to(self.device)
        self.local_model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.testloader):
                print(self.task_flag)
                if self.task_flag in ['IV', 'Med','IVTest','MF']:
                    img1, img2 = data
                else:
                    img1, img2, _ = data

                img1, img2 = img1.to(self.device), img2.to(self.device)
                input = torch.cat((img1, img2), dim=1)
                save_dir = r"E:\FedMIF\code\FedDecomp\results/"
                # 模型推理
                outputs = self.local_model(input, mode)  # shape: [B, ...]
                print(outputs.max(), outputs.min(), outputs.mean())
                sd, en = self.savenp(outputs*0.5 + 0.5, save_dir + f"fused_{batch_idx}_{epoch}_{self.task_flag}.png")
                print(sd, en)


