import copy
import torch
import numpy as np
import os

from tqdm import tqdm
from src.client import Client
import matplotlib.pyplot as plt
from utils.get_dataset import get_dataset
from utils.dataloader import getDataset
from utils.util import getUserGroup
import time

class Server:
    def __init__(self, device, local_model, args, logger=None):
        # base parameters
        self.device = device
        self.args = args
        self.get_global_dataset(self.args)
        self.total_clients = self.args.num_users
        self.indexes = [i for i in range(self.total_clients// self.args.task_num)]
        self.logger = logger

        # initialize clients
        self.clients = []

        self.IV_clients = [Client(device=device, local_model=copy.deepcopy(local_model), train_dataset=self.IV_train_dataset,
                               test_dataset=self.IV_test_dataset, train_idxs=self.IV_train_user_groups[idx],
                               test_idxs=self.IV_test_user_groups[idx], args=args, index=idx, logger=logger, task_flag='IV') for idx in self.indexes]

        self.Med_clients = [
            Client(device=device, local_model=copy.deepcopy(local_model), train_dataset=self.Med_train_dataset,
                   test_dataset=self.Med_test_dataset, train_idxs=self.Med_train_user_groups[idx],
                   test_idxs=self.Med_test_user_groups[idx], args=args, index=idx, logger=logger, task_flag='Med') for idx in self.indexes]
        self.ME_clients = [
            Client(device=device, local_model=copy.deepcopy(local_model), train_dataset=self.ME_train_dataset,
                   test_dataset=self.ME_test_dataset, train_idxs=self.ME_train_user_groups[idx],
                   test_idxs=self.ME_test_user_groups[idx], args=args, index=idx, logger=logger, task_flag='ME') for idx in self.indexes]
        self.MF_clients = [
            Client(device=device, local_model=copy.deepcopy(local_model), train_dataset=self.MF_train_dataset,
                   test_dataset=self.MF_test_dataset, train_idxs=self.MF_train_user_groups[idx],
                   test_idxs=self.MF_test_user_groups[idx], args=args, index=idx, logger=logger, task_flag='MF') for idx in self.indexes]

        self.best_accuracy_global_after = 0



    def get_global_dataset(self, args):
        self.IV_train_dataset = getDataset('IV')
        self.IV_test_dataset = getDataset('IVTest')
        self.IV_train_user_groups = getUserGroup(args, self.IV_train_dataset)
        self.IV_test_user_groups = getUserGroup(args, self.IV_test_dataset)

        self.Med_train_dataset = getDataset('Medical')
        self.Med_test_dataset = getDataset('MedicalTest')
        self.Med_train_user_groups = getUserGroup(args, self.Med_train_dataset)
        self.Med_test_user_groups = getUserGroup(args, self.Med_test_dataset)

        self.ME_train_dataset = getDataset('ME')
        self.ME_test_dataset = getDataset('METest')
        self.ME_train_user_groups = getUserGroup(args, self.ME_train_dataset)
        self.ME_test_user_groups = getUserGroup(args, self.ME_test_dataset)

        self.MF_train_dataset = getDataset('MFGT')
        self.MF_test_dataset = getDataset('MFGTTest')
        self.MF_train_user_groups = getUserGroup(args, self.MF_train_dataset)
        self.MF_test_user_groups = getUserGroup(args, self.MF_test_dataset)
        # self.train_dataset, self.test_dataset, self.train_user_groups, self.test_user_groups = get_dataset(args)
        # self.global_test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.local_bs, shuffle=False)


    def average_weights(self):
        w_avg = copy.deepcopy(self.clients[0].local_model.state_dict())
        for key in w_avg.keys():
            for client in range(self.args.num_users):
                if client == 0: continue
                w_avg[key] += self.clients[client].local_model.state_dict()[key]
            w_avg[key] = torch.div(w_avg[key], float(self.args.num_users))
        return w_avg


    def send_parameters(self, w_avg):
        if self.args.policy == 1:   # separate training
            return
        elif self.args.policy == 3:
            print('Not aggregate Lora!!!!!!!!')
            self.logger.info('Not aggregate Lora!!!!!!!!')
            '''
            IV
            '''
            for client in range(self.args.num_users// self.args.task_num):
                w_local = copy.deepcopy(self.clients[client].local_model.state_dict())
                for key in w_avg.keys():
                    if not ('lora_' in key):    # only aggregate full rank weights
                        w_local[key] = copy.deepcopy(w_avg[key])

                self.IV_clients[client].local_model.load_state_dict(w_local)

            '''
            Med
            '''
            for client in range(self.args.num_users // self.args.task_num):
                w_local = copy.deepcopy(self.clients[client].local_model.state_dict())
                for key in w_avg.keys():
                    if not ('lora_' in key):  # only aggregate full rank weights
                        w_local[key] = copy.deepcopy(w_avg[key])

                self.Med_clients[client].local_model.load_state_dict(w_local)

            '''
            ME
            '''
            for client in range(self.args.num_users // self.args.task_num):
                w_local = copy.deepcopy(self.clients[client].local_model.state_dict())
                for key in w_avg.keys():
                    if not ('lora_' in key):  # only aggregate full rank weights
                        w_local[key] = copy.deepcopy(w_avg[key])

                self.ME_clients[client].local_model.load_state_dict(w_local)
            '''
            MF
            '''
            for client in range(self.args.num_users // self.args.task_num):
                w_local = copy.deepcopy(self.clients[client].local_model.state_dict())
                for key in w_avg.keys():
                    if not ('lora_' in key):  # only aggregate full rank weights
                        w_local[key] = copy.deepcopy(w_avg[key])

                self.MF_clients[client].local_model.load_state_dict(w_local)
            return
        else:
            raise NotImplementedError

    def set_client_weight(self):
        self.clients = self.IV_clients + self.Med_clients + self.ME_clients + self.MF_clients
        print(len(self.clients), len(self.IV_clients), len(self.Med_clients), len(self.ME_clients), len(self.MF_clients))

    def train(self):
        train_losses = []
        test_losses_global_after = []
        test_acc_global_after = []


        total_time = 0
        for epoch in tqdm(range(self.args.epochs)):
            print(f'Start Training round: {epoch}')
            self.logger.info(f'Start Training round: {epoch}')
            local_train_losses = []
            local_test_losses_global_after = []
            local_test_acc_global_after = []


            # select clients to train their local model
            idxs = np.random.choice(self.indexes, max(int(self.args.frac * self.total_clients//self.args.task_num), 1), replace=False)
            # print('IV clients for training', self.total_clients// self.args.task_num)
            for client in idxs:
                print(f'client {self.IV_clients[client].task_flag} for training with number {idxs}')
                loss, train_time = self.IV_clients[client].train('IV')
                local_train_losses.append(loss)
                total_time += train_time

            print('Med clients for training')
            for client in idxs:
                print(f'client {self.Med_clients[client].task_flag} for training with number {idxs}')
                loss, train_time = self.Med_clients[client].train('Med')
                local_train_losses.append(loss)
                total_time += train_time

            print('ME clients for training')
            for client in idxs:
                print(f'client {self.MF_clients[client].task_flag} for training with number {idxs}')
                loss, train_time = self.MF_clients[client].train('ME')
                local_train_losses.append(loss)
                total_time += train_time

            print('MF clients for training')
            for client in idxs:
                print(f'client {self.ME_clients[client].task_flag} for training with number {idxs}')
                loss, train_time = self.ME_clients[client].train('MF')
                local_train_losses.append(loss)
                total_time += train_time


            local_train_losses_avg = sum(local_train_losses) / len(local_train_losses)
            train_losses.append(local_train_losses_avg)

            # clients send parameters to the server
            print('Clients send parameters to the server')
            self.set_client_weight()
            w_avg = self.average_weights()

            print('Server aggregate parameters')
            self.send_parameters(w_avg)

            for idx, client in enumerate(self.IV_clients[:self.args.num_users // self.args.task_num]):
                client.inference(epoch, 'all')
                model_path = os.path.join("E:\FedMIF\code\FedDecomp\pth", f"IV_client_{idx}.pth")
                torch.save(client.local_model.state_dict(), model_path)
            for idx, client in enumerate(self.Med_clients[:self.args.num_users // self.args.task_num]):
                client.inference(epoch, 'all')
                model_path = os.path.join("E:\FedMIF\code\FedDecomp\pth", f"Med_client_{idx}.pth")
                torch.save(client.local_model.state_dict(), model_path)
            for idx, client in enumerate(self.ME_clients[:self.args.num_users // self.args.task_num]):
                client.inference(epoch, 'all')
                model_path = os.path.join("E:\FedMIF\code\FedDecomp\pth", f"ME_client_{idx}.pth")
                torch.save(client.local_model.state_dict(), model_path)
            for idx, client in enumerate(self.MF_clients[:self.args.num_users // self.args.task_num]):
                client.inference(epoch, 'all')
                model_path = os.path.join("E:\FedMIF\code\FedDecomp\pth", f"MF_client_{idx}.pth")
                torch.save(client.local_model.state_dict(), model_path)


            # test_losses_global_after.append(
            #     sum(local_test_losses_global_after) / len(local_test_losses_global_after))
            # test_acc_global_after.append(sum(local_test_acc_global_after) / len(local_test_acc_global_after))
            #
            # # update the best accuracy
            # if test_acc_global_after[-1] >= self.best_accuracy_global_after:
            #     self.best_accuracy_global_after = test_acc_global_after[-1]
            #     self.best_epoch = epoch
            #     self.best_time = total_time


            # print the training information in this epoch
        #     print(f'Communication Round: {epoch}   Policy: {self.args.policy}')
        #     print(f'Avg training Loss: {train_losses[-1]}')
        #     print(f'Avg testing Loss. personalized:{test_losses_global_after[-1]}')
        #     print(
        #         f'Avg training Accuracy. personalized after agg:{test_acc_global_after[-1]}')
        #
        #     print(f'Testing Acc for each client: {local_test_acc_global_after}')
        #     print(
        #         f'Best Accuracy up to now. personalized after agg:{self.best_accuracy_global_after}')
        #     print(f'Best time: {self.best_time}  Best epoch: {self.best_epoch}')
        #
        #     self.logger.info(f'Communication Round: {epoch}   Policy: {self.args.policy}')
        #     self.logger.info(f'Avg training Loss: {train_losses[-1]}')
        #     self.logger.info(f'Avg testing Loss. personalized:{test_losses_global_after[-1]}')
        #     self.logger.info(
        #         f'Avg training Accuracy. personalized after agg:{test_acc_global_after[-1]}')
        #
        #     self.logger.info(f'Testing Acc for each client: {local_test_acc_global_after}')
        #     self.logger.info(
        #         f'Best Accuracy up to now. personalized after agg:{self.best_accuracy_global_after}')
        #     self.logger.info(f'Best time: {self.best_time}  Best epoch: {self.best_epoch}')
        #
        #
        # self.train_losses = train_losses
        # self.test_losses = test_losses_global_after
        # self.test_acc = test_acc_global_after
        return


