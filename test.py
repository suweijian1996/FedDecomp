import os
import torch
import copy
from utils.options import args_parser
from models.LoraResnet1 import resnet8, resnet10
from src.client import Client
from utils.dataloader import getDataset
from utils.util import getUserGroup

def setup_device(args):
    return torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

def load_model(args, device):
    if args.model == 'resnet8':
        return resnet8(num_labels=args.num_classes, Conv_r=args.Conv_r, Linear_r=args.Linear_r).to(device)
    elif args.model == 'resnet10':
        return resnet10(num_labels=args.num_classes, Conv_r=args.Conv_r, Linear_r=args.Linear_r).to(device)
    else:
        raise NotImplementedError(f'Model {args.model} not supported.')

def test_clients(args, task_flag, model_class, device):
    print(f"\nTesting clients from task: {task_flag}")

    # 加载数据和索引


    for idx in range(args.num_users // args.task_num):
        print(f"  - Testing {task_flag}_client_{idx}")
        model = model_class()
        model_path = os.path.join("pth", f"{task_flag}_client_{idx}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        test_dataset = getDataset("MFTest")
        test_user_groups = getUserGroup(args, test_dataset)
        # 创建客户端对象，调用 inference
        client = Client(
            device=device,
            local_model=model,
            train_dataset=None,
            test_dataset=test_dataset,
            train_idxs=None,
            test_idxs=test_user_groups[idx],
            args=args,
            index=idx,
            logger=None,
            task_flag=task_flag
        )
        client.inference(epoch=idx, mode='all')

def main():
    args = args_parser()
    args.num_classes = 10 if args.dataset == 'cifar' else 100 if args.dataset == 'cifar-100' else 200
    device = setup_device(args)

    # 定义模型构造器
    def model_class():
        return load_model(args, device)

    for task_flag in ['MF']:
        test_clients(args, task_flag, model_class, device)

if __name__ == '__main__':
    main()
