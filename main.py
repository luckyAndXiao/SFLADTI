from models import MIFDTI
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from sklearn.utils import shuffle
from copy import deepcopy
from datetime import datetime
from cv import *
cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="MFDTI for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='biosnap')
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    output_path = f"./output2/{args.data}/{cv}/"
    mkdir(output_path)

    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'../../dataset/'
    dataset = pd.read_csv(dataFolder + f"{args.data}.txt", header=None, sep=" ")
    dataset.columns = ["SMILES", "Protein", "Y"]
    result = {"auroc":[], "auprc":[], "f1":[], "precision":[], "recall":[], "accuracy":[], "mcc":[],
                              "test_loss":[]}
    result = pd.DataFrame(result)


    for seed, times in enumerate(range(6, 11)):
        print("*" * 60 + str(times) + "-Random" + "*" * 60)
        torch.cuda.empty_cache()
        times_output_path = output_path + f"random_{times}/"
        cfg.RESULT.OUTPUT_DIR = times_output_path
        os.makedirs(times_output_path, exist_ok=True)

        # if data_name == "biosnap" and cv_id == 2 and times in [1, 2]:
        #     continue

        """split data"""
        data = deepcopy(dataset)
        data = shuffle(data, random_state=seed + 42)
        block_size = len(data) // 10
        dataset_train, dataset_test = data[: block_size * 7], data[block_size * 7:]
        size = len(dataset_test)
        if cv == "cv1":
            train_set, val_set, test_set = dataset_train, \
                dataset_test[:int(size * 1 / 3)], dataset_test[int(size * 1 / 3):]
        elif cv == "cv2":
            train_set, val_set, test_set = cv2(data)
        elif cv == "cv3":
            train_set, val_set, test_set = cv3(data)
        elif cv == "cv4":
            train_set, val_set, test_set = cv4(data)
        else:
            train_set, val_set, test_set = None, None, None

        train_set.reset_index(drop=True, inplace=True)
        val_set.reset_index(drop=True, inplace=True)
        test_set.reset_index(drop=True, inplace=True)

        print(f"dataset: {args.data}")
        print(f"train_set: {len(train_set)}")
        print(f"val_set: {len(val_set)}")
        print(f"test_set: {len(test_set)}")

        set_seed(cfg.SOLVER.SEED)
        train_dataset = DTIDataset(train_set.index.values, train_set)
        val_dataset = DTIDataset(val_set.index.values, val_set)
        test_dataset = DTIDataset(test_set.index.values, test_set)

        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                                                                   'drop_last': True, 'collate_fn': graph_collate_func}
        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        val_generator = DataLoader(val_dataset, **params)
        test_generator = DataLoader(test_dataset, **params)

        model = MIFDTI(device=device, **cfg).to(device=device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        torch.backends.cudnn.benchmark = True

        trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, **cfg)

        res = trainer.train()
        pd.DataFrame(res).to_csv(times_output_path + "test.csv", index=False)
        if os.path.exists(output_path + "result.csv"):
            result = pd.read_csv(output_path + "result.csv")
        result = pd.concat((result, pd.DataFrame(res)), axis=0)
        result.to_csv(output_path + "result.csv", index=False)



if __name__ == '__main__':
    cfg = get_cfg_defaults()
    DATASETS = ["biosnap"]
   # DATASETS = ["celegans"]
    # device = "cpu"
    for data_name in DATASETS:
        for cv_id in [4, 1]:

            args.data = data_name
            cv = f"cv{cv_id}"
            print(data_name, cv)
            main()