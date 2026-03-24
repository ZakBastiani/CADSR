import math
import pickle
import torch
from trainer import CADSR
import os
import numpy as np
import json
import warnings
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from enums import *
import pickle as pkl
import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


if __name__ == '__main__':

    datasets = ["feynman_I_47_23"] 

  
    seeds = [32310] #,  899, 7890, 5646, 34287]

    with open("CADSR_settings/CADSR_default_parameters.json") as file:
        kwargs = json.load(file)

    # for dataset in datasets:
    for seed in seeds:
        for dataset in datasets:
            kwargs["base_name"] = f"{dataset}_{seed}"

            print("Trial:",kwargs["base_name"])

            torch.manual_seed(seed)
            np.random.seed(seed)

            df = pd.read_csv(fr"{dataset}\{dataset}.tsv.gz", compression='infer', header=0, sep='\t')
            train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2)
            target = "target"
            x = train_df.drop(columns=target).to_numpy().T
            y = train_df[target].to_numpy()
            noise = 0.0 # torch.normal(0, y.std() * 0.1, (1, y.shape[0]))[0].numpy()

            model = CADSR(len(x), **kwargs)

            equ = model.train(x, y + noise, epochs=500, batch=1000, save_timings=True, save_epoch_info=True,
                              save_eq_dict=False, verbose=True, max_runtime=8*3600, termination_acc=None)
            model.add_test_info(test_df.drop(columns=target).to_numpy().T, test_df[target].to_numpy())

