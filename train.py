import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import prepare_data


def train():

    # os.makedirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    begin = time.time()
    data = prepare_data()
    split_point = int(len(data) * .8)
    print(split_point)
    train, validate = data[:split_point], data[split_point:]


if __name__ == '__main__':
    main()

    # make directory for