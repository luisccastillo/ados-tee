# -*- coding: utf-8 -*-
# import pdb # Para hacer debug, donde queramos parar se inserta pdb.set_trace()
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split, Subset
from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset

from models.GDN import GDN
from test import test

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random


class Main():
    def __init__(self, train_config, env_config, debug=False):

        print('>>> main.py # Inicio de Constructor <<<')

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset']

        # input csv file with IoT data
        test_orig = pd.read_csv(f'{dataset}/data_in.csv', sep=',', index_col=0)

        # copy the original
        test = test_orig
        print('input file length')
        print(len(test))

        # delete attack column if exists
        if 'attack' in test.columns:
            test = test.drop(columns=['attack'])

        print('input file columns')
        print(test.columns)

        # Read list.txt to create graph of devices.
        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        # set device as GPU or CPU
        set_device(env_config['device'])
        self.device = get_device()

        # create tensor
        fc_edge_index = build_loc_net(fc_struc, list(
            test.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        self.feature_map = feature_map

        # construct input dataset
        #test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())
        test_dataset_indata = construct_data(test, feature_map, labels=0)

        # slide win and slide stride configuration from args
        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        # convert to timedataset using cfg
        test_dataset = TimeDataset(
            test_dataset_indata, fc_edge_index, mode='test', config=cfg)
        #test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='train', config=cfg)

        self.test_dataset = test_dataset

        # creeate DataLoader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                          shuffle=False, num_workers=0)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        # Create GDN
        self.model = GDN(edge_index_sets, len(feature_map),
                         dim=train_config['dim'],
                         input_dim=train_config['slide_win'],
                         out_layer_num=train_config['out_layer_num'],
                         out_layer_inter_dim=train_config['out_layer_inter_dim'],
                         topk=train_config['topk']
                         ).to(self.device)

    def run(self):

        print('>>> main.py # Inicio de run() <<<')

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

        print('>>> main.py # Imprimimos  model_save_path <<<')
        print(model_save_path)
        # print(self.env_config['load_model_path'])

        # test
        # para cargar el modelo que ha sido entrenado

        print('>>> main.py # load_state_dict <<<')
        # self.model.load_state_dict(torch.load(model_save_path))
        self.model.load_state_dict(torch.load(
            model_save_path, map_location=self.device))
        best_model = self.model.to(self.device)
        print('>>> main.py # test_result <<<')
        _, self.test_result = test(best_model, self.test_dataloader)
        # print(self.test_result)
        # seguir por aquí
        print(len(self.test_result[2:][0][0:]))
        print(self.test_result)

        # print('>>> main.py # val_result <<<')
        #_, self.val_result = test(best_model, self.val_dataloader)

        # para guardar en json o en csv
        #dir_path = self.env_config['save_path']
        # df.to_csv(f'{output_directory}/output.csv')
        df = pd.DataFrame(self.test_result[2:][0][0:])
        result = df.to_json(orient='columns')

        # save result for iexec
        with open(os.path.join(output_directory, "result.json"), 'w+') as f:
            json.dump(result, f)
        # este parece q es necesario
        with open(os.path.join(output_directory, "computed.json"), 'w+') as f:
            json.dump(
                {"deterministic-output-path": os.path.join(output_directory, "result.json")}, f)

    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        print(dir_path)

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr

        paths = [
            f'{dir_path}/pretrained/best_{datestr}.pt',
            f'{dir_path}/results/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == "__main__":

    # for iexec
    # meter en directorio de entrada el test.csv y el list.txt
    # estas dos líneas son para pruebas... luego para dockerizar, comentar
    #os.environ['IEXEC_IN'] = args.dataset
    #os.environ['IEXEC_OUT'] = args.dataset

    input_directory = os.environ['IEXEC_IN']
    output_directory = os.environ['IEXEC_OUT']
    #input_filename = os.environ['IEXEC_INPUT_FILE_NAME_1']

# TODO: for args
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type=int, default=32)
    parser.add_argument('-epoch', help='train epoch', type=int, default=30)
    parser.add_argument('-slide_win', help='slide_win', type=int, default=5)
    parser.add_argument('-dim', help='dimension', type=int, default=64)
    parser.add_argument(
        '-slide_stride', help='slide_stride', type=int, default=1)
 #   parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
 #  parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cpu')
    parser.add_argument('-random_seed', help='random seed',
                        type=int, default=5)
    parser.add_argument(
        '-comment', help='experiment comment', type=str, default='')
    parser.add_argument(
        '-out_layer_num', help='outlayer num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim',
                        help='out_layer_inter_dim', type=int, default=128)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-val_ratio', help='val ratio',
                        type=float, default=0.2)
    parser.add_argument('-topk', help='topk num', type=int, default=5)
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-load_model_path', help='trained model path',
                        type=str, default=input_directory + '/best_mls.pt')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    #device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
    }

    env_config = {
        # 'save_path': args.save_path_pattern,
        # 'save_path': args.dataset,
        # 'dataset': args.dataset,
        'save_path': input_directory,
        'dataset': input_directory,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }

    main = Main(train_config, env_config, debug=False)
    main.run()
