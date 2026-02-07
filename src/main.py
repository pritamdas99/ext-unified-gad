import time
import numpy as np
import pandas
import os
import sys
import warnings
from utils import *
warnings.filterwarnings("ignore")
from e2e_model import *

seed_list = list(range(3407, 10000, 10))

def work(dataset: Dataset, dataset_name, cross_mode, kernels, args):
    full_model_name = 'gcn-transformer'
    hop = args.khop
    dataset_name = dataset.name.replace('/', '.')
    print('Dataset: {}, Cross_mode: {}, Hop: {}, Kernels: {}, Model: {}'.format(dataset_name, cross_mode, hop, kernels, full_model_name))
    dataset.prepare_dataset()
    dataset.make_sp_matrix_graph_list(args.khop, args.sp_type, load_kg=True)
    train_dataloader, val_dataloader, test_dataloader =  dataset.get_graph_and_sp_dataloaders()

    e2e_model = UnifyMLPDetector(dataset.total_nodes, dataset, (train_dataloader, val_dataloader, test_dataloader), cross_mode=cross_mode, args=args)
    ST = time.time()
    print(f"training...")
    score_test = e2e_model.train()

    ED = time.time()
    time_cost += ED - ST
    
    
    model_result = {'dataset name': dataset_name,
                    'model_name': full_model_name,
                    'cross mode': cross_mode,
                    'time cost': time_cost}

    return score_test

    

def main():
    print('=================> Starting experiments ...')
    args = get_args()
    # parse data
    if args.kernels is not None:
        # kernels = args.kernels.split(',')
        print('All kernels: ', args.kernels)

    if args.datasets is not None:
        if '-' in args.datasets:
            st, ed = args.datasets.split('-')
            dataset_names = DATASETS[int(st):int(ed)+1]
        else:
            dataset_names = [DATASETS[int(t)] for t in args.datasets.split(',')]
        print('173:========>> All Datasets: ', dataset_names)

    if args.cross_modes is not None:
        cross_modes = args.cross_modes.split(',')
        print('All Cross_modes: ', cross_modes)

    if args.khop == 1:
        sp_type = "star+norm"
    elif args.khop == 2:
        sp_type = "convtree+norm"
    else:
        raise NotImplementedError
    
    # evaluate all parameters
    for dataset_name in dataset_names:
        # parse dataset
        # dataset_name = DATASETS[dataset_id]
        print('190:================> Evaluating dataset: ', dataset_name)

        ### settings
        # load dataset 
        if dataset_name == 'uni-tsocial' \
            or dataset_name == 'mnist/dgl/mnist0' \
            or dataset_name == 'mnist/dgl/mnist1' \
            or dataset_name == 'mutag/dgl/mutag0' \
            or dataset_name == 'bm/dgl/bm_mn_dgl' \
            or dataset_name == 'bm/dgl/bm_ms_dgl' \
            or dataset_name == 'bm/dgl/bm_mt_dgl' \
            :
            dataset = Dataset(dataset_name, prefix='../datasets/unified/', sp_type=sp_type) #, debugnum=10000)
        elif dataset_name == 'reddit' \
            or dataset_name == 'weibo' \
            or dataset_name == 'amazon' \
            or dataset_name == 'yelp' \
            or dataset_name == 'tfinace' \
            or dataset_name == 'tolokers' \
            or dataset_name == 'questions' \
            or dataset_name == 'tfinance' \
            :
            dataset = Dataset(dataset_name+'-els', prefix='../datasets/edge_labels/', sp_type=sp_type, labels_have='ne')
        else:
            dataset = Dataset(dataset_name)

        for cross_mode in cross_modes:
            model_result = work(dataset, dataset_name, cross_mode, args.kernels, args)
            results = pandas.concat([results, model_result])

            # save result for each dataset-model-pair
            full_model_name = kernel + '-transformer'
            save_file_name = f"{args.tag}.{args.act_ft}.dataset_{dataset_name}.premodel_{full_model_name}.preepochs_{args.epoch_pretrain}.hop_{args.khop}.sp_type_{sp_type}.lr_ft_{args.lr_ft}.epochft_{args.epoch_ft}.wd_{args.l2}.crossmode_{cross_mode}.mlplayers_{args.stitch_mlp_layers}_{args.final_mlp_layers}.lossweights_{str(args.node_loss_weight)+'-'+str(args.edge_loss_weight)+'-'+str(args.graph_loss_weight)}"
            save_results(results, save_file_name)
            print(results)

if __name__ == "__main__":
    main()  