import argparse
import logging
from framework import FrameWorker

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='Diagnosis', choices=['Diagnosis', 'Prediction', 'Decompensation'])
    # parser.add_argument('--prediction_task', default='MCI', choices=['MCI', 'CN'])
    parser.add_argument('--data_ratio', default=1, type=float)
    parser.add_argument('--features', default='config/selected_features.csv', type=str)
    # parser.add_argument('--pid', default=0, type=int)
    parser.add_argument('--method', default='LR', type=str, choices=['LR', 'SVM', 'RF', 'DT', 'RS', 'XGB', 'CNN', 'LSTM', 'MLP', 'PLM', 'MoE'])
    parser.add_argument('--pretrain_path', default='', type=str)
    # parser.add_argument('--loss', default='softmax', choices=['softmax', 'sigmoid'])
    parser.add_argument('--input_type', default='float', choices=['float', '100', '3'])
    parser.add_argument('--prompt_type', default='desc', choices=['none', 'flag', 'label', 'desc'])
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--P', action='store_true')
    parser.add_argument('--D', action='store_true')
    parser.add_argument('--M', action='store_true')
    args = parser.parse_args()
    if args.D: args.task_name = 'Diagnosis'
    if args.P: args.task_name = 'Prediction'
    if args.M: args.task_name = 'Decompensation'

    logging.info('[START] {}'.format(args))
    frameworker = FrameWorker(args)
    logging.info('[END]\n')