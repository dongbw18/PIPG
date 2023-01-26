import logging
import torch
from data_loader import DataLoader
from models import LogisticRegressionModel, SupportVectorMachineModel, RandomForestModel, DecisionTreeModel, RandomSubspaceModel
from models import ConvolutionalNeuralNetworkModel, LongShortTermMemoryModel
from models import MultiLayerPerceptronModel, PretrainedLanguageModel, MixtureOfExpertsModel

class FrameWorker:
    def __init__(self, args):
        self.args = args
        dataloader = DataLoader(args)
        self.log_name = 'logs/Roc_{}_{}{}_{}_{}.json'.format(args.task_name, args.method, args.pretrain_path.split('/')[-1], args.features.split('/')[-1].split('.')[0], args.data_ratio)
        if args.method == 'LR':
            self.model = LogisticRegressionModel(args, dataloader)
        elif args.method == 'SVM':
            self.model = SupportVectorMachineModel(args, dataloader)
        elif args.method == 'RF':
            self.model = RandomForestModel(args, dataloader)
        elif args.method == 'DT':
            self.model = DecisionTreeModel(args, dataloader)
        elif args.method == 'RS':
            self.model = RandomSubspaceModel(args, dataloader)
        elif args.method == 'XGB':
            self.model = ExtremeGradientBoostingDecisionTreeModel(args, dataloader)
        elif args.method == 'MLP':
            self.model = MultiLayerPerceptronModel(args, dataloader)
        elif args.method == 'CNN':
            self.model = ConvolutionalNeuralNetworkModel(args, dataloader)
        elif args.method == 'LSTM':
            self.model = LongShortTermMemoryModel(args, dataloader)
        elif args.method == 'PLM':
            self.model = PretrainedLanguageModel(args, dataloader)
        elif args.method == 'MoE':
            self.model = MixtureOfExpertsModel(args, dataloader)
        self.Train()

    def Train(self):
        logging.info('[START] Training.')
        best_acc = 0
        for i in range(self.args.epoch):
            logging.debug('[START] Train epoch {}.'.format(i))
            self.model.Train()
            logging.debug('[END] Train epoch {}.'.format(i))
            acc, confusion_matrix, logit, label = self.model.Test()
            logging.debug('Test acc: {}.'.format(acc))
            logging.debug('Confusion Matrix: \n{}'.format(confusion_matrix))
            if acc > best_acc:
                best_acc = acc
                self.model.SaveModel('ckpt/{}_{}_{}.pt'.format(self.args.task_name, self.args.features.split('/')[-1].split('_')[0], self.args.data_ratio))
                if self.args.task_name == 'Prediction': 
                    self.model.SaveRoc(self.log_name, logit, label, )
        logging.info('Best acc: {}.'.format(best_acc))
        logging.info('[END] Training.')