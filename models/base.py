import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
class BaseModel:

    def __init__(self, args):
        self.args = args
        if args.task_name == 'Diagnosis':
            self.sorted_label = [0, 1, 2]
            self.N = 3
        else:
            self.sorted_label = [0, 1]
            self.N = 2
        # if args.task_name == 'Decompensation':
        #     self.score_func = roc_auc_scorek
        # else:
        #     self.score_func = accuracy_score
        self.score_func = accuracy_score

    def CalcResult(self, label, pred):
        return self.score_func(label, pred), confusion_matrix(label, pred, labels=self.sorted_label)
    
    def SaveRoc(self, log_name, logit, label):
        fpr, tpr, thresholds = roc_curve(label, logit[:,1])
        json.dump({'fpr':fpr.tolist(), 'tpr':tpr.tolist()}, open(log_name, 'w'))

class SklearnModel(BaseModel):

    def __init__(self, args, dataloader):
        super(SklearnModel, self).__init__(args)
        # self.feature_names = dataloader.train_data.columns[1:-1]
        self.train_data, self.test_data = dataloader.train_data.fillna(-1).values, dataloader.test_data.fillna(-1).values
        self.model = 'SklearnModel'
        b = int(args.task_name != 'Decompensation')
        self.train_data, self.train_label = self.train_data[:, b:-1], self.train_data[:, -1]
        self.test_data, self.test_label = self.test_data[:, b:-1], self.test_data[:, -1]
        # train_len = len(self.train_data)
        # datas = np.array(self.train_data + self.test_data)
        # maxx, minn = np.max(datas, dim=0), np.min(datas, dim=0)
        # datas = (datas - minn) / (maxx - minn)
        # self.train_data, self.test_data = datas[:train_len], datas[train_len:]

    def Train(self):
        self.model.fit(self.train_data, self.train_label)

    def Test(self):
        logit = self.model.predict_proba(self.test_data)
        pred = self.model.predict(self.test_data)
        acc, confusion_matrix = self.CalcResult(self.test_label, pred)
        return acc, confusion_matrix, logit, self.test_label

class TextModel(BaseModel):

    def __init__(self, args, dataloader):
        super(TextModel, self).__init__(args)
        self.original_dataloader = dataloader
        if args.task_name == 'Decompensation':
            self.train_data, self.train_label = self.MIMICDataProcess(dataloader.features, dataloader.train_data.values)
            self.test_data, self.test_label = self.MIMICDataProcess(dataloader.features, dataloader.test_data.values)
        else:
            self.train_data, self.train_label = self.TextDataProcess(dataloader.features, dataloader.train_data)
            self.test_data, self.test_label = self.TextDataProcess(dataloader.features, dataloader.test_data)
    
    def MIMICDataProcess(self, features, data):
        res_data, labels = [], []
        for line in data:
            sentence = ''
            for i, feature in enumerate(features):
                if self.args.prompt_type == 'none':
                    sentence += '{} '.format(line[i])
                elif self.args.prompt_type == 'flag':
                    sentence += '{} ,'.format(line[i])
                else:
                    sentence += '{} is {} , '.format(feature, line[i])
            if self.args.prompt_type != 'none':
                sentence = sentence[:-2] + '. '
            res_data.append(sentence)
            labels.append(int(line[-1]))
        return res_data, labels

    def TextDataProcess(self, features, dataloader):
        res_data, labels = [], []
        col_name = 'Feature' if self.args.prompt_type == 'label' else 'Description'
        for i, data in dataloader.iterrows():
            sentence = ''
            for j, feature in features.iterrows():
                val = data[feature['Feature']]
                if feature['Type'] == 'continuous' and val == val and self.args.input_type != 'float':
                    val = int(val)
                    if self.args.input_type == '3':
                        val = 'low' if val == 0 else ('middle' if val == 1 else 'high')
                if self.args.prompt_type == 'none':
                    sentence += '{} '.format(val)
                elif self.args.prompt_type == 'flag':
                    sentence += '{} ,'.format(val)
                else:
                    sentence += '{} is {} , '.format(feature[col_name], val)
            if self.args.prompt_type != 'none':
                sentence = sentence[:-2] + '. '
            res_data.append(sentence)
            labels.append(int(data['DX']))
        return res_data, labels
