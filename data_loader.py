import os
import logging
import pandas as pd
import numpy as np
from collections import defaultdict

class DataLoader:

    def __init__(self, args):
        self.args = args
        logging.info('[START] Prepare for {} DataLoader.'.format(args.task_name))
        
        if args.task_name == 'Decompensation':
            self.MIMICDataLoader()
            logging.info('[END] Prepare for {} DataLoader.'.format(args.task_name))
            return

        self.data_name = 'data/{}/{}_{}_{}_{}'.format(args.task_name, args.features.split('/')[-1].split('.')[0], args.data_ratio, args.input_type, args.norm)
        if os.path.isfile(args.features): self.features = pd.read_csv(args.features)
        if not os.path.exists(self.data_name):
            if not os.path.exists('data/{}'.format(args.task_name)): 
                data = self.DiagnosisInit() if args.task_name == 'Diagnosis' else self.PredictionInit()
                self.DataSplit(args.task_name, data)
            self.train_data = pd.read_csv('data/{}/train.csv'.format(args.task_name))
            self.test_data = pd.read_csv('data/{}/test.csv'.format(args.task_name))

            self.DropColumns()
            if args.input_type == 'float': 
                self.Discrete2Num()
            else:
                self.Float2Level(int(args.input_type))
            self.Label2Num()
            self.Fewshot(args.data_ratio)

            os.system('mkdir -p {}'.format(self.data_name))
            self.train_data.to_csv(self.data_name + '/train.csv', index=False)
            self.test_data.to_csv(self.data_name + '/test.csv', index=False)
        else:
            self.train_data = pd.read_csv(self.data_name + '/train.csv')
            self.test_data = pd.read_csv(self.data_name + '/test.csv')

        if args.task_name == 'Diagnosis':
            logging.info('train data: {}, CN: {}, MCI: {}, AD: {}.'.format(self.train_data.shape, len(self.train_data[self.train_data['DX'] == 0]), len(self.train_data[self.train_data['DX'] == 1]), len(self.train_data[self.train_data['DX'] == 2])))
            logging.info('test data: {}, CN: {}, MCI: {}, AD: {}.'.format(self.test_data.shape, len(self.test_data[self.test_data['DX'] == 0]), len(self.test_data[self.test_data['DX'] == 1]), len(self.test_data[self.test_data['DX'] == 2])))
        else:
            logging.info('train data: {}, 0Non-Dementia: {}, 1Dementia: {}.'.format(self.train_data.shape, len(self.train_data[self.train_data['DX'] == 0]), len(self.train_data[self.train_data['DX'] == 1])))
            logging.info('test data: {}, 0Non-Dementia: {}, 1Dementia: {}.'.format(self.test_data.shape, len(self.test_data[self.test_data['DX'] == 0]), len(self.test_data[self.test_data['DX'] == 1])))
        
        logging.info('[END] Prepare for {} DataLoader.'.format(args.task_name))

    def MIMICDataLoader(self):
        self.data_name = 'data/MIMIC3/{}_{}'.format(self.args.data_ratio, self.args.norm)
        if not os.path.exists(self.data_name):
            self.train_data = pd.read_csv('data/MIMIC3/train.csv')
            self.test_data = pd.read_csv('data/MIMIC3/test.csv')
            self.Fewshot(self.args.data_ratio)
            if self.args.norm:
                for col_name in self.train_data.columns.values:
                    if col_name == 'label': continue
                    maxx = max(self.train_data[col_name].max(), self.test_data[col_name].max())
                    minn = min(self.train_data[col_name].min(), self.test_data[col_name].min())
                    _range = maxx - minn
                    self.train_data[col_name] = self.train_data[col_name].apply(lambda x: (x - minn) / _range)
                    self.test_data[col_name] = self.test_data[col_name].apply(lambda x: (x - minn) / _range)

            os.system('mkdir -p {}'.format(self.data_name))
            self.train_data.to_csv(self.data_name + '/train.csv', index=False)
            self.test_data.to_csv(self.data_name + '/test.csv', index=False)
        else:
            self.train_data = pd.read_csv(self.data_name + '/train.csv')
            self.test_data = pd.read_csv(self.data_name + '/test.csv')
        self.features = self.train_data.columns.values[:-1]

    def ReadData(self):
        data = pd.read_csv('data/original/ADNIMERGE.csv')
        data = data.replace(['NaN', ], np.nan)
        data = data[data['PTRACCAT'] != 'Hawaiian/Other PI']
        data.sort_values(by=['RID', 'EXAMDATE'], inplace=True)
        data.columns = data.columns.str.replace("_", ".")
        logging.debug('original data: {}'.format(data.shape))
        return data

    def DropColumns(self):
        columns_names = ['RID',] + list(self.features['Feature']) + ['DX',]
        self.train_data = self.train_data.loc[:,columns_names]
        self.test_data = self.test_data.loc[:,columns_names]

        for i, row in self.features.iterrows():
            if row['Type'] == 'continuous':
                self.train_data[row['Feature']] = self.train_data[row['Feature']].astype(str).str.extract(r'([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+))')[0].astype(float)
                self.test_data[row['Feature']] = self.test_data[row['Feature']].astype(str).str.extract(r'([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+))')[0].astype(float)
                if self.args.norm:
                    maxx = max(self.train_data[row['Feature']].max(), self.test_data[row['Feature']].max())
                    minn = min(self.train_data[row['Feature']].min(), self.test_data[row['Feature']].min())
                    _range = maxx - minn
                    self.train_data[row['Feature']] = self.train_data[row['Feature']].apply(lambda x: (x - minn) / _range)
                    self.test_data[row['Feature']] = self.test_data[row['Feature']].apply(lambda x: (x - minn) / _range)

    def Fewshot(self, data_ratio):
        self.train_data = self.train_data.sample(frac=data_ratio, random_state=42)
        logging.debug('{}% of data: {}'.format(data_ratio * 100, self.train_data.shape))

    def DiagnosisInit(self):
        data = self.ReadData()
        data = data[(data['DX'] == 'CN') | (data['DX'] == 'MCI') | (data['DX'] == 'Dementia')]
        logging.info('original data: {}, CN: {}, MCI: {}, AD: {}.'.format(data.shape, len(data[data['DX'] == 'CN']), len(data[data['DX'] == 'MCI']), len(data[data['DX'] == 'Dementia'])))
        return data

    def PredictionInit(self):
        data = self.ReadData()
        # Get all the DX for one RID at different VISCODE
        month_dx = defaultdict(dict)
        for i, row in data.iterrows():
            # print(row)
            month_dx[row['RID']][row['VISCODE']] = row['DX']
            if pd.isna(row['DX']): continue

            if row['VISCODE'] != 'bl':
                if month_dx[row['RID']]['bl'] == 'CN':
                    if row['DX'] == 'MCI' or row['DX'] == 'Dementia':
                        month_dx[row['RID']]['DX'] = '1Non-CN'
                elif month_dx[row['RID']]['bl'] == 'MCI':
                    if row['DX'] == 'Dementia':
                        month_dx[row['RID']]['DX'] = '1Dementia'
                else:
                    month_dx[row['RID']]['DX'] = 'Dementia'
                
                # print(row['VISCODE'])
                mon = int(row['VISCODE'][1:])
                if 'last_time' not in month_dx[row['RID']] or month_dx[row['RID']]['last_time'] < mon:
                    month_dx[row['RID']]['last_time'] = mon
            else:
                if month_dx[row['RID']]['bl'] == 'CN':
                    month_dx[row['RID']]['DX'] = '0CN'
                elif month_dx[row['RID']]['bl'] == 'MCI':
                    month_dx[row['RID']]['DX'] = '0Non-Dementia'
                else:
                    month_dx[row['RID']]['DX'] = 'Dementia'
                
                month_dx[row['RID']]['last_time'] = 0
        bl_data = data[data['VISCODE'] == 'bl']
        bl_data.rename(columns={'DX': 'bl_DX'}, inplace=True)
        mdx_data = pd.DataFrame(month_dx).T
        mdx_data['RID'] = mdx_data.index
        
        mdx_data = mdx_data[~((mdx_data['DX'] == '0Non-Dementia') & (mdx_data['last_time'] < 48))]
        res_data = bl_data.join(mdx_data.set_index('RID'), on='RID')
        
        if 'Unnamed: 0' in res_data.columns:
            res_data.drop(columns=['Unnamed: 0'], inplace=True)
        
        res_data = res_data[res_data['bl'] == 'MCI']
        res_data.dropna(subset=['DX', ], inplace=True)
        logging.info('original data: {}, 0Non-Dementia: {}, 1Dementia: {}.'.format(res_data.shape, len(res_data[res_data['DX'] == '0Non-Dementia']), len(res_data[res_data['DX'] == '1Dementia'])))
        return res_data

    def DataSplit(self, task_name, data):
        test_data = data.sample(2000 if task_name == 'Diagnosis' else 80, random_state=42)
        train_data = data[~data.index.isin(test_data.index)]
        logging.debug('train data: {}, test data: {}'.format(train_data.shape, test_data.shape))
        os.mkdir('data/{}'.format(task_name))
        train_data.to_csv('data/{}/train.csv'.format(task_name), index=False)
        test_data.to_csv('data/{}/test.csv'.format(task_name), index=False)

    def Discrete2Num(self):
        for i, row in self.features.iterrows():
            if row['Type'] == 'discrete':
                cnt, trans_map = 0, {}
                for j, val in self.train_data.iterrows():
                    if val[row['Feature']] not in trans_map:
                        trans_map[val[row['Feature']]] = cnt
                        cnt += 1
                self.train_data[row['Feature']].replace(trans_map, inplace=True)
                self.test_data[row['Feature']].replace(trans_map, inplace=True)

    def Float2Level(self, level):
        for i, row in self.features.iterrows():
            if row['Type'] == 'continuous':
                f_index = list(self.train_data.columns).index(row['Feature'])
                mx = max(self.train_data[row['Feature']].max(), self.test_data[row['Feature']].max())
                mn = min(self.train_data[row['Feature']].min(), self.test_data[row['Feature']].min())
                delta = (mx - mn) / level
                for j, val in self.train_data.iterrows():
                    if val[row['Feature']] != val[row['Feature']]: continue
                    try:
                        self.train_data.iloc[j, f_index] = ((val[row['Feature']] - mn) / delta).astype(int)
                    except:
                        print(row['Feature'])
                        print(j, f_index)
                        exit()
                for j, val in self.test_data.iterrows():
                    if val[row['Feature']] != val[row['Feature']]: continue
                    self.test_data.iloc[j, f_index] = ((val[row['Feature']] - mn) / delta).astype(int)

    def Label2Num(self):
        self.train_data['DX'].replace({'CN':0, 'MCI':1, 'Dementia':2, '0Non-Dementia':0, '1Dementia': 1}, inplace=True)
        self.test_data['DX'].replace({'CN':0, 'MCI':1, 'Dementia':2, '0Non-Dementia':0, '1Dementia': 1}, inplace=True)
