import logging
import torch
import torch.nn.functional as F
from .base import BaseModel

class MLPModel(torch.nn.Module): # lr = 5e-2

    def __init__(self, input_dim, output_dim):
        super(MLPModel, self).__init__()
        hidden_size1 = 512 # 256 # 128 # 512
        hidden_size2 = 128 # 64 # 32 # 128
        self.fc1 = torch.nn.Linear(input_dim, hidden_size1)  
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)
        
    def forward(self, inputs):
        x = self.dropout(F.relu(self.fc1(inputs)))
        x = self.dropout(F.relu(self.fc2(x)))
        logits = self.fc3(x)
        return logits

class MultiLayerPerceptronModel(BaseModel):

    def __init__(self, args, dataloader):
        super(MultiLayerPerceptronModel, self).__init__(args)
        self.train_data = self.DataProcess(dataloader.train_data.values)
        self.test_data = self.DataProcess(dataloader.test_data.values) 
        self.softmax = torch.nn.Softmax(-1)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.model = MLPModel(self.input_length, self.N).cuda()
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr)

    def DataProcess(self, data):
        inputs, labels = torch.tensor(data[:, 1:-1]).float(), torch.tensor(data[:, -1]).long()
        inputs = torch.nan_to_num(inputs, nan=-1, posinf=-1, neginf=-1)
        self.input_length = inputs.shape[-1]
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        sampler = torch.utils.data.SequentialSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=self.args.batch_size)
        return dataloader

    def Train(self):
        self.model.train()
        tr_loss, tr_steps = 0, 0
        for step, batch in enumerate(self.train_data):
            batch = tuple(t.to('cuda') for t in batch)
            b_inputs, b_labels = batch

            self.optimizer.zero_grad()
            b_logits = self.model(b_inputs)
            b_loss = self.loss_func(b_logits, b_labels)

            b_loss.backward()
            self.optimizer.step()

            tr_loss += b_loss.item()
            tr_steps += 1
        logging.info("Train loss: {}".format(tr_loss / tr_steps))

    def Test(self):
        self.model.eval()
        logits, labels = [], []
        for step, batch in enumerate(self.test_data):
            batch = tuple(t.to('cuda') for t in batch)
            b_inputs, b_labels = batch
            with torch.no_grad():
                b_logits = self.model(b_inputs)
            logits.append(b_logits.cpu())
            labels.append(b_labels.cpu())
        logits = torch.cat([_ for _ in logits], dim=0)
        labels = torch.cat([_ for _ in labels], dim=0)
        logits = self.softmax(logits)
        preds = torch.argmax(logits, -1)
        acc, confusion_matrix = self.CalcResult(labels, preds)
        return acc, confusion_matrix, logits, labels
        