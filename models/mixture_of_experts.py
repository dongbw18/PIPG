from .base import TextModel
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AdamW, AutoTokenizer
from keras.preprocessing.sequence import pad_sequences

class Expert(torch.nn.Module):
    # def __init__(self, max_length, pretrain_path='google/electra-base-discriminator'):
    # def __init__(self, max_length, prompt, pretrain_path='microsoft/deberta-v3-base'):
    def __init__(self, N, pretrain_path, config):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.N = N
        self.pretrain_path = pretrain_path
        self.hidden_size = config.hidden_size
        self.model = AutoModel.from_pretrained(pretrain_path, config=config)
        self.fc1 = torch.nn.Linear(self.hidden_size * self.N, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.N)

    def forward(self, token, att_mask, pos):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
        Return:
            (B, 2H), representations for sentences
        """
        if 'deberta' in self.pretrain_path:
            hidden = self.model(token, token_type_ids=None, attention_mask=att_mask, return_dict=False)[0]
        else:
            hidden, _ = self.model(token, token_type_ids=None, attention_mask=att_mask, return_dict=False)
        # Get entity start hidden state
        onehot_hiddens = []
        pos = pos.unsqueeze(-1)
        offset = torch.ones(pos.size()).long().to(pos.device)
        for i in range(self.N):
            onehot = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
            onehot = onehot.scatter_(1, pos + offset * i, 1)
            onehot_hiddens.append((onehot.unsqueeze(2) * hidden).sum(1))  # (B, H)
        x = torch.cat(onehot_hiddens, 1)  # (B, H * N)
        x = self.fc1(x) # (B, H)
        x = self.fc2(x) # (B, N)
        return x


class MixtureOfExpertsModel(TextModel):
    
    def __init__(self, args, dataloader):
        super(MixtureOfExpertsModel, self).__init__(args, dataloader)
        self.num_experts = 1
        self.config = AutoConfig.from_pretrained(args.pretrain_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path, config=self.config)
        self.train_dataloader = self.tokenize(self.train_data, self.train_label)
        self.test_dataloader = self.tokenize(self.test_data, self.test_label)

        self.experts = [Expert(self.N, args.pretrain_path, self.config).cuda() for _ in range(self.num_experts)]
        params = []
        for expert in self.experts:
            params += expert.named_parameters()
        
        self.train_data_original = self.OriginalDataProcess(self.original_dataloader.train_data.values)
        self.test_data_original = self.OriginalDataProcess(self.original_dataloader.test_data.values)
        self.gate = torch.nn.Linear(self.input_length, self.num_experts).cuda()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.softmax = torch.nn.Softmax(-1)

        self.SetupOptimizer(params, lr=args.lr, warmup=.1, gate_lr=1e-3)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def get_prompt(self, task_name):
        if task_name == 'Diagnosis':
            return self.tokenizer.tokenize('health status is :') + ['[unused101]', '[unused102]', '[unused103]'] + self.tokenizer.tokenize('[SEP]'), 4
        if task_name == 'Prediction':
            return self.tokenizer.tokenize('condition will :') + ['[unused101]', '[unused102]', '[unused103]'] + self.tokenizer.tokenize('[SEP]'), 3
        if task_name == 'Decompensation':
            return self.tokenizer.tokenize('patient will :') + ['[unused101]', '[unused102]', '[unused103]'] + self.tokenizer.tokenize('[SEP]'), 3
        return '_'

    def tokenize(self, sentences, labels, shuffle=False):
        sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
        sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        prompt, offset = self.get_prompt(self.args.task_name)
        pos = [min(len(_) + offset, self.args.max_length - 1) for _ in sentences]
        tokenized_texts = [sentence + prompt for sentence in sentences]

        print(tokenized_texts[0])

        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        out_size = sum([len(sequence) >= self.args.max_length for sequence in input_ids])
        print('{} / {} sentences exceeds length limit.'.format(out_size, len(input_ids)))
        input_ids = pad_sequences(input_ids, maxlen=self.args.max_length, dtype="long", truncating="post", padding="post")

        attention_masks = [[float(i > 0) for i in sequence] for sequence in input_ids]
        
        dataset = torch.utils.data.TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(pos), torch.tensor(labels))
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=self.args.batch_size)
        return dataloader

    def OriginalDataProcess(self, data):
        inputs, labels = torch.tensor(data[:, 1:-1]).float(), torch.tensor(data[:, -1]).long()
        inputs = torch.nan_to_num(inputs, nan=-1, posinf=-1, neginf=-1)
        self.input_length = inputs.shape[-1]
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        sampler = torch.utils.data.SequentialSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=self.args.batch_size)
        return dataloader

    def SetupOptimizer(self, params, lr, warmup, gate_lr):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.optimizer = AdamW([
            { 'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': lr, 'ori_lr': lr },
            { 'params': [p for n, p in params if any(nd in n for nd in no_decay)],  'weight_decay': 0.0, 'lr': lr, 'ori_lr': lr },
            # { 'params': [p for n, p in self.gate.named_parameters()], 'weight_decay': 0.0, 'lr': gate_lr, 'ori_lr': gate_lr}
        ], correct_bias=False)


    def Train(self):
        for expert in self.experts:
            expert.train()
        self.gate.train()
        tr_loss, tr_steps = 0, 0
        for step, (batch, batch_original) in enumerate(zip(self.train_dataloader, self.train_data_original)):
            batch = tuple(t.to('cuda') for t in batch)
            b_input_ids, b_input_mask, b_pos, b_labels = batch
            batch_original = tuple(t.to('cuda') for t in batch_original)
            b_inputs, _ = batch_original

            self.optimizer.zero_grad()
            b_gating_score = self.softmax(self.gate(b_inputs))
            b_logits = [self.softmax(expert(b_input_ids, b_input_mask, b_pos)).unsqueeze(1) for expert in self.experts]
            b_logits = torch.cat(b_logits, dim=1)
            b_logits = (b_gating_score.unsqueeze(-1) * b_logits).sum(1)
            b_loss = self.loss_func(b_logits, b_labels)
            b_loss.backward()
            self.optimizer.step()

            tr_loss += b_loss.item()
            tr_steps += 1
        logging.info("Train loss: {}".format(tr_loss / tr_steps))

    def Test(self):
        for expert in self.experts:
            expert.eval()
        self.gate.eval()
        logits, labels = [], []
        gate_scores = torch.zeros(self.num_experts)
        for step, (batch, batch_original) in enumerate(zip(self.test_dataloader, self.test_data_original)):
            batch = tuple(t.to('cuda') for t in batch)
            b_input_ids, b_input_mask, b_pos, b_labels = batch
            batch_original = tuple(t.to('cuda') for t in batch_original)
            b_inputs, _ = batch_original

            with torch.no_grad():
                b_gating_score = self.softmax(self.gate(b_inputs))
                gate_scores += b_gating_score.sum(0).cpu()
                b_logits = [self.softmax(expert(b_input_ids, b_input_mask, b_pos)).unsqueeze(1) for expert in self.experts]
                b_logits = torch.cat(b_logits, dim=1)
                b_logits = (b_gating_score.unsqueeze(-1) * b_logits).sum(1)
            logits.append(b_logits.cpu())
            labels.append(b_labels.cpu())
        logging.info('scores of experts : {}'.format(gate_scores))
        logits = torch.cat([_ for _ in logits], dim=0)
        labels = torch.cat([_ for _ in labels], dim=0)
        preds = torch.argmax(logits, -1)
        acc, confusion_matrix = self.CalcResult(labels, preds)
        return acc, confusion_matrix, logits, labels
    
    def SaveModel(self, path):
        torch.save(self.experts[0], path)
