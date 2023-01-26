import os, json, math
import logging
import torch
import numpy as np
from transformers import BertTokenizer
from .base import TextModel

class WordTokenizer:

    def __init__(self, vocab, max_length):
        self.vocab = vocab
        self.max_length = max_length
        self.unk_id = vocab['[UNK]']
        self.pad_id = vocab['[PAD]']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def Tokenize(self, text):
        text = text.strip()
        if not text: return []
        tokens = text.replace(',', ' , ').split()

        ids = []
        for token in tokens:
            token = token.lower()
            if token in self.vocab:
                ids.append(self.vocab[token])
                continue
            for _ in self.tokenizer.tokenize(token):
                if _.startswith('##'): _ = _[2:]
                ids.append(self.unk_id if _ not in self.vocab else self.vocab[_])
        while(len(ids) < self.max_length): ids.append(self.pad_id)
        return ids[:self.max_length]

class Word2VectorModel(TextModel):

    def __init__(self, args, dataloader):
        super(Word2VectorModel, self).__init__(args, dataloader)
        self.args = args
        self.model = 'Word2VectorModel'
        self.softmax = torch.nn.Softmax(-1)
        self.loss_func = torch.nn.CrossEntropyLoss()

        self.InitWord2Vec()
        self.train_data = self.DataLoaderProcess(self.train_data, self.train_label)
        self.test_data = self.DataLoaderProcess(self.test_data, self.test_label)

    def InitWord2Vec(self):
        if not os.path.isdir('config/glove'):
            os.system('config/download_glove.sh')
        self.token2id = json.load(open('config/glove/glove.6B.50d_word2id.json'))
        self.word2vec = np.load('config/glove/glove.6B.50d_mat.npy')
        self.max_length = self.args.max_length
        self.num_token = len(self.token2id)
        self.embedding_dim = self.word2vec.shape[-1]
        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
            self.num_token += 1
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
            self.num_token += 1
        
        self.word_embedding = torch.nn.Embedding(self.num_token, self.embedding_dim)
        self.word2vec = torch.from_numpy(self.word2vec)
        if self.num_token == len(self.word2vec) + 2:            
            unk = torch.randn(1, self.embedding_dim) / math.sqrt(self.embedding_dim)
            blk = torch.zeros(1, self.embedding_dim)
            self.word_embedding.weight.data.copy_(torch.cat([self.word2vec, unk, blk], 0))
        else:
            self.word_embedding.weight.data.copy_(self.word2vec)
        self.tokenizer = WordTokenizer(self.token2id, self.args.max_length)

    def DataLoaderProcess(self, inputs, labels):
        tokens = [self.tokenizer.Tokenize(_) for _ in inputs]
        tokens, labels = torch.tensor(tokens).long(), torch.tensor(labels).long()
        dataset = torch.utils.data.TensorDataset(tokens, labels)
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
