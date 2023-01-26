import torch
import logging
import numpy as np
from .base import TextModel
from transformers import AutoConfig, AutoTokenizer, AdamW, AutoModelForSequenceClassification
from keras.preprocessing.sequence import pad_sequences

class PretrainedLanguageModel(TextModel):

    def __init__(self, args, dataloader, ModelInit=True):
        super(PretrainedLanguageModel, self).__init__(args, dataloader)
        self.config = AutoConfig.from_pretrained(args.pretrain_path)
        self.config.num_labels = self.N
        logging.debug(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path, config=self.config)
        
        self.train_data = self.DataProcess(self.train_data, self.train_label)
        self.test_data = self.DataProcess(self.test_data, self.test_label)
        
        self.softmax = torch.nn.Softmax(-1)
        if ModelInit:
            self.model = AutoModelForSequenceClassification.from_pretrained(args.pretrain_path, config=self.config)
            self.model = self.model.cuda()
            self.SetupOptimizer(self.model.named_parameters(), lr=5e-5 if args.prompt_type == 'none' else args.lr, warmup=.1)

    def SaveModel(self, path):
        torch.save(self.model, path)

    def SetupOptimizer(self, params, lr, warmup):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.optimizer = AdamW([
            { 'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': lr, 'ori_lr': lr },
            { 'params': [p for n, p in params if any(nd in n for nd in no_decay)],  'weight_decay': 0.0, 'lr': lr, 'ori_lr': lr }
        ], correct_bias=False)
            
    def DataProcess(self, sentences, labels):
        sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
        tokenized_texts = [self.tokenizer.tokenize(sentence) for sentence in sentences]

        print(tokenized_texts[0])
        # exit(0)

        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        out_size = sum([len(sequence) >= self.args.max_length for sequence in input_ids])
        logging.info('{} / {} sentences exceeds length limit.'.format(out_size, len(input_ids)))
        input_ids = pad_sequences(input_ids, maxlen=self.args.max_length, dtype="long", truncating="post", padding="post")

        attention_masks = [[float(i > 0) for i in sequence] for sequence in input_ids]

        dataset = torch.utils.data.TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels))
        sampler = torch.utils.data.SequentialSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=self.args.batch_size)
        return dataloader

    def Train(self):
        self.model.train()
        tr_loss, tr_steps = 0, 0
        for step, batch in enumerate(self.train_data):
            batch = tuple(t.to('cuda') for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            self.optimizer.zero_grad()
            outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            b_loss, b_logits = outputs[0], outputs[1]
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
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                b_logits = self.softmax(outputs[0])
            logits.append(b_logits.cpu())
            labels.append(b_labels.cpu())
        logits = torch.cat([_ for _ in logits], dim=0)
        labels = torch.cat([_ for _ in labels], dim=0)
        logits = self.softmax(logits)
        preds = torch.argmax(logits, -1)
        acc, confusion_matrix = self.CalcResult(labels, preds)
        return acc, confusion_matrix, logits, labels

