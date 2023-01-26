import torch
import torch.nn.functional as F
from .word_to_vector import Word2VectorModel

class LSTMModel(torch.nn.Module):

    def __init__(self, word_embedding, embedding_dim, max_length, output_dim):
        super(LSTMModel, self).__init__()
        self.word_embedding = word_embedding
        hidden_size = 128
        self.max_length = max_length
        self.lstm = torch.nn.LSTM(embedding_dim, 
                           hidden_size, 
                           num_layers=2, 
                           bidirectional=True, 
                           dropout=0.5,
                           batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_dim)
    
    def forward(self, inputs):
        x = self.word_embedding(inputs)
        # len_seq = torch.tensor([len(_) for _ in inputs]).long()
        # packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, len_seq, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(x)
        outputs = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=-1)
        logits = self.fc2(F.relu(self.fc1(outputs)))
        return logits

class LongShortTermMemoryModel(Word2VectorModel):
    
    def __init__(self, args, dataloader):
        super(LongShortTermMemoryModel, self).__init__(args, dataloader)
        self.model = LSTMModel(self.word_embedding, self.embedding_dim, self.max_length, self.N).cuda()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
