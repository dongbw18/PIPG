import torch
import torch.nn.functional as F
from .word_to_vector import Word2VectorModel

class CNNModel(torch.nn.Module):
    
    def __init__(self, word_embedding, embedding_dim, max_length, output_dim):
        super(CNNModel, self).__init__()
        self.word_embedding = word_embedding
        kernel_sizes, num_channels = [3, 4, 5], [128, 128, 128]
        self.convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=embedding_dim, out_channels=c, kernel_size=k), 
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=max_length-k+1)
            ) for c, k in zip(num_channels, kernel_sizes)
        ])
        self.fc = torch.nn.Linear(sum(num_channels), output_dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, inputs):
        x = self.word_embedding(inputs).permute(0, 2, 1)
        outputs = torch.cat([conv(x).squeeze(-1) for conv in self.convs], dim=-1)
        logits = self.fc(self.dropout(outputs))
        return logits

class ConvolutionalNeuralNetworkModel(Word2VectorModel):
    def __init__(self, args, dataloader):
        super(ConvolutionalNeuralNetworkModel, self).__init__(args, dataloader)
        self.model = CNNModel(self.word_embedding, self.embedding_dim, self.max_length, self.N).cuda()
        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr)
