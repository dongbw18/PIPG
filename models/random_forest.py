import numpy as np
from .base import SklearnModel
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(SklearnModel):

    def __init__(self, args, dataloader):
        super(RandomForestModel, self).__init__(args, dataloader)
        self.n_estimators = 100
        self.model = RandomForestClassifier(random_state=42, n_estimators=self.n_estimators)

    # def Train(self):
    #     self.model.fit(self.train_data[:, 1:-1], self.train_data[:, -1])
    #     print(self.model.n_estimators)
        
        # N = 10
        # indicators, n_nodes_ptr = self.model.decision_path(self.train_data[:N, 1:-1])
        # cnt = np.zeros([N, self.n_estimators])
        # indicators = indicators.toarray()
        # for i in range(N):
        #     for j in range(self.n_estimators):
        #         cnt[i][j] = indicators[i][n_nodes_ptr[j]:n_nodes_ptr[j + 1]].sum()
        # print(cnt)