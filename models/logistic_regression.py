from .base import SklearnModel
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(SklearnModel):

    def __init__(self, args, dataloader):
        super(LogisticRegressionModel, self).__init__(args, dataloader)

        self.model = LogisticRegression(random_state=42)