from .base import SklearnModel
from sklearn.ensemble import BaggingClassifier

class RandomSubspaceModel(SklearnModel):
    def __init__(self, args, dataloader):
        super(RandomSubspaceModel, self).__init__(args, dataloader)
        self.model = BaggingClassifier(random_state=42)