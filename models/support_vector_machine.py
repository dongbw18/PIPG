from .base import SklearnModel
from sklearn import svm

class SupportVectorMachineModel(SklearnModel):

    def __init__(self, args, dataloader):
        super(SupportVectorMachineModel, self).__init__(args, dataloader)
        
        self.model = svm.SVC(probability=True, random_state=42)
