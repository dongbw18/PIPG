import numpy as np
from .base import SklearnModel
from xgboost.sklearn import XGBClassifier

class ExtremeGradientBoostingDecisionTreeModel(SklearnModel):

    def __init__(self, args, dataloader):
        super(ExtremeGradientBoostingDecisionTreeModel, self).__init__(args, dataloader)
        self.n_estimators = 100
        self.model = XGBClassifier(random_state=42, n_estimators=self.n_estimators)
