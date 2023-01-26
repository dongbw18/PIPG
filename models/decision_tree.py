import logging
from .base import SklearnModel
from sklearn.tree import DecisionTreeClassifier, export_text

class DecisionTreeModel(SklearnModel):

    def __init__(self, args, dataloader):
        super(DecisionTreeModel, self).__init__(args, dataloader)
        self.model = DecisionTreeClassifier(random_state=42)

    # def Train(self):
    #     self.model.fit(self.train_data[:, 1:-1], self.train_data[:, -1])
        
    #     for name, p in zip(self.feature_names, self.model.feature_importances_):
    #         logging.debug(name + (': %.2f%%' % (p * 100)))
    #     tree_str = export_text(self.model, max_depth=3)
    #     for i, name in enumerate(self.feature_names):
    #         tree_str = tree_str.replace(('feature_%d' % i), name)
    #     logging.debug('\n' + tree_str)
