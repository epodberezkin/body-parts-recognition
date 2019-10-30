
from deepImageFeatures import RFRModel
import os

class Solver:

    def __init__(self, train=False):
        self.model = RFRModel(train)

    def process(self, data):
        return self.model.process(data)

    def test(self):
        return self.model.test_train()
