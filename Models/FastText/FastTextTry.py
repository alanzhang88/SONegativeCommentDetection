from FastTextUtil import FastText
import pandas as pd
import numpy as np

instance = FastText()
(train, test, trainlabel, testlabel) = instance.preprocessData()
texts = instance.extractText(test)
labels = instance.predict(texts)
