import json
import numpy as np
import os

# load models
print(os.path.abspath("trainDump"))
# fileNames = ["/dev/shm/first.json", "/dev/shm/second.json"]
fileNames = [file for file in os.scandir("AverageTrainResults/trainDump")]
print(fileNames)

models = list()
for name in fileNames:
    with open(name, "r") as file:
        model = json.load(file)
        models.append(model)

# average
averageModel = list()

for i in range(1, len(models[0])):
    currentWeights = list()
    for model in models:
        currentWeights.append(np.array(model[i]))

    currentWeights = np.stack(currentWeights)
    averageModel.append(np.average(currentWeights, axis=0))

# remove np.arrays
for i in range(len(averageModel)):
    if type(averageModel[i]) is np.ndarray:
        averageModel[i] = averageModel[i].tolist()

# add model description
description = models[0][0]
averageModel.insert(0, description)

# dump
with open("/dev/shm/average.json", "w") as file:
    json.dump(averageModel, file)
