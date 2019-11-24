import os
import math
import random

import numpy as np

from tqdm import tqdm
from easyesn import OneHotEncoder
from easyesn import ClassificationESN


def getIO(directory, output):

	inputs = []

	for filename in os.listdir(directory):
		if filename.endswith(".csv"): 

			csv = np.genfromtxt (os.path.join(directory, filename), delimiter=",", dtype=str)
			dates = csv[:,1][1:]
			activities = csv[:,2][1:]

			days = {}

			for i in range(len(dates)):
				
				if dates[i] in days:
					days[dates[i]].append(int(activities[i]))
				else:
					days[dates[i]] = [int(activities[i])]

			for day in days.items():
				inputs.append([day[1]])

	outputs = [output] * len(inputs)

	return inputs, outputs


def trainSingle(esn, inputData, outputData):

	ohe = OneHotEncoder()

	# dont ask
	inputData = np.array(np.array(inputData))

	if outputData:
		esn.fit(np.array(inputData), np.array([[1, 0]]), verbose=0)
	else:
		esn.fit(np.array(inputData), np.array([[0, 1]]), verbose=0)

	return esn


def testSingle(esn, inputData):

	# dont ask
	inputData = np.array(np.array(inputData))

	res = esn.predict(inputData, verbose=0)

	if np.array_equal(res, [1, 0]):
		return True
	else:
		return False



# Parse
inputs_condition, outputs_condition = getIO('./condition', True)
inputs_control, outputs_control = getIO('./control', False)

inputs_bulk = inputs_control + inputs_condition
outputs_bulk = outputs_control + outputs_condition

tmp_shuffle = list(zip(inputs_bulk, outputs_bulk))
random.shuffle(tmp_shuffle)
inputs_bulk, outputs_bulk = zip(*tmp_shuffle)

# Shuffle / Split
N = len(inputs_bulk)

train_inputs =  inputs_bulk[math.floor(N/4):]
train_outputs = outputs_bulk[math.floor(N/4):]

test_inputs =  inputs_bulk[:math.floor(N/4)]
test_outputs = outputs_bulk[:math.floor(N/4)]


# Big spec = slow decay
esn = ClassificationESN(1, 500, 2, spectralRadius=0.5)


# Train ESN
print("Training ESN")

for i in tqdm(range(len(train_inputs))):
	esn = trainSingle(esn, train_inputs[i], train_outputs[i])

# Test
correct = 0

print("Testing ESN")

for i in tqdm(range(len(test_inputs))):

	res = testSingle(esn, test_inputs[i])

	if res == test_outputs[i]:
		correct += 1

accuracy = 100 * correct / len(test_inputs)
print("\nModel accuracy: %.2f" % accuracy + "%")