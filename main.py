from easyesn import OneHotEncoder
from easyesn import ClassificationESN
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os

ext = '.csv'

def import_data(ext, dir_path):
    # Conditions
    condition = []    # Loop through files:
    for (dirpath, dirnames, filenames) in os.walk(dir_path):        # Loop through Annotations:
        for i in range(len(filenames)):
            # Open files of interest:
            if filenames[i].endswith(ext):

                # Read CSV
                current_activity = pd.read_csv(os.path.join(dirpath, filenames[i]))['activity']
                # Convert to numpy + Append to conditions
                condition.append(current_activity.to_list())
    return condition

condition_path = './condition/'
control_path = './control/'

control = import_data(ext, control_path)
control_label = [False]*len(control)

condition = import_data(ext, condition_path)
condition_label = [True]*len(condition)

inputData = control + condition
outputData = control_label + condition_label

longest = max([len(x) for x in inputData])
inputData = np.array([[0]*(longest-len(x)) + x for x in inputData])


ohe = OneHotEncoder()
outputData = ohe.fit_transform(np.array(outputData))

# Split data into training:
X_train, X_test, y_train, y_test = train_test_split(inputData, outputData,
                                                    test_size=0.25,
                                                    random_state=42)

esn = ClassificationESN(1, 500, 2)
esn.fit(X_train, y_train, verbose=1)
res = esn.predict(X_test, verbose=1)

correct = 0

for i, r in enumerate(res):

    if np.array_equal(y_test[i], res[i]):
        correct += 1


accuracy = 100 * correct / len(y_test)
print("\nModel accuracy: %.2f" % accuracy + "%")
