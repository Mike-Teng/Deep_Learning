import csv
import numpy as np

class_list = []
with open('train_label.csv', 'r') as f:
    for row in f.readlines():
        class_list.append(int(row))

# count each class's number
class_count = np.array([class_list.count(i) for i in range(5)])

print(max(class_count)/class_count)

