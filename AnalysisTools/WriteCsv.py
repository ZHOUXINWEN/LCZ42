import csv
import os
import torch

out = torch.zeros(17)
out[3] = 1

with open('trys.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    csvrow = []
    for i in out.numpy():
        csvrow.append(i)
    spamwriter.writerow(csvrow)

