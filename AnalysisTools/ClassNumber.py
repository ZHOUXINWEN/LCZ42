'''
plot the bar graph to visualise the distribution of class in dataset
'''
import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.subplot(1, 1, 1)
class_names = ['1', '2', '3','4', '5', '6', '7', '8', '9',
               '10', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
#values = ( 5068, 24431, 31693, 8651, 16493, 35290, 3269, 39326, 13584, 11954, 42902, 9514, 9165, 41377, 2392, 7898, 49359)
values = np.load('data/number_of_class_inval.npy')
N = 17

index = np.arange(N)


width = 0.35


p2 = plt.bar(index, values, width, label="Amount", color="#87CEFA")


plt.xlabel('LCZ classes')

plt.ylabel('Amount')


plt.title('Amount of Different LCZ Classes')


plt.xticks(index, class_names)
plt.yticks(np.arange(0, 3500, 500))


plt.legend(loc="upper right")

plt.show()

#print(std_for_class)
