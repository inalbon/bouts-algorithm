import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# test = np.array([1, 2, 4, 7, 0])
# print('test as numpy.array\n', test)
# test1 = pd.DataFrame(test)
# print('test converted to dataframe\n', test1)
# test2 = test1.to_numpy()
# print('test converted back to numpy.array\n', test2)
# test3 = test1[0].values
# print('test converted to numpy.array with values\n', test3)
# print('diff test\n', np.diff(test3))
#
# a = [1, 4, 1, 7, 5]
# b = a[:-2]
# print(a, b)
#
# sds = pd.DataFrame([[1, 1, 4, 2, 5, 5, 3, 2, 2, 8], [4, 2, 6, 4, 2, 5, 7, 8, 5, 4, ]]).T
# print('sds\n', sds)
# bouts = pd.DataFrame([[4, 2], [2, 1], [4, 3]]).astype(int)
# print('bouts\n', bouts)
# #print('le fameux\n', sds[[1, 4] [1, 2]])
#
# for n in range(3):
#     rev = reversed(range(3))
#     print(n)
#
# array = np.array([[[1, 3, 1],
#                    [4, 6, 1],
#                    [5, 1, 3]],
#                   [[10, 3, 1],
#                    [4, 67, 1],
#                    [5, 1, 3]],
#                   [[10, 300, 1],
#                    [43, 6, 10],
#                    [5, 1, 30]]])
# print(array)
# max_val = np.argmax(array)
# print('max value of array = ', max_val)
# ind = np.unravel_index(np.argmax(array, axis=None), array.shape)
# print('index of max = ', ind)
#
# error1 = np.linalg.norm(np.array([0, 2, 3])-np.array([1, 1, 3]))
# error2 = np.linalg.norm(np.array([3, 2, 0])-np.array([3, 1, 1]))
s = np.array([0, 0.5, 2, 3, 4, 3, -1, 0, 1, 0, -2, 5, 6, 8, 1, -4, -5, -2, 0, 1, 5, 2])
sd_thr = 0
sd = np.diff(s)
plt.plot(s, label=r'Smoothed signal $x_s$')
plt.plot(sd, label=r'Smoothed derivative signal $\dot{x}_s$')
plt.legend()
plt.grid(True)
sd_pos = sd >= sd_thr  # positive part of the derivative
signchange = np.diff(np.array(sd_pos, dtype=int))  #Change neg->pos=1, pos->neg=-1 ?
print(f"signchange = {signchange}")
pos_changes = np.nonzero(signchange > 0)[0]
print(f"pos_changes = {pos_changes}")
neg_changes = np.nonzero(signchange < 0)[0]
print(f"neg_changes = {neg_changes}")

# have to ensure that first change is positive, and every pos. change is complemented by a neg. change
if pos_changes[0] > neg_changes[0]: #first change is negative
    #discard first negative change
    neg_changes = neg_changes[1:]
if len(pos_changes) > len(neg_changes): # lengths must be equal
    difference = len(pos_changes) - len(neg_changes)
    pos_changes = pos_changes[:-difference]

posneg = np.zeros((2, len(pos_changes)))
posneg[0, :] = pos_changes
posneg[1, :] = neg_changes

for b in range(len(pos_changes)):
    plt.plot(np.arange(pos_changes[b], neg_changes[b]+1), sd[pos_changes[b]:neg_changes[b]+1], color='r')

for b in range(len(pos_changes)):
    plt.plot(pos_changes[b], sd[pos_changes[b]], marker='.',
             markerfacecolor='None', markeredgecolor="tab:green", markersize=10)

    plt.plot(neg_changes[b], sd[neg_changes[b]], marker='.',
             markerfacecolor='None', markeredgecolor="tab:red", markersize=10)
print(posneg)

test_histo = [4.1, 4.3, 5, 4.2, 5, 3, 3, 5, 1, 1, 1]
plt.figure()
plt.hist(test_histo)

list_fruits = []
for i in range(10):
    list_fruits.append('apple')

print(list_fruits)
list_fruits = []
print(list_fruits)


plt.show()
