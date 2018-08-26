import random
import matplotlib.pyplot as plt
import numpy as np

epsilon = 0.1
N = 2000
count_of_one = 0

for i in range(N):
    if random.random() < (0.5+epsilon):
        count_of_one += 1
  
Majority_percent = (count_of_one/N)*100


epsilon = 0.005
Total_percent_arr = []
majority_vote = []
N_arr = []

for N in range(5000,105000,5000):
    count_of_ones = 0
    for i in range(N):
        if random.random() < (0.5+epsilon):
            count_of_ones += 1
    if (count_of_ones/N)*100 > 50:
        majority_vote.append(1)
    else:
        majority_vote.append(0)
    Total_percent_arr.append((count_of_ones/N)*100)
    N_arr.append(N)

plt.plot(N_arr,Total_percent_arr)
plt.plot(N_arr,majority_vote)


epsilon_chk = 0.005
N_chk_max = 100000
x = []
y = []

for i in range(N_chk_max):
    x.append(i)
    yy = 1-2*np.exp(-2*epsilon_chk*epsilon_chk*i)
    y.append(yy)
    
plt.plot(x,y)


    