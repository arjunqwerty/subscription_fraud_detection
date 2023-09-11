import random 
import matplotlib.pyplot as plt 
    
# store the random numbers in a list 
nums = [] 
mu = 1.0
sigma = 0.175
    
for i in range(10000): 
    # temp = random.lognormvariate(mu, sigma) 
    temp = int(10 * random.weibullvariate(3, 4))
    nums.append(10*temp) 
        
# plotting a graph 
plt.hist(nums, bins = 200) 
plt.show()
# print(nums)