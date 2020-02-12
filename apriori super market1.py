import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
#apriori takes in input as list of lists containing the items.
#[[a,b],[c,d]] example and the items should be strings.

dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])


# The support of item I is defined as the ratio between the number of transactions containing the item I by the total number of transactions.
# let a item is bought 4 times a day then 4*7/7501 = 0.0029 as this dataset for 7days.

#This is measured by the proportion of transactions with item I1, in which item I2 also appears. The confidence between two items I1 and I2,  
in a transaction is defined as the total number of transactions containing both items I1 and I2 divided by the total number of transactions containing I1.
from apyori import apriori

#Lift is the ratio between the confidence and support

#for more info https://analyticsindiamag.com/beginners-guide-to-understanding-apriori-algorithm-with-implementation-in-python/

rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

im = list(rules)

print(im[1])  



  