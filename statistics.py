# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:00:35 2024

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

np.random.seed(42) #for generating random data
#Dataframe
data = pd.DataFrame({
'Product_ID' : np.arange(1,101),#creating 100 product ids
'Category' : np.random.choice(['Electronics','Clothing','Home'], size=100),
'Price' : np.random.normal(loc=50, scale=15, size=100).round(2),
'Units_Sold' : np.random.poisson(lam=20, size = 100),
'Customer_Rating' : np.random.uniform(1,5,100).round(1),
'In_stock' : np.random.choice([True,False],size=100, p=[0.8,0.2])
    })

df = data.copy()

#print("First 5 rows:")
#print(df.head())

price_mean = df['Price'].mean()

#Calculate median
price_median = df["Price"].median()

print(f"Mean price : ${price_mean : 2f}")
print(f"Median price : ${price_median : 2f}")

#Calculate mode
price_mode = df["Price"].mode()[0]
print(f"Mode price : ${price_mode : 2f}")

favourite_colours = pd.Series(["Blue", "Green", "Blue", "Red", "BLue"])
colour_mode = favourite_colours.mode()[0]
print(f"Most Common favourite colour : {colour_mode}")

max_price = df['Price'].max()
min_price = df['Price'].min()
price_range = max_price - min_price
print(f"Price : ${min_price : 2f} ${max_price : .2f}")

print(f"Price Range : ${price_range : .2f}")

price_variance = df["Price"].var()
price_std = df["Price"].std()
print(f"Price variance : ${price_variance : .2f}")
print(f"Price standard deviation : ${price_std : .2f}")

ex1 = pd.Series([100,102,98,101,99])
ex2 = pd.Series([80,120,70,130,60])
ex1_std = ex1.std() #Standard deviation is low
ex2_std = ex2.std() #Standard deviation is high

print(f"ex1 std : ${ex1_std : .2f} ex2 std : ${ex2_std : .2f}")

price_skew = df['Price'].skew()
print(f"Price skewness : ${price_skew : .2f} ")

# plt.figure(figsize = (14,5))
# plt.subplot(1,2,1)
# sns.histplot(df['Price'], kde=True)
# plt.title("Price Distribution")

# plt.show()
"""
Data is concentrated on lower side (right-skewed data distribution) 
and the outliers (skewness) lie on the right side or the higher side.
"""
right_skewed = np.random.exponential(scale=2,size=1000)
right_skewed_df = pd.DataFrame({"Value" : right_skewed})

right_skew_val = right_skewed_df['Value'].skew()
print(f"Right-Skewed Data Skewness : ${right_skew_val : 0.2f}")

#Plotting
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
sns.histplot(right_skewed_df['Value'], kde = True)
plt.title("Right-Skewed Distribution")
plt.show()

plt.subplot(1,2,2)
sns.boxplot(x=right_skewed_df["Value"])
plt.title("Right-Skewed Box Plot")
