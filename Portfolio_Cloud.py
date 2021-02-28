
# This Python code generates a cloud of portfolios that are build from stocks.
# This cloud, together with the stocks and the least-varaince portfolio, will be plotted
# a diagram. The minimum variance is not computed by means of optimization. 
# It is simply the random prortfolio with least variance.
# In this code we print some intermediate results. Also the calculation of the average deviation
# is shown. 

import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sn

# In what follows we specify the time interval for which we want to download 
# the financial data
start='2015-1-1'
end='2020-12-4'

# If we want to apply a function that returns today's date, then we can apply the 
# datetime library. We can then specify the time interval as follows:
start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()

# Now we specify a list of tickers:
tickers = ["AAPL", 'AMZN', "FB", "GOOG", 'MCD', 'MSFT']

# Download all tickers into a pandas dataframe. 
# For this we apply the pandas_datareader library
df = pdr.get_data_yahoo(tickers, start, end)["Adj Close"]

# Let us have a look how this dataframe looks like
df.head(2)
df.tail(2)

# In the following we compute the returns of the assets (stocks)
# We can here apply the pct_change (percentage change) 
Returns=df.pct_change()

# Again we will have a look how the returns look like
Returns.head(2)
Returns.tail(2)

# Let us also check the data type of the Returns
type(Returns)

# The following calculates the covariance matrix
CovarMatrix=Returns.cov()

# Note that we also can apply several functions at once. Hence, we do not have to
# produce the intermediate Returns data frame 
CovarMatrix=df.pct_change().cov()
print(CovarMatrix)

CorrelMatrix=df.pct_change().corr()
print(CorrelMatrix)

# We now create a heatmap for the correlation matrix
# To do this, we need the seaborn library
sn.heatmap(CorrelMatrix, annot = True)

# Standard Deviations, Mean abs. dev. and Mean Returns of the single stocks
AssetStandDev=Returns.std()
print(AssetStandDev)

AssetMeanAbsDev=Returns.mad()
print(AssetMeanAbsDev)

AssetMeanReturns=Returns.mean()
print(AssetMeanReturns)

# We will now plot the expected returns against the standard dev.
# Below we will make a more complete plot with axis titles and legend.
plt.subplots(figsize = [10,10])
plt.scatter(AssetStandDev, AssetMeanReturns, marker = "o", s = 10, alpha = 0.3)

for i, label in enumerate(tickers):
    plt.annotate(label, (AssetStandDev[i], AssetMeanReturns[i]))

plt.show()

# Let us calculate the number of assets in the dataframe
Num_Assets = len(df.columns)

# The number of portfolios needs to be big enough to make a representative efficient frontier
Num_Portfolios = 20000

# We will now create a numpy array of random variables
p_weights=np.random.rand(Num_Assets, Num_Portfolios)
print(p_weights)

# The random numbers do not add up to One, as we see in the following:
column_sums = p_weights.sum(axis=0)
print(column_sums)

# We will now normalize the weights. To do this, we  devide each of the 
# entries in the random_numbers by the sum of the weights in the same 
# column.

Rand_Weights=p_weights/column_sums
print(Rand_Weights)

# Let us check that this is correct
column_sums = Rand_Weights.sum(axis=0)
print(column_sums)

# In order to calculate the variance of each portfolio, we
# need to compute x*C*x, where x is the vector with weights and 
# C is the Covariance matrix, and * means the dot-product

# Let us first initialize the vectors/lists for the 
# means and standard deviations  
PortVariances=np.zeros( (1, Num_Portfolios) )
PortMeanReturns=np.zeros( (1, Num_Portfolios) )

# Before we generate a loop, let us do this operation
# for the first column of weights
# Check the following web-page for matrix calculation with numpy
# https://www.programiz.com/python-programming/matrix

PortVariances[:,0]=Rand_Weights[:,0].dot(CovarMatrix).dot(Rand_Weights[:,0])
print(PortVariances)

PortMeanReturns[:,0]=Rand_Weights[:,0].dot(AssetMeanReturns)
print(PortMeanReturns)


for i in range(Num_Portfolios):
    PortVariances[:,i]=Rand_Weights[:,i].dot(CovarMatrix).dot(Rand_Weights[:,i])
    PortMeanReturns[:,i]=Rand_Weights[:,i].dot(AssetMeanReturns)

print(PortVariances)
print(PortMeanReturns)

# Now we calculate the standard deviation which is the square root of the variance
PortStandDev=PortVariances**(1/2)


# Now we look-up the portfolio with least standard deviation.
# This is an approximation, of the min-var-portfolio, which is the better the more
# portfolios we have  
# The following delivers the minimum element (not index)
minElement = np.amin(PortStandDev)
print(minElement)

# The index of the minimum element can be retrieved by:
MinIndex = np.where(PortStandDev == np.amin(PortStandDev))
print(MinIndex)

MinStandDev=PortStandDev[MinIndex]
MinMeanReturn=PortMeanReturns[MinIndex]


#  We will here work with the scaling of the plot
Min1=np.min(PortMeanReturns)
Min2=np.min(AssetMeanReturns)
MinTotal=np.min([Min1,Min2])
Max1=np.max(PortMeanReturns)
Max2=np.max(AssetMeanReturns)
MaxTotal=np.max([Max1,Max2])
bottom = MinTotal-(MaxTotal-MinTotal)*0.05
top = MaxTotal+(MaxTotal-MinTotal)*0.05


# Finally we create the plots

plt.subplots(figsize = [10,10])
plt.scatter(PortStandDev, PortMeanReturns, marker = "o", s = 10, color = "b", alpha = 0.3, label='Random Portfolios')
plt.scatter(AssetStandDev, AssetMeanReturns, marker = "o", s = 20, color = "r", label='Assets/Stocks')
plt.scatter(MinStandDev, MinMeanReturn, marker = "*", s = 100, color = "g", label='Min Variance Portfolio')

plt.title("Portfolio Analysis")
plt.xlabel("Standard Deviation")
plt.ylabel("Expected Return")
plt.legend(loc="upper left")

for i, label in enumerate(tickers):
    plt.annotate(label, (AssetStandDev[i], AssetMeanReturns[i]))


# The following is ignored (at least in VS Code) if the plot is generated by this code.
# Alternative implementation is shown below.
plt.ylim=(bottom, top)  # This was somehow ignored in VS Code

# Now the figure will be saved
plt.savefig('PortfolioCloud.png')


# Here comes the alternative implemenation where the adjsutment of the axis works

fig, ax=plt.subplots(figsize = [10,10])

ax.scatter(PortStandDev, PortMeanReturns, marker = "o", s = 10, color = "b", alpha = 0.3, label='Random Portfolios')
ax.scatter(AssetStandDev, AssetMeanReturns, marker = "o", s = 20, color = "r", label='Assets/Stocks')
ax.scatter(MinStandDev, MinMeanReturn, marker = "*", s = 100, color = "g", label='Min Variance Portfolio')

ax.set(xlabel='Standard Deviation', ylabel='Expected Return',
       title='Portfolio Analysis')

ax.legend(loc="upper left")

for i, label in enumerate(tickers):
    ax.annotate(label, (AssetStandDev[i], AssetMeanReturns[i]))

ax.set_ylim(bottom, top)

# Now the figure will be saved
plt.savefig('PortfolioCloud.png')