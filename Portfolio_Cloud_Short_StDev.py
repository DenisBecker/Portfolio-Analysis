
# This Python code generates a cloud of portfolios that are build from stocks.
# This cloud, together with the stocks and the least-varaince portfolio, will be plotted
# a diagram. The minimum variance is not computed by means of optimization. 
# It is simply the random prortfolio with least variance.

import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


# Specify the time interval:
start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()

# Now we specify a list of tickers:
tickers = ["AAPL", 'AMZN', "FB", "GOOG", 'GM', 'MCD', 'MSFT']

# Download all tickers into a pandas dataframe. 
# For this we apply the pandas_datareader library
df = pdr.get_data_yahoo(tickers, start, end)["Adj Close"]


# In the following we compute the returns of the assets (stocks)
# We can here apply the pct_change (percentage change) 
Returns=df.pct_change()

# The following calculates the covariance matrix
CovarMatrix=Returns.cov()

# Standard Deviations, Mean abs. dev. and Mean Returns of the single stocks
AssetStandDev=Returns.std()

AssetMeanReturns=Returns.mean()

# Let us calculate the number of assets in the dataframe
Num_Assets = len(df.columns)

# The number of portfolios needs to be big enough to make a representative efficient frontier
Num_Portfolios = 20000

# We will now create a numpy array of random variables
p_weights=np.random.rand(Num_Assets, Num_Portfolios)

# We will now normalize the weights.
column_sums = p_weights.sum(axis=0)
Rand_Weights=p_weights/column_sums

# In order to calculate the variance of each portfolio, we
# need to compute x*C*x, where x is the vector with weights and 
# C is the Covariance matrix, and * means the dot-product

# Let us first initialize the vectors/lists for the 
# means and standard deviations  
PortVariances=np.zeros( (1, Num_Portfolios) )
PortMeanReturns=np.zeros( (1, Num_Portfolios) )

# Generating all the portfolios

for i in range(Num_Portfolios):
    PortVariances[:,i]=Rand_Weights[:,i].dot(CovarMatrix).dot(Rand_Weights[:,i])
    PortMeanReturns[:,i]=Rand_Weights[:,i].dot(AssetMeanReturns)


# Now we calculate the standard deviation which is the square root of the variance
PortStandDev=PortVariances**(1/2)

# Now we look-up the portfolio with least standard deviation.
# This is an approximation, of the min-var-portfolio, which is the better the more
# portfolios we have  
# The following delivers the minimum element (not index)
minElement = np.amin(PortStandDev)

# The index of the minimum element can be retrieved by:
MinIndex = np.where(PortStandDev == np.amin(PortStandDev))

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
fig, ax=plt.subplots(figsize = [10,10])

ax.scatter(PortStandDev, PortMeanReturns, marker = "o", s = 10, color = "b", alpha = 0.3, label='Random Portfolios')
ax.scatter(AssetStandDev, AssetMeanReturns, marker = "o", s = 20, color = "r", label='Assets/Stocks')
ax.scatter(MinStandDev, MinMeanReturn, marker = "*", s = 100, color = "g", label='Min Variance Portfolio')

ax.set(xlabel='Standard Deviation', ylabel='Expected Return',
       title='Portfolio Analysis')

ax.legend(loc="upper left")

for i, label in enumerate(tickers):
    ax.annotate(label, (AssetStandDev[i], AssetMeanReturns[i]))

ax
ax.set_ylim(bottom,top)

# Now the figure will be saved
plt.savefig('PortfolioCloud.png')