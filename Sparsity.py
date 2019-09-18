# Demonstrating Noise, Sparsity, BVT
# 2019Sep18
# Nick Hertle

# First Imports
import numpy as np
import pandas as pd

dataFrameLowP = pd.DataFrame(
	data = {'Feat1' : np.random.normal(loc = 0.0, scale = 3.0, size = 30),
			'Feat2' : np.random.normal(loc = 0.0, scale = 3.0, size = 30),
			'Feat3' : np.random.normal(loc = 0.0, scale = 3.0, size = 30)})

dataFrameLowP['Resp'] = dataFrameLowP['Feat1'] * 0.1 - dataFrameLowP['Feat2'] * 0.2 + dataFrameLowP['Feat3'] * 0.05 + np.random.normal(0.0, 5.0, size = 30)
dataFrameLowP.head()
dataFrameLowP.std(axis = 0)

# import some learning tools
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression

featureCols = ['Feat1', 'Feat2', 'Feat3']

# learn on the training data
## ordinary least squares ##
ols = LinearRegression().fit(dataFrameLowP[featureCols].values, dataFrameLowP['Resp'].values)
## lasso w/ lambda chosen by 5-fold CV ##
# feature standardization not necessary but will simulate "real-world"

# standardize features for penalized regression
def featureStandardizer(dataFrame, colIndices = None):
	if(colIndices is None):
		colIndices = np.arange(len(dataFrame.columns))
	newDF = dataFrame.copy(deep = True)
	for index in colIndices:
		newDF.iloc[:, index] = (newDF.iloc[:, index] - np.mean(newDF.iloc[:, index])) / (np.std(newDF.iloc[:, index]))
	return(newDF)

dataFrameLowP_std = featureStandardizer(
	dataFrameLowP, colIndices = np.array([0, 1, 2]))

# default for ddof for stdev calculation in pandas is 1, numpy 0
dataFrameLowP_std.std(axis = 0, ddof = 0)
dataFrameLowP_std.mean(axis = 0)

lasso = ElasticNetCV(l1_ratio = 1, cv = 5).fit(dataFrameLowP_std[featureCols].values, dataFrameLowP_std['Resp'].values)

# back transform to non-standardized predictors
# B.j = B.Std.j / S.std.j
# B.0 = B.Std.0 - sumj(B.Std.J * (Xbar.std.j/S.std.j))

# boosted decision tree regressor

# test on some new data - 
# easy to do when we control the data creation process
