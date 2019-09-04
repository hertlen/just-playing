# Abalone ML and Python Practice
# 2019Sep03
# Nick Hertle

# Imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split

# Columns to use
abaloneColnames = ['sex', 'length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell', 'rings']
# Columns denoting continuous predictors
abaloneContPred = ['length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell']
# Columns denoting predictors of any data type
abalonePred = ['length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell', 'sexM', 'sexF', 'sexI']

# 	Sex				nominal			M, F, and I (infant)
# 	Length			continuous	mm	Longest shell measurement
# 	Diameter		continuous	mm	perpendicular to length
# 	Height			continuous	mm	with meat in shell
# 	Whole weight	continuous	grams	whole abalone
# 	Shucked weight	continuous	grams	weight of meat
# 	Viscera weight	continuous	grams	gut weight (after bleeding)
# 	Shell weight	continuous	grams	after being dried
# 	Rings			integer				+1.5 gives the age in years

# load data into a pd dataframe
abalonePD = pd.read_csv('abalone.csv', header = None, names = abaloneColnames)

# Age of the abalone in years is defined as # rings + 1.5
abalonePD['age'] = abalonePD['rings'] + 1.5

abalonePD['sex'].unique()

# write a quick function to turn a string column into one-hot encoding
# takes a pandas DF and a variable name, adds a new column for each
# unique value of the variable. New column is 1/0 (one-hot) encoding
def oneHotEncode(pdDF, varName):
	newDF = pdDF.copy(deep = True)
	uniqueVals = newDF[varName].unique()
	for name in uniqueVals:
		newVarname = varName + "." + name
		newDF[newVarname] = (newDF[varName] == name).astype('int')
	return(newDF)

abalonePD = oneHotEncode(abalonePD, 'sex')
abalonePD.head()

# # boxplot of values
# # note that cont. values were scaled by factor of 200 for use with ANN
# boxplotDat = abalonePD[abaloneContPred].values
# fig, ax = plt.subplots()
# ax.set_title('Abalones')
# ax.boxplot(boxplotDat, labels = abalonePD[abaloneContPred].columns)

# plt.show()

# Correlation matrix of predictors + age
abaloneCorrMatrix = abalonePred.copy()
abaloneCorrMatrix.append('age')
corrMatrix = abalonePD[abaloneCorrMatrix].corr()
print(corrMatrix)

# Use a boolean mask to plot only the lower values of the correlation
# matrix, not including diagonal, to reduce redundant plotting.
mask = np.zeros_like(corrMatrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# corrHeatmap = sb.heatmap(corrMatrix, mask = mask, annot = True, cmap = "YlGnBu").get_figure()
# corrHeatmap.savefig("Abalone Correlation Heatmap.png", dpi = 800)

# pairedPlot = sb.pairplot(abalonePD[abaloneCorrMatrix])
# fig = pairedPlot.fig
# fig.savefig("Abalone Paired Plot.png", dpi = 300)

# object in pandas -> highest coercable type is python object
abalonePD = abalonePD.astype({'sex': str})

for i in np.arange(len(abalonePD.columns)):
	print(abalonePD.columns[i], ":", abalonePD[abalonePD.columns[i]].dtype)

# Do some learning!

# standardize features for penalized regression
def featureStandardizer(dataFrame, colIndices = None):
	if(colIndices is None):
		colIndices = np.arange(len(dataFrame.columns))
	newDF = dataFrame.copy(deep = True)
	for index in colIndices:
		newDF.iloc[:, index] = (newDF.iloc[:, index] - np.mean(newDF.iloc[:, index])) / (np.std(newDF.iloc[:, index]))
	return(newDF)

# Standardize only the continuous predictive features defined in abaloneContPred
abaloneStandardized = featureStandardizer(
	abalonePD, 
	colIndices = np.where(np.isin(abalonePD.columns, abaloneContPred)))

# # Checks out! Mean is epsilon close to 0 given machine precision
# np.std(abaloneStandardized, axis = 0)
# np.mean(abaloneStandardized, axis = 0)

# Hold out 10% of data (418 cases) to evaluate models
abTrainX, abTestX, abTrainY, abTestY = train_test_split(
	abaloneStandardized[['sexM', 'sexF', 'sexI', 'length', 'diameter', 'height', 'whole', 'shucked', 'viscera']],
	abaloneStandardized['age'],
	test_size = 0.1)

# Using ElasticNet, but fixing l1-ratio at 1 -> Lasso objective function
abElasticNetCV = ElasticNetCV(l1_ratio = 1, cv = 5)
abElasticNetCV.fit(abTrainX, abTrainY)

abElasticNetCV.alpha_


abElasticNetCV.coef_
abTrainX.columns