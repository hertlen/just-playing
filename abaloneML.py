# Abalone ML and Python Practice
# 2019May28
# Nicholas Hertle, Agilent Technologies CDx

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import ElasticNetCV

abaloneColnames = ['sex', 'length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell', 'rings']
abaloneContPred = ['length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell']
abalonePred = ['length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell', 'sexM', 'sexF', 'sexI']

abalonePD = pd.read_csv('abalone.csv', header = None, names = abaloneColnames)

abalonePD['age'] = abalonePD['rings'] + 1.5

# following two lines return the same data
# abalonePD.iloc[0:4, len(abalonePD.columns)-1]
# abalonePD.loc[0:4, 'age']

# abalonePD['sex'].unique()

def oneHotEncode(pdDF, varName):
	newDF = pdDF.copy(deep = True)
	uniqueVals = newDF[varName].unique()
	for name in uniqueVals:
		newVarname = varName+name
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

# Correlation matrix of predictors
abaloneCorrMatrix = abalonePred.copy()
abaloneCorrMatrix.append('age')
corrMatrix = abalonePD[abaloneCorrMatrix].corr()
print(corrMatrix)

mask = np.zeros_like(corrMatrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
corrHeatmap = sb.heatmap(corrMatrix, mask = mask, annot = True, cmap = "YlGnBu").get_figure()
corrHeatmap.savefig("Abalone Correlation Heatmap.png", dpi = 800)

# Paired Scatterplots/Boxplots along Diag

pairedPlot = sb.pairplot(abalonePD[abaloneCorrMatrix])
fig = pairedPlot.fig
fig.savefig("Abalone Paired Plot.png", dpi = 300)

# object in pandas -> highest coercable type is python object
# abalonePD = abalonePD.astype({'sex': str})

# for i in np.arange(len(abalonePD.columns)):
# 	print(abalonePD.columns[i], ":", abalonePD[abalonePD.columns[i]].dtype)

abTrainX, abTestX, abTrainY, abTestY = train_test_split(
	abalonePD[['sexM', 'sexF', 'sexI', 'length', 'diameter', 'height', 'whole', 'shucked', 'viscera']],
	abalonePD['age'],
	test_size = 0.25)

abElasticNetCV = ElasticNetCV(l1_ratio = np.arange(start = 0.1, stop = 1.1, step = 0.1), cv = 5)
abElasticNetCV.fit(abaloneTrainX, abaloneTrainY)

# 	Sex		nominal			M, F, and I (infant)
# 	Length		continuous	mm	Longest shell measurement
# 	Diameter	continuous	mm	perpendicular to length
# 	Height		continuous	mm	with meat in shell
# 	Whole weight	continuous	grams	whole abalone
# 	Shucked weight	continuous	grams	weight of meat
# 	Viscera weight	continuous	grams	gut weight (after bleeding)
# 	Shell weight	continuous	grams	after being dried
# 	Rings		integer			+1.5 gives the age in years