from megafs import megafs


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

dataset = 'RNASeqHIVFeb25.xlsx'
#megafs(dataset=dataset, model_list = {'QDA': QuadraticDiscriminantAnalysis(),'KNNC': KNeighborsClassifier()}, percent_feat_total = 0.1)
megafs(dataset=dataset, model_list = {'QDA': QuadraticDiscriminantAnalysis(), 'KNNC': KNeighborsClassifier()}, percent_feat_total = 0.05, output_file='Prueba')
