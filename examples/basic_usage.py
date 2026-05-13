from megafs import megafs


dataset = 'RNASeqHIVFeb25.xlsx'
#dataset = 'RNASeqFeb25.xlsx'
#BioStackFS(dataset, model_list = {'SVC': preprocessing.MinMaxScaler()}, percent_feat_total = 0.1)
megafs(dataset=dataset)
