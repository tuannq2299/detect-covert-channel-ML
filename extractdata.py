# import numpy as np
# import pandas as pd
# dataset = pd.read_csv('out.csv')
# y_test = dataset.iloc[:,0].values
# print(dataset)
# y_check=vstack((y_test,y_pred))

from csv import writer
from csv import reader

giong = 0 
khac = 0

with open('out_SVM.csv', 'r') as read:
	csv_reader = reader(read)
	for row in csv_reader:
		test = row[0].split("'")[0]
		test1 = test.split(" ")[9]
		test2 = test.split(" ")[19]
		print(test1+" "+test2)
		if test1 == test2:
			giong+=1
		else:
			khac+=1

percent = (giong / (giong + khac) * 100)

print (str(percent))
