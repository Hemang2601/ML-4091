import pandas as pd

winedatac=pd.read_csv('C:/Users/student/Downloads/ML-4091/wine.csv')

print(winedatac)
print(winedatac.head())
print("shape\n",winedatac.shape)
print("columns\n",winedatac.columns)
print("dtypes\n",winedatac.dtypes)
print("ndim\n",winedatac.ndim)
print("size\n",winedatac.size)


winedatae=pd.read_excel('C:/Users/student/Downloads/ML-4091/wine.xls')

print('\n')


