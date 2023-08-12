import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
data=pd.read_csv('power.csv')
print(data.columns)
print(len(data))
print(data.info())
print(data.isna().sum())
print(data.describe())

for i in data.columns.values:
    print(data[i].value_counts())

'''for i in data.select_dtypes(include='number').columns.values:
    sn.boxplot(data[i])
    plt.show()'''

'''for i in data.columns.values:
    if len(data[i].value_counts())<=20:
        val=data[i].value_counts().values
        index=data[i].value_counts().index
        plt.pie(val,labels=index,autopct='%1.1f%%')
        plt.title(f'{i} column values')
        plt.legend()
        plt.show()'''

lab=LabelEncoder()
for i in data.columns.values:
    data[i]=lab.fit_transform(data[i])

'''for i in data.columns.values:
    for j in data.columns.values:
        sn.distplot(data[i],color='red',label=f'{i}')
        sn.distplot(data[j],color='blue',label=f'{j}')
        plt.title(f"The {i} vs  {j}")
        plt.xlabel(f'{i}')
        plt.ylabel(f'{j}')
        plt.legend()
        plt.show()'''

'''plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()'''

for i in data.columns.values:
    data['z-scores']=(data[i]-data[i].mean())/data[i].std()
    outliers=np.abs(data['z-scores'])>3
    print(f'The number of outliers in {i} column {outliers.sum()}')



sn.pairplot(data)
plt.show()

# 2. Pair Plot
sn.pairplot(data, hue='HVACSystem')
plt.show()



# 4. Bubble Plot
for i in data.columns.values:
    for j in data.columns.values:
        plt.scatter(data[i], data[j], s=data['Z'] * 50, alpha=0.5)
        plt.title(f'{i} vs {j}')
        plt.xlabel(i)
        plt.ylabel(j)
        plt.legend()
        plt.show()


pd.plotting.parallel_coordinates(data, 'HVACSystem')
plt.show()

correlation_matrix = data.corr()
sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()



for i in data.columns.values:
    for j in data.columns.values:
        sn.kdeplot(data[i], data[j], cmap='Blues', fill=True)
        plt.xlabel(i)
        plt.ylabel(j)
        plt.title(f"its {i} vs {j}")
        plt.show()


pd.plotting.andrews_curves(data, 'HVACSystem')
plt.show()


pd.plotting.radviz(data, 'HVACSystem')
plt.show()