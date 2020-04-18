#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
ds=pd.read_csv('cricket dat.csv')
x=ds.iloc[:,[4,17]].values

#number of clusters-elbow method
from sklearn.cluster import KMeans
w=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(x)
    w.append(km.inertia_)
plt.plot(range(1,11),w)
plt.title('elbow method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

#apllying k-means 
km=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
ykm=km.fit_predict(x)

#scatter plot graph
plt.scatter(x[ykm==0,0],x[ykm==0,1],s=100,c='red',label='batsman')
plt.scatter(x[ykm==1,0],x[ykm==1,1],s=100,c='blue',label='bowlers')
plt.scatter(x[ykm==2,0],x[ykm==2,1],s=100,c='green',label='allrounders')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=300,c='yellow',label='centroid')
plt.title('RUNS VS WKT')
plt.xlabel('RUNS')
plt.ylabel('WKT')
plt.legend()
plt.show()