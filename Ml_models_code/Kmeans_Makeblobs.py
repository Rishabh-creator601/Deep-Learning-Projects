from sklearn.datasets import make_blobs 

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

x,y = make_blobs()

kmeans = KMeans(n_clusters = 4)

kmeans.fit(x)

y_means = kmeans.predict(x)

plt.style.use('ggplot')

plt.scatter(x[y_means==0,0],x[y_means==0,1],s=100,color='red',label="first")

plt.scatter(x[y_means==1,0],x[y_means==1,1],s=100,color='green',label="Second") # (first cluster , column 0 ), (first cluster ,column one)

plt.scatter(x[y_means==2,0],x[y_means==2,1],s=100,color='blue',label="Third")

plt.scatter(x[y_means==3,0],x[y_means==3,1],s=100,color='pink',label="fourth")

plt.xlabel("x-axis")

plt.ylabel('y-axis')

plt.title("KMeans !")

plt.legend()

plt.show()

