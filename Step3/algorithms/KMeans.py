import matplotlib as matplotlib
from sklearn.cluster import KMeans as KMeansAlgorithm # Algorithm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# matplotlib.use('MacOSX')

class KMeans:
    def __init__(self, vectors_3dim, color):
        self.vectors_3dim = vectors_3dim
        self.color = color
        self.km = KMeansAlgorithm(n_clusters=self.find_elbow(), init='k-means++', max_iter=300, n_init=10, random_state=0)
        self.km.fit_predict(self.vectors_3dim)

    def find_elbow(self):
        mms = MinMaxScaler()
        mms.fit(self.vectors_3dim)
        data_transformed = mms.transform(self.vectors_3dim)

        Sum_of_squared_distances = []
        K = range(1, int(0.01*len(data_transformed)))
        for k in K:
            km = KMeansAlgorithm(n_clusters=k)
            km = km.fit(self.vectors_3dim)
            Sum_of_squared_distances.append(km.inertia_)
            if (len(Sum_of_squared_distances))>1:
                gradient = Sum_of_squared_distances[k-1] - Sum_of_squared_distances[k-2]
                if gradient > -1200:
                    return k-1

    def getPlot(self):

        # Create a scatter plot
        fig = plt.figure(dpi=120, figsize=(8.0, 5.0))
        ax = fig.add_subplot( projection='3d')

        # drow all the nodes in the grapg
        ax.scatter(self.vectors_3dim[:, 0], self.vectors_3dim[:, 1], self.vectors_3dim[:, 2], s=1)

        # drow the clusters
        ax.scatter(self.km.cluster_centers_[:, 0], self.km.cluster_centers_[:, 1],
                            self.km.cluster_centers_[:, 2], s=700, c=self.color, marker = 'o', depthshade=False)
        # the axis labels
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        return plt

