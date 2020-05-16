import matplotlib as matplotlib
import networkx as nx
import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from Step3.algorithms import KMeans, BaseGraph, \
    ConnectedComponents, SpectralClustering, Combined

import matplotlib.pyplot as plt

# matplotlib.use('MacOSX')

# =============================================== Help Methods ===============================================
def make_PCA(G, model):
    pca = decomposition.PCA(n_components=3)
    all_vectors_after_pca = pca.fit_transform(StandardScaler().fit_transform(model.vectors))
    return all_vectors_after_pca

# =============================================== Plotter ===============================================
# This class is responsible to plot
# He has algorithms and functions
class Plotter:

    # G is networkx
    # model is after node2vec embedded
    def __init__(self, G, model):
        all_vectors_after_pca = make_PCA(G, model)

        # Make base graph (without algorithm)
        self.BaseGraph = BaseGraph.BaseGraph(all_vectors_after_pca)

        # Make Kmeans
        self.kmeans = KMeans.KMeans(all_vectors_after_pca, "red")

        # Make Connected Componenet
        self.cc = ConnectedComponents.ConnectedComponents(all_vectors_after_pca, G, "green")

        # Make Spectral
        self.spectral = SpectralClustering.SpectralClustering(all_vectors_after_pca, "yellow")

        #Make Combined
        self.Combined = Combined.Combined( self.kmeans,self.spectral,self.cc)

    # Plotting the Graph with no algo
    def showWithBaseGraph(self):
        self.BaseGraph.getPlot().show()

    # Plotting the Graph with KMeans
    def showWithKMeans(self):
        self.kmeans.getPlot().show()

    # Plotting the Graph with ConnectedComponents
    def showWithCC(self):
        self.cc.getPlot().show()

    # Plotting the Graph with SpectralClustering
    def showWithSpectral(self):
        self.spectral.getPlot().show()

    def showCombined(self, mode):
        self.Combined.getPlot(mode).show()

    def getAll(self):
        algorithms = {}
        algorithms["base"] = self.BaseGraph.getPlot()
        algorithms["kmeans"] = self.kmeans.getPlot()
        algorithms["spectral"] = self.spectral.getPlot()
        algorithms["connected"] = self.cc.getPlot()
        algorithms["kmeans+spectral"] = self.Combined.getPlot("kmeans+spectral")
        algorithms["connected+kmeans"] = self.Combined.getPlot("kmeans+connected")
        algorithms["connected+spectral"] = self.Combined.getPlot("spectral+connected")
        algorithms["connected+kmeans+spectral"] = self.Combined.getPlot("kmeans+spectral+connected")
        return algorithms

    def SaveAll(self,prefix):
        self.BaseGraph.getPlot().savefig("." + prefix + "/base.png")
        self.kmeans.getPlot().savefig("." + prefix + "/kmeans.png")
        self.spectral.getPlot().savefig("." + prefix + "/spectral.png")
        self.cc.getPlot().savefig("." + prefix + "/connected.png")
        self.Combined.getPlot("kmeans+spectral").savefig("." + prefix + "/kmeans+spectral.png")
        self.Combined.getPlot("kmeans+connected").savefig("." + prefix + "/connected+kmeans.png")
        self.Combined.getPlot("spectral+connected").savefig("." + prefix + "/connected+spectral.png")
        self.BaseGraph.getPlot().savefig("." + prefix + "/base.png")
        self.Combined.getPlot("kmeans+spectral+connected").savefig("." + prefix + "/connected+kmeans+spectral.png")
