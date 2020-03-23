#coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
import time
from  scipy.spatial.distance import cdist

class KMeans:
    """
        K-Means clustering.
        
        Parameters
        ----------
        n_cluters : int, default=4
            The number of clusters to form as well as the number of centroids to generate.
            
        init : {'k-means++', 'random'}, default='k-means++'
            Method for initialization, defaults to 'k-means++':
            
            'k-means++' : selects initial cluster centers for k-mean 
            clustering in a smart way to speed up convergence.
            
            'random': choose k observations (rows) at random from data for
            the initial centroids.
 
        max_iter : int, default=100
            Maximum number of iterations of the k-means algorithm for a single run.
                 
        dist : str, default='euclidean'
            The distance metric to use.
        
        Attributes
        ----------
        cluster_centers_ : ndarray of shape (n_clusters, n_features)
            Coordiantes of cluster center.
        
        labels_ : ndarray of shape (n_samples, )
            Labels of each point.
        
        inertia_ : float
            Sum of squared distances of samples to their closest cluster center.
            
        n_iter_ : int
            Number of iterations run.   
    """
    
    
    def __init__(self, n_cluters=4, init='k-means++', max_iter=100, dist="euclidean"):
        """
            Initialize self.
        """
        self.__n_cluters = n_cluters
        self.__init = init
        self.__max_iter = max_iter
        self.__dist = dist

    def __initialisation(self, X):
        if self.__init == 'k-means++':
            self.cluster_centers_ = X[np.random.randint(0, X.shape[0], 1), :]
            for i in range(self.__n_cluters-1):
                dist = np.min(cdist(X, self.cluster_centers_, self.__dist), axis=1)
                self.cluster_centers_ = np.concatenate([self.cluster_centers_, 
                                                       X[np.argmax(dist), :].reshape(1,X.shape[1])], 
                                                       axis=0)
        else:
            self.cluster_centers = X[np.random.randint(0, X.shape[0], 3), :]

    def __affecte_cluster(self, X):
        dist = cdist(X, self.cluster_centers_, self.__dist)
        return np.argmin(dist, axis=1), np.min(dist, axis=1)

    def nouveaux_centroides(self, X):
        return np.array([list((X[self.labels_ == i]).mean(axis=0)) for i in np.unique(self.labels_)])

    def __inertie_globale(self, cluster_dist):
        return np.sum([np.power(cluster_dist[self.labels_ == i], 2).sum() for i in np.unique(self.labels_)])


    def fit(self, X):
        """
            Compute k-means clustering.
            
            Parameters
            ----------
            X : array, shape=(n_samples, n_features)
                Training instances to cluster.
            
            Returns 
            -------
            self
                Fitted estimator.
        """
        self.__initialisation(X)
        self.inertia_ = float('inf')
        self.n_iter_ = 0
        for i in range(self.__max_iter):
            self.n_iter_ += 1
            if self.n_iter_ != 1:
                self.cluster_centers_ = self.nouveaux_centroides(X)
            self.labels_, cluster_dist = self.__affecte_cluster(X)
            new_inertia = self.__inertie_globale(cluster_dist)
            if new_inertia == self.inertia_:
                self.inertia_ = new_inertia
                break
                
    def fit_predict(self, X):
        """
            Compute cluster centers and predict cluster index for each sample.
            
            Parameters
            ----------
            X : array, shape=(n_samples, n_features)
            
            Returns
            -------
            labels : array, shape [n_samples,]
                Index of the cluster each sample belongs to.
            
        """
        self.fit(X)
        return self.labels_
    
class WKMeans(KMeans):
    """
        Weighted K-Means clustering.
        In addition to the kmeans algorithm, each object has a weight, 
        so that the center of the clusters will be moved towards the objects with more weight,
        when the weights are equal the behavior is the same as the k-means
        
        Parameters
        ----------
        n_cluters : int, default=4
            The number of clusters to form as well as the number of centroids to generate.
        
        w : ndarray of shape(n_samples, )
            Weight of each object.
            
        init : {'k-means++', 'random'}, default='k-means++'
            Method for initialization, defaults to 'k-means++':
            
            'k-means++' : selects initial cluster centers for k-mean 
            clustering in a smart way to speed up convergence.
            
            'random': choose k observations (rows) at random from data for
            the initial centroids.
 
        max_iter : int, default=100
            Maximum number of iterations of the k-means algorithm for a single run.
                 
        dist : str, default='euclidean'
            The distance metric to use.
        
        Attributes
        ----------
        cluster_centers_ : ndarray of shape (n_clusters, n_features)
            Coordiantes of cluster center.
        
        labels_ : ndarray of shape (n_samples, )
            Labels of each point.
        
        inertia_ : float
            Sum of squared distances of samples to their closest cluster center.
            
        n_iter_ : int
            Number of iterations run.   
    """
   
    def __init__(self, n_cluters=4, w=None, max_iter=100, dist="euclidean"):
        KMeans.__init__(self, n_cluters=n_cluters, max_iter=max_iter, dist=dist)
        self.w = w
    
    def nouveaux_centroides(self, X):
        return np.array([list((X[self.labels_ == i] * self.w[self.labels_ == i].reshape((len(self.w[self.labels_ == i]), 1))).sum(axis=0)/self.w[self.labels_ == i].sum()) for i in np.unique(self.labels_)])

  
