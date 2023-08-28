import cv2 as cv
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import pandas as pd



class PoreSeperator():
    def __init__(self, contour, img, segmentsize=0.1):
        self.image = img
        self.contour = contour
        self.segmentsize = segmentsize
        self.pore = self.__seperate()

    def __seperate(self):
        width = round(self.segmentsize * self.image.shape[1])
        height = round(self.segmentsize * self.image.shape[0])
        blanc = np.zeros_like(self.image)
        filled = cv.fillPoly(blanc, [self.contour], color=(255,255,255))
        
        M = cv.moments(self.contour)
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        
        filled = filled[y-round(height/2):y+round(height/2), x-round(width/2):x+round(width/2)]
        
        return filled
    
    def check_size(self):
        if self.pore.shape[0] == round(self.segmentsize * self.image.shape[0]) and self.pore.shape[1] == round(self.segmentsize * self.image.shape[1]):
            return True
        else:
            return False
        
    def save(self, name):
        cv.imwrite(name, self.pore)
    
    
    
class Preprocessing():
    def __init__(self, pores):
        self.pores = pores
        self.pores_reshpd = [np.float32(pore.reshape((-1,1))).T for pore in pores] # Bilddaten zu Vektor umwandeln
        self.data = [data[0] for data in self.pores_reshpd]
        self.dataframe = self.__create_dataframe()
        
    def __create_dataframe(self):
        data_dict = {}
        for i in range(len(self.data[0])):
            data_dict['x{}'.format(i)] = [data[i] for data in self.data]
        return pd.DataFrame.from_dict(data_dict)
        
        
        
class DimensionReductionPCA():
    def __init__(self, data, pca_model=None, k=3):
        self.data = data
        self.k = k
        self.pca_model, self.explanation = self.__create_pca_model() if pca_model == None else self.__load_pca_model(pca_model)
        self.pca = self.__calc_pca()
        self.dataframe = self.__create_dataframe()
    
    def __create_pca_model(self):
        if self.data.shape[0] < self.data.shape[1]:
            n  = self.data.shape[0]
        else: 
            n = self.data.shape[1]
            
        pca_ = PCA(n_components=n, random_state=2020)
        pca_.fit(self.data)
        explanation = np.cumsum(pca_.explained_variance_ratio_ * 100)
    
        pca_m = PCA(n_components=self.k, random_state=2020)
        pca_m.fit(self.data)
        return pca_m, explanation
        
    def __calc_pca(self):
        return self.pca_model.transform(self.data)
    
    def __load_pca_model(self, name):
        pca_ = pickle.load(open('{}.pickle'.format(name), 'rb'))
        explanation = np.cumsum(pca_.explained_variance_ratio_ * 100)
        return pca_, explanation
    
    def __create_dataframe(self):
        data_dict = {}
        for i in range(len(self.pca[0])):
            data_dict['pca{}'.format(i)] = [elmnt[i] for elmnt in self.pca]
        return pd.DataFrame.from_dict(data_dict)
    
    def save_pca_model(self, name):
        pickle.dump(self.pca_model, open('{}.pickle'.format(name), 'wb'))
        print('PCA Model saved as {}.pickle'.format(name))
        
    def pca_explain(self):
        print('Varianz explained by PCA model with {} Components is {:.3f} %.'.format(self.k, self.explanation[self.k]))
        
    def scale(self):
        # copy the dataframe
        df_norm = self.dataframe.copy()
        # apply min-max scaling
        for column in df_norm.columns:
            df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
        return df_norm
        
        
        
class DBSCANClassifier():
    def __init__(self, data, n_neighbors=2, min_samples = 3, knee=None):
        self.data = data
        self.n_neighbors = n_neighbors
        self.min_samples = min_samples
        self.knee = self.__calc_knee() if knee == None else knee
        self.n_clusters = None
        self.n_outliers = None
        self.labels = self.__dbscan()
        
    def __calc_knee(self):
        # finding best epsilon for DBSCAN: https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf
        nbrs = NearestNeighbors(n_neighbors = self.n_neighbors, metric='euclidean').fit(self.data) # Anzahl an Nachbarn sollte etwa das Doppelte von der Featureanzahl sein
        neigh_dist, neigh_ind = nbrs.kneighbors(self.data)
        sort_neigh_dist = np.sort(neigh_dist, axis=0)

        k_dist = sort_neigh_dist[:, self.n_neighbors-1]
        x = [i for i in range(len(k_dist))]
    
        kneedle = KneeLocator(x=x, y=k_dist, S=1.0, curve='concave', direction='increasing', online=True)
        knee = kneedle.knee_y
        return knee
    
    def __dbscan(self):
        knee = ( self.knee if self.knee != None else 0.1 )
        db = DBSCAN(eps=knee, min_samples=self.min_samples).fit(self.data)
        labels = db.labels_
        
        i = 0
        for element in labels: # Anzahl an Ausreißern zählen
            if element==-1:
                i+=1
                            
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_noise = list(labels).count(-1)
        return labels
    
    
    
class KMeansClassifier():
    def __init__(self, data):
        self.data = data
        
    def find_knee(self, k=5):
        K = range(1, k)
        distortions = []
        
        for k in K:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=None)
            kmeans.fit(self.data)
            distortions.append(kmeans.inertia_)
            
        # kneedle = KneeLocator(x=K, y=distortions, S=1.0, curve='concave', direction='decreasing', online=True)
        
        plt.figure(figsize=(16,8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('Am Ellenbogen liegt die perfekte Anzahl an Clustern')
        plt.show()
        
        # return distortions# kneedle.knee_y
            
    def train(self, k=3):    
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=None)
        kmeans.fit(self.data)
        self.model = kmeans
        
    def predict(self, data):
        self.model.predict(data)
        clusters = self.model.labels_
        
        return clusters
    
    def save_model(self, name):
        pickle.dump(self.model, open(name+'.pickle', 'wb'))
        print('Model saved as {}.pickle!'.format(name))