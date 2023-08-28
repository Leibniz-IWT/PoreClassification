######################################################## F E A T U R E   E X T R A K T I O N ########################################################
######                                                                                                                                         ######
###### A U T H O R   I N F O R M A T I O N S                                                                                                   ######
###### Mika Leon Altmann                                                                                                                       ######
###### 31th of March, 2023                                                                                                                     ######
###### Leibniz-Institute for Materials Science, Bremen, Germany                                                                                ######
######                                                                                                                                         ######
###### D E S C R I B T I O N                                                                                                                   ######
###### Feature exraction of micrographs for powder bed fusion with laser beam of metals.                                                       ######
######                                                                                                                                         ######
#####################################################################################################################################################

import cv2 as cv
import numpy as np
from numpy.linalg import norm
from scipy.stats import skew
import pandas as pd
from heatmap import heatmap, corrplot

from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
from sklearn.preprocessing import QuantileTransformer

import matplotlib.pyplot as plt
import os
 

# Klasse zur Beschreibung der Porenfeatures (lokale Features) wie der Größe, Form und Position 
class Pore():
    def __init__(self, contour, scale=1.79173):
        self.contour = contour
        self.scale = scale
        self.area = self.__area()
        self.perimeter = self.__perimeter()
        self.convex_hull = self.__convex_hull()
        self.convexity_defects = self.__convexity_defects()
        self.defect_density = self.__defect_density()
        self.mean_defect = self.__mean_defect()
        self.solidity = self.__solidity()
        self.bounding_box = self.__bounding_box()
        self.x = self.bounding_box[0]
        self.y = self.bounding_box[1]
        self.label = None
        
    def __area(self):
        area = cv.contourArea(self.contour)
        return area / (self.scale**2)
    
    def __perimeter(self):
        return cv.arcLength(self.contour, True) / self.scale
    
    def __convex_hull(self):
        convex_hull_ret = cv.convexHull(self.contour, returnPoints=False) # returning indices of  the contour points making the convex hull
        convex_hull = cv.convexHull(self.contour) # returning the coordinates of the point making the convex hull
        return convex_hull_ret, convex_hull
        
    def __convexity_defects(self):
        convexity_defects = cv.convexityDefects(self.contour, self.convex_hull[0])
        return convexity_defects
    
    def __defect_density(self):
        try:
            defects = len(self.convexity_defects)
        except:
            defects = 0
            
        defect_density = defects / self.area * 100
        return defect_density
    
    def __mean_defect(self):
        # print('pore')
        # print(self.convexity_defects)
        try:
            defects = self.__defect_size()
        except: 
            defects = [0]
            
        mean_defect = sum(defects) / len(defects)
        return mean_defect / self.scale
    
    def __defect_size(self): # Berechnung der Größe der Konvexitätsfehler --> Bug in der openCV Berechnung --> Faktor 255 größer als wirklich 
        s = [ dfct[0][0] for dfct in self.convexity_defects ]
        e = [ dfct[0][1] for dfct in self.convexity_defects ]
        d = [ dfct[0][2] for dfct in self.convexity_defects ]
        distance = [ norm(np.cross(self.contour[s[idx]][0]-self.contour[e[idx]][0], self.contour[e[idx]][0] - self.contour[d[idx]][0])/norm(self.contour[s[idx]][0]-self.contour[e[idx]][0])) for idx in range(len(d)) ]
        return distance
        
    
    def __solidity(self):
        solidity = self.area / (cv.contourArea(self.convex_hull[1])/self.scale**2)
        return solidity
    
    def __bounding_box(self):
        bounding_box = cv.boundingRect(self.contour)
        return bounding_box
    
    def set_label(self, label):
        self.label = label


# Klasse zur allgemeinen Beschreibung eines Schliffbildes, Zuschnitt, Binarisierung, Skalierung und Poren
class MicrographBase():
    def __init__(self, image, scale=1.79173, cropsize_microns=2000):
        self.img = image
        self.scl = scale
        self.crpsz = cropsize_microns
        self.bnry = self.__binary()
        self.cnt_crp = self.__center_crop()
        self.cnt_hrrchy = self.__contours_hierarchy()
        self.prs = self.__pores()
        self.img_cnt = self.__img_contour()
        
    def __binary(self):
        try:
            image = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
        except: 
            image = self.img
            
        img_blur = cv.GaussianBlur(image, (5, 5), 0) 
        threshold, img_binary = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return img_binary
    
    def __center_crop(self):
        height = self.bnry.shape[0]
        width = self.bnry.shape[1]
        
        w = round( self.crpsz * self.scl )
        h = round( self.crpsz * self.scl )
    
        x = round( width/2 - w/2 )
        y = round( height/2 - h/2 )
    
        binary_img = self.bnry[y:y+h, x:x+w]
        binary_img = cv.copyMakeBorder(binary_img, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, value=255)
        binary_img = cv.copyMakeBorder(binary_img, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, value=0)
        return binary_img
    
    def __contours_hierarchy(self):
        cnt, hrrchy = cv.findContours(self.cnt_crp, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return cnt, hrrchy[0]
    
    def __pores(self):
        # Überprüfung ob die gefundenen Konturen die Probenkontur als Elternkontur haben --> nur dann Pore
        prs_cnts = [ cnt for idx, cnt in enumerate(self.cnt_hrrchy[0]) if self.cnt_hrrchy[1][idx][3] == 0 and idx != 0 ]
        prs = []
        for cnt in prs_cnts:
            prs.append(Pore(cnt, self.scl))
        return prs
    
    def __img_contour(self):
        img_cnt = cv.cvtColor(self.cnt_crp, cv.COLOR_GRAY2RGB)
        for idx, pore in enumerate(self.prs):
            # x, y = tuple(cnt[cnt[:, :, 0].argmin()][0])
            x = pore.x
            y = pore.y
            img_cnt = cv.putText(img_cnt, str(idx), (x, y-round(0.01*self.crpsz)), cv.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
        return img_cnt
    
    def set_center_crop(self, crop_size_microns=2000): # Zuschnittsgröße von außen verändern
        self.crpsz = crop_size_microns
        self.cnt_crp = self.__center_crop()
        
    def save_pore_segments(self, path=os.getcwd()):
        path=path+'\Pores'
        if not os.path.exists(path):
            os.makedirs(path)
        binary_image = self.cnt_crp
        bounding_rect = [pore.bounding_box for pore in self.prs]
        rois = [None]*len(bounding_rect)
    
        # Über alle Boundingboxen iterieren und die ROIs in dem Array speichern
        for i, rect in enumerate(bounding_rect):
            rois[i] = binary_image[int(bounding_rect[i][1]): int(bounding_rect[i][1]+int(bounding_rect[i][3])),
                              int(bounding_rect[i][0]): int(bounding_rect[i][0])+int(bounding_rect[i][2])]
        
        for i, roi in enumerate(rois):
            cv.imwrite(path+'/'+str(i)+'.jpg', roi)
            
        print('Saved segmented pores. \n Path: {}'.format(path))
        

# Klasse zur Beschreibung der statistischen Kenngrößen in einem Schliffbild, bzgl. der Poren
class Micrograph(MicrographBase):
    def __init__(self, image, scale=1.79173, cropsize_microns=2000):
        super().__init__(image, scale, cropsize_microns)
        self.rltv_dnsty = self.__relative_density()
        self.pr_dnsty = self.__pore_density()
        self.sldty  = self.calc_stats([pore.solidity for pore in self.prs])
        self.area = self.calc_stats([pore.area for pore in self.prs])
        self.prmtr = self.calc_stats([pore.perimeter for pore in self.prs])
        self.dfct_dnsty = self.calc_stats([pore.defect_density for pore in self.prs])
        self.mn_dfct = self.calc_stats([pore.mean_defect for pore in self.prs])
        
    def __relative_density(self):
        area_mcrgrph = self.crpsz**2
        area_pr = sum( [ pr.area for pr in self.prs ] )
        rltv_dnsty = 100 - area_pr / area_mcrgrph * 100
        return rltv_dnsty
    
    def __pore_density(self):
        nbr_prs = len(self.prs)
        pr_dnsty = nbr_prs / (self.crpsz**2) * 100
        return pr_dnsty
    
    def calc_stats(self, lst):  
        stts = {'Count': len(lst),
                'Unique': len(np.unique(lst)),
                'Max': max(lst),
                'Min': min(lst),
                'Mean': sum(lst) / len(lst),
                'Median': np.median(lst),
                'Std_Dev': np.std(lst),
                'Varianz': sum((elmnt-(sum(lst)/len(lst)))**2 for elmnt in lst) / len(lst),
                'Skewness': skew(lst)}
        return stts
    
    def stats(self):
        lst = [self.sldty, self.area, self.prmtr, self.dfct_dnsty, self.mn_dfct]
        stats = pd.DataFrame.from_records(lst, index=['Solidity', 'Area', 'Perimeter', 'Defect Density', 'Mean Defect']).round(decimals=3)
        return stats
        


# Klasse zur Beschreibung der Schliffbilder mit Identifizierung der typischen und atypischen Poren
class MicrographDBSCAN(Micrograph):
    def __init__(self, image, scale=1.79173, cropsize_microns=2000, n_neighbors=2, min_samples = 3):
        super().__init__(image, scale, cropsize_microns)
        self.n_neighbors = ( n_neighbors if n_neighbors <= len(self.prs) else len(self.prs))
        self.min_samples = min_samples
        self.pore_features = self.__pore_features()
        self.pore_ftrs_nrmlzd = self.__normalization()
        self.knee = self.__calc_knee()
        self.pca = self.__pca()
        self.pores_dbscan = self.__dbscan()
        
    def __pore_features(self):
        ftrs = {'Area': [ pore.area for pore in self.prs ],
                'Solidity': [ pore.solidity for pore in self.prs ],
                'Perimeter': [ pore.perimeter for pore in self.prs ],
                'Defect_Density': [ pore.defect_density for pore in self.prs ],
                'Mean_Defect': [ pore.mean_defect for pore in self.prs ]}
        
        pores_ftrs = pd.DataFrame.from_dict(ftrs)
        return pores_ftrs
        
    def __normalization(self):
        pores_ftrs = self.pore_features.copy()
        cols = pores_ftrs.columns.tolist()
        ftrs = pores_ftrs[cols]
        
        sclr = QuantileTransformer(n_quantiles = pores_ftrs.shape[0])
        pores_ftrs[cols] = sclr.fit_transform(ftrs.values)
        return pores_ftrs
    
    def __pca(self):
        if len(self.prs) < 5:
            raise CustomException('Less than 5 pores found, unable to calculate stats.')
        pca_ = PCA(n_components=len(self.pore_ftrs_nrmlzd.columns.tolist()), random_state=2020)
        pca_.fit(self.pore_ftrs_nrmlzd)
        explanation = np.cumsum(pca_.explained_variance_ratio_ * 100)
    
        pca3 = PCA(n_components=3, random_state=2020)
        pca3.fit(self.pore_ftrs_nrmlzd)
        pores_pca3 = pca3.transform(self.pore_ftrs_nrmlzd)
        return explanation, pores_pca3
    
    def __calc_knee(self):
        # finding best epsilon for DBSCAN: https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf
        nbrs = NearestNeighbors(n_neighbors = self.n_neighbors, metric='euclidean').fit(self.pore_ftrs_nrmlzd) # Anzahl an Nachbarn sollte etwa das Doppelte von der Featureanzahl sein
        neigh_dist, neigh_ind = nbrs.kneighbors(self.pore_ftrs_nrmlzd)
        sort_neigh_dist = np.sort(neigh_dist, axis=0)

        k_dist = sort_neigh_dist[:, self.n_neighbors-1]
        x = [i for i in range(len(k_dist))]
    
        kneedle = KneeLocator(x=x, y=k_dist, S=1.0, curve='concave', direction='increasing', online=True)
        knee = kneedle.knee_y
        return knee
        
    def __dbscan(self):
        knee = ( self.knee if self.knee != None else 0.1 )
        db = DBSCAN(eps=knee, min_samples=self.min_samples).fit(self.pore_ftrs_nrmlzd)
        labels = db.labels_
        
        i = 0
        for element in labels: # Anzahl an Ausreißern zählen
            if element==-1:
                i+=1
                
        if i > 0.1*len(labels): # Wenn Ausreißer Anzahl mehr als 10 % der Poren sind, alle Ausreißer zu Klasse 0 zuweisen
            for i in range(len(labels)):
                labels[i] = 0            
        
        for i, pore in enumerate(self.prs): # Label zu jeder Pore speichern
            pore.set_label(labels[i])
    
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        return n_clusters, n_noise
    
    def explain_pca(self):
        print('The varianz of the pores captured by the principle components: \n 1 principle component: {:.3f} % \n 2 principle components: {:.3f} % \n 3 principle components: {:.3f} % \n 4 principle components: {:.3f} % \n 5 principle components: {:.3f} %'.format(self.pca[0][0], self.pca[0][1],self.pca[0][2], self.pca[0][3], self.pca[0][4]))
        
    def visualize_clusters(self):
        labels = [pore.label for pore in self.prs]
        
        fig = plt.figure(figsize=(12,8))
        ax = plt.axes(projection="3d")
    
        sctt = ax.scatter3D(self.pca[1][:,0], self.pca[1][:,1], self.pca[1][:,2], c=labels, s=25, alpha=0.6, cmap='viridis')

        plt.title('3D Scatterplot: {:.2f} % of the variability captured'.format(self.pca[0][2]))
        ax.set_xlabel('PC 1', labelpad=15, weight='bold')
        ax.set_ylabel('PC 2', labelpad=10, weight='bold')
        ax.set_zlabel('PC 3', labelpad=10, weight='bold')
        ax.view_init(25, 10)
        return fig
    
    def hist_pore_features(self, normalized=False, include_outliers=True):
        if normalized == False:
            data = self.__pore_features()
            data['Label'] = [pore.label for pore in self.prs]
            if include_outliers == False:
                data = data[data['Label'] >= 0]
            
        else: 
            data = self.__normalization()
            data['Label'] = [pore.label for pore in self.prs]
            if include_outliers == False:
                data = data[data['Label'] >= 0]
                
        data.hist(bins=30, figsize=(8,8))
        
    def get_corr_plot(self, include_outliers=True):
        data = self.__pore_features()
        data['Label'] = [pore.label for pore in self.prs]
        if include_outliers==True:
            data = data
        else: 
            data = data[data['Label'] >= 0]
            
        fig = plt.figure(figsize=(16,8))
        corr = data.corr(numeric_only=True)
        corrplot(corr, size_scale=100, marker="s")
        
    def get_stats(self, include_outliers=True):
        if include_outliers==True:
            lst = [self.sldty, self.area, self.prmtr, self.dfct_dnsty, self.mn_dfct]
        else: 
            solidity = self.calc_stats([pore.solidity for i, pore in enumerate(self.prs) if pore.label >= 0])
            area = self.calc_stats([pore.area for i, pore in enumerate(self.prs) if pore.label >= 0])
            perimeter = self.calc_stats([pore.perimeter for i, pore in enumerate(self.prs) if pore.label >= 0])
            defect_density = self.calc_stats([pore.defect_density for i, pore in enumerate(self.prs) if pore.label >= 0])
            mean_defect = self.calc_stats([pore.mean_defect for i, pore in enumerate(self.prs) if pore.label >= 0])
            
            lst = [solidity, area, perimeter, defect_density, mean_defect]
            
        stats = pd.DataFrame.from_records(lst, index=['Solidity', 'Area', 'Perimeter', 'Defect Density', 'Mean Defect']).round(decimals=3)
        return stats
        

class MicrographFullDescription(MicrographDBSCAN): #Ausreißer ein- oder ausschließen
    def __init__(self, image, laserpower, scanspeed, hatchdistance, layerthickness, scale=1.79173, cropsize_microns=2000, n_neighbors=2, min_samples = 3):
        super().__init__(image, scale, cropsize_microns, n_neighbors, min_samples)
        self.laserpower = laserpower
        self.scanspeed = scanspeed
        self.hatchdistance = hatchdistance
        self.layerthickness = layerthickness
        self.global_ftrs = self.__global_ftrs()
        self.global_stts = self.__global_stts()
        self.local_ftrs = self.__local_frts()
        
    def __global_ftrs(self):
        lst = {'LaserPower': self.laserpower,
               'ScanSpeed': self.scanspeed,
               'HatchDistance': self.hatchdistance,
               'LayerThickness': self.layerthickness,
               'RelativeDensity': self.rltv_dnsty,
               'PoreDensity': self.pr_dnsty,
               'ED': self.laserpower/(self.scanspeed*self.hatchdistance*self.layerthickness),
               'TE': self.laserpower/self.scanspeed,
               'CountOutliers': self.pores_dbscan[1],
               'CountClusters': self.pores_dbscan[0],
               'EpsilonDBSCAN': self.knee}
        global_ftrs = pd.DataFrame(data=lst, index=[0])
        return global_ftrs

    def __local_frts(self):
        local_ftrs = self.pore_features
        local_ftrs['Label'] = [pore.label for pore in self.prs]
        return local_ftrs
    
    def __global_stts(self):
        keys = self.area.keys()
        area = {}
        solidity = {}
        perimeter = {}
        defect_density = {}
        mean_defect = {}
        
        for key in keys:
            area['Area'+key] = [self.area[key]]
            solidity['Solidity'+key] = [self.sldty[key]]
            perimeter['Perimeter'+key] = [self.prmtr[key]]
            defect_density['DefectDensity'+key] = [self.dfct_dnsty[key]]
            mean_defect['MeanDefect'+key] = [self.mn_dfct[key]]
            
        global_stts = pd.DataFrame.from_dict(area)
        lst = [solidity, perimeter, defect_density, mean_defect]
        for elmnt in lst:
            global_stts = pd.concat([global_stts, pd.DataFrame.from_dict(elmnt)], axis=1)
        
        return global_stts
    
    def get_full_description(self, include_stats=True, include_pore_features=True):
        if include_stats == True:
            if include_pore_features == True:
                data = pd.concat([self.local_ftrs, self.global_ftrs, self.global_stts], axis=1)
            else:
                data = pd.concat([self.global_ftrs, self.global_stts], axis=1)
        else: 
            if include_pore_features == True:
                data = pd.concat([self.local_ftrs, self.global_ftrs], axis=1)
            else:
                data = pd.concat([self.global_ftrs], axis=1)
            
        nans = data.isna().any().tolist()
        cols = data.columns.tolist()

        for i, col in enumerate(cols):
            if nans[i] == True:
                data[col] = data[col][0]

        return data