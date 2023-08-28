######################################################## F E A T U R E   E X T R A K T I O N ########################################################
######                                                                                                                                         ######
###### A U T H O R   I N F O R M A T I O N S                                                                                                   ######
###### Mika Altmann                                                                                                                            ######
###### 20th of February, 2023                                                                                                                  ######
###### Leibniz-Institute for Materials Science, Bremen, Germany                                                                                ######
######                                                                                                                                         ######
###### D E S C R I B T I O N                                                                                                                   ######
###### These functions containing methods to extract multiple features from metallurgical micrographs, especially for porosity evaluations     ######
###### in PBF-LB/M processes.                                                                                                                  ######
######                                                                                                                                         ######
#####################################################################################################################################################

import cv2 as cv
import numpy as np
from numpy import median
import matplotlib.pyplot as plt
from pywt import dwt2
import pywt
import pandas as pd
import math
from scipy.stats import skew
import imutils

#####################################################################################################################################################

###### Alle Konturen in dem Schliffbild bestimmen ######
def get_Contours(image):
    ### Bild in Graustufen umwandeln, wenn es ein RGB Bild ist ###
    try:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    except: 
        image=image
        
    ### Weichzeichnen mit Gaussfilter und binarisieren ###
    # bei 5x5 Gausszeichner gehen viele kleine Poren/Defekte verloren
    # bei 3x3 Gausszeichner werden mehr kleine Poren/Defekte erkannt
    img_blur = cv.GaussianBlur(image, (5, 5), 0) 
    threshold, img_binary = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
    ### Bild zuschneiden ###
    # 1/3 des Bildes wird oben und 1/3 unten weg geschnitte
    # es bleibt nur noch ein Streifen in der Mitte (idealerweise stabilder Prozesszustand)
    # Entfernung der Skalen
    height = img_binary.shape[0]
    width = img_binary.shape[1]
    
    #upper_crop = round(height * 1/3)
    #lower_crop = round(height * 2/3)

    #Center Crop
    # gleich großen Untersuchungsbereich aus den Bildern ausschneiden
    w = round( 2000 * 1.79173 )
    h = round( 2000 * 1.79173 )
    
    x = round( width/2 - w/2 )
    y = round( height/2 - h/2 )
    
    img_binary = img_binary[y:y+h, x:x+w]
    img_binary = cv.copyMakeBorder(img_binary, 5, 5, 5, 5, cv.BORDER_CONSTANT, None, value=0)
    
    ###### Konturen und deren Hierarchien bestimmen und speichern ######
    # contours, hierarchy = cv.findContours(img_binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
#     ### Winkel der Probenkontur bestimmen ###
#     contour_area = [cv.contourArea(contour) for contour in contours]
#     specimen_index = [i for i in range(len(contours)) if contour_area[i] == max(contour_area)][0]
#     specimen_bounding = cv.minAreaRect(contours[specimen_index])
#     specimen_angle = specimen_bounding[2]
    
#     ### originales Bild rotieren ###
#     img_rot = imutils.rotate(image, angle= -1*specimen_angle)
        
    ### hierarchy ###
    # hierarchy[i][0]: the index of the next contour of the same level
    # hierarchy[i][1]: the index of the previous contour of the same level
    # hierarchy[i][2]: the index of the first child
    # hierarchy[i][3]: the index of the parent
    
    # print('Anzahl an Konturen: {}'.format(len(contours)))

    return contours, hierarchy[0], img_binary, threshold

#####################################################################################################################################################

###### Bounding Boxen und Polygone um alle Poren/Defekte legen ######
def get_BoundingBox(contours):
    # leere Arrays erzeugen in denen alle Boundingboxen und Polygone gepseichert werden
    bounding_poly = [None]*len(contours)
    bounding_rect = [None]*len(contours)

    # für jede Kontur eine Boundingbox und ein umschließendes Polygon erzeugen
    for i, contour in enumerate(contours):
        bounding_poly[i] = cv.approxPolyDP(contour, 1, True)
        bounding_rect[i] = cv.boundingRect(bounding_poly[i])
        
    # Koordinaten der Bounding Boxen zurück geben
    return bounding_rect

#####################################################################################################################################################

###### die Poren mithilfe von den Boundingboxen segmentieren ######
def segment_Contours(bounding_rect, binary_image):
    # leeren Array für alle Region of Interests erzeugen
    roi = [None]*len(bounding_rect)
    
    # Über alle Boundingboxen iterieren und die ROIs in dem Array speichern
    for i, rect in enumerate(bounding_rect):
        roi[i] = binary_image[int(bounding_rect[i][1]): int(bounding_rect[i][1]+int(bounding_rect[i][3])),
                              int(bounding_rect[i][0]): int(bounding_rect[i][0])+int(bounding_rect[i][2])]
     
    # alle einzeln segmentierten Konturen zurückgeben
    return roi

#####################################################################################################################################################

###### Nach Poren und Partikeln sortieren ######
# Die Konuturen sind sowohl Partikel als auch Poren
# Partikel können dabei auf der Probenaußenseite als auch innerhalb von Poren vorliegen
# eine Unterscheidung zwischen Partikeln und Poren ist zwingend notwendig für die Ableitung statistischer Größen

def sort_Contours(contours, hierarchy, binary_image):
    
    ### Non Parents finden ###
    # haben keine Eltern-Kontur
    # sind damit die Probenkontur oder Anhaftungen außerhalb der Probenkontur
    # Partikel innerhalb von Poren haben allerdings ebenfalls keine Eltern-Kontur
    # es müssen innere von äußeren Partikeln getrennt werden
    
    non_parents_shape = []
    non_parents_index = []
    
    for i, contour in enumerate(contours):
        if hierarchy[i][3] <= 0:
            non_parents_shape.append(contour.shape[0]*contour.shape[1]) # Fläche aller non_parents berechnen (Größe der Boundingbox)
            non_parents_index.append(i) # Liste der "realen" Indizes in non_parents_index speichern
            
    ### Probenkontur finden ###
    # erheblich größer als alle anderen Konturen
    # Index in der Liste alle Konut
    specimen_contour_shape = max(non_parents_shape)
    
    for i, shape in enumerate(non_parents_shape):
        if shape == specimen_contour_shape:
            specimen_contour_index = non_parents_index[i]
            non_parents_shape.pop(i) # Probenkontur aus den non_parents_shape löschen
            non_parents_index.pop(i) # Probenkontur aus den non_parents_index löschen
            
    ### Kontur und Hierarchy der Probenkontur abspeichern ### 
    # specimen_contour[0] enthält die Kontur
    # specimen_contour[1] enthält die hierarchy
    # specimen_contour[2] enthält den realen Index 
    specimen_contour = [contours[specimen_contour_index], hierarchy[specimen_contour_index], specimen_contour_index]
    
#     ### inner Poren entfernen und Differenz bestimmen ####
#     img = np.zeros_like(binary_image)
#     particle_contours = [contours[index] for index in non_parents_index]
#     img = cv.drawContours(img, particle_contours, -1, (255,255,255), -1) # alle Partikel (non_parents) in weiß einzeichnen
#     img = cv.drawContours(img, specimen_contour[0], 0, (2555,255,255), -1)
#     ## Konturen der äußeren Partikel finden ##
#     outer_particles, _ = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
#     outer_particles_shape = [contour.shape[0]*contour.shape[1] for contour in outer_particles]
#     ## Indizes der äußeren Partikel bestimmen ##
#     outer_particles_index = [non_parents_index[i] for i, shape in enumerate(non_parents_shape) if shape in outer_particles_shape]
#     inner_particles_index = [non_parents_index[i] for i, shape in enumerate(non_parents_shape) if not shape in outer_particles_shape]
    
#     print('Anzahl an Partikeln: {}'.format(len(non_parents_shape)))
#     print('Anzahl äußerer Partikel: {}'.format(len(outer_particles)))
#     print('Anzahl innerer Partikel:{}'.format(len(inner_particles_index)))
#     cv.imwrite('Images/outer_Particles.jpg', img)
    
    ### Poren finden ###
    # müssen alle innerhalb der Probenkontur liegen
    # müssen als Eltern-Kontur die Probenkontur haben
    inner_pores_index = []
    
    for i, contour in enumerate(hierarchy):
        if contour[3] == specimen_contour_index: # Filtern ob die Pore als Elternkontur die Probenkontur hat
            inner_pores_index.append(i)
           
    ### Ausgeben von Informationen über die Ergebnisse ###
    # print('Probenkontur-Index: {}'.format(specimen_contour_index))
    # print('Anzahl an Partikelanhaftungen: {}'.format(len(non_parents_index)))
    # print('Anzahl an innerer Poren: {}'.format(len(inner_pores_index)))
    
    ### Probenkontur und Anhaftungen zurückgeben ###
    # Rückgabe der realen Indizes der anhaftenden Partikel
    # Größen etc. der Anhaftungen können dann über den Index aus der Konturliste ausgelesen werden.
    outer_particles_index = non_parents_index # Anhaftungen sind alle Konturen ohne Eltern und ohne die Probenkontur
    return specimen_contour, outer_particles_index, inner_pores_index

#####################################################################################################################################################

###### Porenfeatures bestimmen und Porenzuschnitte speichern ######
# Schliffbild maskieren, sodass alle äußeren Partikel weg fallen
def get_Pore_Features(binary_img, specimen_contour, circularity_threshold, save_pores, save_path, sizeFilter):
    ### idealisiertes Schliffbild erzeugen ###
    # leeres Bild [None] --> Partikel in Schwarz [0] --> Poren in weiß [255] ==> äußere Partikel verschwinden
    # Poren in äußeren Partikeln könnten im idealisierten Bild auftreten
    # mask = specimen_contour # Probenkontur als Maske --> äußere Partikel fallen weg
    # blanc = np.zeros_like(binary_img) # leeres schwarzes Bild erzeugen
    # color = [255,255,255] # Farbe (Weiß) für zu maskierenden Bereich
    # cv.fillPoly(blanc, [mask], color) # zu maskierenden Bereich einfärben --> alles innerhalb der Maske bzw. weiße bleibt bestehen
    # result = cv.bitwise_and(binary_img, blanc) # Binärbild maskieren, nur Probe mit inneren Poren bleibt bestehen
    
    ### Konturen suchen ###
    # kein bluring oder binarisieren notwendig, wurde bei binary_image bereits angewendet
    # contours, hierarchy = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(binary_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    ### Poren und innere Partikel separieren ###
    # Partikel haben nicht die Probe als Elternkontur sondern die Poren
    # Probenkontur finden
    contour_area = [cv.contourArea(contour) for contour in contours]
    specimen_index = [i for i in range(len(contours)) if contour_area[i] == max(contour_area)][0]
    pores_index = [i for i, hierarch in enumerate(hierarchy[0]) if hierarch[3] == specimen_index]
    # Konturflächenfilter
    if sizeFilter != None:
        for index in pores_index:
            if contour_area[index] <= 5:
                pores_index.remove(index)
        # (pores_index.remove(index) for index in pores_index if contour_area[index] < sizeFilter)
        
    ### Konturen der Poren als Liste speichern ###
    pores = [contours[index] for index in pores_index]
    
    ## Anzahl und Größe der inneren Partikel für jede Pore bestimmen
    # Partikel mit Poren finden
    particle_pores = [index for index in pores_index if hierarchy[0, index, 2] >= 0]
    # Partikel aus allen Konturen selektieren
    particles = [index for index in range(len(contours)) if hierarchy[0, index, 3] != specimen_index and hierarchy[0, index, 3] > 0]
    # Die Elternkontur für jedes Partikel bestimmen
    particle_parents = [hierarchy[0, particle, 3] for particle in particles]
    # Anzahl an Partikeln in jeder Partikelpore bestimmen
    particles_in_pores = [particle_parents.count(index) for index in particle_pores]
    
    ## Für jede Pore die Anzahl an Partikeln speichern
    pores_particles = [None] * len(pores_index)
    for i, index in enumerate(pores_index):
        if index in particle_pores:
            j = particle_pores.index(index)
            pores_particles[i] = particles_in_pores[j]
        else:  
            pores_particles[i] = 0
    
    ## für jedes Partikel die Größe bestimmen
    particle_size = [cv.contourArea(contours[particle]) for particle in particles]
    ## Für jede Pore die Größen der enthaltenen Partikel als Subliste speichern
    particle_sizes_pores = [None] * len(pores_index)
    for i, index in enumerate(pores_index):
        if index in particle_pores:
            sizes = [particle_size[i] for i, parent in enumerate(particle_parents) if parent == index]
            particle_sizes_pores[i] = sizes
        else: 
            particle_sizes_pores[i] = [0]
    ## Statistik zu den Partikelgrößen 
    max_particle_size_pore = [max(element) if element[0] != 0 else 0 for element in particle_sizes_pores]
    min_particle_size_pore = [min(element) if element[0] != 0 else 0 for element in particle_sizes_pores]
    mean_particle_size_pore = [sum(element)/len(element) if element[0] != 0 else 0 for element in particle_sizes_pores]
    median_particle_size_pore = [np.median(element) if element[0] != 0 else 0 for element in particle_sizes_pores]
    std_particle_size_pore = [np.std(element) if element[0] != 0 else 0 for element in particle_sizes_pores]
    
    ## Porengrößen
    pore_areas = [contour_area[index] for index in pores_index]
    ## Konturlängen
    pore_perimeters = [cv.arcLength(contours[index], True) for index in pores_index]
    ## Approximierte Kontur
    
    ## Konvexitätsfehler
    pore_convex_hulls_ret = [cv.convexHull(contours[index], returnPoints=False) for index in pores_index]
    pore_convex_hulls = [cv.convexHull(contours[index]) for index in pores_index]
    convexity_defects = [cv.convexityDefects(contours[index], pore_convex_hulls_ret[i]) for i, index in enumerate(pores_index)]   
    pore_convexity_defects = []
    for convexity_defect in convexity_defects:
        try:
            pore_convexity_defects.append(len(convexity_defect))
        except:
            pore_convexity_defects.append(0)
            
    convexity_defect_sizes = []
    for convexity_defect in convexity_defects:
        try:
            convexity_defect_sizes.append([defect[0][3] for defect in convexity_defect])
        except:
            convexity_defect_sizes.append(0)
            
    max_convexity_defect = []
    min_convexity_defect = []
    mean_convexity_defect = []
    median_convexity_defect = []
    std_convexity_defect = []
    for convexity_defect in convexity_defect_sizes:
        try: 
            max_convexity_defect.append(max(convexity_defect))
            min_convexity_defect.append(min(convexity_defect))
            mean_convexity_defect.append(sum(convexity_defect)/len(convexity_defect))
            median_convexity_defect.append(np.median(convexity_defect))
            std_convexity_defect.append(np.std(convexity_defect))
        except:
            max_convexity_defect.append(0)
            min_convexity_defect.append(0)
            mean_convexity_defect.append(0)
            median_convexity_defect.append(0)
            std_convexity_defect.append(0)
            
    
    ## Zirkularität
    pore_circularities = [4*math.pi*area/(pore_perimeters[i])**2 for i, area in enumerate(pore_areas)]
    
    ## Solidität
    pore_solidities = [pore_areas[i]/cv.contourArea(convex_hull) for i, convex_hull in enumerate(pore_convex_hulls)]
    
    ## Bounding Box
    pore_bounding = [cv.boundingRect(contours[index]) for index in pores_index]
    width_rect = [bounding[2] for bounding in pore_bounding]
    height_rect = [bounding[3] for bounding in pore_bounding]
    density_bounding_rect = [contour_area[index]/(width_rect[i]*height_rect[i]) for i, index in enumerate(pores_index)]
    
    ## Bounding Box mit minimaler Fläche
    pore_min_area_bounding = [cv.minAreaRect(contours[index]) for index in pores_index]
    width_min_rect = [bounding[1][0] for bounding in pore_min_area_bounding]
    height_min_rect = [bounding[1][1] for bounding in pore_min_area_bounding]
    density_min_rect = [contour_area[index]/(width_min_rect[i]*height_min_rect[i]) for i, index in enumerate(pores_index)]
    
    ## Porenrotation    
    pore_rotations = []
    for i, element in enumerate(pore_min_area_bounding):
        # element.width < element.heigth
        if element[1][0] < element[1][1]: 
            pore_rotations.append(abs(pore_min_area_bounding[i][2]) + 90)
        else:
            pore_rotations.append(abs(pore_min_area_bounding[i][2]))

    # Winkel bei Verdrehung der Probe korrigieren
    pore_rotations_corr = [angle - cv.minAreaRect(contours[specimen_index])[2] for angle in pore_rotations]
    
    ## Minimaler einschließender Kreis
    min_circles = [cv.minEnclosingCircle(contours[index]) for index in pores_index]
    radian_min_circle = [int(min_circle[1]) for min_circle in min_circles]
    density_min_circle = [contour_area[index]/(math.pi*radian_min_circle[i]**2) for i, index in enumerate(pores_index)]
    
    particle_density = np.divide(pores_particles, [get_ContourArea_Microns(area) for area in pore_areas])
    defect_density = np.divide(pore_convexity_defects, [get_ContourArea_Microns(area) for area in pore_areas])
    
    ## Mittelpunktkoordinaten auslesen (von min_area_rect) ###
    # print(pore_bounding[0])
    x_coordinates = [bounding[0] for bounding in pore_bounding]
    y_coordinates = [bounding[1] for bounding in pore_bounding]
    
    features = {'Pore_Index': pores_index,
                'x_Coordinate': x_coordinates,
                'y_Coordinate': y_coordinates,
                'No_Particles': pores_particles,
                'Particle_Density': particle_density,
                'Max_Particle': [get_ContourArea_Microns(element) for element in max_particle_size_pore] ,
                'Min_Particle': [get_ContourArea_Microns(element) for element in min_particle_size_pore],
                'Mean_Particle': [get_ContourArea_Microns(element) for element in mean_particle_size_pore],
                'Median_Particle': [get_ContourArea_Microns(element) for element in median_particle_size_pore],
                'STD_Particle': [get_ContourArea_Microns(element) for element in std_particle_size_pore],
                'Area': [get_ContourArea_Microns(element) for element in pore_areas],
                'Area_PX': pore_areas,
                'Perimeter': [get_Length_Mircons(element) for element in pore_perimeters],
                'Circularity': pore_circularities,
                'Solidity': pore_solidities,
                'Angle': pore_rotations,
                'Angle_corr': pore_rotations_corr,
                # 'No_Convexity_Defects': pore_convexity_defects,
                'Defect_Density': defect_density,
                'Max_Convexity_Defect': [get_Length_Mircons(element) for element in max_convexity_defect],
                'Min_Convexity_Defect': [get_Length_Mircons(element) for element in min_convexity_defect],
                'Mean_Convexity_Defect': [get_Length_Mircons(element) for element in mean_convexity_defect],
                'Median_Convexity_Defect': [get_Length_Mircons(element) for element in median_convexity_defect],
                'STD_Convexity_Defect': [get_Length_Mircons(element) for element in std_convexity_defect],
                'Width_Rect': [get_Length_Mircons(element) for element in width_rect],
                'Height_Rect': [get_Length_Mircons(element) for element in height_rect],
                'Width_Min_Rect': [get_Length_Mircons(element) for element in width_min_rect],
                'Height_Min_Rect': [get_Length_Mircons(element) for element in height_min_rect],
                'Density_Rect': density_bounding_rect,
                'Density_min_Rect': density_min_rect,
                'Radius_Circle': [get_Length_Mircons(element) for element in radian_min_circle],
                'Density_min_Circle': density_min_circle,
                }
    
    ## Zuschnitt jeder Pore in Ordner speichern
    if save_pores == True:
        rois = segment_Contours(bounding_rect=pore_bounding, binary_image=binary_img)
        for i, roi in enumerate(rois):
            cv.imwrite(save_path+'/'+str(pores_index[i])+'.jpg', roi)
            
    
    return features, binary_img, pores 

#####################################################################################################################################################

###### Kontur in leeres Bild zeichnen ######
# Zum Überprüfen, wie die entsprechende Kontur aussieht
def plot_Contour(contours, image, contour_index, bounding_rect, save, crop, name):
    ### leeres Bild erzeugen mit der Größe des Eingangsbildes ###
    empty_segment = np.zeros_like(image)
    
    # gewünschte Konturen in einer Konturenliste speichern
    # anhand der contour_index Angabe
    cnt = []
    [cnt.append(contours[contour]) for i, contour in enumerate(contour_index)]
    
    # alle Konturen in der erstellten Liste plotten
    segment = cv.drawContours(empty_segment, cnt, -1, (255,255,255), -1)
    
    ### Bild auf relevanten Bereich zuschneiden ###
    # nur wenn crop=True, sonst wird Kotur in das Ursprungsbild geplottet
    if crop == True:
        segment = segment[int(bounding_rect[contour_index][1]): int(bounding_rect[contour_index][1]+int(bounding_rect[contour_index][3])), \
                          int(bounding_rect[contour_index][0]): int(bounding_rect[contour_index][0])+int(bounding_rect[contour_index][2])]
    
    if save == True:
        cv.imwrite('Images/segmented_contour_{}.jpg'.format(name), segment)
    
    fig = plt.figure(figsize=(8,8))
    plt.imshow(segment, cmap='gray')
    plt.axis('off')
    plt.title('Segmentierte Kontur {}'.format(name))
    plt.show()
    
#####################################################################################################################################################
    
####### Konturfläche als Mikrometer ausgeben lassen ######
def get_ContourArea_Microns(area):
    one_micron = 1.79173 # Umrechnungsfaktor PX in Mikrometer --> mithilfe der BA Bildanalyse bestimmt
    one_sq_micron = one_micron**2 # Umrechnung in Quadratmikrometer
    
    area_contour_microns = area/one_sq_micron # Umrechnun der Flöche in Quadratmikrometer
    
    return area_contour_microns

#####################################################################################################################################################

###### Längenmaße in Mikrometer umrechnen ######

def get_Length_Mircons(data):
    one_micron = 1.79173
    data_microns = one_micron * data
    
    return data_microns

#####################################################################################################################################################

###### relative Dichte berechnen ######
# Berechnung der Dichte aus der Fläche der Probenkontur und der der inneren Poren
# Partikel innerhalb von Poren haben so keine Auswirkung auf die Dichte
# zweiten Dichtewert mit inneren Partikeln bestimmen (ggf. sinnvoll wenn Proben gehipt werden)
def get_relative_Density(specimen_contour, inner_pores):
    
    ### für die gesamte Probenfläche ###
    specimen_area = round( 2000 * 1.79173 )**2
    pores_area = 0
    
    for element in inner_pores:
        pores_area += cv.contourArea(element)
        
    rel_density = (specimen_area - pores_area) / specimen_area * 100
    
    return rel_density

#####################################################################################################################################################

###### Probe in Kern und Randbereich zerlegen ######
# Ziel ist zwei Binärbilder mit der Ursprungsgröße zu erzeugen
# Dabei ist jeweils einmal die Kontur und einmal der Kern weiß mit schwarzen Poren
# die  Binärbilder können anschließend wie das Ursprungsbild durch die Funktionen untersucht werden

### I N P U T S ###
# contours: alle Konturen die in dem Originalbild gefunden wurden
# contours_index: Indizes aller Poren
# specimen_contour: die spezifische Contour der Probe
# binary_image: das Eingabebild als Binärbild, um die gleiche Bildgröße zu kriegen
# relative_core_size: Die relative Fläche des Kernbereichs zur Boundingbox der Probenflöche
# relative_y_offset: relative Erweiterung des Kernbereichs in y-Richtung
def get_Core_Border(contours, contours_index, specimen_contour, binary_image, relative_core_size, relative_y_offset):
    ### Flächenschwerpunkt der Probe finden ###
    # Ausgangspunkt für Centercropping
    ## Momente der  Probenkontur bestimmen ##
    moments = cv.moments(specimen_contour)
    ## Koordinaten des Zentrums bestimmen ##
    x_center = int(moments['m10']/moments['m00'])
    y_center = int(moments['m01']/moments['m00'])
    
    ### Porenkonturen ###
    contours_des = [contours[index] for index in contours_index]
    
    ## Kernbereich segmentieren ##
    core_img = get_Core(specimen_contour, relative_core_size, relative_y_offset, binary_image, x_center, y_center, contours_des)
    
    ## Konturbereich segmentieren ##
    border_img = get_Border(specimen_contour, relative_core_size, relative_y_offset, binary_image, x_center, y_center, contours_des)
    
    return core_img, border_img


def get_Core(specimen_contour, relative_core_size, relative_y_offset, binary_image, x_center, y_center, contours_des):
    ### Kernbereich segmentieren ###
    # Bereich wird relativ zur Boundingbox als rechteckiger Centercrop gewählt
    ## Shape von der Probenfläche bestimmen
    specimen_boundingRect = cv.boundingRect(specimen_contour) # = (x, y, w, h)
    specimen_height = specimen_boundingRect[3]
    specimen_width = specimen_boundingRect[2]
    ## Zuschnittsgröße festlegen ##
    core_width = relative_core_size * specimen_width
    core_height = relative_core_size * specimen_height
    core_height = core_height+relative_y_offset*core_height
    ## Bild zusammenbauen ##
    plane_img = np.zeros_like(binary_image)
    core_img = plane_img
    core_img = cv.rectangle(core_img, (int(x_center-core_width/2), int(y_center-core_height/2)), 
                            (int(x_center+core_width/2), int(y_center+core_height/2)), (255,255,255), -1)
    core_img = cv.drawContours(core_img, contours_des, -1, (0,0,0), -1)
    core_img = core_img
    
    return core_img

def get_Border(specimen_contour, relative_core_size, relative_y_offset, binary_image, x_center, y_center, contours_des):
    ## Shape von der Probenfläche bestimmen
    specimen_boundingRect = cv.boundingRect(specimen_contour) # = (x, y, w, h)
    specimen_height = specimen_boundingRect[3]
    specimen_width = specimen_boundingRect[2]
    ## Zuschnittsgröße Kern festlegen ##
    core_width = relative_core_size * specimen_width
    core_height = relative_core_size * specimen_height
    core_height = core_height+relative_y_offset*core_height
    ## Bild zusammenbauen ##
    plane_img = np.zeros_like(binary_image)
    border_img = plane_img
    border_img = cv.drawContours(border_img, [specimen_contour], 0, (255,255,255), -1)
    border_img = cv.drawContours(border_img, contours_des, -1, (0,0,0), -1)
    border_img = cv.rectangle(border_img, (int(x_center-core_width/2), int(y_center-core_height/2)), 
                              (int(x_center+core_width/2), int(y_center+core_height/2)), (0,0,0), -1)
    
    return border_img

#####################################################################################################################################################

###### Größen bestimmen ######
# Ausreißer bzgl. der Porengröße finden
# Einfluss von Ausreißern auf die Dichte/Porosität untersuchen 
def get_Sizes(contour):
    
    ### Flächenberechnungen ###
    area_contour = get_ContourArea_Microns(cv.contourArea(contour))
    
    return area_contour

#####################################################################################################################################################

###### Mittelpunkt(-abstand) und Schwerpunkt(-abstand) ######
def get_Positions(x_coordinates, y_coordinates, contour_areas):
    
    ### Mittelpunkt von allen Porenzentren ###
    x_mean = sum(x_coordinates) / len(x_coordinates)
    y_mean = sum(y_coordinates) / len(y_coordinates)
    
    ### Mittelpunkt von allen Porenzentren gewichtet mit der Konturflöche ###
    x_gravities = [x_coordinates[i]*contour_areas[i] for i in range(len(contour_areas))]
    y_gravities = [y_coordinates[i]*contour_areas[i] for i in range(len(contour_areas))]
    x_gravity = sum(x_gravities) / sum(contour_areas)
    y_gravity = sum(y_gravities) / sum(contour_areas)
    
    ### Ausgabe der Verschiebung in Prozent ###
    x_diff = (x_mean-x_gravity)/x_mean*100
    y_diff = (y_mean-y_gravity)/y_mean*100
    
    ### Entfernungen der Porenzentren ###
    ## zum Mittelpunkt ##
    x_distances = [abs(x_mean-coordinate) for coordinate in x_coordinates]
    y_distances = [abs(y_mean-coordinate) for coordinate in y_coordinates]
    # Länge der direkten Verbinungslinien
    z_distances = [(x**2 + y_distances[i]**2)**(1/2) for i, x in enumerate(x_distances)]
    ## zum Schwerpunkt ##
    x_distances_gravity = [abs(x_gravity-coordinate) for coordinate in x_coordinates]
    y_distances_gravity = [abs(y_gravity-coordinate) for coordinate in y_coordinates]
    # Länge der direkten Verbinungslinien
    z_distances_gravity = [(x**2 + y_distances_gravity[i]**2)**(1/2) for i, x in enumerate(x_distances_gravity)]
    
    ### Rückgabe der Positionen und Entfernungen ###
    positions = {'center': (x_mean, y_mean),
                 'center_of_mass': (x_gravity, y_gravity),
                 'center_distances': (x_distances, y_distances),
                 'center_of_mass_distances': (x_distances_gravity, y_distances_gravity)}
    
    return positions

#####################################################################################################################################################

###### Statistische Werte einer Liste berechnen ######
def get_Statistics(data):
    no_elements = len(data)
    maximum = max(data)
    minimum = min(data)
    average = sum(data) / len(data)
    median = np.median(data)
    standard_deviation = np.std(data)
    varianz = sum((element-average)**2 for element in data) / len(data)
    skew_val = skew(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # z_score = [(element-average) / standard_deviation for element in data]
    unique = len(np.unique(data))
    
    ### Dictionary mit allen Werten anlegen ###
    statistics = {'NoElements': no_elements,
                  'Unique_Elements': unique,
                  'Maximum': maximum,
                  'Minimum': minimum,
                  'Average': average,
                  'Median': median,
                  'STD': standard_deviation,
                  'Varianz': varianz,
                  'Skewness': skew_val,
                  # 'Z_Score': z_score,
                  'Q1': q1,
                  'Q3': q3}
    
    return statistics

#####################################################################################################################################################

###### Größengewichtete statistische Werte ableiten ######
# def get_Weighted_Statistics():

#####################################################################################################################################################
###### Ausreißer detektieren ######
def get_Outliers(data):
    
    ### Interquartilsabstand ###
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    
    q_distance = q3-q1
    
    iq_high = q3+1.5*q_distance
    iq_low = q1-1.5*q_distance
    
    iq_outliers = []
    for element in data:
        if element > iq_high:
            iq_outliers.append(element)
        if element < iq_low: 
            iq_outliers.append(element)
            
    ### Standardabweichung ###
    std = np.std(data)
    mean = sum(data)/len(data)
    
    std_high = mean + 3 * std # 99.7 % der Datenpunkte liegen innerhalb von 3 Standardabweichungen --> Ausreißer liegen bei circa 0.3 %
    std_low = mean - 3 * std
    
    std_outliers = []
    for element in data:
        if element > std_high:
            std_outliers.append(element)
        if element < std_low: 
            std_outliers.append(element)      
            
    ### z-Score ###
    z = [(element - mean) / std for element in data]
    
    z_outliers = []
    for element in data:
        if element > 3:
            z_outliers.append(element)
        if element < -3: 
            z_outliers.append(element)
            
    outliers = {'IQ': len(iq_outliers),
                'STD': len(std_outliers),
                'Z': len(z_outliers)}
    
    return outliers

#####################################################################################################################################################

###### Umlaufende Histogramme ######
# Histogramme von 0° - 180° erzeugen, ggf. nur 0° und 90°
# Bestimmung der statistischen Kenngrößen
# so können Informationen über die lokalität der Poren gewonnen werden

# Zeilen- und Spaltenweise die Anzahl an schwarzen Pixeln bestimmen --> Welche Zeile/ Spalte hat welchen Schwarzanteil, ggf. relativ umsetzbar
# Statistische Größen aus dem resultierenden Vektor bestimmen
def get_Position_Histograms(binary_image):
    img_height = binary_image.shape[0]
    img_width = binary_image.shape[1]
    
    vertical_histogram = np.sum(binary_image==255, axis=1).tolist()#[np.sum(binary_image[row]==0) for row in range(img_height)]
    
    # print('Bildhöhe: {}, Länge vertikales Histogramm: {}'.format(img_height, len(vertical_histogram)))
        
    horizontal_histogram = np.sum(binary_image==255, axis=0).tolist()
        
    # print('Bildbreite: {}, Länge horizontales Histogramm: {}'.format(img_width, len(horizontal_histogram)))
    
    return vertical_histogram, horizontal_histogram 

#####################################################################################################################################################

###### Repräsentative Pore erzeugen ######
# innere Poren aus originalem Binärbild ausschneiden
# Größten Zuschnitt ermitteln und Bild aus [None]s erzeugen
# segmentierte Poren ggf. invertieren 
# Poren in weiß (255) in das None einzeichnen 
# Bild an Funktion zum überlagern übergeben
# alle Porenbilder überlagern
## ! https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil ! ##

# Primär sinnvoll für alle großen Poren (z.B. die Ausreißer die wesentlich Größer sind)
def get_Average_Pore(segmented_Contours, contour_Index, threshold, plot):
    
    ### nur die Contouren der inneren Poren segmentieren ###
    contours_des = []
    [contours_des.append(segmented_Contours[index]) for index in contour_Index]
    
    ### Breiten und Höhen der Boundingboxen aller Poren bestimmen ###    
    widths = []
    [widths.append(contours_des[i].shape[1]) for i in range(len(contours_des))]
    heights = []
    [heights.append(contours_des[i].shape[0]) for i in range(len(contours_des))]
    
    ### leeres Bild erzeugen ###
    shape = np.ones([max(heights), max(widths)]) * 255
    avr_img = np.zeros([max(heights), max(widths)])
    
    ### Mittlere Pixelintensität bestimmen ###
    for contour in contours_des:
        scaled = np.ones_like(shape)*255 # leere Skalierungsmatrix erzeugen
        
        ## Fehlende Zeilen und Spalten bestimmen ##
        missing_left = int((shape.shape[1] - contour.shape[1]) / 2)
        missing_up = int((shape.shape[0] - contour.shape[0]) / 2)
        
        ## Werte der Contour in scaled eintragen ##
        # Start (oben links) in shape ist [missing_up-1, missing_left-1]
        # len(contour) = Zeilen/Höhe
        # len(contour[0]) = Spalten/Breite
        for i in range(len(contour)):
            for j in range(len(contour[0])):
                scaled[missing_up-1+i][missing_left-1+j] = contour[i][j]
           
        ## Mittelwertbild erzeugen ##
        avr_img = avr_img + scaled/len(contours_des)
        
    ### Mittelwertbild binarisieren ###
    _, avr_img = cv.threshold(avr_img, threshold, 255, cv.THRESH_BINARY)
    
    ### Average Pore plotten ###
    if plot == True:
        plt.imshow(avr_img, cmap='gray')
        plt.axis('off')
        plt.title('Average Pore')
        plt.show()
    
    ### Mittelwertbild zurückgeben ###
    return avr_img

#####################################################################################################################################################

###### Repräsentative Pore erzeugen mit Gewichtung der Porengröße in die Überlagerung ######
def get_Average_Pore_weighted(contours, segmented_Contours, contour_Index, threshold, plot):
    
    if contour_Index != -1:
        ### nur die Contouren der inneren Poren segmentieren ###
        contours_des = []
        [contours_des.append(segmented_Contours[index]) for index in contour_Index]
        contours_des_area = [contours[index] for index in contour_Index]
        contours_area = [cv.contourArea(contour) for contour in contours_des_area]
    else:
        contours_des = segmented_Contours
        contours_area = [cv.contourArea(contour) for contour in contours]
    
    ### Breiten und Höhen der Boundingboxen aller Poren bestimmen ###    
    widths = []
    [widths.append(contours_des[i].shape[1]) for i in range(len(contours_des))]
    heights = []
    [heights.append(contours_des[i].shape[0]) for i in range(len(contours_des))]
    
    ### leeres Bild erzeugen ###
    shape = np.ones([max(heights), max(widths)]) * 255
    avr_img = np.zeros([max(heights), max(widths)])

    
    ### Mittlere Pixelintensität bestimmen ###
    for index, contour in enumerate(contours_des):
        scaled = np.ones_like(shape)*255 # leere Skalierungsmatrix erzeugen
        
        ## Fehlende Zeilen und Spalten bestimmen ##
        missing_left = int((shape.shape[1] - contour.shape[1]) / 2)
        missing_up = int((shape.shape[0] - contour.shape[0]) / 2)
        
        ## Werte der Contour in scaled eintragen ##
        # Start (oben links) in shape ist [missing_up-1, missing_left-1]
        # len(contour) = Zeilen/Höhe
        # len(contour[0]) = Spalten/Breite
        for i in range(len(contour)):
            for j in range(len(contour[0])):
                scaled[missing_up-1+i][missing_left-1+j] = contour[i][j]
           
        ## Mittelwertbild erzeugen ##
        avr_img = avr_img + scaled * contours_area[index] / sum(contours_area)
        
    ### Mittelwertbild binarisieren ###
    _, avr_img = cv.threshold(avr_img, threshold, 255, cv.THRESH_BINARY)
    
    ### Average Pore plotten ###
    if plot == True:
        plt.imshow(avr_img, cmap='gray')
        plt.axis('off')
        plt.title('Average Pore weighted by Poresize')
        plt.show()
    
    ### Mittelwertbild zurückgeben ###
    return avr_img

#####################################################################################################################################################

###### lokale Dichte bestimmen ######
# Zerlegung des Bildes in Quadrate
# Konturen in Quadrat bestimmen
# Anzahl an Poren, Form, Typ und Größe bestimmen
# relative Dichte im Quadrat bestimmen

#####################################################################################################################################################