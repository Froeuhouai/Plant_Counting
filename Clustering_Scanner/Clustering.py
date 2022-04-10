# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:30:47 2021

@author: Groupe Tournesol PFR 2020/2021

"""

# Import of librairies
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.gridspec as gridspec
import pandas as pd
import sys
import os
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.patches as patches
import seaborn as sns
import time
import json
import math
import random
from kneed import KneeLocator

# pip install -U scikit-fuzzy
import skfuzzy as fuzz


# If import does work, use the following lines
os.chdir("../Utility/")
import general_IO as gIO

# else
# if "PATH/TO/REPOSITORY/FOLDER/Plant_counting" not in sys.path:
#     sys.path.append("PATH/TO/REPOSITORY/FOLDER/Plant_counting")

# os.chdir("PATH/TO/REPOSITORY/FOLDER/Plant_counting")
# import Utility.general_IO as gIO


def DBSCAN_clustering(img, epsilon, min_point, image):
    """
    The objective of this function is to differenciate rows of a field.
    As input, it needs a binarized picture (cf function/package...) taken by an
    unmanned aerial vehicle (uav). The white pixels representing the plants in
    are labelled with the DBSCAN algorithm according to the row they belong to.
    The package Scikit-learn is used to do so.
    The intensity value of the pixels is 255.

    Extraction of the white pixels of an Image object
    - Extract white pixel (intensity of 255) from
    the image matrix.
    - Clustering coordinates using DBSCAN algorithm


    Parameters
    ----------
    img : PIL Image
        Image opened with the PIL package. It is an image of plants cultivated
        in a field.
        The picture can be taken by an unmanned aerial vehicle (uav).

    epsilon : INTEGER
        Two points are considered neighbors if the distance between the two
        points is below the threshold epsilon.

    min_point : INTEGER
        The minimum number of neighbors a given point should have in order to
        be classified as a core point.


    Returns
    -------
    dataframe_coord : PANDAS DATAFRAME
        Dataframe with the X and Y coordinates of the plants' pixels
        and their cluster's label.

    """

    # extraction of white pixels coordinates
    img_array = np.array(img)
    mat_coord = np.argwhere(img_array[:, :, 0] == 255)

    # clustering using DBSCAN
    mat_clustered = DBSCAN(eps=epsilon, min_samples=min_point).fit(mat_coord)
    print("mat_clustered : ", mat_clustered)
    # Panda dataframe
    dataframe_coord = pd.DataFrame(mat_coord)
    dataframe_coord = dataframe_coord.rename(columns={0: "X", 1: "Y"})

    label = pd.DataFrame(mat_clustered.labels_)
    label = label.rename(columns={0: "label"})


    # Dataframe gathering each plants pixel its label
    dataframe_coord = pd.concat([dataframe_coord, label], axis=1)
    #print("mat_clustered.get_params(deep=True) : ", mat_clustered.get_params(deep=True)
    #Plot(dataframe_coord, 0, image, "DBSCan clustering for {0}".format(image))
    return dataframe_coord

def OPTICS_clustering(img, epsilon, min_point, image):
    """
    The objective of this function is to differenciate rows of a field.
    As input, it needs a binarized picture (cf function/package...) taken by an
    unmanned aerial vehicle (uav). The white pixels representing the plants in
    are labelled with the OPTICS algorithm according to the row they belong to.
    The package Scikit-learn is used to do so.
    The intensity value of the pixels is 255.

    Extraction of the white pixels of an Image object
    - Extract white pixel (intensity of 255) from
    the image matrix.
    - Clustering coordinates using DBSCAN algorithm


    Parameters
    ----------
    img : PIL Image
        Image opened with the PIL package. It is an image of plants cultivated
        in a field.
        The picture can be taken by an unmanned aerial vehicle (uav).

    epsilon : INTEGER
        Two points are considered neighbors if the distance between the two
        points is below the threshold epsilon.

    min_point : INTEGER
        The minimum number of neighbors a given point should have in order to
        be classified as a core point.


    Returns
    -------
    dataframe_coord : PANDAS DATAFRAME
        Dataframe with the X and Y coordinates of the plants' pixels
        and their cluster's label.

    """

    # extraction of white pixels coordinates
    img_array = np.array(img)
    mat_coord = np.argwhere(img_array[:, :, 0] == 255)
    
    n_points_per_cluster = 250
    C1 = [-5, -2] + 0.8 * np.random.randn(n_points_per_cluster, 2)
    C2 = [4, -1] + 0.1 * np.random.randn(n_points_per_cluster, 2)
    C3 = [1, -2] + 0.2 * np.random.randn(n_points_per_cluster, 2)
    C4 = [-2, 3] + 0.3 * np.random.randn(n_points_per_cluster, 2)
    C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
    C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
    X = np.vstack((C1, C2, C3, C4, C5, C6))
    print(type(X[0]))
    print(type(mat_coord[0]))
    
    # test_matrix = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.05).fit(X)
    # reach= test_matrix.reachability_[test_matrix.ordering_]
    # labls = test_matrix.labels_[test_matrix.ordering_]
    # colors = ["g", "r", "b", "y", "c","g", "r", "b", "y", "c","g", "r", "b", "y", "c","g", "r", "b", "y", "c","g", "r", "b", "y", "c"]
    # for klass, color in zip(range(0, max(labls)), colors):
    #     print("[labels == klass] : ", [labls == klass])
    #     Xk = np.arange(len(X))[labls == klass]
    #     Rk = reach[labls == klass]
    #     plt.scatter(Xk, Rk, c = color, alpha=0.3, plotnonfinite=True, s=0.2)
    # plt.plot(np.arange(len(X))[labls == -1], reach[labls == -1], "k.", alpha=0.3)
    # plt.plot(np.arange(len(X)), np.full_like(np.arange(len(X)), 2.0, dtype=float), "k-", alpha=0.5)
    # plt.plot(np.arange(len(X)), np.full_like(np.arange(len(X)), 0.5, dtype=float), "k-.", alpha=0.5)
    # plt.ylabel("Reachability (epsilon distance)")
    # plt.title("Test Reachability Plot on random matrix")
    # plt.show()

    # clustering using OPTICS
    #mat_clustered = OPTICS(max_eps=epsilon, min_samples=min_point, metric = 'euclidean', cluster_method="dbscan").fit(mat_coord)
    mat_clustered = OPTICS(xi=0.05, min_samples=min_point, min_cluster_size=500, predecessor_correction = True).fit(mat_coord)
    #mat_clustered = OPTICS(min_samples = min_point, min_cluster_size = 500).fit(mat_coord)
    space = np.arange(len(mat_coord))
    reachability = mat_clustered.reachability_[mat_clustered.ordering_]
    core_distances = mat_clustered.core_distances_[mat_clustered.ordering_]
    labels = mat_clustered.labels_[mat_clustered.ordering_]
    predecessor = mat_clustered.predecessor_[mat_clustered.ordering_]
    
    # LOF = LocalOutlierFactor(n_neighbors=100)
    # decision_LOF = LOF.fit_predict(reachability.reshape(-1,1))
    # score_LOF = LOF.negative_outlier_factor_
    
    # plt.figure()
    # plt.scatter(decision_LOF, score_LOF, s=0.2)
    # plt.title("LOF")
    # plt.show()
    
    
    plt.figure()
    plt.scatter([i for i in range(len(mat_clustered.predecessor_[mat_clustered.ordering_]))], mat_clustered.predecessor_[mat_clustered.ordering_], s=0.2)
    plt.title("Predecessors")
    plt.show()
    
    print("mat_clsutered.ordering_ : ", mat_clustered.ordering_)
    print("Reachability : ", len(reachability), reachability)
    print("Labels : ", len(labels), labels)
    print("Space : ", len(space), space)
    print("predecessors : ", predecessor)
    
    colors = ["g", "r", "b", "y", "c","g", "r", "b", "y", "c","g", "r", "b", "y", "c","g", "r", "b", "y", "c","g", "r", "b", "y", "c"]
    
    #Core distances plot
    plt.figure()
    for klass, color in zip(range(0, max(labels)), colors) :
        Xk = space[labels == klass]
        Rk = core_distances[labels == klass]
        plt.scatter(Xk, Rk, s =0.2, c = color, alpha = 0.3, plotnonfinite=True)
    plt.title("Core_dsitances" + image + " epsilon : " + str(epsilon) + " min_point = " + str(min_point))
    plt.show()

    
    # Reachability plot
    plt.figure()
    for klass, color in zip(range(0, max(labels)), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        plt.scatter(Xk, Rk, c = color, alpha=0.3, s=0.2)
    #plt.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
    plt.plot(space, np.full_like(space, 50, dtype=float), "k-", alpha=0.5)
    plt.plot(space, np.full_like(space, 20, dtype=float), "k-.", alpha=0.5)
    plt.ylabel("Reachability (epsilon distance)")
    plt.title("Reachability Plot " + image + " epsilon : " + str(epsilon) + " min_point = " + str(min_point))
    plt.show()

    
    
    print("mat_clustered : ", mat_clustered)
    # Panda dataframe
    dataframe_coord = pd.DataFrame(mat_coord)
    dataframe_coord = dataframe_coord.rename(columns={0: "X", 1: "Y"})

    label = pd.DataFrame(mat_clustered.labels_)
    label = label.rename(columns={0: "label"})


    # Dataframe gathering each plants pixel its label
    dataframe_coord = pd.concat([dataframe_coord, label], axis=1)
    #print("mat_clustered.get_params(deep=True) : ", mat_clustered.get_params(deep=True))
    Plot(dataframe_coord, 0, image, "OPTICS clustering for {0} epsilon = {1} min point = {2}".format(image, epsilon, min_point))
    return dataframe_coord

def Nb_neighbours_identifier(img, image_name, epsilon, min_point, dataframe_coord) : 
    # extraction of white pixels coordinates
    img_array = np.array(img)
    mat_coord = np.argwhere(img_array[:, :, 0] == 255)
    
    list_cluster = []
    list_dataframe_coord = dataframe_coord.values.tolist() #On récupère la liste des clusters auxquels appartiennent les pixels
    for j in range(0, len(list_dataframe_coord),20) : 
        list_cluster.append(list_dataframe_coord[j][2])

    #Initializing object to save results 
    dic = {}
    dic2 = {}
    dic3 = {}
    
    
    txt_file = open(r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\Nb_neighbours_identifier\_" + image_name.split(".")[0] + "_epsilon_" + str(epsilon)+ "min_point_" + str(min_point) + ".txt", "w")
    
    print("--Strating Neighbours identification--")
    start_time = time.time()
    nb_neighbours = []
    dist_neighbours = []
    total_iteration = 0
    
    print("Epsilon : ", epsilon, " Min_point : ", min_point)
    #For each white pixel in mat coord, checks how many neighbours it has regarding of the circle radius epsilon
    for pixel in range(0, int(len(mat_coord)/50)) :
        neighbours = [] #Re-initialise this list for each pixel
        neighbours_count = 0
        nb_iteration = 0
        X_center = mat_coord[pixel][0] #X position of the point we want to find the neighbours
        Y_center = mat_coord[pixel][1] #Y position of the point we want to find the neighbours
        _list_dist = []
        
        for pixels in range(0,len(mat_coord),2) :
                X_test = mat_coord[pixels][0] #X position of the pixel we want to test the membership
                Y_test = mat_coord[pixels][1] #Y position of the point we want to test the membership
                dist = (X_center - X_test)**2 + (Y_center - Y_test)**2
                if dist <= epsilon**2 : #If the test pixel is a member of the circle 
                    neighbours.append(mat_coord[nb_iteration]) #We add the pixel to the neighbours
                    _list_dist.append(int(dist))
                    neighbours_count += 1   
        dist_neighbours.append(_list_dist)
        nb_neighbours.append(neighbours_count)
        if pixel%1000 == 0 :
            print("Temps par pixel : ", (time.time() - start_time)/(pixel+1), " for pixel n°", pixel)
            
    print("nb_neighbours : ", nb_neighbours)
    end_time = time.time()
    Process_time = end_time-start_time
    print("total_iteration : ", total_iteration)    
    print("Process time of neighbours identifier : ", Process_time)
    
    #Save results
    dic[str(min_point)] = {"nb_neighbours_per_pixel" : nb_neighbours, 
                          "Process_time" : Process_time,
                          "list_cluster" : list_cluster,
                          "dist_neighbours_per_pixel" : dist_neighbours} 
    
    dic2[str(epsilon)] = dic
    
    dic3[image_name] = dic2
    json.dump(dic3, txt_file, indent = 3)
    txt_file.close()
    
    #Ploting
    # plt.figure()
    # plt.plot(range(0,len(nb_neighbours)), nb_neighbours) #Essayer d'associer sur les plots tous les points d'un cluster en une seule couleur pour repérer plus distinctement les pics/creux
    # plt.title("Nb neighbours / epsilon " + image_name + " epsilon = " + str(epsilon))
    # axis = plt.gca()
    # axis.set_xlabel("Pixel n° ")
    # axis.set_ylabel("Nb neighbours")
    # plt.savefig("Nb_neighbours_epsilon " + image_name + " epsilon_" + str(epsilon)+".jpg")
    # plt.show()
    
def Dist_Neighbours_Identifier(img, image_name, dataframe_coord) :
        # extraction of white pixels coordinates
    img_array = np.array(img)
    mat_coord = np.argwhere(img_array[:, :, 0] == 255)
    

    
    start_time = time.time()
    print("--Strating Neighbours identification--")
    
    #For each white pixel in mat coord, checks how many neighbours it has regarding of the circle radius epsilon
    print("Nombre de pixel à tester : ", len(mat_coord))
    print("Temps estimé pour l'image en minutes : ", (len(mat_coord)/1000)*90/60)
    dist_neighbours = [] #Re-initialise this list for each pixel
    for pixel in range(4500, 5000) :
        X_center = mat_coord[pixel][0] #X position of the point we want to find the neighbours
        Y_center = mat_coord[pixel][1] #Y position of the point we want to find the neighbours
        _list_dist = [] 
        for pixels in range(0,len(mat_coord)) :
                X_test = mat_coord[pixels][0] #X position of the pixel we want to test the membership
                Y_test = mat_coord[pixels][1] #Y position of the point we want to test the membership
                dist = math.sqrt((X_center - X_test)**2 + (Y_center - Y_test)**2)
                _list_dist.append(int(dist))
        dist_neighbours.append(_list_dist)
        if pixel%10 == 0 :
            print("Temps pour 10 pixels : ", (time.time() - start_time))
            
    print("Temps total pour {0} pixels : ".format(len(mat_coord)), time.time() - start_time)
    print("len(dist_neighbours) : ", len(dist_neighbours))
    print("len(dist_neighbours)[0] : ", len(dist_neighbours[0]))
    print("Nombre de pixel : ", len(mat_coord))
    
    previous_i = 0
    count = 0
    for i in range(int(len(dist_neighbours)/10), len(dist_neighbours), int(len(dist_neighbours)/10)) :
        txt_file = open(r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\Dist_neighbours_identifier\_" + image_name.split(".")[0] + "_" + str(count) + ".txt", "w")
        dic = {}
        print("len(dist_neighbours[previous_i:i]) : ", len(dist_neighbours[previous_i:i]))
        dic["list_distances"] = dist_neighbours[previous_i:i]
        print("len(dic[list_distances] ", len(dic["list_distances"]))
        json.dump(dic, txt_file, indent = 3)
        txt_file.close()
        previous_i = i
        count += 1
        

    
def Plants_Detection(dataframe_coord, e, max_iter, m_p, threshold, image):
    """
    The objective of this function is to differenciate the plants in each row
    of the binarized picture. It is based on predefined rows, the corresponding
    pixels labelled.
    It calls for the Fuzzy_Clustering function.
    The final result cen be used to initialize a grid for a multiple agents
    system to count more precisely the number of plants in the picture.


    Parameters
    ----------
    dataframe_coord : Panda dataframe
        Dataframe with the X and Y coordinates and as a third column, the row
        the pixel belongs to.
        It is obtained with the function DBSCAN_clustering.

    e : FLOAT
        Stopping criterion for the partitionning between two iterations,
        set at 0.005

    max_iter : INTEGER
        maximum iteration number, set at 100

    m_p : INTEGER
        Fuzzy parameters, power apply to U (d'appartenance) matrix. It is
        often set at 2.

    Threshold : INTEGER
        Threshod in order to determine if there are enough pixels in the result
        of the DBSCAN_clustering to considere a cluster as a row


    Returns
    -------
    JSON_final : JSON file that can be used as a grid to initialize the agents
        of a multiple agents system. Its size is the number of rows and the
        size of a row is the number of plants (centroïd coordinates).
    """
    # Number of rows and their label
    label_row = np.unique(dataframe_coord[["label"]].to_numpy())
    # Quality metrics of the Fuzzy clustering method, include in the interval
    # [0,1], 1 representing a good result.
    # fpcs = []

    # Historic_cluster save for each row the number of estimate clusters
    # and the final number of clusters
    historic_cluster = [[], []]

    # JSON_final contain final plants positions to initialize the MAS.
    # Each list is a row and contain couple of coordinates representing plants.
    JSON_final = []

    # Centers of all plants clusters coordinates
    XY = [[], []]

    # For each rows, do the plants detection.
    for row in label_row:
        # row_pixels is a matrix with pixels coordinates belonging to a row.
        row_pixels = (
            dataframe_coord[dataframe_coord["label"] == row][["X", "Y"]].to_numpy().T #Get the X and Y coords for each pixels with the same "label" value == belonging to the same cluster == Forming a row
        )
        
        # If the row is really a row.
        # if Threshold_Pixels_Row(row_pixels, threshold) is False:
        #     dataframe_coord.drop(
        #         dataframe_coord[dataframe_coord["label"] == row].index, inplace=True
        #     )
        # else:
            # Determination of the initial estimating number of clusters,
            # and adding to historic_cluster
        estimate_nb_clusters = Automatic_Cluster_Number(row_pixels, _RAs_group_size=20)
        historic_cluster[0].append(estimate_nb_clusters)

        # Clustering using Fuzzy_Clustering, and adding to historic_cluster
        results_fuzzy_clustering, final_nb_clusters = Fuzzy_Clustering(
            row_pixels, estimate_nb_clusters, e, m_p, max_iter
        )
        historic_cluster[1].append(final_nb_clusters)
        #print("results_fuzzy_clustering : ")
        #print(results_fuzzy_clustering)
        reversed_fuzzy_matrix = []
        for pt in results_fuzzy_clustering:
            XY[0].append(pt[1])
            XY[1].append(pt[0])
            reversed_fuzzy_matrix.append([pt[1], pt[0]])

        # Append to the final JSON positions of plants
        # for the considered row.
        #JSON_final.append(results_fuzzy_clustering)
        JSON_final.append(reversed_fuzzy_matrix)
    
    # Plot
    Plot(dataframe_coord, XY, image, "Fuzzy clustering for {0}".format(image))
    return JSON_final

def Dataframe_to_Json(dataframe_coord) :
    
    _list_coords = []
    _x = dataframe_coord["X"].tolist()
    _y = dataframe_coord["Y"].tolist()
    _cluster = dataframe_coord["label"].tolist()
    for i in range(0, len(dataframe_coord)):   
        _list_coords.append([_x[i], _y[i], _cluster[i]])
    
    return _list_coords
        
def Threshold_Pixels_Row(row_pixels, threshold):
    """
    Determine if there are enough pixels in the result of the
    DBSCAN_clustering function. Define if a cluster can be considered
    as a row according to a threshold pixels.

    Parameters
    ----------
    row_pixels : List
        List of 2 lists, the values of X and Y, respectively, coordinates of
        pixels representing plants in the row.

    Return
    -------
    isRow : Boolean
        Boolean indicating if there are enough pixel in a row to count at least
        a plant. A VERIFIER A PARTIR DE COMBIEN DE PLANTES DANS LE RANG ON
        CONSIDERE OK POUR INITIALISATION D'UN AGENT PLANTE

    """
    # There are enough pixels.
    if len(row_pixels[0]) > threshold:
        return True
    else:
        return False


def Automatic_Cluster_Number(row_pixels, _RAs_group_size):
    """
    Objective : to estimate the number of clusters to initialize the fuzzy
    clustering function. To do so we divide the number of pixels in a row by
    an estimation of the surface of one plant : 40% of a lenght of a plant agent squared.

    Parameters
    ----------
    row_pixels : list of np.array
        Values of X and Y, the coordinates of the pixels in a row.

    Returns
    -------
    Estimated_nb_clusters : Integer
        An estimation of the number of clusters (plants) in a row.
    """
    print("Nombre de pixels dans le cluster : ", len(row_pixels[0]))
    estimated_nb_clusters = int(
        len(row_pixels[0]) / ((_RAs_group_size * _RAs_group_size) * 0.4)
    )
    print("Estimated cluster number : ", estimated_nb_clusters)
    # If too few pixels, supplementary security
    if estimated_nb_clusters < 3 :
        estimated_nb_clusters = 3
    return estimated_nb_clusters


def Fuzzy_Clustering(row_pixels, estimate_nb_clusters, e, m_p, max_i):
    """
    Apply the Fuzzy clustering algorithm in order to determine the number
    of clusters, ie the number of plants in a row. Take as an input pixels
    coordinate for on row : row_pixels, the initial number of cluster :
    estimate_nb_clusters, and parameters e, m_p, max_i required for
    cmeans algorithm.

    Returns
    -------
    Return information about plants positions for one row. Its also
    possible to return other parameters like fpc quality score.

    position_cluster_center : LIST
        List of coordinate couple representing center of clusters
        position ie the position of each plant.

    final_nb_clusters : INTEGER
        The final number of cluster, ie of plants, in a row.


    """
    position_cluster_center = []

    centres, u, u0, d, jm, p, fpc = fuzz.cmeans(
        row_pixels, c=estimate_nb_clusters, m=m_p, error=e, maxiter=max_i
    )

    final_nb_clusters = len(u)

    for position in centres:
        position_cluster_center.append([int(position[0]), int(position[1])])

    return position_cluster_center, final_nb_clusters


def Plot(mat_coord, centresCoordinates, image, _title):
    fig = plt.figure(figsize=(8, 12))
    label_cluster = np.unique(mat_coord[["label"]].to_numpy())
    ax = fig.add_subplot(111)
    txts = []
    for i in label_cluster:
        xtext = np.median(mat_coord[mat_coord["label"] == i]["Y"])
        ytext = np.median(mat_coord[mat_coord["label"] == i]["X"])
        txt = ax.text(ytext, xtext, str(i))
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
        )
        txts.append(txt)
            
    scatter_row = ax.scatter(
        mat_coord["X"].tolist(),
        mat_coord["Y"].tolist(),
        c=mat_coord["label"].tolist(),
        s=0.5,
        cmap="Paired",
    )
    if centresCoordinates != 0 :
        print("center coordinates : ")
        print(centresCoordinates)
        scatter_plant = plt.scatter(
            centresCoordinates[1], centresCoordinates[0], s=10, marker="x", color="k"
        )
    #plt.title("Announced number of clusters = " + str(max(label_cluster)+1))
    plt.ylabel(_title)
    plt.show()
    #fig.savefig("../../Resultats_Visual" + image.split(".")[0] + "cluster.png")
    return

def Plot_end_of_rows(mat_coord, centresCoordinates, image, _title, end_of_rows):
    fig = plt.figure(figsize=(8, 12))
    label_cluster = np.unique(mat_coord[["label"]].to_numpy())
    ax = fig.add_subplot(111)
    txts = []
    for i in label_cluster:
        xtext = np.median(mat_coord[mat_coord["label"] == i]["Y"])
        ytext = np.median(mat_coord[mat_coord["label"] == i]["X"])
        txt = ax.text(ytext, xtext, str(i))
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w", alpha = 1), PathEffects.Normal()]
        )
        txts.append(txt)
        # if i <= 2 : #Normally no condition on that, did it to pass through missing rows index in the dataframe
        #     txt = ax.text(ytext, xtext, str(i))
        #     txt.set_path_effects(
        #         [PathEffects.Stroke(linewidth=5, foreground="w", alpha = 1), PathEffects.Normal()]
        #     )
        #     txts.append(txt)
        # if i > 2 and i <= 9 :
        #     txt = ax.text(ytext, xtext, str(i-1))
        #     txt.set_path_effects(
        #     [PathEffects.Stroke(linewidth=5, foreground="w", alpha = 1), PathEffects.Normal()]
        #     )
        #     txts.append(txt)
        # if i > 9 : 
        #     txt = ax.text(ytext, xtext, str(i-2))
        #     txt.set_path_effects(
        #         [PathEffects.Stroke(linewidth=5, foreground="w", alpha = 1), PathEffects.Normal()]
        #     )
        #     txts.append(txt)
    f = 0
    for _RAL in end_of_rows :
        
        xtext = _RAL.x
        ytext = _RAL.y
        txt = ax.text(ytext, xtext, str(f), color="r")
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w", alpha = 1), PathEffects.Normal()]
        )
        txts.append(txt)
        f+=1
        _RAL.Compute_neighbours_distances()
        _RAL.Compute_closest_neighbour_translation()
        print(_RAL.x, _RAL.y, _RAL.CN_translation_x + _RAL.x, _RAL.CN_translation_y + _RAL.y)
        ax.plot([_RAL.y,_RAL.CN_translation_y + _RAL.y],[_RAL.x,_RAL.CN_translation_x + _RAL.x], marker = "o", markersize = 5)

        end_of_rows_distances = []
        for _RAL_2 in end_of_rows :
            euclidean_distance = math.sqrt((_RAL.x - _RAL_2.x)**2 + (_RAL.y - _RAL_2.y)**2)
            end_of_rows_distances.append(euclidean_distance)
        end_of_rows_distances[end_of_rows_distances.index(0)] = max(end_of_rows_distances)
        nearest_neighbour = end_of_rows[end_of_rows_distances.index(min(end_of_rows_distances))]
        ax.plot([_RAL.y, nearest_neighbour.y], [_RAL.x, nearest_neighbour.x])

    scatter_row = ax.scatter(
        mat_coord["X"].tolist(),
        mat_coord["Y"].tolist(),
        c=mat_coord["label"].tolist(),
        s=0.5,
        cmap="Paired",
    )
    if centresCoordinates != 0 :
        print("center coordinates : ")
        print(centresCoordinates)
        scatter_plant = plt.scatter(
            centresCoordinates[1], centresCoordinates[0], s=10, marker="x", color="k"
        )
    #plt.title("Announced number of clusters = " + str(max(label_cluster)+1))
    plt.ylabel(_title)
    plt.show()
    #fig.savefig("../../Resultats_Visual" + image.split(".")[0] + "cluster.png")
    return

def Automatic_Epsilon(img,image):
    
    for i in range(1,300,5) :
        dataframe_coord = DBSCAN_clustering(img, i, int(0.1*math.pi*(i**2)))
        dataframe_coord.drop( #Remove all the pixels which have -1 as label value
        dataframe_coord[dataframe_coord["label"] == -1].index, inplace=True
        )
        print("plot")
        Plot(dataframe_coord,0,image, "Clustering de {0} pour epsilon = {1} Nb de cluster = ".format(image,i))
        print("plot")
        print(max(dataframe_coord["label"].tolist()))
        
        
        if max(dataframe_coord["label"].tolist()) == 0 :
            inter_row_dist = i
            optimal_epsilon = int(0.7*inter_row_dist) #0.7 is arbitrary, this optimal distance is a distance that allow to make longest row as possible without allowing parallel rows to form a cluster
            return optimal_epsilon

        

def add_array_in_json(path_input_root, dataframe_coord, i) : 
    list_cluster = []
    #print("current i : ", i)
    if 0 <= i <= 19 :
        path = path_input_root + "\RGB_34" 
        os.chdir(path)
        list_img = os.listdir(path)
        for k in [0 for j in range(0,10)] : #On réorganise la liste de manière à ce que les epsilon soient dans l'ordre
            list_img.append(list_img[0])
            list_img.pop(0)
        read_file = open(list_img[i], "r") #On ouvre en read pour copier les données
        print(list_img[i], " ", i )
        json_file = json.load(read_file)
        read_file.close()
        list_dataframe_coord = dataframe_coord.values.tolist()
        for j in range(0, len(list_dataframe_coord)) : 
            list_cluster.append(list_dataframe_coord[j][2])
        dic = {"cluster_list" : list_cluster}
        key1 = list(json_file)[0]
        key2 = list(json_file[key1])[0]
        json_file[key1][key2].update(dic) #On modifie les données
        append_file = open(list_img[i], "w") #On ouvre en write pour effacer les données précédentes
        json.dump(json_file, append_file, indent=1) #On dump les nouvelles données
        append_file.close()
        print("len(list_dataframe_coord) : ", len(list_dataframe_coord))
        print("len(list_cluster) : ", len(list_cluster))
        
        
    if 20 <= i <= 39 :
        path = path_input_root + "\RGB_36"
        os.chdir(path)
        list_img = os.listdir(path)
        for k in [0 for j in range(0,10)] :
            list_img.append(list_img[0])
            list_img.pop(0)
        read_file = open(list_img[i%20], "r") #On ouvre en read pour copier les données
        json_file = json.load(read_file)
        read_file.close()
        list_dataframe_coord = dataframe_coord.values.tolist()
        for j in range(0, len(list_dataframe_coord)) : 
            list_cluster.append(list_dataframe_coord[j][2])
        dic = {"cluster_list" : list_cluster}
        key1 = list(json_file)[0]
        key2 = list(json_file[key1])[0]
        json_file[key1][key2].update(dic) #On modifie les données
        append_file = open(list_img[i%20], "w") #On ouvre en write pour effacer les données précédentes
        json.dump(json_file, append_file, indent = 1) #On dump les nouvelles données
        append_file.close()
        print("len(list_dataframe_coord) : ", len(list_dataframe_coord))
        print("len(list_cluster) : ", len(list_cluster))
        
    if 40 <= i <= 59 :
        path = path_input_root + "\RGB_38"
        os.chdir(path)
        list_img = os.listdir(path)
        for k in [0 for j in range(0,10)] :
            list_img.append(list_img[0])
            list_img.pop(0)
        read_file = open(list_img[i%20], "r") #On ouvre en read pour copier les données
        json_file = json.load(read_file)
        read_file.close()
        list_dataframe_coord = dataframe_coord.values.tolist()
        for j in range(0, len(list_dataframe_coord)) : 
            list_cluster.append(list_dataframe_coord[j][2])
        dic = {"cluster_list" : list_cluster}
        key1 = list(json_file)[0]
        key2 = list(json_file[key1])[0]
        json_file[key1][key2].update(dic) #On modifie les données
        append_file = open(list_img[i%20], "w") #On ouvre en write pour effacer les données précédentes
        json.dump(json_file, append_file, indent = 1) #On dump les nouvelles données
        append_file.close()
        print("len(list_dataframe_coord) : ", len(list_dataframe_coord))
        print("len(list_cluster) : ", len(list_cluster))
        
    if 60 <= i <= 79 :
        path = path_input_root + "\RGB_40"
        os.chdir(path)
        list_img = os.listdir(path)
        for k in [0 for j in range(0,10)] :
            list_img.append(list_img[0])
            list_img.pop(0)
        read_file = open(list_img[i%20], "r") #On ouvre en read pour copier les données
        json_file = json.load(read_file)
        read_file.close()
        list_dataframe_coord = dataframe_coord.values.tolist()
        for j in range(0, len(list_dataframe_coord)) : 
            list_cluster.append(list_dataframe_coord[j][2])
        dic = {"cluster_list" : list_cluster}
        key1 = list(json_file)[0]
        key2 = list(json_file[key1])[0]
        json_file[key1][key2].update(dic) #On modifie les données
        append_file = open(list_img[i%20], "w") #On ouvre en write pour effacer les données précédentes
        json.dump(json_file, append_file, indent = 1) #On dump les nouvelles données
        append_file.close()
        print("len(list_dataframe_coord) : ", len(list_dataframe_coord))
        print("len(list_cluster) : ", len(list_cluster))
        
def Total_Plant_Position_Bis(
    _path_input_root,
    _path_output_root,
    _set,
    session,
    growth_stage,
    epsilon,
    min_point,
    e,
    max_iter,
    m_p,
    threshold,
    _RAs_group_size,
):
    # Open the binarized image
    list_image = listdir(
        _path_input_root.format(_stage)
    )
    list_row = []
    i = 0
    for image in list_image:
        plt.figure()
        nb_cluster = []
        for epsilon in range(1,51) :
            print("--- start --- ", image)
            #os.chdir(r"C:\Users\prje\OneDrive - SUPERETTE\Bureau\Stages\Stage INRAE\Plant_Counting\Utility")
            img = Image.open(_path_input_root.format(_stage) + "/" + image)
            
            print("--- DBSCAN ---")
            print(epsilon)
            _time = time.time()
            dataframe_coord = DBSCAN_clustering(img, epsilon, min_point,image)
            #Plot(dataframe_coord,[0,0],0, "DBScan image {0}, Epsilon = {1} Avant drop des pixels sans cluster".format(image, epsilon))
            print("Process time DBScan : ", time.time() - _time)
            # dataframe_coord.drop( #Remove all the pixels which have -1 as label value
            #     dataframe_coord[dataframe_coord["label"] == -1].index, inplace=True
            # )
            try :
                nb_cluster.append(max(dataframe_coord["label"])+1)
            except ValueError :
                nb_cluster.append(1)

            Plot(dataframe_coord,0,image, "Clustering de {0} pour epsilon = {1} Nb de cluster = {2}".format(image,epsilon,nb_cluster[-1]))
                
                
            #Dist_Neighbours_Identifier(img, image, dataframe_coord)
            
            #Plot(dataframe_coord,[0,0],0, "DBScan image {0}, Epsilon = {1} Après drop des pixels sans cluster".format(image, epsilon))
            # print("--- OPTICS ---")
            # _time = time.time()
            # dataframe_coord_2 = OPTICS_clustering(img, epsilon+50, min_point)
            # print("Process time OPTICS : ", time.time() - _time)
            # dataframe_coord_2.drop( #Remove all the pixels which have -1 as label value
            #     dataframe_coord[dataframe_coord["label"] == -1].index, inplace=True
            # )
            
            #list_row.append(len(np.unique(dataframe_coord[["label"]].to_numpy()))-1)


            #plt.scatter(list_epsilon, list_mp, s=0.5 , c = list_row, norm = plt.Normalize(vmin=0, vmax=max(list_row)), cmap = 'nipy_spectral')
            # print("list_row : ", list_row)
            # print("dataframe_coord : ", dataframe_coord)
            #print("len(dataframe_coord) : ", len(dataframe_coord))
            
            #add_array_in_json(r"D:\Datasets\Datasets rangs courbes\Nb_neighbours_identifier", dataframe_coord, i) #ATTENTION A NE DECOMMENTER CETTE LIGNE QU'AVEC dataframe_coord.drop DE COMMENTé
            #i+=1
            
            #Nb_neighbours_identifier(img, image, epsilon, min_point, dataframe_coord)
            
            # print("--- Plant_detection DBScan ---")
            # JSON_final = Plants_Detection(
            #     dataframe_coord, e, max_iter, m_p, threshold, image)
            # )
            # # print("--- Plant_detection OPTICS ---")
            # # JSON_final = Plants_Detection(
            # #     dataframe_coord_2, e, max_iter, m_p, threshold, image
            # # )
            
            # print("--- write_json ---")
            # path_JSON_output = (
            #     _path_output_root.format(_set, session)
            #     # + "/outputCL/session"
            #     # + str(session)
            #     # + "/Plant_CL_Predictions"
            # )
            # gIO.check_make_directory(path_JSON_output)
            # gIO.WriteJson(
            #     path_JSON_output,
            #     "Predicting_initial_plant_" + image.split(".")[0],
            #     JSON_final
            # )
        # plt.title("Nombre de cluster pour {0}".format(image))
        # plt.xlabel("Epsilon")
        # plt.ylabel("Nombre de cluster")
        # plt.plot([i for i in range(41,151)], nb_cluster)
        # plt.grid(visible=True)
        # plt.show()
        
        # cluster_speed = []
        # for i in range(0, len(nb_cluster)-1) :
        #     cluster_speed.append(nb_cluster[i+1] - nb_cluster[i])
        
        # plt.figure()
        # plt.plot([i for i in range(41, len(cluster_speed)+41)], cluster_speed)
        # plt.title("Vistesse d'augmentation du nombre de cluster pour {0}".format(image))
        # plt.xlabel("Epsilon")
        # plt.ylabel("Vitesse")
        # plt.grid(visible=True)
        # plt.show()
        
        # cluster_acceleration = []
        # for i in range(0, len(cluster_speed)-1) :
        #     cluster_acceleration.append(cluster_speed[i+1] - cluster_speed[i])
            
        # plt.figure()
        # plt.plot([i for i in range(41, len(cluster_acceleration)+41)], cluster_acceleration)
        # plt.title("Acceleration de l'augmentation du nombre de cluster pour {0}".format(image))
        # plt.xlabel("Epsilon")
        # plt.ylabel("Acceleration")
        # plt.grid(visible=True)
        # plt.show()
        
    #return
        
def Total_Plant_Position(
    _path_input_root,
    _path_output_root,
    _set,
    session,
    growth_stage,
    epsilon,
    min_point,
    e,
    max_iter,
    m_p,
    threshold,
    _RAs_group_size,
):
    
    # Open the binarized image
    list_image = listdir(
        _path_input_root.format(growth_stage)
    )
    list_row = []
    i = 0
    for image in list_image :
        
        
            print("--- start --- ", image)
            #os.chdir(r"C:\Users\prje\OneDrive - SUPERETTE\Bureau\Stages\Stage INRAE\Plant_Counting\Utility")
            img = Image.open(_path_input_root.format(growth_stage) + "/" + image)
            
            # print("------Epsilon parameter-------")
            # epsilon = Automatic_Epsilon(img,image)
            # min_point = 0.1*math.pi*(epsilon**2) #0.1 is arbitrary, it make the hypothesis that plants have more than 0.1*surface covered by a circle of radius = epsilon white pixels
            # print("Epsilon/min_point : ", epsilon, min_point)
            
            print("--- DBSCAN ---")
            dataframe_coord = DBSCAN_clustering(img, epsilon, min_point, image)
            dataframe_coord.drop( #Remove all the pixels which have -1 as label value
                dataframe_coord[dataframe_coord["label"] == -1].index, inplace=True
            )
            Plot(dataframe_coord,0,image, "Clustering after {3} errosions and dilatations for {0} with eps = {1} and min_point = {2}".format(image, epsilon, min_point, image[0]))
            
            # print("--- OPTICS ---")
            # _time = time.time()
            # dataframe_coord_2 = OPTICS_clustering(img, epsilon, min_point, image)
            # print("Process time OPTICS : ", time.time() - _time)
            # dataframe_coord_2.drop( #Remove all the pixels which have -1 as label value
            #     dataframe_coord[dataframe_coord["label"] == -1].index, inplace=True
            # )
            
            # print(type(dataframe_coord))
            # os.chdir(r"D:/Datasets/Datasets rangs courbes/Champs_courbes_2/DIR_gt_DIP/DBScan_csv")
            # dataframe_coord.to_csv("{2}_eps{0}_mp{1}_DBSCAN.csv".format(epsilon, min_point,image[:-4]))
            
            # list_row.append(len(np.unique(dataframe_coord[["label"]].to_numpy()))-1)
            
            # list_row.append(len(np.unique(dataframe_coord[["label"]].to_numpy()))-1)

            # print("list_row : ", list_row)
            # print("dataframe_coord : ", dataframe_coord)
            # #print("len(dataframe_coord) : ", len(dataframe_coord))
            
            print("--- Plant_detection ---")
            JSON_final = Plants_Detection(
                dataframe_coord, e, max_iter, m_p, threshold, image
            )
            
            JSON_dataframe = Dataframe_to_Json(dataframe_coord)
            
            # print("--- write_json ---")
            # path_JSON_output = (
            #     _path_output_root.format(growth_stage)
            #     # + "/outputCL/session"
            #     # + str(session)
            #     # + "/Plant_CL_Predictions"
            # )
            # gIO.check_make_directory(path_JSON_output + "_3") #Ces deux là servent à stocker de novueau essais sans effacer les anciens (le premier pour la positions des plantes et le deuxieme pour le dataframe coord entier)
            # gIO.check_make_directory(path_JSON_output + "_4")
            # gIO.WriteJson(
            #     path_JSON_output + "_3",
            #     "Predicting_initial_plant_" + image.split(".")[0],
            #     JSON_final,
            # )
            
            # gIO.WriteJson(path_JSON_output + "_4",
            #               "Dataframe_coord_" + image.split(".")[0],
            #               JSON_dataframe)

    #return #Permet de limiter la boucle à une seule image pour le calibrage et les tests


if __name__ == "__main__":


    #for _stage in [0,1,2] :
        _stage = 2
        Total_Plant_Position(
            #  _path_input_root = r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\processed\Field_0\GrowthStage_{0}\Output\Session_1\Otsu_Erroded\RGB_58", #Pour Bis
            # _path_output_root= r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\processed\Field_0\GrowthStage_{0}\Output_general_clustering_Erroded", 
            _path_input_root = r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\processed\Field_0\GrowthStage_{0}\Output\Session_1\Otsu_erroded",
            _path_output_root= r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\processed\Field_0\GrowthStage_{0}\Output_general_clustering_Erroded", 
            _set = 2,
            session=1,
            growth_stage = _stage,
            epsilon= 90,
            min_point=80,
            e=0.05,
            max_iter=100,
            m_p=2,
            threshold=200, #1/1000 de la taille de l'image en pixel
            _RAs_group_size=20,
        )
