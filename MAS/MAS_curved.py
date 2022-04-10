# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import json
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import ttest_ind
import math
import statistics 
import random
import pandas as pd
from beautifultable import BeautifulTable


os.chdir("../Utility")
import general_IO as gIO
os.chdir("../Clustering_Scanner")
import Clustering 

# =============================================================================
# Utility Functions
# =============================================================================

def rotation_matrix(_theta):
    """
    Counter clock wise rotation matrix
    """
    return np.array([[np.cos(_theta), -np.sin(_theta)],
                     [np.sin(_theta),  np.cos(_theta)]])

def rotate_coord(_p, _pivot, _R):
    """
    gives the rotated coordinates of the point _p relatively to the _pivot point
    based on the rotation matrix _R.
    _p, _pivot and _R must be numpy arrays
    """
    _r_new_point = np.dot(_R, _p - _pivot) + _pivot

    return _r_new_point

# =============================================================================
# Agents Definition
# =============================================================================
class ReactiveAgent(object):
    """
    Agents pixels

    _RAL_x (int):
        leader column index in the image array
    _RAL_y (int):
        leader line index in the image array
    _local_x (int):
        column index relatively to the RAL
    _local_y (int):
        line index relatively to the RAL
    _img_array (numpy.array):
        array containing the image on which the Multi Agent System is working
    """

    def __init__(self,
                 _RAL_x, _RAL_y,
                 _local_x, _local_y,
                 _img_array):


        self.local_x = _local_x
        self.local_y = _local_y
        
        self.global_x = _RAL_x + _local_x
        self.global_y = _RAL_y + _local_y
        
        self.outside_frame = False

        self.img_array = _img_array

        self.Move_Based_On_RAL(_RAL_x, _RAL_y)

        self.decision = False

    def Otsu_decision(self):
        """
        Sets self.decision to True if the pixel where the RA is present is white.
        Sets to False otherwise.
        """
        if (self.img_array[self.global_y, self.global_x][0] > 220):
            self.decision = True
        else:
            self.decision = False


    def Move_Based_On_RAL(self, _RAL_x, _RAL_y):
        """
        Update the position of the RAL based on the order given by the AD (agent
        director).
        _ADO_x (int):
            X coordinate of the target point (column of the image array)

        _ADO_y (int):
            Y coordinate of the target point (line of the image array)
        """
        self.global_x = _RAL_x + self.local_x
        self.global_y = _RAL_y + self.local_y

        self.Is_Inside_Image_Frame()

    def Is_Inside_Image_Frame(self):

        if (self.global_x < 0 or
            self.global_x >= self.img_array.shape[1] or
            self.global_y < 0 or
            self.global_y >= self.img_array.shape[0]):

            self.outside_frame = True

        else:
            self.outside_frame = False


class ReactiveAgent_Leader(object):
    """
    Agent Plante

    _x (int):
        column index in the image array
    _y (int):
        lign index in the image array
    _img_array (numpy.array):
        array containing the image on which the Multi Agent System is working
    _group_size (int, optional with default value = 50):
        distance to the farthest layer of reactive agents from the RAL

    _group_step (int, optional with default value = 5):
        distance between two consecutive reactive agents

    _field_offset (list size 2, optional with default value [0, 0]):
        the offset to apply to all the positioned agents of the simulation to
        be coherent at the field level.

    """
    def __init__(self, _x, _y, _reduction_step, _img_array, _group_size = 50, _group_step = 5,
                 _field_offset = [0,0]):

# =============================================================================
#         print()
#         print("Initializing Reactive Agent Leader at position [{0},{1}]...".format(_x, _y), end = " ")
# =============================================================================

        self.x = int(_x)
        self.y = int(_y)
        self.img_array = _img_array
        self.group_size = _group_size
        self.group_step = _group_step
        self.correct_RAL_position()

        self.nb_contiguous_white_pixel = 0
        self.white_contigous_surface = 0

        self.field_offset = _field_offset

        self.decision = False

        self.active_RA_Point = np.array([self.x, self.y])
        self.movement_vector = np.zeros(2)

        self.recorded_positions = [[self.x, self.y]]
        self.field_recorded_positions = [[self.x + int(self.field_offset[0]),
                                          self.y + int(self.field_offset[1])]]

        self.used_as_filling_bound = False

        # curved
        self.neighbours = []
        self.other_neighbours = []
        self.end_of_row = False
# =============================================================================
#         print("Done")
#         print("Initializing the Reactive Agents under the RAL supervision...", end = " ")
# =============================================================================

        self.RAs_square_init()
        #self.RAs_circle_init()
        

        self.list_active_RAs = []
        self.Get_RAs_Otsu_Prop()
        self.recorded_Decision_Score = [self.decision_score]

# =============================================================================
#         print("Done")
# =============================================================================

    def correct_RAL_position(self):
        """
        adapt the self.x and self.y values (position of the RAL on the image)
        to avoid the instanciation of RAs outside the frame of the image
        """
        if (self.x-self.group_size < 0):
            self.x = self.group_size

        if (self.y-self.group_size < 0):
            self.y = self.group_size

        if (self.x+self.group_size > self.img_array.shape[1]):
            self.x = self.img_array.shape[1]-self.group_size

        if (self.y+self.group_size > self.img_array.shape[0]):
            self.y = self.img_array.shape[0]-self.group_size

    def RAs_square_init(self):
        """
        Instanciate the RAs
        """
        self.nb_RAs = 0
        self.RA_list = []
        for i in range (-self.group_size,
                        self.group_size+self.group_step,
                        self.group_step):

            for j in range (-self.group_size,
                            self.group_size+self.group_step,
                            self.group_step):

                _RA = ReactiveAgent(self.x, self.y, i, j, self.img_array)
                self.RA_list += [_RA]
                self.nb_RAs += 1

    def RAs_circle_init(self):
        self.nb_RAs = 0
        self.RA_list = []
        for i in range (-self.group_size,
                        self.group_size,
                        ):

            for j in range (-self.group_size,
                            self.group_size,
                            ):

                if i**2 + j**2 <= self.group_size**2 :
                    _RA = ReactiveAgent(self.x, self.y, i, j, self.img_array)
                    self.RA_list += [_RA]
                    self.nb_RAs += 1
            
    def Get_RAs_Otsu_Prop(self):
        """
        Computing the proportion of subordinates RAs that are positive
        """
        nb_true_votes = 0
        nb_outside_frame_RAs = 0
        for _RA in self.RA_list:
            if not _RA.outside_frame:
                _RA.Otsu_decision()
                if (_RA.decision):
                    nb_true_votes+=1
                    self.list_active_RAs += [_RA]
            else:
                nb_outside_frame_RAs += 1
        if self.nb_RAs - nb_outside_frame_RAs != 0 :
            self.decision_score = nb_true_votes/(self.nb_RAs-nb_outside_frame_RAs)
        else : 
            self.decision_score = 0
            
    def Get_RAL_Otsu_Decision(self, _threshold = 0.5):
        """
        Gathering the information from the RAs based on their Otsu decision
        """
        self.Get_RAs_Otsu_Prop()

        if (self.decision_score > _threshold):
            self.decision = True

    def Get_RAs_Mean_Point(self):
        """
        compute the mean point of the RAs that gave a positive answer to the
        stimuli
        """
        active_RA_counter = 0
        mean_x = 0
        mean_y = 0

        nb_outside_frame_RAs = 0
        for _RA in self.RA_list:
            if not _RA.outside_frame:
                _RA.Otsu_decision()
                if (_RA.decision):
                    mean_x += _RA.global_x
                    mean_y += _RA.global_y
                    active_RA_counter += 1
            else:
                nb_outside_frame_RAs += 1

        self.recorded_Decision_Score += [active_RA_counter/(self.nb_RAs-nb_outside_frame_RAs)]

        if (active_RA_counter != 0):
            self.active_RA_Point[0] = mean_x/active_RA_counter
            self.active_RA_Point[1] = mean_y/active_RA_counter


    def Move_Based_on_AD_Order(self, _ADO_x, _ADO_y):
        """
        Update the position of the RAL based on the order given by the AD (agent
        director).
        _ADO_x (int):
            X coordinate of the target point (column of the image array)

        _ADO_y (int):
            Y coordinate of the target point (line of the image array)
        """
        self.x = _ADO_x
        self.y = _ADO_y

        self.recorded_positions += [[int(self.x), int(self.y)]]
        self.field_recorded_positions += [[int(self.x + self.field_offset[0]),
                                           int(self.y + self.field_offset[1])]]

        for _RA in self.RA_list:
            _RA.Move_Based_On_RAL(self.x, self.y)

    def Do_Scores(self, _reduction_step) :
        self.Compute_Center_Score()
        self.Compute_Symetricity_Score()
    
        self.identity_vector = [self.center_score, self.symetricity_score]

        self.Compute_size_reduction_score(_reduction_step)

    def Compute_Center_Score(self):
        
        list_distances = []
        
        for _RA in self.list_active_RAs :
            a = _RA.local_x
            b = _RA.local_y
            distance = math.atan2(a,b)
            #distance = math.sqrt((self.x - _RA.global_x)**2 + (self.y - _RA.global_y)**2)
            list_distances.append(distance)
            
        mean_distance = sum(list_distances)/len(list_distances)
        
        mean_distance_precision = statistics.pstdev(list_distances)/math.sqrt(len(list_distances)) #permet de sur selectionner les RALs avec beaucoup de point et une petite moyenne
        
        self.center_score = mean_distance*mean_distance_precision #(sum(distances au centre)*écart-type)/taille de l'échantillon^1.5
        
        
    def Compute_Symetricity_Score(self):
        
        list_x_translations = []
        list_y_translations = [] 
        
        for _RA in self.list_active_RAs :
            list_x_translations.append(_RA.local_x)
            list_y_translations.append(_RA.local_y)
            
        self.symetricity_score = sum(list_x_translations) + sum(list_y_translations)
    
    def Compute_size_reduction_score(self, _reduction_step):
        
        # RAL_three_quarter_sized = ReactiveAgent_Leader(self.x, self.y, self.img_array, _group_size = int(0.75*self.group_size), do_scores = False)
        # RAL_half_sized = ReactiveAgent_Leader(self.x, self.y, self.img_array, _group_size = int(0.5*self.group_size), do_scores = False)
        # RAL_quarter_sized = ReactiveAgent_Leader(self.x, self.y, self.img_array, _group_size = int(0.25*self.group_size), do_scores = False)
        
        # self.three_quarter_RA_proportion = RAL_three_quarter_sized.decision_score
        # self.half_RA_proportion = RAL_half_sized.decision_score
        # self.quarter_RA_proportion = RAL_quarter_sized.decision_score
        
        
        for i in np.arange(0,1.0001,_reduction_step) : #A little more than 1 to not discriminate the 1 value when the step arrives to it
            new_RAL = ReactiveAgent_Leader(self.x, 
                                           self.y, 
                                           _reduction_step, 
                                           self.img_array, 
                                           _group_size = int(i*self.group_size))
            self.identity_vector.append(new_RAL.decision_score)
        #Il faudrait améliorer la fonction pour pouvoir initialiser automatiquement des RAL dans une range et avec un pas donné
        #De manière à pouvoir selectionner la précision de la réduction
        #Il faudrait aussi que les attributs soient créés et ajoutés à l'identity_vector automatiquement
        
        
    def Compute_Surface(self):
        """
        Counts the number of white pixels in the area scanned by the RAL. The
        search of white pixels uses the Pixel agents as seeds.
        """
        self.nb_contiguous_white_pixel = 0 #reset

        #print("self.group_size", self.group_size)
        square_width = 2*self.group_size+1
        surface_print=np.zeros((square_width,square_width))

        directions = [(0,1), (0,-1), (1,0), (-1,0)] #(x, y)

        explorers = []
        for _RA in self.RA_list:
            explorers += [(_RA.local_x, _RA.local_y)]

        nb_explorers=self.nb_RAs
        #print("nb_explorers", nb_explorers)
        #nb_op = 0
        while nb_explorers > 0:
            print_row = explorers[0][1]+self.group_size#row coord in surface print array
            print_col = explorers[0][0]+self.group_size#column coord in surface print array

            image_row = self.y+explorers[0][1]#row coord in image array
            image_col = self.x+explorers[0][0]#col coord in image array

            ####### curved: set the condition that fits regarding the orientation of the image
            # if image_row < 1920 and image_col < 1080: 
            if image_row < 1080 and image_col < 1920:          
                if (self.img_array[image_row][image_col][0] > 220):#if the pixel is white
                    surface_print[print_row][print_col]=2
                    self.nb_contiguous_white_pixel +=1
                    
                    for _d in directions:
                        if (0 <= print_row + _d[1] < square_width and
                            0 <= print_col + _d[0]< square_width):#if in the bounds of the surface_print array size
                            if (surface_print[print_row + _d[1]][print_col + _d[0]] == 0):#if the pixel has not an explorer already
                                
                                surface_print[print_row+_d[1]][print_col+_d[0]]=1#we indicate that we have added the coords to the explorers
                                
                                new_explorer_x = print_col-self.group_size + _d[0]
                                new_explorer_y = print_row-self.group_size + _d[1]
                                explorers += [(new_explorer_x, 
                                            new_explorer_y)]
                                nb_explorers += 1
            
            explorers = explorers[1:]
            nb_explorers -= 1

            #nb_op+=1
        self.white_contigous_surface = self.nb_contiguous_white_pixel/(square_width*square_width)
        #print(surface_print)
        #print("nb_white_pixels", self.nb_contiguous_white_pixel)
        #print("surface_white_pixels", self.white_contigous_surface)
    def Compute_neighbours_distances(self) :
        self.neighbours_distances = []
        for neighbour in self.neighbours : 
            euclidean_distance = math.sqrt((self.x - neighbour.x)**2 + (self.y - neighbour.y)**2)
            self.neighbours_distances.append(euclidean_distance)
    
    def Compute_closest_neighbour_translation(self) :
        Closest_neighbour = self.neighbours[self.neighbours_distances.index(min(self.neighbours_distances))]
        self.CN_translation_x = Closest_neighbour.x - self.x
        self.CN_translation_y = Closest_neighbour.y - self.y

class Row_Agent(object):
    """
    Agent rang de culture

    _plant_FT_pred_per_crop_rows (list of lists extracted for a JSON file):
        array containing the predicted position of plants organized by rows.
        The lists corresponding to rows contain other lists of length 2 giving
        the predicted position of a plant under the convention [image_line, image_column]

    _OTSU_img_array (numpy.array):
        array containing the OSTU segmented image on which the Multi Agent System
        is working

    _group_size (int, optional with default value = 5):
        number of pixels layers around the leader on which we instanciate
        reactive agents

    _group_step (int, optional with default value = 5):
        distance between two consecutive reactive agents

    _field_offset (list size 2, optional with default value [0, 0]):
        the offset to apply to all the positioned agents of the simulation to
        be coherent at the field level.

    """
    def __init__(self, _plant_FT_pred_in_crop_row, _OTSU_img_array, 
                 _group_size = 50, _group_step = 5,
                 _field_offset = [0,0], 
                 recon_policy="global"):
        
# =============================================================================
#         print()
#         print("Initializing Row Agent class...", end = " ")
# =============================================================================

        self.plant_FT_pred_in_crop_row = _plant_FT_pred_in_crop_row

        self.OTSU_img_array = _OTSU_img_array

        self.group_size = _group_size
        self.group_step = _group_step

        self.field_offset = _field_offset

        self.RALs = []

        self.extensive_init = False

        # curved rows
        self.recon_policy = recon_policy
        
# =============================================================================
#         print("Done")
# =============================================================================

        self.Initialize_RALs()

        self.Get_Row_Mean_X()


    def Initialize_RALs(self):
        """
        Go through the predicted coordinates of the plants in self.plant_FT_pred_par_crop_rows
        and initialize RALs at these places.

        """
# =============================================================================
#         print()
# =============================================================================
        
        for _plant_pred in self.plant_FT_pred_in_crop_row :
                RAL = ReactiveAgent_Leader(_x = _plant_pred[0],
                                           #_y = self.OTSU_img_array.shape[0] - _plant_pred[1],
                                           _y = _plant_pred[1],
                                           _reduction_step = 0,
                                           _img_array = self.OTSU_img_array,
                                           _group_size = self.group_size,
                                           _group_step = self.group_step,
                                           _field_offset = self.field_offset)
    
                self.RALs += [RAL]
                
    def Local_Exploration(self) :
 
            
        # _RAL_ref_index_last = -1
        # _RAL_ref_1 = self.RALs[_RAL_ref_index_last]
        
        # _RAL_ref_index_first = 0
        # _RAL_ref_2 = self.RALs[_RAL_ref_index_first]
        
        # _Row_direction_y = _RAL_ref_2.y - _RAL_ref_1.y
        # _Row_direction_x = _RAL_ref_1.x - _RAL_ref_2.x
        # slope = _Row_direction_y/_Row_direction_x
        # ordered_at_the_origin = _RAL_ref_1.y/(slope*_RAL_ref_1.x)
        # min_y = ordered_at_the_origin
        # max_y = self.OTSU_img_array.shape[0]*slope + ordered_at_the_origin
        
        # x_min = min([_RAL_ref_1.x,_RAL_ref_2.x])
        # x_max = max([_RAL_ref_1.x,_RAL_ref_2.x])
        
        list_x = []
        list_y = []
        for _RAL in self.RALs :
            if _RAL.end_of_row == True :
                list_x.append(_RAL.x)
                list_y.append(_RAL.y)
        x_min = list_x[0]
        min_y = list_y[0]
        x_max = list_x[1]
        max_y = list_y[1]
        
        return x_min, x_max, min_y, max_y
         
        
        

    def Edge_Exploration(self, _filling_step):
        """
        Uses the first and last RALs in the self.RALs list to extensively instanciate
        RALs at the edges of the rows.
        """

        _RAL_ref_index = -1
        _RAL_ref = self.RALs[_RAL_ref_index]

        y_init = _RAL_ref.y
        while y_init + _filling_step < self.OTSU_img_array.shape[0]:
            new_RAL = ReactiveAgent_Leader(_x = self.Row_Mean_X,
                                           _y = int(y_init + _filling_step),
                                           _reduction_step = 0,
                                           _img_array = self.OTSU_img_array,
                                           _group_size = self.group_size,
                                           _group_step = self.group_step,
                                           _field_offset = self.field_offset)
            new_RAL.used_as_filling_bound = True
            y_init += _filling_step

            self.RALs += [new_RAL]

        _RAL_ref_index = 0
        _RAL_ref = self.RALs[_RAL_ref_index]
        y_init = _RAL_ref.y
        new_RALs = []
        new_diffs = []
        while y_init - _filling_step > 0:
            new_RAL = ReactiveAgent_Leader(_x = self.Row_Mean_X,
                                           _y = int(y_init + _filling_step),
                                           _reduction_step = 0,
                                           _img_array = self.OTSU_img_array,
                                           _group_size = self.group_size,
                                           _group_step = self.group_step)
            new_RAL.used_as_filling_bound = True

            new_RALs += [new_RAL]
            new_diffs += [_filling_step]

            y_init -= _filling_step

        self.RALs = new_RALs + self.RALs

        a = np.array([RAL.y for RAL in self.RALs])
        b = np.argsort(a)
        self.RALs = list(np.array(self.RALs)[b])

    # curved
    def Compute_Distance_Matrix(self):
        """
        Compute the pairwise distance matrix between all RAL
        """
        # compute the distance matrix
        distance_matrix = np.zeros((len(self.RALs), len(self.RALs)))
        for i in range(len(self.RALs)):
            for j in range(len(self.RALs)):
# =============================================================================
#                 print((self.RALs[i].x, self.RALs[i].y),
#                       (self.RALs[j].x, self.RALs[j].y))
# =============================================================================
                if distance_matrix[i, j] == 0:
                    distance_matrix[i, j] = self.euclidean_distance(self.RALs[i], self.RALs[j])
        return distance_matrix

    # curved
    # def Check_irrelevant_neighbours(self, i ):
    #     problematic_neighbouring = []
        
    #     plt.figure(dpi=300)
    #     plt.imshow(self.OTSU_img_array)
        
    #     for _RAL in self.RALs :
    #         plt.scatter(_RAL.x, _RAL.y, color="g", marker = "+")
    #         print("_RAL.neighbours : ", _RAL, _RAL.neighbours)
    #         if len(_RAL.neighbours) != 2 :
    #             problematic_neighbouring.append(len(_RAL.neighbours))
                
    #             plt.scatter(_RAL.x, _RAL.y, color="pink", marker = "x")
    #             try : 
    #                 plt.scatter(_RAL.neighbours[0].x, _RAL.neighbours[0].y, color="lightgreen", marker = "x")
    #             except IndexError : 
    #                 pass
    #             try :
    #                 plt.scatter(_RAL.neighbours[0].x, _RAL.neighbours[0].y, color="lightgreen", marker = "x")
    #             except IndexError :
    #                 pass
                
    #             print("_RAL.neighbours : ", _RAL, _RAL.neighbours)
    #         try :   
    #             if _RAL.neighbours[0] == _RAL.neighbours[1] : 
    #                 problematic_neighbouring.append("same neighbours, {0}, {1}".format(_RAL, _RAL.neighbours[0]))
                    
    #                 plt.scatter(_RAL.x, _RAL.y, color="b", marker = "x")
    #                 try : 
    #                     plt.scatter(_RAL.neighbours[0].x, _RAL.neighbours[0].y, color="lightblue", marker = "x")
    #                 except ValueError : 
    #                     pass
    #                 try :
    #                     plt.scatter(_RAL.neighbours[0].x, _RAL.neighbours[0].y, color="lightblue", marker = "x")
    #                 except ValueError :
    #                     pass
    #         except IndexError :
    #             pass
             
    #         try : 
    #             if _RAL.neighbours[0] == _RAL or _RAL.neighbours[1] == _RAL :
    #                 problematic_neighbouring.append("_RAL same as neighbour")
                    
    #                 plt.scatter(_RAL.x, _RAL.y, color="r", marker = "x")
    #                 try : 
    #                     plt.scatter(_RAL.neighbours[0].x, _RAL.neighbours[0].y, color="lightred", marker = "x")
    #                 except ValueError : 
    #                     pass
    #                 try :
    #                     plt.scatter(_RAL.neighbours[0].x, _RAL.neighbours[0].y, color="lightred", marker = "x")
    #                 except ValueError :
    #                     pass
    #         except IndexError :
    #             pass
                
    #     plt.title("Rang {0}".format(i))
    #     plt.show()
                
            
    #     return problematic_neighbouring
    
    def Check_irrelevant_neighbours(self):

        for _RAL in self.RALs :
            if len(_RAL.neighbours) != 2 :
                print("len(_RAL.neighbours) = ", len(_RAL.neighbours))
                self.Compute_single_RAL_neighbours(_RAL)
                print("Solved : len(_RAL.neighbours) = ", len(_RAL.neighbours))
                break
 
            if _RAL.neighbours[0] == _RAL.neighbours[1] : 
                print("Same neighbours for RAL {1} : {0} / {0}".format(_RAL.neighbours[0], _RAL))
                self.Compute_single_RAL_neighbours(_RAL)
                print("Solved Same neighbours for_RAL {2}: {0} / {1}".format(_RAL.neighbours[0], _RAL.neighbours[1], _RAL))
                break
                        
            if _RAL.neighbours[0] == _RAL or _RAL.neighbours[1] == _RAL :
                print("Himself in his neighbours for RAL {2}: {0} / {1}".format(_RAL.neighbours[0], _RAL.neighbours[1], _RAL))
                self.Compute_single_RAL_neighbours(_RAL)
                print("Solved Himself in his neighbours for RAL {2}: {0} / {1}".format(_RAL.neighbours[0], _RAL.neighbours[1], _RAL))
                break
    
    def Get_End_Of_Row(self) : 
        Tot_translations = []
        for _RAL_1 in self.RALs : #A la fin de cette boucle on est censé avoir la somme des translations aux autres RALs du rang pour chaque RAL
            translations = [] 
            for _RAL_2 in self.RALs :
                if _RAL_1 != _RAL_2 : #Sa transaltion à lui même étant égale à 0 et identique pour tous les RALs, il n'est pas nécéssaire de la considérer
                    translation_x = _RAL_2.x - _RAL_1.x
                    translation_y = _RAL_2.y - _RAL_1.y
                    translation = math.atan2(translation_y,translation_x) #On calcule le sens de la translation
                    translations.append(translation*math.sqrt(translation_x**2 + translation_y**2)) #On le multiplie par la distance entre les deux RALs
            Tot_translations.append(sum(translations))
        #print("Tot_transations : ",Tot_translations)
        #print("Tot_transations.sort() : ",sorted(Tot_translations))
        #print("median(Tot_transations.sort()) : ", statistics.median(sorted(Tot_translations)))
        idx_1 = Tot_translations.index(max(Tot_translations)) #Celui avec la somme des translations la plus importante dans ]0,+infini[ est un bout de rang
        idx_2 = Tot_translations.index(min(Tot_translations)) #Idem pour celui avec la somme des translations la plus basse dans ]-infini, 0[ 
        self.RALs[idx_1].end_of_row = True #On stocke l'information dans l'attribut end_of_row des RALs concernés
        self.RALs[idx_2].end_of_row = True
        
        
        
        
        
        
#     def Set_RALs_Neighbours(self):
#         """
#         Here we link each RAL with its neighbours which are the two (maximum) closest RALs. One potential
#         issue comes when deciding whether an RAL is an extremity (then it should have only one neighbour)
#         or not (then it should have two). To do this, use the following thumb: if the second closest RAL
#         is in the same direction than the closest (NE, NW, SE, SW), then the target RAL is an extremity.
#         It is not a perfect rule but works 99% of the time for quasi-linear rows.
#         """
        
#         #print("In Set_RALs_Neighbours")
        
#         if len(self.RALs) < 2:
#             return

#         def get_direction_of_others_RAL(RAL, list_of_RALs):
#             """
#             Return the direction of each RAL in list_of_RALs compared with one RAL.
#             """
#             directions = []
#             for other in range(len(list_of_RALs)):
#                 direction = ""
#                 if list_of_RALs[other].x < RAL.x:
#                     direction += "W"
#                 else:
#                     direction += "E"
#                 if list_of_RALs[other].y < RAL.y:
#                     direction += "S"
#                 else:
#                     direction  += "N"
#                 directions.append(direction)
#             return directions        
        
#         # use to compute the neighbours
#         distance_matrix = self.Compute_Distance_Matrix()    
#         #print(distance_matrix)

#         for _RAL in self.RALs: # reinitialize the neighbours at each turn to avoid cumulating
#             _RAL.neighbours = []

#         # to avoid visiting the same neighbour twice
#         for i, _RAL in enumerate(self.RALs):
#             #print(i, self.RALs[i].neighbours)
#             if len(self.RALs[i].neighbours) == 2: # already found two neighbours
#                 continue

#             already_seen = [i]  # contains the idx of the already seen RAL, including itself
#             for n in _RAL.neighbours: # we already saw the neighbours that saw the RAL before
#                 idx = np.nonzero([self.RALs[k] == n for k in range(len(self.RALs))])[0]
#                 already_seen.extend([i for i in idx])

#             # get the remaining closest neighbours
#             direction_of_neighbours = get_direction_of_others_RAL(_RAL, self.RALs)  # record the direction (N, W, E, S) from RAL to neighbour 
#             for k in range(2 - len(_RAL.neighbours)): 

#                 # finding the next closest RAL
#                 mask = [True if n not in already_seen else False for n in range(distance_matrix.shape[0])]
#                 if True not in mask: # if we exhausted the RAL list
#                     break
#                 else:
#                     min_dist = np.min(distance_matrix[i, mask]) # distance to closest unseen neighbour
#                     closest_idx = np.nonzero(distance_matrix[i, :] == min_dist)[0] # get the index(s) of the closest neighbour
#                     close_idx = closest_idx[0]

#                 if _RAL.neighbours == []: # first neighbour found
#                     if self.RALs[close_idx] not in _RAL.neighbours:
#                         _RAL.neighbours.append(self.RALs[close_idx])
#                     if _RAL not in self.RALs[close_idx].neighbours and len(self.RALs[close_idx].neighbours) < 2:
#                         self.RALs[close_idx].neighbours.append(_RAL) # also add _RAL to its neighbour's list
#                     already_seen.append(close_idx)
                    
#                     #print(i, self.RALs[i].neighbours)
#                 else: # find the second neighbour
#                     first_neighbour_direction = direction_of_neighbours[already_seen[-1]]
#                     candidates = [True if n not in already_seen else False for n in range(distance_matrix.shape[0])]
                    
#                     while True in candidates:
#                         min_dist = np.min(distance_matrix[i, candidates]) # distance to closest unseen neighbour
#                         closest_idx = np.nonzero(distance_matrix[i, :] == min_dist)[0] # get the index(s) of the closest neighbour
#                         #print("closest_idx", closest_idx)
#                         # get the second closest neighbour for sure
# # =============================================================================
# #                         if closest_idx[0] not in already_seen:
# #                             close_idx = closest_idx[0]
# #                         else:
# #                             close_idx = closest_idx[-1]
# # =============================================================================
#                         for ii in closest_idx:
#                             if (ii not in already_seen):
#                                 close_idx = ii
#                                 break
#                         # the second closest is a neighbour of one RAL neighbour -> extremity
#                         is_closest_in_neighbour_neighbourhood = False
#                         for n in _RAL.neighbours:
#                             if self.RALs[close_idx] in n.neighbours:
#                                 is_closest_in_neighbour_neighbourhood = True
                        
#                         #print(direction_of_neighbours[close_idx], first_neighbour_direction, is_closest_in_neighbour_neighbourhood)
#                         if direction_of_neighbours[close_idx] != first_neighbour_direction and not is_closest_in_neighbour_neighbourhood:
#                             if self.RALs[close_idx] not in _RAL.neighbours and len(_RAL.neighbours) < 2:
#                                 _RAL.neighbours.append(self.RALs[close_idx])
#                             if _RAL not in self.RALs[close_idx].neighbours and len(self.RALs[close_idx].neighbours) < 2:
#                                 self.RALs[close_idx].neighbours.append(_RAL)
#                             break # the second neighbour has been found -> get out of the while loop

#                         already_seen.append(close_idx)
#                         candidates = [True if n not in already_seen else False for n in range(distance_matrix.shape[0])]
#                         #print(already_seen)

#             # if not len(_RAL.neighbours) <= 2:
#             #     print(i, _RAL.neighbours)
#             #     print(direction_of_neighbours)
#             #     self.Show_RALs_Position(title=f"Three neighbours or more for agent {i}")
#             assert(len(_RAL.neighbours) <= 2)

#     # curved
#     # def Sort_RALs(self):
#     #     """
#     #     The sorting stepis used initially, to have the RALs sorted in the right
#     #     order. This step is required since after we make several computations
#     #     using RALs indices (distances between i and i+1)
#     #     """
#     #     distance_matrix = self.Compute_Distance_Matrix()
        
#     #     # we sequentially visit the neighboors de proche en proche, on both sides of the origin agent and store the indices
#     #     origin = 0
#     #     closest_idx1 = np.argmin(distance_matrix[0, 1:]) + 1 # closest neighbour (without itself)
#     #     mask = [True if n > 0 and n != closest_idx1 else False for n in range(distance_matrix.shape[0])]
#     #     dist = np.min(distance_matrix[0, mask]) # second closest neighbour
#     #     closest_idx2 = np.argwhere(distance_matrix[0, :] == dist)[0, 0] # get the index of the second closest neighbour

#     #     # closest1 and closest2 are on each side of origin : we visit each side separatly and give negative indices to the left list (visited2) and positive to the 
#     #     # right list (visited1) 
#     #     if distance_matrix[closest_idx1, closest_idx2] > distance_matrix[origin, closest_idx1] and distance_matrix[closest_idx1, closest_idx2] > distance_matrix[origin, closest_idx2]:
#     #         visited1, visited2 = [origin], []  # origin is already visited, stored in the first list
#     #         ref1, ref2 = closest_idx1, closest_idx2 # we now examine the neighbours of closest_idx_1 and closest_idx_2
#     #         while len(visited1) + len(visited2) < len(self.RALs): # not visited all the RALs
#     #             for ref, visited in zip([ref1, ref2], [visited1, visited2]):
#     #                 closest_neighbour_dist = distance_matrix[origin, ref]
#     #                 memory = closest_neighbour_dist # memory helps to stop adding RALs to visited when we arrived at an extremity of the row
#     #                 non_visited_neighbours = [True if n not in visited1 and n not in visited2 else False for n in range(distance_matrix.shape[0])] # mask to get only non visited RALs
#     #                 while True in non_visited_neighbours:
#     #                     visited.append(ref)
#     #                     non_visited_neighbours = [True if n not in visited1 and n not in visited2 else False for n in range(distance_matrix.shape[0])] # mask to get only non visited RALs
#     #                     if not True in non_visited_neighbours: # when everythind is visited
#     #                         break
#     #                     closest_neighbour_dist = np.min(distance_matrix[ref, non_visited_neighbours]) # get closest neighbour among non visited RALs
#     #                     if closest_neighbour_dist > 2 * memory:  # TODO:improve this condition
#     #                         break
#     #                     closest_neighbour_idx = np.argwhere(distance_matrix[ref, :] == closest_neighbour_dist)[0, 0]  # get the closest neighbour idx in RALs
#     #                     ref = closest_neighbour_idx # pass to next RALs
#     #                     memory = closest_neighbour_dist
#     #         # build sorted list of RALs and assign it to self.RALs
#     #         sorted_RALs = [0 for i in range(len(self.RALs))] 
#     #         for i in range(len(self.RALs)):
#     #             if i < len(visited1):
#     #                 sorted_RALs[i] = self.RALs[visited1[i]]
#     #             else:
#     #                 sorted_RALs[i] = self.RALs[visited2[-(i - len(visited1) + 1)]] # count the elements in visited2 starting from the end (modular counting over the row)
#     #         self.RALs = sorted_RALs
#     #     else: # 0 is on the extremity of the row, so we do only one pass
#     #         visited = [origin, closest_idx1]
#     #         ref = closest_idx2
#     #         while len(visited) < len(self.RALs): # while not visited all the RALs
#     #             visited.append(ref)
#     #             non_visited_neighbours = [True if n not in visited else False for n in range(distance_matrix.shape[0])] # mask to get only non visited RALs
#     #             if not True in non_visited_neighbours:
#     #                 break
#     #             closest_neighbour_dist = np.min(distance_matrix[ref, non_visited_neighbours]) # get closest neighbour among non visited RALs
#     #             closest_neighbour_idx = np.argwhere(distance_matrix[ref, :] == closest_neighbour_dist)[0, 0]  # get the closest neighbour idx in RALs
#     #             ref = closest_neighbour_idx # pass to next RAL
#     #         sorted_RALs = [0 for i in range(len(self.RALs))] # visited is ordered so now the RALs are sorted in RALs.
#     #         for i in range(len(sorted_RALs)):
#     #             sorted_RALs[i] = self.RALs[visited[i]]
#     #         self.RALs = sorted_RALs

    # curved
    def Compute_single_RAL_neighbours(self, _RAL) :
        _RAL.neighbours = []
        distances = []
        list_theta = []
        list_RALs = []
        for _RAL_2 in self.RALs :
            euclidean_distance = math.sqrt((_RAL.x - _RAL_2.x)**2 + (_RAL.y - _RAL_2.y)**2)
            distances.append(euclidean_distance)
        #Récupération de l'index des voisins
        #print(distances)
        distances[distances.index(0)] = max(distances)+1 #Arbitraire pour fixer la valeur à une valeur trop grande pour que _RAL soit pris comme son propre voisin, +1 obligatoire pour les rang avec 3 plantes, les rangs avec 2 plantes vont poser problème
        idx_neighbour_1 = distances.index(min(distances))
        #print(distances)
        distances[idx_neighbour_1] = max(distances)+2 #Arbitraire pour fixer la valeur à une valeur trop grande pour être reprise pour le deuxième voisin
        idx_neighbour_2 = distances.index(min(distances))
        #print(distances)
        distances[idx_neighbour_2] = max(distances)+3 #Arbitraire pour fixer la valeur à une valeur trop grande pour être reprise pour le troisième voisin en cas de non linéarité entre les voisins
        
        #Check pour que les voisins enregistrés soient bien ceux qui sont à droite et à gauche du plant (pour éviter que les RALs initialisés sur des adventices, décalés par rapport au rang, soient pris comme voisins)
        # cos_theta = (self.RALs[idx_neighbour_1].x*self.RALs[idx_neighbour_2].x + self.RALs[idx_neighbour_1].y*self.RALs[idx_neighbour_2].y)/(math.sqrt(self.RALs[idx_neighbour_1].x**2 + self.RALs[idx_neighbour_1].y**2)*(math.sqrt(self.RALs[idx_neighbour_2].x**2 + self.RALs[idx_neighbour_2].y**2)))
        # theta_rad = math.acos(cos_theta)
        # theta_deg = theta_rad*(180/math.pi)
        # list_theta.append(theta_deg)
        # #print("Voisins envisagés 1/2: ", _RAL, self.RALs[idx_neighbour_1],self.RALs[idx_neighbour_2])
        # if (theta_deg > 165 and theta_deg < 195) or (theta_deg < 15 or theta_deg > 365) : 
        #     #Ajout des voisins dans l'attribut "voisins" de chaque RAL
        #     _RAL.neighbours.append(self.RALs[idx_neighbour_1])
        #     _RAL.neighbours.append(self.RALs[idx_neighbour_2])
        # else : 
        #     sorted_distances = sorted(distances)
        #     idx_neighbour_3 = distances.index(min(distances))
        #     #print(idx_neighbour_1, idx_neighbour_2, idx_neighbour_3)
            
        #     #Calcul de l'angle formé par le troisième RAL le plus proche et le premier voisin
        #     cos_theta_1 = (self.RALs[idx_neighbour_1].x*self.RALs[idx_neighbour_3].x + self.RALs[idx_neighbour_1].y*self.RALs[idx_neighbour_3].y)/(math.sqrt(self.RALs[idx_neighbour_1].x**2 + self.RALs[idx_neighbour_1].y**2)*(math.sqrt(self.RALs[idx_neighbour_3].x**2 + self.RALs[idx_neighbour_3].y**2)))
        #     theta_rad_1 = math.acos(cos_theta_1)
        #     theta_deg_1 = theta_rad_1*(180/math.pi)
            
        #     #Calcul de l'angle formé entre le troisième RAL le plus proche et le deuxième voisin
        #     cos_theta_2 = (self.RALs[idx_neighbour_2].x*self.RALs[idx_neighbour_3].x + self.RALs[idx_neighbour_2].y*self.RALs[idx_neighbour_3].y)/(math.sqrt(self.RALs[idx_neighbour_2].x**2 + self.RALs[idx_neighbour_2].y**2)*(math.sqrt(self.RALs[idx_neighbour_3].x**2 + self.RALs[idx_neighbour_3].y**2)))
        #     theta_rad_2 = math.acos(cos_theta_2)
        #     theta_deg_2 = theta_rad_2*(180/math.pi)
            
        #     if theta_deg_1 > theta_deg_2 :
        #         _RAL.neighbours.append(self.RALs[idx_neighbour_3])
        #         _RAL.neighbours.append(self.RALs[idx_neighbour_2])
        #     if theta_deg_1 < theta_deg_2 :
        #         _RAL.neighbours.append(self.RALs[idx_neighbour_1])
        #         _RAL.neighbours.append(self.RALs[idx_neighbour_3])
            
        #     list_RALs.append([_RAL,self.RALs[idx_neighbour_1],self.RALs[idx_neighbour_2]])
            
        
        #Assertion pour limiter les erreurs plus tard dans le MAS
        assert(_RAL.neighbours[0] != _RAL and _RAL.neighbours[1] != _RAL) #Le RAL ne fait pas lui même partie de ses voisins
        assert(len(_RAL.neighbours) == 2) #Le RAL a 2 voisins
        assert(_RAL.neighbours[0] != _RAL.neighbours[1]) #Les voisins ne sont pas identiques
    
    def Set_RALs_Neighbours(self):

        if len(self.RALs) >= 3 : #Pour éviter les problèmes de définition des voisins pour les rangs avec 1 ou 2 plantes seulement, normalement plus un problème avec la fixation du nombre de RALs par rang minimum en sortie de clustering à 3
            
            print("len(self.RALs) : ",len(self.RALs))
            list_theta = []
            list_RALs = []
            for _RAL in self.RALs : #On prend un RAL
                distances = []
                neighbours = _RAL.neighbours
                _RAL.neighbours = []
                for _RAL_2 in self.RALs : #Pour chaque autre RALs du rang
                    euclidean_distance = math.sqrt((_RAL.x - _RAL_2.x)**2 + (_RAL.y - _RAL_2.y)**2) #On calcule la distance entre les deux
                    if euclidean_distance == 0 and _RAL != _RAL_2: #Condition pour afficher les RALs impliqués dans un des cas où deux RALs étaient initialisés à la même position, peut être plus d'actualité
                        print("RALs différents avec une distance de 0 : ", _RAL, _RAL_2)
                        
                        plt.figure(dpi=300)
                        plt.imshow(self.OTSU_img_array)
                        for _RALs in self.RALs :
                            plt.scatter(_RALs.x, _RALs.y, color = "g", marker = "+")
                        plt.scatter(_RAL.x, _RAL.y, color ="r", marker = "x")
                        plt.scatter(neighbours[0].x, neighbours[0].y, color ="b", marker = "x")
                        plt.scatter(neighbours[1].x, neighbours[1].y, color ="b", marker = "x")
                        plt.title("Rouge = double RAL")
                        plt.show()
                        
                    distances.append(euclidean_distance) #La distance est ajoutée à la liste compléte des distances entre un RAL et tous les autres RALs du rang
                
                #Récupération de l'index des voisins
                #print(distances)
                distances[distances.index(0)] = max(distances)+1 #Arbitraire pour fixer la valeur à une valeur trop grande pour que _RAL soit pris comme son propre voisin, +1 obligatoire pour les rang avec 3 plantes
                idx_neighbour_1 = distances.index(min(distances))
                #print(distances)
                distances[idx_neighbour_1] = max(distances)+2 #Arbitraire pour fixer la valeur à une valeur trop grande pour être reprise pour le deuxième voisin
                idx_neighbour_2 = distances.index(min(distances))
                #print(distances)
                distances[idx_neighbour_2] = max(distances)+3 #Arbitraire pour fixer la valeur à une valeur trop grande pour être reprise pour le troisième voisin en cas de non linéarité entre les voisins
                
                
                #La partie de fonction commentée qui suit n'est utile que dans le cas d'images sur lesquelles ont aurait pas réussi à erroder les adventices. De plus, même avec des adventices, ce processus ne permettait pas de régler tous les problèmes et en créait même d'autres
                #Check pour que les voisins enregistrés soient bien ceux qui sont à droite et à gauche du plant (pour éviter que les RALs initialisés sur des adventices, décalés par rapport au rang, soient pris comme voisins)
                # cos_theta = (self.RALs[idx_neighbour_1].x*self.RALs[idx_neighbour_2].x + self.RALs[idx_neighbour_1].y*self.RALs[idx_neighbour_2].y)/(math.sqrt(self.RALs[idx_neighbour_1].x**2 + self.RALs[idx_neighbour_1].y**2)*(math.sqrt(self.RALs[idx_neighbour_2].x**2 + self.RALs[idx_neighbour_2].y**2)))
                # theta_rad = math.acos(cos_theta)
                # theta_deg = theta_rad*(180/math.pi)
                # list_theta.append(theta_deg)
                # _RAL.neighbours.append(self.RALs[idx_neighbour_1])
                # _RAL.neighbours.append(self.RALs[idx_neighbour_2])
                # #print("Voisins envisagés 1/2: ", _RAL, self.RALs[idx_neighbour_1],self.RALs[idx_neighbour_2])
                # if (theta_deg > 165 and theta_deg < 195) or (theta_deg < 15 or theta_deg > 365) : 
                #     #Ajout des voisins dans l'attribut "voisins" de chaque RAL
                #     _RAL.neighbours.append(self.RALs[idx_neighbour_1])
                #     _RAL.neighbours.append(self.RALs[idx_neighbour_2])
                # else : 
                #     sorted_distances = sorted(distances)
                #     idx_neighbour_3 = distances.index(min(distances))
                #     #print(idx_neighbour_1, idx_neighbour_2, idx_neighbour_3)
                    
                #     #Calcul de l'angle formé par le troisième RAL le plus proche et le premier voisin
                #     cos_theta_1 = (self.RALs[idx_neighbour_1].x*self.RALs[idx_neighbour_3].x + self.RALs[idx_neighbour_1].y*self.RALs[idx_neighbour_3].y)/(math.sqrt(self.RALs[idx_neighbour_1].x**2 + self.RALs[idx_neighbour_1].y**2)*(math.sqrt(self.RALs[idx_neighbour_3].x**2 + self.RALs[idx_neighbour_3].y**2)))
                #     theta_rad_1 = math.acos(cos_theta_1)
                #     theta_deg_1 = theta_rad_1*(180/math.pi)
                    
                #     #Calcul de l'angle formé entre le troisième RAL le plus proche et le deuxième voisin
                #     cos_theta_2 = (self.RALs[idx_neighbour_2].x*self.RALs[idx_neighbour_3].x + self.RALs[idx_neighbour_2].y*self.RALs[idx_neighbour_3].y)/(math.sqrt(self.RALs[idx_neighbour_2].x**2 + self.RALs[idx_neighbour_2].y**2)*(math.sqrt(self.RALs[idx_neighbour_3].x**2 + self.RALs[idx_neighbour_3].y**2)))
                #     theta_rad_2 = math.acos(cos_theta_2)
                #     theta_deg_2 = theta_rad_2*(180/math.pi)
                    
                #     if theta_deg_1 > theta_deg_2 :
                #         _RAL.neighbours.append(self.RALs[idx_neighbour_3])
                #         _RAL.neighbours.append(self.RALs[idx_neighbour_2])
                #     if theta_deg_1 < theta_deg_2 :
                #         _RAL.neighbours.append(self.RALs[idx_neighbour_1])
                #         _RAL.neighbours.append(self.RALs[idx_neighbour_3])
                    
                #     list_RALs.append([_RAL,self.RALs[idx_neighbour_1],self.RALs[idx_neighbour_2]])
                    
                
                #Assertion pour limiter les erreurs plus tard dans le MAS
                assert(_RAL.neighbours[0] != _RAL and _RAL.neighbours[1] != _RAL) #Le RAL ne fait pas lui même partie de ses voisins
                assert(len(_RAL.neighbours) == 2) #Le RAL a 2 voisins
                assert(_RAL.neighbours[0] != _RAL.neighbours[1]) #Les voisins ne sont pas identiques
            # if len(self.RALs) > 40 :
            #     plt.figure(dpi=300)
            #     plt.imshow(self.OTSU_img_array)
            #     for _RALs in self.RALs :
            #         if _RALs.end_of_row :
            #             plt.scatter(_RALs.x, _RALs.y, color = "b", marker = "+")
            #         else :
            #             plt.scatter(_RALs.x, _RALs.y, color = "g", marker = "+")
            #     plt.title("Après le premier appel de Set_RALs_Neighbours")
            #     plt.show()
            return list_theta, list_RALs
    
    # def Fuse_RALs(self, RAL1, RAL2):
    #     """
    #     Fuse two RALs, by initializing a new RAL at the barycenter of them. We cannot
    #     use indices to fuse RALs in curved mode since adjacent RALs have not necessarily
    #     contiguous neighbours
    #     """
    #     if (RAL2 not in RAL1.neighbours and RAL1 not in RAL2.neighbours):
    #         RAL1_idx = np.nonzero(self.RALs[k] == RAL1 for k in range(len(self.RALs)))
    #         RAL2_idx = np.nonzero(self.RALs[k] == RAL2 for k in range(len(self.RALs)))
    #         self.Show_RALs_Position(title=f"Fusing non neighbours agents : {RAL1_idx} and {RAL2_idx} at position : ({RAL1.x}, {RAL1.y})")

    #     if (RAL2 == RAL1):
    #         RAL1_idx = np.nonzero(self.RALs[k] == RAL1 for k in range(len(self.RALs)))
    #         self.Show_RALs_Position(title=f"Fusing one agent with himself : agent {RAL1_idx} at position : ({RAL1.x}, {RAL1.y})")

    #     fusion_RAL_x = (RAL1.x + RAL2.x) / 2
    #     fusion_RAL_y = (RAL1.y + RAL2.y) / 2
            
    #     fusion_RAL = ReactiveAgent_Leader(_x = int(fusion_RAL_x),
    #                                        _y = int(fusion_RAL_y),
    #                                        _reduction_step = 0,
    #                                        _img_array = self.OTSU_img_array,
    #                                        _group_size = self.group_size,
    #                                        _group_step = self.group_step)
        
    #     if (RAL1.used_as_filling_bound and
    #         RAL2.used_as_filling_bound):
    #             fusion_RAL.used_as_filling_bound = True
        
    #     # update new RAL neighbours : its neighbours are the ones of the previous two RAL
    #     for n in RAL1.neighbours:
    #         if n != RAL2:
    #             fusion_RAL.neighbours.append(n)
    #             n.neighbours.append(fusion_RAL)
    #             #print("RAL1.neighbours : ", n)

    #     for n in RAL2.neighbours:
    #         if n != RAL1:
    #             fusion_RAL.neighbours.append(n)
    #             n.neighbours.append(fusion_RAL)
    #             #print("RAL2.neighbours : ", n)
    #     #print("fusion_RAL.neighbours : ", fusion_RAL.neighbours)
    #     # remove RAL1 and RAL2 from self.RALs
    #     self.RALs.append(fusion_RAL)
    #     if RAL1 in self.RALs:
    #         self.RALs.remove(RAL1)
    #     if RAL2 in self.RALs:
    #         self.RALs.remove(RAL2)

    #     # remove RAL1 and RAL2 from their neighbours' neighbours list
    #     for _RAL in self.RALs:
    #         if RAL1 in _RAL.neighbours:
    #             _RAL.neighbours.remove(RAL1)
    #         if RAL2 in _RAL.neighbours:
    #             _RAL.neighbours.remove(RAL2)

    #     for n in fusion_RAL.neighbours:
    #         assert (n in self.RALs)

    #     # print(f"Fusion agent {len(self.RALs) - 1} at position : ({fusion_RAL.x},{ fusion_RAL.y})")
    #     # self.Show_RALs_Position(title=f"Fusion agent {len(self.RALs) - 1} at position : ({fusion_RAL.x},{ fusion_RAL.y})")

    # curved
    # def Fill_RALs(self, RAL1, RAL2, _filling_step):
    #     """
    #     The new RALs are initialized on a straight line between RAL1 and RAL2, and evenly spaced.
    #     The step's composantes are computed along each axis (step_x and step_y variables).
    #     """
        
    #     if (not RAL1.used_as_filling_bound or
    #         not RAL2.used_as_filling_bound):
    #         x_0, x_f = RAL1.x, RAL2.x 
    #         y_0, y_f = RAL1.y, RAL2.y
    #         new_RALs = []
    #         nb_new_RALs = 0
    
    #         def get_direction(x_init, x_final):
    #             if x_final - x_init >= 0:
    #                 return 1
    #             else:
    #                 return -1
            
    #         step_x = np.abs(RAL2.x - RAL1.x) / self.euclidean_distance(RAL1, RAL2) * (_filling_step / 2)
    #         step_y = np.abs(RAL2.y - RAL1.y) / self.euclidean_distance(RAL1, RAL2) * (_filling_step / 2)
    #         k_x, k_y = 1, 1
    #         delta_x, delta_y = get_direction(x_0, x_f) * step_x, get_direction(y_0, y_f) * step_y
    
    #         previous_RAL = RAL1
    
    #         while self.euclidean_distance(previous_RAL, RAL2) >= _filling_step \
    #                and ((x_0 <= x_0 + k_x * delta_x <= x_f) or (x_f <= x_0 + k_x * delta_x <= x_0)) \
    #                and ((y_0 <= y_0 + k_y * delta_y <= y_f) or (y_f <= y_0 + k_y * delta_y <= y_0)):
    
    #             new_RAL = ReactiveAgent_Leader(_x = int(x_0 + (k_x) * delta_x),
    #                                 _y = int(y_0 + (k_y) * delta_y),
    #                                 _reduction_step = 0,
    #                                 _img_array = self.OTSU_img_array,
    #                                 _group_size = self.group_size,
    #                                 _group_step = self.group_step)
                
    #             new_RAL.used_as_filling_bound = True
                
    #             new_RALs.append(new_RAL)
    #             nb_new_RALs += 1
    
    #             # update the neighbours
    #             previous_RAL.neighbours.append(new_RAL)
    #             new_RAL.neighbours.append(previous_RAL)
    #             previous_RAL = new_RAL
    
    #             k_x += 1
    #             k_y += 1
    #             # update the deltas
    #             delta_x, delta_y = get_direction(x_0 + k_x * step_x, x_f) * step_x, get_direction(y_0 + k_y * step_y, y_f) * step_y
    #             #print(f"step_x = {step_x}, step_y = {step_y}")
    #             #print(f"new_x : {x_0 + k_x * delta_x}, new_y : {y_0 + k_y * delta_y}")
    #             #print(f"New distance : {np.sqrt((x_f - (x_0 + k_x * delta_x)) ** 2 + (y_f - (x_0 + k_y * delta_y)) ** 2)}")
            
    #         RAL1.used_as_filling_bound = True
    #         RAL2.used_as_filling_bound = True
            
    #         # close neighbours updates
    #         if (nb_new_RALs > 0):
    #             # to update neighbours of RAL1 and RAL2
    #             if RAL2 in RAL1.neighbours:
    #                 RAL1.neighbours.remove(RAL2)
    #             if RAL1 in RAL2.neighbours:
    #                 RAL2.neighbours.remove(RAL1)
    
    #             # link the last initialized RAL with RAL2
    #             RAL2.neighbours.append(new_RALs[-1])
    #             new_RALs[-1].neighbours.append(RAL2)
                
    #             # print(f"Initialized {len(new_RALs)} new RALs, indices : {len(self.RALs)}  to {len(self.RALs) + len(new_RALs) - 1} at position : ({new_RALs[-1].x},{new_RALs[-1].y}).")
    #             self.RALs.extend(new_RALs)
    #             # print("\n")
    
    #             assert (RAL1 in new_RALs[0].neighbours and RAL2 in new_RALs[-1].neighbours
    #                     and new_RALs[0] in RAL1.neighbours and new_RALs[-1] in RAL2.neighbours)
            
    #         # if len(new_RALs) != 0:
    #             # self.Show_RALs_Position(title=f"Initialized {len(new_RALs)} new RALs, indices : {len(self.RALs) - len(new_RALs)}  to {len(self.RALs) - 1} at position : ({new_RALs[-1].x},{new_RALs[-1].y}).")

    # curved
#     def Fill_or_Fuse_RALs(self, _crit_value, _fuse_factor = 0.5, _fill_factor = 1.5):
#         """
#         Calls the Fill and Fuse mechanisms for each RAL of the Row. 
#         _crit_value (float) : The inter-plant distance
#         _fuse_factor (float) : fraction of the _crit_value under which two RALs will be fused
#         _fill_factor (float) : Fraction of the _crit_value over which new RALs will be 
#         instantiated between two RALs
#         """
#         # potentiellmeent dangereux : peut tendre vers 0 et initialiser trop d'agents...
#         # Mais meilleurs resultats empiriques lorsque fill_factor > 1.1
# # =============================================================================
# #         if len(self.RALs) > 1:
# #             d = []
# #             for _RAL in self.RALs:
# #                 for n in _RAL.neighbours:
# #                     d.append(self.euclidean_distance(_RAL, n))
# #             _crit_value = np.median(d)
# #         else:
# #             return
# # =============================================================================

#         nb_RALs = len(self.RALs)
#         i = 0
#         while i < nb_RALs-1:
#             #print(f"Analyzing agent {i}")            
#             has_been_fused = False

#             for n in self.RALs[i].neighbours:
#                 try:
#                     neighbour_idx = np.nonzero([self.RALs[k] == n for k in range(len(self.RALs))])[0][0] 
#                 except IndexError:
#                     print(f"Index Error : {i}")
#                     self.Show_RALs_Position(title="Index Error")
#                 #print(f"Analyzing agent {neighbour_idx}, neighbour of {i} for FUSING.")
#                 #print(f"estimated distance is {self.euclidean_distance(self.RALs[i], n)}.Y compared with {0.9 * _fuse_factor * _crit_value}")
#                 if self.euclidean_distance(self.RALs[i], n) < 0.9 * _fuse_factor * _crit_value:
#                     #print(f"Fusing agents : {i} and {neighbour_idx} into {len(self.RALs) - 2} into agent {len(self.RALs) - 1} at position ({self.RALs[i].x}, {self.RALs[i].y})")
#                     #print (_crit_value)
#                     self.Fuse_RALs(self.RALs[i], n)
#                     has_been_fused = True
#                     break
            
#             if has_been_fused:  # if the RAL has been fused to another, do not use it to detect filling
#                 # print(f"just fused: {i} with {neighbour_idx}")
#                 nb_RALs = len(self.RALs)
#                 continue
            
#             for n in self.RALs[i].neighbours:
#                 #neighbour_idx = np.nonzero([self.RALs[k] == n for k in range(len(self.RALs))])[0][0]
#                 # print(f"Analyzing agent {neighbour_idx}, neighbour of {i} for FILLING")
#                 # print(f"estimated distance is {self.euclidean_distance(self.RALs[i], n)} compared with {1.1 * _fill_factor * _crit_value}")
#                 if self.euclidean_distance(self.RALs[i], n) >= int(1.1 * _fill_factor * _crit_value):
#                     #print(f"Initializing a RAL between agent {neighbour_idx} and agent {i}...")
#                     #print("\n")
#                     self.Fill_RALs(self.RALs[i], n, int(1.1 * _fill_factor * _crit_value))
            
#             i += 1
#             nb_RALs = len(self.RALs)
    
    def Fill_or_Fuse_RALs(self, _crit_value, _fuse_factor = 0.5, _fill_factor = 1.5):
        for _RAL in self.RALs :
            for neighbour in _RAL.neighbours :
                euclidean_distance = math.sqrt((_RAL.x - neighbour.x)**2 + (_RAL.y - neighbour.y)**2)
                if euclidean_distance < _crit_value*_fuse_factor :
                    self.Fuse_RALs(_RAL, neighbour)
                if euclidean_distance > _crit_value*_fill_factor :
                    if _RAL.end_of_row == False and neighbour.end_of_row == False : #Condition pour éviter d'initialiser en boucle entre un bout de rang et son voisin le plus éloigné
                        self.Fill_RALs(_RAL, neighbour)
                    
    
    def Fill_RALs(self, _RAL, neighbour) :
        new_RAL = new_RAL = ReactiveAgent_Leader(_x = (neighbour.x + _RAL.x)/2,
                                         _y = (neighbour.y + _RAL.y)/2,
                                         _reduction_step = 0,
                                         _img_array = self.OTSU_img_array,
                                         _group_size = self.group_size,
                                         _group_step = self.group_step
                                         )
        
        _RAL.neighbours.pop(_RAL.neighbours.index(neighbour)) #On enlève de la liste des voisins du RAL le voisin qui sert à la nouvelle initialisation
        _RAL.neighbours.append(new_RAL) #On ajoute le nouveau RAL à la place du voisin qu'on vient de retirer 
        
        try : 
            neighbour.neighbours.pop(neighbour.neighbours.index(_RAL)) #On enlève le RAL de la liste des voisins de l'autre plante qui sert à l'initialisation 
            neighbour.neighbours.append(new_RAL) #Si le RAL parent fait bien partie des voisins du second RAL parent, on ajoute le nouveau RAL aux voisins du second RAL parent
        except ValueError :
            #print("neighbour.neighbours : ", neighbour, neighbour.neighbours)
            plt.figure(dpi=300)
            plt.imshow(self.OTSU_img_array)
            plt.scatter(neighbour.x, neighbour.y, color="g")
            plt.scatter(_RAL.x, _RAL.y, color="r")
            plt.scatter(neighbour.neighbours[0].x, neighbour.neighbours[0].y, color="lightgreen")
            plt.scatter(neighbour.neighbours[1].x, neighbour.neighbours[1].x, color="lightgreen")
            plt.scatter(_RAL.neighbours[0].x,_RAL.neighbours[0].y, color="pink")
            plt.scatter(_RAL.neighbours[1].x,_RAL.neighbours[1].y, color="pink")
            plt.scatter(new_RAL.x, new_RAL.y, color="orange")
            plt.show()
            neighbour.other_neighbours.append(new_RAL) #Si le premier RAL parent ne fait pas partie des voisins du second RAL parent, on ajoute le nouveau RAL dans les autres voisins du second RAL parent
            try : 
               neighbour.other_neighbours.pop(neighbour.other_neighbours.index(_RAL)) #On essaye d'enlever le RAL de la liste des autres voisins de l'autre plante qui sert à l'initialisation  
            except ValueError :
                pass
            
        new_RAL.neighbours.append(_RAL) #Le nouveau RAL prend comme voisins les deux RALs parents
        new_RAL.neighbours.append(neighbour)
        
        self.RALs.append(new_RAL)
    
    def Fuse_RALs(self, _RAL, neighbour) : 
            new_RAL = ReactiveAgent_Leader(_x = (neighbour.x + _RAL.x)/2,
                                                     _y = (neighbour.y + _RAL.y)/2,
                                                     _reduction_step = 0,
                                                     _img_array = self.OTSU_img_array,
                                                     _group_size = self.group_size,
                                                     _group_step = self.group_step
                                                     )
            
            if _RAL.decision_score > (neighbour.decision_score + new_RAL.decision_score)/2 : 
                print("_RAL.neighbours Fuse : " ,_RAL, _RAL.neighbours)
                try : 
                    self.RALs.pop(self.RALs.index(neighbour)) #On enlève le RAL non conservé de la liste des plantes du rang
                except ValueError :
                    plt.figure(dpi=300)
                    plt.imshow(self.OTSU_img_array)
                    plt.scatter(neighbour.x, neighbour.y, color="g")
                    plt.scatter(_RAL.x, _RAL.y, color="r")
                    plt.scatter(neighbour.neighbours[0].x, neighbour.neighbours[0].y, color="lightgreen")
                    plt.scatter(neighbour.neighbours[1].x, neighbour.neighbours[1].x, color="lightgreen")
                    plt.scatter(_RAL.neighbours[0].x,_RAL.neighbours[0].y, color="pink")
                    plt.scatter(_RAL.neighbours[1].x,_RAL.neighbours[1].y, color="pink")
                    plt.scatter(new_RAL.x, new_RAL.y, color="orange")
                    plt.title("Neighbour not in self.RALs list")
                    plt.show()
                _RAL.neighbours.pop(_RAL.neighbours.index(neighbour)) #On enlève de la liste des voisins du RAL conservé le RAL non conservé
                for neighbours in neighbour.neighbours :
                    if neighbours != _RAL : #Ne prend comme nouveau voisin que le voisin qui n'est pas le RAL conservé
                        _RAL.neighbours.append(neighbours) #On ajoute à la liste des voisins du RAL conservé le voisin du RAL supprimé
                        try :
                            neighbours.neighbours.pop(neighbours.neighbours.index(neighbour)) #On enlève le RAL supprimé de la liste des voisins de son autre voisin
                            neighbours.neighbours.append(_RAL) #On ajoute à la place le RAL conservé
                        except ValueError : #Si le RAL supprimé n'était pas dans les voisins de son voisin, le RAL conservé est ajouté aux other_neighbours 
                            neighbours.other_neighbours.append(_RAL)
                            try :
                                neighbours.other_neighbours.pop(neighbours.other_neighbours.index(neighbour)) #On enlève le RAL supprimé de la liste des voisins de son autre voisin
                            except ValueError : 
                                pass
                if neighbour.end_of_row == True : #Si la plante fusionne avec une plante en bout de rang, place son statut comme bout de rang
                    _RAL.end_of_row = True
                return
                
            if neighbour.decision_score > (_RAL.decision_score + new_RAL.decision_score)/2 :
                self.RALs.pop(self.RALs.index(_RAL)) #On enlève le RAL non conservé de la liste des plantes du rang
                neighbour.neighbours.pop(neighbour.neighbours.index(_RAL)) #On enlève le RAL non conservé de la liste des voisins du RAL conservé 
                for neighbours in _RAL.neighbours :
                    if neighbours != neighbour : #Ne prend comme nouveau voisin que le voisin qui n'est pas le RAL conservé
                        neighbour.neighbours.append(neighbours) 
                        try :
                            neighbours.neighbours.pop(neighbours.neighbours.index(_RAL)) #On enlève le RAL supprimé de la liste des voisins de son autre voisin
                            neighbours.neighbours.append(neighbour) #On ajoute à la place le RAL conservé
                        except ValueError : 
                           neighbours.other_neighbours.append(neighbour) 
                           try : 
                               neighbours.other_neighbours.pop(neighbours.other_neighbours.index(_RAL)) #On enlève le RAL supprimé de la liste des voisins de son autre voisin
                           except ValueError :
                               pass
                if _RAL.end_of_row == True :
                    neighbour.end_of_row = True
                    
                return
            

            if new_RAL.decision_score > (neighbour.decision_score + _RAL.decision_score)/2 :   
                self.RALs.pop(self.RALs.index(neighbour))
                self.RALs.pop(self.RALs.index(_RAL))
                self.RALs.append(new_RAL)
                
                if _RAL.end_of_row == True or neighbour.end_of_row == True :
                    new_RAL.end_of_row = True
                    
                for neighbours in _RAL.neighbours :
                    if neighbours != neighbour :
                        new_RAL.neighbours.append(neighbours)   
                        try : 
                            neighbours.neighbours.pop(neighbours.neighbours.index(_RAL))
                            neighbours.neighbours.append(new_RAL)
                        except ValueError :
                            neighbours.other_neighbours.append(new_RAL)
                            try : 
                                neighbours.other_neighbours.pop(neighbours.other_neighbours.index(_RAL))
                            except ValueError :
                                pass
                for neighbours in neighbour.neighbours : 
                    if neighbours != _RAL :
                        new_RAL.neighbours.append(neighbours)   
                        try : 
                            neighbours.neighbours.pop(neighbours.neighbours.index(neighbours))
                            neighbours.neighbours.append(new_RAL)
                        except ValueError :
                            neighbours.other_neighbours.append(new_RAL)
                            try :
                                neighbours.other_neighbours.pop(neighbours.other_neighbours.index(neighbours))
                            except ValueError :
                                pass
                return

    def Show_RALs_Position(self,
                           _ax = None,
                           _color = 'g',
                           title = ""):
        """
        Display the Otsu image with overlaying rectangles centered on RALs. The
        size of the rectangle corespond to the area covered by the RAs under the 
        RALs supervision.
        
        _ax (matplotlib.pyplot.axes, optional):
            The axes of an image on which we wish to draw the adjusted 
            position of the plants
        
        _color (optional,list of color references):
            The color of the rectangle representing a RAL.
        """
        
        if (_ax == None):
            fig, ax = plt.subplots(1)
            ax.imshow(self.OTSU_img_array)
        else:
            ax = _ax

        for _RAL in self.RALs:
            #coordinates are the upper left corner of the rectangle
            rect = patches.Rectangle((_RAL.x-_RAL.group_size,
                                      _RAL.y-_RAL.group_size),
                                      2*_RAL.group_size,
                                      2*_RAL.group_size,
                                     linewidth=1,
                                     edgecolor=_color,
                                     facecolor='none')
            ax.add_patch(rect)
        
# =============================================================================
#         plt.xlim(170, 270)
#         plt.ylim(1350, 1450)
# =============================================================================

    def Show_RALs_Position_2(self,
                           _ax = None,
                           _recorded_position_indeces = [0, -1],
                           _colors = ['r', 'g'], title = "" ):
        """
        Display the Otsu image with overlaying rectangles centered on RALs. The
        size of the rectangle corespond to the area covered by the RAs under the 
        RALs supervision.
        
        _ax (matplotlib.pyplot.axes, optional):
            The axes of an image on which we wish to draw the adjusted 
            position of the plants
        
        _recorded_position_indeces (optional,list of int):
            indeces of the recored positions of the RALs we wish to see. By defaullt,
            the first and last one
        
        _colors (optional,list of color references):
            Colors of the rectangles ordered indentically to the recorded positons
            of interest. By default red for the first and green for the last 
            recorded position.
        """
        
        if (_ax == None):
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(self.OTSU_img_array)
        else:
            ax = _ax
        
        # nb_indeces = len(_recorded_position_indeces)

        _colors = ["r", "g", "b", "c", "m", "y", "darkorange", "lime", "royalblue",
                   "mediumslateblue", "mediumpurple", "plum", "violet", "crimson",
                   "dodgerblue", "chartreuse", "peru", "lightcoral"]
        #already_seen = []
        for j, _RAL in enumerate(self.RALs):    
            # for k in range (nb_indeces):
            rect = patches.Rectangle((_RAL.x-_RAL.group_size,
                                      _RAL.y-_RAL.group_size),
                                      2*_RAL.group_size,2*_RAL.group_size,
                                      linewidth=2,
                                    #  edgecolor=_colors[k],
                                    edgecolor=_colors[j % len(_colors)],
                                    facecolor='none')
            ax.add_patch(rect)
            ax.text(_RAL.active_RA_Point[0]-_RAL.group_size, 
                    _RAL.active_RA_Point[1]-_RAL.group_size, 
                        str(j), 
                        color=_colors[j % len(_colors)], size=12)
            for n in _RAL.neighbours:
                try:
                    idx = np.nonzero([self.RALs[k] == n for k in range(len(self.RALs))])[0][0]
                    print(f"Agent {j}, neighbour : {idx}")
                    ax.plot([_RAL.x + np.random.random() * 10, n.x + np.random.random() * 10], [_RAL.y+ np.random.random() * 10, n.y+ np.random.random() * 10], c=_colors[j % len(_colors)])
                except IndexError:
                    print("Agent, Non-existing neighbour...")
                #if not idx in already_seen:
                    # idx = np.nonzero([self.RALs[k] == n for k in range(len(self.RALs))])[0][0]
            #already_seen.append(j) 
        ax.set_title(title)
        # fig.tight_layout()
        plt.xlim(250, 1300)
        plt.show()
        # plt.ylim(0, 1150)
    
    def Get_RALs_mean_points(self):
        for _RAL in self.RALs:
            _RAL.Get_RAs_Mean_Point()

    def Get_Row_Mean_X(self):
        RALs_X = []

        for _RAL in self.RALs:
            RALs_X += [_RAL.active_RA_Point[0]]
        
        self.Row_Mean_X = int(np.mean(RALs_X))

    # curved
    def Get_Inter_Plant_Diffs(self):
        self.InterPlant_Diffs = []
        nb_RALs = len(self.RALs)
        if (nb_RALs > 1):
            
            for _RAL in self.RALs:
                for n in _RAL.neighbours:
                    self.InterPlant_Diffs.append(self.euclidean_distance(_RAL, n))
                
    def Get_Most_Frequent_InterPlant_Y(self):
        self.Get_Inter_Plant_Diffs()
        self.InterPlant_Y_Hist_Array = np.histogram(self.InterPlant_Diffs)

    def Is_RALs_majority_on_Left_to_Row_Mean(self):
        left_counter = 0
        for _RAL in self.RALs:
            if (_RAL.active_RA_Point[0] < self.Row_Mean_X):
                left_counter += 1

        return (left_counter/len(self.RALs) > 0.5)

    def Is_RALs_majority_going_up(self):
        up_counter = 0
        for _RAL in self.RALs:
            if (_RAL.active_RA_Point[1] - _RAL.y > 0):
                up_counter += 1

        return (up_counter/len(self.RALs) > 0.5)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Repositioning of agents                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def euclidean_distance(self, RAL1, RAL2):
        """
        Computes the euclidean distance between two points
        RAL1 (RAL)
        RAL2 (RAL)
        """
        return np.sqrt((RAL1.x - RAL2.x) ** 2 + (RAL1.y - RAL2.y) ** 2)

    
    def ORDER_RALs_to_Correct_X(self):
        """
        Interface for repositionning policy : can be global (for linear rows, colinear to Y after rotation)
        or local (for non linear rows). This interface calls to function : (i) Detect which RAL has to be
        repositioned based on the chosen criterion and (ii) reposition the RAL previously detected (once
        again based on the chosen repositionning policy).
        self.recon_policy = "global", "local_XY", "local_weighted_X" or "local_threshold_X"
        """
        if self.recon_policy == "global":
            to_reposition = self.Get_RALs_to_reposition_global()
            self.Global_repositioning(to_reposition)
        elif self.recon_policy.startswith("local"):
            to_reposition = self.Get_RALs_to_reposition_local()
            self.Local_repositioning(to_reposition)
        else:
            print("No implemented repositioning policy")

    ################################################
    # Outlier detection policies : global or local #
    ################################################
    def Get_RALs_to_reposition_global(self):
        """
        Returns the indices of the RAL which should be repositionned in the global repositioning framework
        """
        to_reposition_idx = []
        
        # find where is majority
        majority_left = False
        if (len(self.RALs)>0):
            self.Get_Row_Mean_X()
            majority_left = self.Is_RALs_majority_on_Left_to_Row_Mean()

        for i, _RAL in enumerate(self.RALs):
            if majority_left:
                if _RAL.active_RA_Point[0] > self.Row_Mean_X:
                    to_reposition_idx.append(i)
            else:
                if (_RAL.active_RA_Point[0] < self.Row_Mean_X):
                    to_reposition_idx.append(i)
        
        return to_reposition_idx

    # TODO: add an extra RAL attribute "self.outlier_detection_policy" to implement several 
    # outlier detection policies (similar to "self.recon_policy" for the repositionning policies)
    # one possibility would be to use the inter-plant distance (estimated by the Director agent)
    def Get_RALs_to_reposition_local(self, n_neighbors=1, k=1.5):
        """
        Get the indices of the agents which has to be repositionned according the local
        criterion. The local_XY criterion (not commented) computes the distance between
        each agent and its two adjacent neighbours. If this distance (for both neighbours) is greater than 
        the mean distance + 1 standard deviation of the distances between adjacent RAL in the row, 
        the agent is considered outlier and has to be repositioned. Could be improved (for ex. by taking 
        more neighbours into account) but will do the work for now.
        Returns (list): list of indices of the RAL that need to be repositioned
        """
        to_reposition_idx = []
        # the mean RALs distance will be used to detect which RALs are outliers

        # retrieve mean RAL distances and std deviation
        distances = []
        for _RAL in self.RALs:
            for n in _RAL.neighbours:
                distances.append(self.euclidean_distance(_RAL, n))
        mean_inter_RAL_dist = np.median(distances)
        std = np.std(distances)

        for i, _RAL in enumerate(self.RALs):
            count = 0
            for n in _RAL.neighbours:
                if self.euclidean_distance(_RAL, n) > mean_inter_RAL_dist + 2 * std:
                    count += 1
            if count == len(_RAL.neighbours):  # if the _RAL is too far from each of his neighbours...
                to_reposition_idx.append(i)

        return to_reposition_idx

    ################################################################################################
    # Repositionning policies : global, local_XY, or local_weighted_X/local_threshold_X            #
    ################################################################################################

    def Global_repositioning(self, to_reposition_indices):
        for idx in to_reposition_indices:
            self.RALs[idx].active_RA_Point[0] = self.Row_Mean_X
    
    # curved
    def Local_repositioning(self, to_reposition, n_neighbors=5):
        """
        Local repositioning policy for agents. Several local policies can be implemented
        """
        if self.recon_policy == "local_XY":
            self.Local_XY_Repositionning(to_reposition)
        else:
            raise NotImplementedError("This repositioning mechanism is not yet implemented...")
    
    # curved
    def Local_XY_Repositionning(self, to_reposition):
        """
        Local_XY repositionning policy : the RAL to be repositionned is set at the barycenter
        of its two adjacent neighboors. Thus, we locally approximate the curvature of the row by a 
        linear curve, which is faire if we consider the plants small and the row "smooth"
        Parameters
        ----------
        to_reposition (list) : list of indices of the RAL that need to be repositioned
        """
        for i in to_reposition:
            x, y = 0, 0
            if self.RALs[i].neighbours != []:
                for n in self.RALs[i].neighbours: # barycenter of the two closest neighbours
                    x += n.active_RA_Point[0]
                    y += n.active_RA_Point[1]
                self.RALs[i].active_RA_Point[0] = x / len(self.RALs[i].neighbours) 
                self.RALs[i].active_RA_Point[1] = y / len(self.RALs[i].neighbours)
            # update the neighbours ?? Will be done in PerformSimulationnewEndCrit()
              
    def Get_Mean_Majority_Y_movement(self, _direction):
        """
        computes the average of the movement of the RALs moving in the
        majority direction.

        _direction (int):
            Gives the direction of the majority movement. If set to 1 then
            majority of the RAls are going up. If set to -1 then majority of the
            RALs is going down.
        """
        majority_movement = 0
        majority_counter = 0
        for _RAL in self.RALs:
            if ( _direction * (_RAL.active_RA_Point[1] - _RAL.y) >= 0):
                majority_movement += (_RAL.active_RA_Point[1] - _RAL.y)
                majority_counter += 1

        self.Row_mean_Y = majority_movement/majority_counter

    def ORDER_RALs_to_Correct_Y(self):

        if (len(self.RALs)>0):
            majority_up = self.Is_RALs_majority_going_up()
            if (majority_up):
                self.Get_Mean_Majority_Y_movement(1)
            else:
                self.Get_Mean_Majority_Y_movement(-1)

        for _RAL in self.RALs:
            if (majority_up):
                if (_RAL.active_RA_Point[1] - _RAL.y < 0):
                    _RAL.active_RA_Point[1] = _RAL.y + self.Row_mean_Y
            else:
                if (_RAL.active_RA_Point[1] - _RAL.y > 0):
                    _RAL.active_RA_Point[1] = _RAL.y + self.Row_mean_Y

    def Move_RALs_to_active_points(self):
        for _RAL in self.RALs:
            _RAL.Move_Based_on_AD_Order(_RAL.active_RA_Point[0],
                                        _RAL.active_RA_Point[1])

    # curved
    def Destroy_RALs(self, RAL_idx):
        """
        Destroy the given RAL and update its neighbours
        """
        to_be_destroyed = self.RALs[RAL_idx]
        print("to_be_destroyed : ", to_be_destroyed)
        # update neighbours of the neighbours
        for neighbour in to_be_destroyed.neighbours:
            # remove RAL from its neighbours' neighbours' list
            #print("to be destroyed neighbours : ", to_be_destroyed, to_be_destroyed.neighbours)
            if to_be_destroyed in neighbour.neighbours: #Si le RAL supprimé est dans la liste de voisins de son voisin
                neighbour.neighbours.remove(to_be_destroyed) #On l'en enlève
                
                # doesn't append n himself in his neighbours
                for k in to_be_destroyed.neighbours: # link the neighbours together
                    if k != neighbour : # in case to_be_destroyed is in its neighbours
                        neighbour.neighbours.append(k) 
                        
        # destroy the RAL
        self.RALs.pop(RAL_idx)
        del to_be_destroyed        
    
    def Destroy_Low_Activity_RALs(self):
        nb_RALs = len(self.RALs)
        i = 0
        while i < nb_RALs:
            if (self.RALs[i].recorded_Decision_Score[-1] < 0.1):
                self.Destroy_RALs(i)
                nb_RALs -= 1
            else:
                i += 1
                
    def Destroy_Low_Center_Score_RALs(self) :
        for _RAL in self.RALs :
            if _RAL.center_score > 0.15:
                self.Destroy_RALs(_RAL)
        
    def Adapt_RALs_group_size(self):
        for _RAL in self.RALs:
            if (_RAL.recorded_Decision_Score[-1] < 0.2 and
                _RAL.group_size > 5*_RAL.group_step):
                _RAL.group_size -= 1
                _RAL.RAs_square_init()
            elif (_RAL.recorded_Decision_Score[-1] > 0.8 and
                  _RAL.group_size < 50*_RAL.group_step):
                _RAL.group_size += 1
                _RAL.RAs_square_init()

    def Get_RALs_Surface(self):
        for _RAL in self.RALs:
            _RAL.Compute_Surface()

class Agents_Director(object):
    """
    Agent directeur

    _plant_FT_pred_per_crop_rows (list of lists extracted for a JSON file):
        array containing the predicted position of plants organized by rows.
        The lists corresponding to rows contain other lists of length 2 giving
        the predicted position of a plant under the convention [image_line, image_column]

    _OTSU_img_array (numpy.array):
        array containing the OSTU segmented image on which the Multi Agent System
        is working

    _group_size (int, optional with default value = 5):
        number of pixels layers around the leader on which we instanciate
        reactive agents

    _group_step (int, optional with default value = 5):
        distance between two consecutive reactive agents

    _RALs_fuse_factor(float, optional with default value = 0.5):
        The proportion of the inter-plant Y distance under which we decide to
        fuse 2 RALs of a same Row Agent

    _RALs_fill_factor(float, optional with default value = 1.5):
        The proportion of the inter-plant Y distance above which we decide to
        fill the sapce between 2 RALs of a same Row Agent with new RALs.

    _field_offset (list size 2, optional with default value [0, 0]):
        the offset to apply to all the positioned agents of the simulation to
        be coherent at the field level.

    """
    def __init__(self, _plant_FT_pred_per_crop_rows, _OTSU_img_array, _ADJUSTED_img_plant_positions, _DATAFRAME_COORD,
                 _group_size = 50, _group_step = 5,
                 _RALs_fuse_factor = 0.5, _RALs_fill_factor = 1.5,
                 _field_offset = [0,0], 
                 recon_policy="local_XY"):
        
# =============================================================================
#         print()
#         print("Initializing Agent Director class...", end = " ")
# =============================================================================

        self.plant_FT_pred_par_crop_rows = _plant_FT_pred_per_crop_rows

        self.OTSU_img_array = _OTSU_img_array
        
        self.ADJUSTED_img_plant_positions = _ADJUSTED_img_plant_positions
        
        self.DATAFRAME_COORD = _DATAFRAME_COORD

        self.group_size = _group_size
        self.group_step = _group_step

        self.RALs_fuse_factor = _RALs_fuse_factor
        self.RALs_fill_factor = _RALs_fill_factor

        self.field_offset = _field_offset

        self.RowAs = []

        self.recon_policy = recon_policy

        print(self.OTSU_img_array.shape)
        
# =============================================================================
#         print("Done")
# =============================================================================

    def Initialize_RowAs(self):
        """
        Go through the predicted coordinates of the plants in self.plant_FT_pred_par_crop_rows
        and initialize the Row Agents
        """
        self.RowAs_start_x = []
        self.RowAs_start_nbRALs = []
        for _crop_row in self.plant_FT_pred_par_crop_rows:
            nb_RALs=len(_crop_row)
            if (nb_RALs > 0):
                self.RowAs_start_x += [_crop_row[0][0]]
                self.RowAs_start_nbRALs += [nb_RALs]
                RowA = Row_Agent(_crop_row, self.OTSU_img_array,
                                 self.group_size, self.group_step,
                                 self.field_offset, 
                                 recon_policy=self.recon_policy)
                
                self.RowAs += [RowA]

    def Analyse_RowAs(self):
        """
        Go through the RowAs and check if some of them are not irregulars
        regarding the distance to their neighbours and the number of RALs
        """
        mean_nb_RALs = np.mean(self.RowAs_start_nbRALs)

        X_Diffs = np.diff(self.RowAs_start_x)
        X_Diffs_hist = np.histogram(X_Diffs, int(len(self.RowAs)/2))
        Low_Bounds = X_Diffs_hist[1][:2]
        print ("X_Diffs_hist", X_Diffs_hist)
        print ("Low_Bounds", Low_Bounds)
        print ("mean_nb_RALs", mean_nb_RALs)
        print ("self.RowAs_start_nbRALs", self.RowAs_start_nbRALs)

        nb_diffs = len(X_Diffs)
        to_delete=[]
        for i in range(nb_diffs):
            if (X_Diffs[i] >= Low_Bounds[0] and X_Diffs[i] <= Low_Bounds[1]):
                print("self.RowAs_start_nbRALs[i]", i, self.RowAs_start_nbRALs[i])
                if (self.RowAs_start_nbRALs[i]<0.5*mean_nb_RALs):
                    to_delete += [i]
                elif (self.RowAs_start_nbRALs[i+1]<0.5*mean_nb_RALs):
                    to_delete += [i+1]

        nb_to_delete = len(to_delete)
        for i in range(nb_to_delete):
            self.RowAs = self.RowAs[:to_delete[i]-i] + self.RowAs[to_delete[i]-i+1:]

        print("Rows at indeces", to_delete, "were removed")

    def Analyse_RowAs_Kmeans(self):
        """
        Go through the RowAs and check if some of them are not irregulars
        regarding the distance to their neighbours and the number of RALs
        """
        X_Diffs = np.diff(self.RowAs_start_x)
        print("X_Diffs",X_Diffs)
        X = np.array([[i,0] for i in X_Diffs])
# =============================================================================
#         print("X",X)
# =============================================================================
        kmeans = KMeans(n_clusters=2).fit(X)
        print("kmeans.labels_",kmeans.labels_)
        _indeces_grp0 = np.where(kmeans.labels_ == 0)
        _indeces_grp1 = np.where(kmeans.labels_ == 1)
        grp0 = X_Diffs[_indeces_grp0]
        grp1 = X_Diffs[_indeces_grp1]
# =============================================================================
#         print("grp0", grp0)
#         print("grp1", grp1)
# =============================================================================
        test_stat, p_value = ttest_ind(grp0, grp1)
        print("test_stat", test_stat, "p_value", p_value)
        means_grp = np.array([np.mean(grp0), np.mean(grp1)])
        print("mean_nb_RALs", means_grp)

        if (p_value < 0.0001):

            index_small_grp = list(np.array((_indeces_grp0,_indeces_grp1))[np.where(means_grp == min(means_grp))][0][0])
            print(index_small_grp)

            nb_indeces = len(index_small_grp)
            to_delete = []
            if (nb_indeces == 1):
                to_delete += [index_small_grp[0]]
            else:
                if not index_small_grp[0] == index_small_grp[1]-1:
                    to_delete += [index_small_grp[0]]
                    index_small_grp = index_small_grp[1:]
                    nb_indeces -= 1
            k = 0
            while k < nb_indeces:
                sub_indeces = []
                i = k
# =============================================================================
#                 print(index_small_grp[i], index_small_grp[i+1], index_small_grp[i] == index_small_grp[i+1]-1)
# =============================================================================
                while (i < nb_indeces-1 and
                       index_small_grp[i] == index_small_grp[i+1]-1):
                    sub_indeces+=[index_small_grp[i], index_small_grp[i+1]]
                    i+=2

                nb_sub_indeces = len(sub_indeces)
                print("sub_indeces", sub_indeces)
                if (nb_sub_indeces%2 == 0):
                    for j in range (1,nb_sub_indeces,2):
                        to_delete += [sub_indeces[j]]
                else:
                    for j in range (0,nb_sub_indeces,2):
                        to_delete += [sub_indeces[j]]

                if (i>k):
                    k=i
                else:
                    k+=2

            print("Rows to_delete", to_delete)
            print("\n")
            nb_to_delete = len(to_delete)
            for i in range(nb_to_delete):
                self.RowAs = self.RowAs[:to_delete[i]-i] + self.RowAs[to_delete[i]-i+1:]

    # curved : used
    def ORDER_RowAs_to_Set_RALs_Neighbours(self):
        theta_to_histo = []
        list_theta = []
        list_RALs = []
        for _RowA in self.RowAs:
            theta, RALs = _RowA.Set_RALs_Neighbours()
            list_theta.append(theta)
            list_RALs.append(RALs)
        
        for _list in list_theta :
            for element in _list :
                theta_to_histo.append(element)



        for _list in list_RALs[0] :
            fig, ax = plt.subplots(1, dpi=300)
            ax.imshow(self.OTSU_img_array)
            #print("list : ", _list)
            #coordinates are the upper left corner of the rectangle
            rect = patches.Rectangle((_list[0].x-_list[0].group_size,
                                      _list[0].y-_list[0].group_size),
                                      2*_list[0].group_size,
                                      2*_list[0].group_size,
                                     linewidth=1,
                                     edgecolor="r",
                                     facecolor='none')
            ax.add_patch(rect)
            
            rect = patches.Rectangle((_list[1].x-_list[1].group_size,
                          _list[1].y-_list[1].group_size),
                          2*_list[1].group_size,
                          2*_list[1].group_size,
                         linewidth=1,
                         edgecolor="darkgreen",
                         facecolor='none')
            ax.add_patch(rect)
            
            rect = patches.Rectangle((_list[2].x-_list[2].group_size,
                          _list[2].y-_list[2].group_size),
                          2*_list[2].group_size,
                          2*_list[2].group_size,
                         linewidth=1,
                         edgecolor="darkgreen",
                         facecolor='none')
            ax.add_patch(rect)
            plt.show()
            
        plt.figure()
        plt.hist(theta_to_histo)
        plt.title("Histogramme des thetas pour les RALs de tous les rangs")
        plt.show()
        
    def ORDER_RowAs_to_Check_irrelevant_neighbours(self):
        #problematic_neighbouring = []
        i = 0
        for _RowA in self.RowAs :
            #problematic_neighbouring.append(_RowA.Check_irrelevant_neighbours(i))
            _RowA.Check_irrelevant_neighbours()
            i+=1
        #print("Problematic_neighbouring : ", problematic_neighbouring)
        
        
            
    def ORDER_RowAs_to_Get_End_Of_Rows(self) :
        for _RowA in self.RowAs :
            _RowA.Get_End_Of_Row()
            
    # def Aggregate_Rows(self):
    #     print("---- Aggregate RowAs ----")
        
    #     #Affichage des rangs dans l'image
    #     dataframe_coord = pd.DataFrame(self.DATAFRAME_COORD, columns = ["X", "Y", "label"])
    #     print(dataframe_coord)
        
        
    #     end_of_rows_positions = []
    #     for _RowA in self.RowAs :
    #         for _RAL in _RowA.RALs :
    #             if _RAL.end_of_row :
    #                 end_of_rows_positions.append(_RAL)
                    
    #     Clustering.Plot_end_of_rows(dataframe_coord, 0, "RGB_38", "Rows in RGB_38", end_of_rows_positions)
    #     rows_distances = []
    #     thetas = []
    #     distancess = []
    #     for _RAL in end_of_rows_positions :#Calcul de la distance aux autres bouts de rang de chaque bout de rang
    #         distances = []
    #         euclidean_distance_1 = math.sqrt((_RAL.neighbours[0].x - _RAL.x)**2 + (_RAL.neighbours[0].y - _RAL.y)**2)
    #         euclidean_distance_2 = math.sqrt(((_RAL.neighbours[1].x - _RAL.x)**2 + (_RAL.neighbours[1].y - _RAL.y)**2))
    #         if euclidean_distance_1 > euclidean_distance_2 : #On calcule la translation avec son plus proche voisin
    #             _RAL_translation_x = _RAL.x - _RAL.neighbours[0].x
    #             _RAL_translation_y = _RAL.y - _RAL.neighbours[0].y
    #         if euclidean_distance_1 < euclidean_distance_2 : #On calcule la translation avec son plus proche voisin
    #             _RAL_translation_x = _RAL.x - _RAL.neighbours[1].x
    #             _RAL_translation_y = _RAL.y - _RAL.neighbours[1].y
    #         distances = []
    #         for _RAL_2 in end_of_rows_positions :
    #             if _RAL != _RAL_2 :
    #                 _RAL_2_translation_x = _RAL_2.x - _RAL.x
    #                 _RAL_2_translation_y = _RAL_2.y - _RAL.y
    #                 cos_theta = (_RAL_translation_x*_RAL_2_translation_x + _RAL_translation_y*_RAL_2_translation_y)/(math.sqrt(_RAL_translation_x**2 + _RAL_translation_y**2)*(math.sqrt(_RAL_2_translation_x**2 + _RAL_2_translation_y**2)))
    #                 print("cos_theta : ",cos_theta)
                    
    #                 if cos_theta > 1 :
    #                     print("Problème cos theta : ", cos_theta) #Gestion de l'exception cos_theta > 1
    #                     cos_theta = 1
                        
    #                 theta_rad = math.acos(cos_theta)
    #                 theta_deg = theta_rad*(180/math.pi)
    #                 if theta_deg > 90 :
    #                     print("math.pi - cos_theta : ", -cos_theta)
    #                     theta_rad = math.acos(-cos_theta) #On ramène l'angle entre 0 et 90
    #                     print("theta_rad : ", theta_rad)
    #                     theta_deg = theta_rad*(180/math.pi)
                    
    #                 print("theta_deg : ", theta_deg)
    #                 euclidean_distance = math.sqrt(((_RAL.x - _RAL_2.x)**2 + (_RAL.y - _RAL_2.y)**2))
    #                 distances.append(euclidean_distance + 5*theta_deg)
    #                 thetas.append(theta_deg)
    #                 distancess.append(euclidean_distance)

    #             else : 
    #                 distances.append(0)
    #         rows_distances.append(distances)
            
        
    #     plt.figure()
    #     plt.hist(thetas)
    #     plt.title("Histo des angles en °")
    #     plt.show()
        
    #     plt.figure()
    #     plt.scatter(thetas, distancess)
    #     plt.title("Couples Theta/Distance pour tous les bouts de rang")
    #     plt.xlabel("Theta")
    #     plt.ylabel("Distance")
    #     plt.show()
    #     # new_rows = []
    #     # for i in range(len(rows_distances)) :
    #     #     new_rows.append(rows_distances[i][i:len(rows_distances[i])])
    #     new_list = []
    #     for i in range(1, len(rows_distances[0])) : 
    #         #print("rows_distances ",rows_distances[i-1][i:])
    #         for element in rows_distances[i-1][i:]:
    #             new_list.append(element)
    #     print("new_list : ", new_list)
    #     #distance_matrix = pdist(rows_distances, metric="euclidean")
    #     #print("distance matrix : ", distance_matrix)
    #     model = linkage(new_list, method = "single", metric = "euclidean", optimal_ordering=True)
        
    #     print(model)         
        
    #     plt.title("CAH")
    #     dendrogram(model,orientation='left',color_threshold=250)
    #     plt.show()
        
    def Aggregate_Rows(self) :
        
        def Compute_dists_and_angles(RALs_list,end_of_rows_positions) :

            infos = []
            angles = []
            for _RAL in RALs_list[:2] :
                _RAL.Compute_neighbours_distances()
                _RAL.Compute_closest_neighbour_translation()
                
                for _RAL_2 in RALs_list[2:] :
                    if _RAL != _RAL_2 :
                        _RAL_2.Compute_neighbours_distances()
                        _RAL_2.Compute_closest_neighbour_translation()
                            
                        euclidean_distance = math.sqrt((_RAL_2.x - _RAL.x)**2 + (_RAL_2.y - _RAL.y)**2) #On calcule la translation entre _RAL_2 et _RAL
                        _RALs_translation_x = _RAL_2.x - _RAL.x
                        _RALs_translation_y = _RAL_2.y - _RAL.y
                        _dist = math.sqrt(_RALs_translation_x**2 + _RALs_translation_y**2)

                        cos_theta_1 = (_RAL.CN_translation_x*_RALs_translation_x + _RAL.CN_translation_y*_RALs_translation_y)/(math.sqrt(_RAL.CN_translation_x**2 + _RAL.CN_translation_y**2)*math.sqrt(_RALs_translation_x**2 + _RALs_translation_y**2)) #Angle entre Voisin_RAL_1--RAL_1--RAL_2
                        cos_theta_2 = (_RAL_2.CN_translation_x*_RALs_translation_x + _RAL_2.CN_translation_y*_RALs_translation_y)/(math.sqrt(_RAL_2.CN_translation_x**2 + _RAL_2.CN_translation_y**2)*math.sqrt(_RALs_translation_x**2 + _RALs_translation_y**2)) #Angle entre Voisin_RAL_1--RAL_1--RAL_2
                        
                        #print("cos_theta : ",cos_theta_1, cos_theta_2)
                        sin_theta_1 = math.sqrt(1 - cos_theta_1**2)
                        sin_theta_2 = math.sqrt(1 - cos_theta_2**2)
                        
                        #theta_rad_1 = math.acos(cos_theta_1) 
                        theta_rad_1 = math.asin(sin_theta_1)
                        theta_deg_1 = theta_rad_1*(180/math.pi)

                        
                        #theta_rad_2 = math.acos(cos_theta_2)
                        theta_rad_2 = math.asin(sin_theta_2)
                        theta_deg_2 = theta_rad_2*(180/math.pi)

                        mean_theta_deg = (theta_deg_1 + theta_deg_2)/2

                        infos.append([_dist,end_of_rows_positions.index(_RAL),end_of_rows_positions.index(_RAL_2), mean_theta_deg])

            _list = []
            for info in infos : 
                _list.append(info[0])
            return infos[_list.index(min(_list))]
        
        def Merge_Rows(model, min_clusters_infos):
            
            rows_to_merge = []
            for results in model :
                nb_initial_rows = results[2]
                if nb_initial_rows == 2 :
                    first_row = results[0]
                    second_row = results[1]
                    rows_to_merge.append([first_row, second_row])
            
            for rows in rows_to_merge :
                    
                pass
            
        print("---- Aggregate RowAs ----")
        
        #Affichage des rangs dans l'image
        dataframe_coord = pd.DataFrame(self.DATAFRAME_COORD, columns = ["X", "Y", "label"])
        label_cluster = np.unique(dataframe_coord[["label"]].to_numpy())
        print(dataframe_coord)
        print(label_cluster)
        
        
        end_of_rows_positions = []
        for _RowA in self.RowAs :
            for _RAL in _RowA.RALs :
                if _RAL.end_of_row :
                    end_of_rows_positions.append(_RAL)
                    
        Clustering.Plot_end_of_rows(dataframe_coord, 0, "RGB_38", "Rows in RGB_38", end_of_rows_positions)
        
        min_cluster_infos = []
        for i,j in zip([2*x for x in range(0, int(len(end_of_rows_positions)/2))], [-1+2*y for y in range(1, int((len(end_of_rows_positions)/2)+1))]) :
            RAL_1 = end_of_rows_positions[i]
            RAL_2 = end_of_rows_positions[j]
            _list_dist = []
            print(len(end_of_rows_positions),i,j)
            for h,k in zip([2*x for x in range(0, int(len(end_of_rows_positions)/2))], [-1+2*y for y in range(1, int((len(end_of_rows_positions)/2)+1))]) :
                
                RAL_3 = end_of_rows_positions[h]
                RAL_4 = end_of_rows_positions[k]
                min_dist = Compute_dists_and_angles([RAL_1,RAL_2,RAL_3,RAL_4],end_of_rows_positions)
                _list_dist.append(min_dist)
            min_cluster_infos.append(_list_dist)
        

        for i in range(11) : 
            for j in range(11) :   
                print("Angle entre {0} et {1} = ".format(i,j), min_cluster_infos[i][j][3])
                
        
        
    
        upper_triangular_CAH_matrix = []
        count  = 1
        angle_coefficient = 5
        for row_info in min_cluster_infos :
            _transition_list = []
            for element in row_info :
                distance = element[0]
                mean_angle = element[3]
                #_transition_list.append(distance + mean_angle) #Sans poid de l'angle
                #_transition_list.append(distance + mean_angle*angle_coefficient) #Avec angle sans pondération de l'angle
                _transition_list.append(distance + mean_angle*angle_coefficient) #Avec angle pondération de l'angle*5


            #print("input Full_CAH_matrix : ", _transition_list[count:]) #count sert à selectionne segmenter chaque ligne pour obtenir la matrice triangulaire supérieure 
            for value in _transition_list[count:] :
                upper_triangular_CAH_matrix.append(value) #On stocke la matrice triangulaire supérieure sous la forme d'un flattened array
            count += 1

        model = linkage(upper_triangular_CAH_matrix, method = "single", metric = "euclidean", optimal_ordering=True)
        
        print(model)         
        
        plt.title("CAH")
        dendrogram(model,orientation='left',color_threshold=250)
        plt.show()
        
        #Merge_Rows(model, min_cluster_infos)
        
        fusion_weights = []
        for fusion_data in model :
            fusion_weights.append(fusion_data[2])
        
        fusion_angle = [2.62,3.93, 30.89,32.73,46.97,69.29, 69.21,70.88, 73.02, 85.39] # 9-7; 2-4; 2-3; 2-1; 1-0; 3-5; 7-6; 8-10; 7-8, 5-6
        plt.figure()
        plt.plot(range(len(fusion_angle)), fusion_angle)
        plt.xlabel("Fusion n°")
        plt.ylabel("Angle de fusion")
        plt.title("Angles entre les rangs pour chaque fusion sur RGB_38")
        plt.show()
        
        #print("Angle entre 7 et 5 = ", min_cluster_infos[7][i][1]) #Distance minimale entre le rang 7 et le rang i        
                
            # if fusion_data[3] == 0 :
            #     fusion_angle.append(min_cluster_infos[fusion_data[0]][fusion_data[1]][3])
            # if fusion_data[3] !: 0 :
            #     if fusion_data[0] > 10 :
            #         index_initial_fusion = fusion_data[0] - 10
            #     if fusion_data[1] > 10 :
            #         index_initial_fusion_2 = fusion_data[1] - 10
                    
                

        
        plt.figure()
        plt.plot(range(len(fusion_weights)), fusion_weights)
        plt.title("Fusion_weights")
        plt.xlabel("Fusion n°")
        plt.ylabel("Fusion weight")
        plt.show()
        
        
        

        
        
    # curved : not used in practice

    
    def ORDER_RowAs_to_Sort_RALs(self):
        for _RowA in self.RowAs:
            _RowA.Sort_RALs()
            
    def ORDER_RowAs_for_RALs_mean_points(self):
        for _RowA in self.RowAs:#[10:11]:
            _RowA.Get_RALs_mean_points()

    def ORDER_RowAs_to_Correct_RALs_X(self):
        for _RowA in self.RowAs:#[10:11]:
            _RowA.ORDER_RALs_to_Correct_X()

    def ORDER_RowAs_to_Correct_RALs_Y(self):
        for _RowA in self.RowAs:
            _RowA.ORDER_RALs_to_Correct_Y()

    def ORDER_RowAs_to_Update_InterPlant_Y(self):
        for _RowA in self.RowAs:#[10:11]:
            _RowA.Get_Most_Frequent_InterPlant_Y()

    def ORDER_RowAs_for_Moving_RALs_to_active_points(self):
        for _RowA in self.RowAs:#[10:11]:
            _RowA.Move_RALs_to_active_points()

    def Summarize_RowAs_InterPlant_Y(self):
        SumNbs = np.zeros(10, dtype=np.int32)
        SumBins = np.zeros(11)
        for _RowA in self.RowAs:#[10:11]:
            SumNbs += _RowA.InterPlant_Y_Hist_Array[0]
            SumBins += _RowA.InterPlant_Y_Hist_Array[1]
        SumBins /= len(self.RowAs)#[10:11])

        print("max of SumNbs", SumNbs, np.max(SumNbs))
        print("index of max for SumBins", np.where(SumNbs == np.max(SumNbs)))
        print("SumBins", SumBins)
        max_index = np.where(SumNbs == np.max(SumNbs))[0]
        if(max_index.shape[0]>1):
            max_index = max_index[:1]
        print("max_index", max_index)
        self.InterPlant_Y = int(SumBins[max_index][0])
        print("InterPlant_Y before potential correction", self.InterPlant_Y)
        while (max_index < 10 and self.InterPlant_Y < 5):
            max_index += 1
            self.InterPlant_Y = int(SumBins[max_index])
            print("Correcting InterPlant_Y", self.InterPlant_Y)

    def ORDER_RowAs_Fill_or_Fuse_RALs(self):
        for _RowA in self.RowAs:
            _RowA.Fill_or_Fuse_RALs(self.InterPlant_Y,
                                    self.RALs_fuse_factor,
                                    self.RALs_fill_factor)

    def ORDER_RowAs_to_Destroy_Low_Activity_RALs(self):
        for _RowA in self.RowAs:#[10:11]:
            _RowA.Destroy_Low_Activity_RALs()
            
    def ORDER_RowAs_to_Destroy_Low_Center_Score_RALs(self) : 
        for _RowA in self.RowAs :
            _RowA.Destroy_Low_Center_Score_RALs()

    def ORDER_RowAs_to_Adapt_RALs_sizes(self):
        for _RowA in self.RowAs:#[10:11]:
            _RowA.Adapt_RALs_group_size()

    def ORDER_RowAs_for_Edges_Exploration(self):
        for _RowA in self.RowAs:#[10:11]:
            _RowA.Edge_Exploration(1.1*self.RALs_fuse_factor*self.InterPlant_Y)
    
    def ORDER_RowAs_for_Local_Exploration(self) :
        # plt.figure()
        fig, ax = plt.subplots(1,dpi=300)
        ax.imshow(self.OTSU_img_array)
        for _RowA in self.RowAs :
            x_min, x_max, min_y, max_y = _RowA.Local_Exploration()
            #print("points : ({3},{0}), ({1}, {2})".format(min_y, x_max, max_y, x_min))
            plt.plot([x_min,x_max], [min_y,max_y], 'g')
            #plt.plot([x_min,x_max], [min_y, max_y], color='g')
        plt.title("Droites d'intersection entre les rangs")
        plt.show()
            

    def ORDER_RowAs_for_Global_Exploration(self) :
        
        def Init_Grid(self) :
            """
            Here we initialize the grid formed with RALs
            Returns
            -------
            """
            for y_init in range(20, len(self.OTSU_img_array[0]), 20):
                for x_init in range(20, len(self.OTSU_img_array[1]), 20):
                    new_RAL = ReactiveAgent_Leader(_x = x_init,
                                                   _y = y_init,
                                                   _reduction_step = reduction_step,
                                                   _img_array = self.OTSU_img_array,
                                                   _group_size = self.group_size,
                                                   _group_step = self.group_step
                                                   )
                    _transition_RowA.RALs.append(new_RAL)
                    
        def Normalize_Identity_Vectors(self, input_1, input_2, input_3) : 
            #for i in range(0, len(_transition_RowA.RALs[0].identity_vector)) :
            for i in range(0, 2) : #No need to normalize decision scores
                _list_for_normalizing = []
                _normalized_list = []
                for _RAL in _transition_RowA.RALs :
                    _list_for_normalizing.append(_RAL.identity_vector[i])
                
                for _RAL in _transition_RowA.RALs :
                    to_normalize = _RAL.identity_vector[i]
                    _RAL.identity_vector[i] = (to_normalize - min(_list_for_normalizing))/(max(_list_for_normalizing) - min(_list_for_normalizing))
                    _normalized_list.append(_RAL.identity_vector[i])              
                
                to_normalize = input_1[i]  
                input_1[i] = (to_normalize - min(_list_for_normalizing))/(max(_list_for_normalizing) - min(_list_for_normalizing))
                to_normalize = input_1[i]  
                input_2[i] = (to_normalize - min(_list_for_normalizing))/(max(_list_for_normalizing) - min(_list_for_normalizing))
                to_normalize = input_1[i]  
                input_3[i] = (to_normalize - min(_list_for_normalizing))/(max(_list_for_normalizing) - min(_list_for_normalizing))
                
                plt.figure()
                plt.hist(_list_for_normalizing)
                plt.title("A normaliser")
                plt.show
                
                plt.figure()
                plt.hist(_normalized_list)
                plt.title("Normalisées")
                plt.show
                
            print("_transition_RowA.RALs[0].identity_vector after normalization : ", _transition_RowA.RALs[0].identity_vector)
            return input_1, input_2, input_3
        
        def Ponderate_reduction_scores(self, reduction_step) :
            for i in range(2, len(_transition_RowA.RALs[0].identity_vector)):
                for _RAL in _transition_RowA.RALs :
                    value_to_ponderate = _RAL.identity_vector[i]
                    _RAL.identity_vector[i] = value_to_ponderate*reduction_step
            
        def Get_Identity_Vectors(self) : 
            _list_identity_vectors = []
            for _RAL in _transition_RowA.RALs :
                _list_identity_vectors.append(_RAL.identity_vector)
            
            return _list_identity_vectors
        
        def Select_input_cluster(self, Cluster_list, Input_cluster, _RALs_list) :
            cluster_index = []
            cluster_index = np.argwhere(Cluster_list == Input_cluster)
            
            _transition_RowA.RALs = []
            print("len(_RALs_list)/max(cluster_index) : ", len(_RALs_list)," / ", max(cluster_index))
            for index in cluster_index : 
                _transition_RowA.RALs.append(_RALs_list[index[0]-3])
            Show_Debugage(self,_color = "g", _title = "Results for cluster n°{0}".format(Input_cluster))
        
        def Fuse_RALs(self, Fuse_Factor, reduction_step) :
            for _RAL in _transition_RowA.RALs :
                Fuse_test(self, _RAL.x, _RAL.y, _RAL.decision_score, _RAL, Fuse_Factor, reduction_step)
                    
        
        def Fuse_test(self, _RAL_x, _RAL_y, _RAL_decision_score, _RAL, Fuse_Factor, reduction_step) : 
            for _RAL_2 in _transition_RowA.RALs : 
                if math.sqrt((_RAL_2.x - _RAL_x)**2 + (_RAL_2.y - _RAL_y)**2) <= self.group_size*Fuse_Factor :
                    new_RAL = ReactiveAgent_Leader(_x = (_RAL_2.x - _RAL_x)/2,
                                                             _y = (_RAL_2.y - _RAL_y)/2,
                                                             _reduction_step = reduction_step,
                                                             _img_array = self.OTSU_img_array,
                                                             _group_size = self.group_size,
                                                             _group_step = self.group_step
                                                             )
                    
                    if _RAL_decision_score > (_RAL_2.decision_score + new_RAL.decision_score)/2 :
                        _transition_RowA.pop(_transition_RowA.index(_RAL_2))
                        return
                        
                    if _RAL_2.decision_score > (_RAL_decision_score + new_RAL.decision_score)/2 :
                        _transition_RowA.pop(_transition_RowA.index(_RAL))
                        return
        
                    if new_RAL.decision_score > (_RAL_2.decision_score + _RAL_decision_score)/2 :   
                        _transition_RowA.pop(_transition_RowA.index(_RAL_2))
                        _transition_RowA.pop(_transition_RowA.index(_RAL))
                        _transition_RowA.append(new_RAL)
                        return
                    
        def Show_Adjusted_Plant_Informations(self, reduction_step, _color):
            
            positions = []
            for _dic in self.ADJUSTED_img_plant_positions :
                positions.append((_dic["image_x"],_dic["image_y"]))
            
            Labelled_RowA = Row_Agent(positions, self.OTSU_img_array,
                _group_size = self.group_size, _group_step = 5,
                _field_offset = [0,0],
                recon_policy="global"
                )
            
            fig, ax = plt.subplots(1,dpi=300)
            ax.imshow(self.OTSU_img_array)
            
            i = 0
            # for pos in positions :
            #     i+=1
            #     circle = patches.Circle(pos,
            #                             radius = self.group_size,
            #                             linewidth=1,
            #                             edgecolor=_color,
            #                             facecolor='none')
            #     ax.add_patch(circle)
            # #ax.title("Adjusted Plant Positions")
            # plt.title("Adjusted Plant Positions")
            # plt.show()
            # print("Printed Labelled positions")
            i=0
            _RALs_size = []
            for _RAL in Labelled_RowA.RALs :
                _RALs_size.append(len(_RAL.list_active_RAs))
                if i%10 == 0 :  
                    print(i, " sur {0} RALs".format(len(Labelled_RowA.RALs)))
                    print(len(_RAL.list_active_RAs))
                    j = 0
                    circle = patches.Circle((_RAL.x, _RAL.y),
                                            radius = self.group_size,
                                            linewidth=0.4,
                                            edgecolor="r",
                                            facecolor='none')
                    ax.add_patch(circle)
                    j = 0
                    for _RA in _RAL.list_active_RAs :
                        if j%10 == 0 :
                            circle = patches.Circle((_RA.global_x, _RA.global_y),
                                                    radius = 20,
                                                    linewidth=0.05,
                                                    edgecolor=_color,
                                                    facecolor='none')
                            ax.add_patch(circle)
                        j+=1
                i+=1
            #ax.title("Adjusted Plant Positions")
            plt.title("Adjusted Plant Positions Radius = 20")
            plt.show()
            plt.figure()
            plt.hist(_RALs_size)
            plt.ylabel("Nombre de pixels dans le RAL")
            plt.title("Répartition du nombre de pixel pour chaque RAL")
            print("Printed Labelled positions")


            #self.RowAs.append(Labelled_RowA)

            for _RAL in Labelled_RowA.RALs :
                _RAL.Do_Scores(reduction_step)
                
            for i in range(0, 2):
                list_to_hist = []
                for _RAL in Labelled_RowA.RALs :
                    list_to_hist.append(_RAL.identity_vector[i])
                plt.figure()
                plt.hist(list_to_hist)
                plt.title(["Center_Score", "Symetricity_Score"][i])
                plt.show()   
             
            for i in range(2, len(Labelled_RowA.RALs[0].identity_vector)):
                list_to_hist = []
                for _RAL in Labelled_RowA.RALs :
                    list_to_hist.append(_RAL.identity_vector[i])
                plt.figure()
                plt.hist(list_to_hist)
                plt.title("Size of RALs compare to original one : {0}".format(reduction_step*(i-2)))
                plt.show()
                
            
            input_1 = Labelled_RowA.RALs[random.randrange(0,len(Labelled_RowA.RALs))].identity_vector
            input_2 = Labelled_RowA.RALs[random.randrange(0,len(Labelled_RowA.RALs))].identity_vector
            input_3 = Labelled_RowA.RALs[random.randrange(0,len(Labelled_RowA.RALs))].identity_vector
            
            #######################
            #Affichage des empreintes des adventices
            #######################
            weeds_positions = [(115,41),(67,86),(85,173),(116,199),(247,80),(431,87),(305,197),(535,156),(574,120),(675,44),(803,48),(728,248),(786,199),(935,112),(1098,85),(893,315),(1029,320),(1408,102),(1495,145),(37,878),(235,818),(276,842),(299,872),(232,962),(458,862)] #38
            
            Weeds_RowA = Row_Agent(weeds_positions, self.OTSU_img_array,
                _group_size = self.group_size, _group_step = 5,
                _field_offset = [0,0],
                recon_policy="global"
                )
            
            # plt.figure(dpi=300)
            # v = 0
            # for _weed in Weeds_RowA.RALs :
            #     times = time.time()
            #     nb_neighbours = []
            #     for epsilon in range(101,152, 5) :
            #         new_RAL = ReactiveAgent_Leader(_x = _weed.x,
            #                                  _y = _weed.y,
            #                                  _reduction_step = 0,
            #                                  _img_array = self.OTSU_img_array,
            #                                  _group_size = epsilon,
            #                                  _group_step = self.group_step
            #                                  )
            #         nb_neighbours.append(len(new_RAL.list_active_RAs)) 
            #     plt.plot([96 + i*5 for i in range(0, len(nb_neighbours))], nb_neighbours, color = ["b", "c", "m", "y", "r", "green", "deeppink", "lightblue", "brown", "lightgreen"][v%10])
            #     v+=1
            #     print("Nb RALs processed : ",v, "/", len(Labelled_RowA.RALs))
            #     print("Temps pour 1 RAL : ", time.time() - times)
            # plt.title("Empreintes des adventices labélisées pour RGB_38")
            # plt.xlabel("Epsilon")
            # plt.ylabel("Nb de voisins")
            # plt.show()
            
            plt.figure(dpi=300)
            Q01 = []
            Q025 = []
            med = []
            Q075 = []
            Q09 = []
            for epsilon in range(1,152,5) :
                nb_neighbours = []
                times = time.time()
                for _weed in Weeds_RowA.RALs :
                    

                    new_RAL = ReactiveAgent_Leader(_x = _weed.x,
                                             _y = _weed.y,
                                             _reduction_step = 0,
                                             _img_array = self.OTSU_img_array,
                                             _group_size = epsilon,
                                             _group_step = self.group_step
                                             )
                    nb_neighbours.append(len(new_RAL.list_active_RAs)) 
                Q01.append(np.quantile(nb_neighbours,0.1))
                Q025.append(np.quantile(nb_neighbours,0.25))
                med.append(np.quantile(nb_neighbours,0.5))
                Q075.append(np.quantile(nb_neighbours,0.75))
                Q09.append(np.quantile(nb_neighbours,0.9))
                print("Nb Epsilon processed : ",epsilon, "/", len([i for i in range(1,152)]))
                print("Temps pour 1 epsilon : ", time.time() - times)
            plt.plot([i for i in range(1,152,5)],Q01, color = "r")
            plt.plot([i for i in range(1,152,5)],Q09, color = "r")
            plt.plot([i for i in range(1,152,5)],Q025, color = "b")
            plt.plot([i for i in range(1,152,5)],Q075, color = "b")
            plt.plot([i for i in range(1,152,5)],med, color = "g")
            plt.title("Empreintes des adventices labélisées pour RGB_46")
            plt.xlabel("Epsilon")
            plt.ylabel("Nb de voisins")
            plt.show()
            
            #####################
            #Affichage des empreintes pour chaque plantes
            #####################
            # print("Entrée Empreintes")
            # plt.figure(dpi=300)
            # v = 0
            # for _RAL in Labelled_RowA.RALs :
            #     if v%2 == 0 :
            #         times = time.time()
            #         nb_neighbours = []
            #         for epsilon in range(101, 152, 5) :
            #             new_RAL = ReactiveAgent_Leader(_x = _RAL.x,
            #                                  _y = _RAL.y,
            #                                  _reduction_step = 0,
            #                                  _img_array = self.OTSU_img_array,
            #                                  _group_size = epsilon,
            #                                  _group_step = self.group_step
            #                                  )
                        
            #             nb_neighbours.append(len(new_RAL.list_active_RAs))
            #         plt.plot([96 + i*5 for i in range(1, len(nb_neighbours))], nb_neighbours, color = ["b", "c", "m", "y", "r", "green", "deeppink", "lightblue", "brown", "lightgreen"][v%10])
            #         print("Nb RALs processed : ",v, "/", len(Labelled_RowA.RALs))
            #         print("Temps pour 1 RAL : ", time.time() - times)
            #     v+=1
            # plt.title("Empreintes des plantes labélisées pour RGB_38")
            # plt.xlabel("Epsilon")
            # plt.ylabel("Nb de voisins")
            # plt.show()
            
            plt.figure(dpi=300)
            Q01 = []
            Q025 = []
            med = []
            Q075 = []
            Q09 = []
            for epsilon in range(1,152,5) :
                nb_neighbours = []
                times = time.time()
                for _weed in Labelled_RowA.RALs :
                    

                    new_RAL = ReactiveAgent_Leader(_x = _weed.x,
                                             _y = _weed.y,
                                             _reduction_step = 0,
                                             _img_array = self.OTSU_img_array,
                                             _group_size = epsilon,
                                             _group_step = self.group_step
                                             )
                    nb_neighbours.append(len(new_RAL.list_active_RAs)) 
                Q01.append(np.quantile(nb_neighbours,0.1))
                Q025.append(np.quantile(nb_neighbours,0.25))
                med.append(np.quantile(nb_neighbours,0.5))
                Q075.append(np.quantile(nb_neighbours,0.75))
                Q09.append(np.quantile(nb_neighbours,0.9))
                print("Nb Epsilon processed : ",epsilon, "/", len([i for i in range(1,152)]))
                print("Temps pour 1 epsilon : ", time.time() - times)
            plt.plot([i for i in range(1,152,5)],Q01, color = "r")
            plt.plot([i for i in range(1,152,5)],Q09, color = "r")
            plt.plot([i for i in range(1,152,5)],Q025, color = "b")
            plt.plot([i for i in range(1,152,5)],Q075, color = "b")
            plt.plot([i for i in range(1,152,5)],med, color = "g")
            plt.title("Empreintes des adventices labélisées pour RGB_46")
            plt.xlabel("Epsilon")
            plt.ylabel("Nb de voisins")
            plt.show()
            

                
                
            
            return input_1, input_2, input_3, Labelled_RowA
            
        def Show_Sample_Histo(self, Labelled_RowA) :
            list_to_hist_sample = []
            list_to_hist_labelled = []
            for i in range(0,2) :
                for _RAL in _transition_RowA.RALs :
                    list_to_hist_sample.append(_RAL.identity_vector[i])
                for _RAL in Labelled_RowA.RALs :
                    list_to_hist_labelled.append(_RAL.identity_vector[i])
                    
                plt.figure()
                plt.hist(list_to_hist_sample, color = "b", alpha = 0.5)
                plt.hist(list_to_hist_labelled, color = "orange", alpha = 0.5)
                plt.title(["Center_Score", "Symetricity_Score"][i])
                plt.show()
             
            for i in range(2, len(_transition_RowA.RALs[0].identity_vector)) :
                for _RAL in _transition_RowA.RALs :
                    list_to_hist_sample.append(_RAL.identity_vector[i])
                for _RAL in Labelled_RowA.RALs :
                    list_to_hist_labelled.append(_RAL.identity_vector[i])
                    
                plt.figure()
                plt.hist(list_to_hist_sample, color = "b", alpha = 0.5)
                plt.hist(list_to_hist_labelled, color = "orange", alpha = 0.5)
                plt.title("Size of RALs compare to original one : {0}".format(reduction_step*(i-2)))
                plt.show()
                
        def Show_Debugage(self, _color, _title):
            print("Show_Debugage_{0}".format(_title))
            fig, ax = plt.subplots(1,dpi=300)
            ax.imshow(self.OTSU_img_array)
                
            for _RAL in _transition_RowA.RALs :
                circle = patches.Circle((_RAL.x,_RAL.y),
                                         radius = self.group_size,
                                         linewidth=1,
                                         edgecolor=_color,
                                         facecolor='none')
                ax.add_patch(circle)
            plt.title(_title)
            plt.show()
        
        self.RowAs = []
        reduction_step = 0.25
        Fuse_Factor = 0.9
        
        
        plants_list = []
        for rows in self.plant_FT_pred_par_crop_rows : #Need to transform self.plant_FT_pred_par_crop_rows to a list of list to a list for using native MAS functions
            for plants in rows :
                plants_list.append(plants)
                
        _transition_RowA = Row_Agent(plants_list, self.OTSU_img_array,
                                    _group_size = 50, _group_step = 5,
                                    _field_offset = [0,0],
                                    recon_policy="global"
                                    )
        
        self.RowAs.append(_transition_RowA)
        
        print("Printing Labelled Plant positions")
        input_1, input_2, input_3, labelled_RowA = Show_Adjusted_Plant_Informations(self, reduction_step, _color = "g")
        
        grid_time = time.time()
        Init_Grid(self)
        print("Time to initialize a grid of {0} RALs : ".format(len(_transition_RowA.RALs)), time.time() - grid_time)
        
        print("Nombre de RALs avant Destroy : ", len(_transition_RowA.RALs))
        #Show_Debugage(self, _color = "g", _title = "Avant Destroy")
        destroy_time = time.time()
        nb_RALs_before_destroy = len(_transition_RowA.RALs)
        _transition_RowA.Destroy_Low_Activity_RALs()
        print("Time to destroy {0} RALs : ".format(nb_RALs_before_destroy - len(_transition_RowA.RALs)), time.time() - destroy_time)
        print("Nombre de RALs après Destroy : ", len(_transition_RowA.RALs))
        Show_Debugage(self, _color = "g", _title = "Après Destroy Seuil d'activité")

        
        for _RAL in _transition_RowA.RALs :
            _RAL.Do_Scores(reduction_step)
        _transition_RowA.Destroy_Low_Center_Score_RALs()
        Show_Debugage(self, _color = "g", _title = "Après Destroy Center_Score")
        vectors_time = time.time()
        input_1, input_2, input_3 = Normalize_Identity_Vectors(self, input_1, input_2, input_3)
        Ponderate_reduction_scores(self, reduction_step)
        Identity_Vectors = Get_Identity_Vectors(self)
        Show_Sample_Histo(self, labelled_RowA)
        print("Time to compute and get {0} identity vectors : ".format(len(_transition_RowA.RALs)), time.time() - vectors_time)
        
        Identity_Vectors.append(input_1)
        Identity_Vectors.append(input_2)
        Identity_Vectors.append(input_3)
        
        kmeans_time = time.time()
        nb_cluster = 2
        Kmeans = KMeans(n_clusters = nb_cluster, n_init = 10, max_iter = 300)
        Kmeans.fit(np.array(Identity_Vectors))
        print("Process time for KMeans with {0} dimensions for {1} RALs : ".format(len(_transition_RowA.RALs[0].identity_vector), len(_transition_RowA.RALs)), time.time() - kmeans_time)
        
        print("Inputs Clusters : ", Kmeans.labels_[-3], Kmeans.labels_[-2], Kmeans.labels_[-1])
        
        plt.figure()
        plt.hist(Kmeans.labels_)
        plt.title("Clusters après KMeans, le bon est le n°{0}".format(Kmeans.labels_[-3]))
        plt.show()
        
        _RALs_list = _transition_RowA.RALs
        Select_input_cluster(self, Kmeans.labels_, Kmeans.labels_[-3], _RALs_list)
        Select_input_cluster(self, Kmeans.labels_, Kmeans.labels_[-2], _RALs_list)
        Select_input_cluster(self, Kmeans.labels_, Kmeans.labels_[-1], _RALs_list)
        
        Fuse_time = time.time()
        for i in range(0,20) :
            Fuse_RALs(self, Fuse_Factor, reduction_step)
            print("Time 1 round of Fusing : ", time.time-Fuse_time)
            print("Nb de RALs restants après le Fuse n°{0}: ".format(i+1), len(_transition_RowA.RALs))
        
            Show_Debugage(self, _color = "g", _title = "Après 1 tour de Fuse")
        
    def Check_RowAs_Proximity(self):
        nb_Rows = len(self.RowAs)
        i=0
        while i < nb_Rows-1:
            if (abs(self.RowAs[i].Row_Mean_X-self.RowAs[i+1].Row_Mean_X) < self.group_size):
                print(f"removing row {i}, distance to row {i+1} is {abs(self.RowAs[i].Row_Mean_X-self.RowAs[i+1].Row_Mean_X)} greater lower than {self.group_size}")
                new_plant_FT = self.RowAs[i].plant_FT_pred_in_crop_row + self.RowAs[i].plant_FT_pred_in_crop_row
                new_plant_FT.sort()

                RowA = Row_Agent(new_plant_FT,
                                 self.OTSU_img_array,
                                 self.group_size,
                                 self.group_step,
                                 self.field_offset)
                if (i<nb_Rows-2):
                    self.RowAs = self.RowAs[:i]+ [RowA] + self.RowAs[i+2:]
                else:
                    self.RowAs = self.RowAs[:i]+ [RowA]
                i+=2
                nb_Rows-=1
            else:
                i+=1

    def ORDER_RowAs_for_RALs_Surface_Compute(self):
        for _RowA in self.RowAs:
            _RowA.Get_RALs_Surface()


# =============================================================================
# Simulation Definition
# =============================================================================
class Simulation_MAS(object):
    """
    This class manages the multi agent simulation on an image.
    In particular, it instanciate the Agent Director of an image, controls the
    flow of the simulation (start, stop, step), and rthe results visualization
    associated.

    _RAW_img_array (numpy.array):
        array containing the raw RGB image. This would be mostly used for results
        visualization.

    _plant_FT_pred_per_crop_rows (list of lists extracted for a JSON file):
        array containing the predicted position of plants organized by rows.
        The lists corresponding to rows contain other lists of length 2 giving
        the predicted position of a plant under the convention [image_line, image_column]

    _OTSU_img_array (numpy.array):
        array containing the OSTU segmented image on which the Multi Agent System
        is working

    _group_size (int, optional with default value = 5):
        number of pixels layers around the leader on which we instanciate
        reactive agents

    _group_step (int, optional with default value = 5):
        distance between two consecutive reactive agents

    _RALs_fuse_factor(float, optional with default value = 0.5):
        The proportion of the inter-plant Y distance under which we decide to
        fuse 2 RALs of a same Row Agent

    _RALs_fill_factor(float, optional with default value = 1.5):
        The proportion of the inter-plant Y distance above which we decide to
        fill the sapce between 2 RALs of a same Row Agent with new RALs.

    _field_offset (list size 2, optional with default value [0, 0]):
        the offset to apply to all the positioned agents of the simulation to
        be coherent at the field level.

    _ADJUSTED_img_plant_positions (list, optional with default value = None):
        The list containing the adjusted positions of the plants coming from
        the csv files. So the positions are still in the string format.
    
    _follow_simulation (bool, optional with default value = False):
        Generates the plot showing all RALs and target positions at every steps
        of the simulation to follow the movements and theevolution of the number
        RALs
    
    _follow_simulation_save_path(string, optional with default value ""):
        The path where the plots following the steps of the simulation will be 
        saved.
    
    _simulation_name (string, optional with default value = ""):
        Name given to the simulation. used as a prefix of the some saved files.
        
    _recon_policy (string, optional with default value = "local_XY"):
        keyword to chose which policy RAL agents should use to control their
        positions.
        TODO describe values
    """

    def __init__(self, _RAW_img_array,
                 _plant_FT_pred_per_crop_rows, _OTSU_img_array, _data_input_DATAFRAME_COORD,
                 _group_size = 50, _group_step = 5,
                 _RALs_fuse_factor = 0.5, _RALs_fill_factor = 1.5,
                 _field_offset = [0,0],
                 _ADJUSTED_img_plant_positions = None,
                 _follow_simulation = False,
                 _follow_simulation_save_path = "",
                 _simulation_name = "",
                 _recon_policy="local_XY"):
        
        print("Initializing Simulation class...", end = " ")

        self.RAW_img_array = _RAW_img_array

        self.plant_FT_pred_par_crop_rows = _plant_FT_pred_per_crop_rows

        self.OTSU_img_array = _OTSU_img_array

        self.data_input_DATAFRAME_COORD = _data_input_DATAFRAME_COORD

        self.group_size = _group_size
        self.group_step = _group_step

        self.RALs_fuse_factor = _RALs_fuse_factor
        self.RALs_fill_factor = _RALs_fill_factor

        self.recon_policy = _recon_policy
        
        self.ADJUSTED_img_plant_positions = _ADJUSTED_img_plant_positions
        if (self.ADJUSTED_img_plant_positions != None):
            self.Correct_Adjusted_plant_positions()
            self.labelled=True
        else:
            self.labelled=False

        self.field_offset = _field_offset

        self.simu_steps_times = []
        self.simu_steps_time_detailed=[]
        self.RALs_recorded_count = []
        self.nb_real_plants=0
        self.TP=0
        self.FP=0
        self.FN=0
        self.real_plant_detected_keys = []
        
        self.follow_simulation = _follow_simulation
        if (_follow_simulation):
            self.follow_simulation_save_path = _follow_simulation_save_path
            gIO.check_make_directory(self.follow_simulation_save_path)
            
        self.simulation_name = _simulation_name

        print("Done")

    def Initialize_AD(self):
        self.AD = Agents_Director(self.plant_FT_pred_par_crop_rows,
                             self.OTSU_img_array,
                             self.ADJUSTED_img_plant_positions,
                             self.data_input_DATAFRAME_COORD,
                             self.group_size, self.group_step,
                             self.RALs_fuse_factor, self.RALs_fill_factor,
                             self.field_offset, recon_policy=self.recon_policy)
        self.AD.Initialize_RowAs()


# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# PERFORM SIMULATION NEW END CRIT                   # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    # curved : _coerced_Y and _check_rows_proximity have to be False
    def Perform_Search_Simulation(self, _steps = 10,
                                  _coerced_X = False,
                                  _coerced_Y = False,
                                  _analyse_and_remove_Rows = False,
                                  _edge_exploration = True,
                                  _check_rows_proximity=False):
        
        print()
        print("Starting MAS Search Simulation:")
        self.steps = _steps
        self.max_steps_reached = False
        if (_analyse_and_remove_Rows):
            self.AD.Analyse_RowAs_Kmeans()

        # curved
        self.AD.ORDER_RowAs_to_Get_End_Of_Rows()

        self.AD.ORDER_RowAs_to_Set_RALs_Neighbours()
        self.AD.Aggregate_Rows()
        
        print("After first Set_RALs_neighbours")
        self.AD.ORDER_RowAs_to_Check_irrelevant_neighbours()
        
        
        self.AD.ORDER_RowAs_to_Update_InterPlant_Y()
        self.AD.Summarize_RowAs_InterPlant_Y()
        
        if (self.follow_simulation):
            self.Show_Adjusted_And_RALs_positions(_save=True,
                                                  _save_name=self.simulation_name+"_A")

        if (_edge_exploration):
            #self.Show_Adjusted_And_RALs_positions()
            #self.AD.ORDER_RowAs_for_Edges_Exploration()
            self.AD.ORDER_RowAs_for_Global_Exploration()
            #self.AD.ORDER_RowAs_for_Local_Exploration()
            if (self.follow_simulation):
                self.Show_Adjusted_And_RALs_positions(_save=True,
                                                      _save_name=self.simulation_name+"_B")
                
        print("After Edges exploration")
        self.AD.ORDER_RowAs_to_Check_irrelevant_neighbours()
        
        self.AD.ORDER_RowAs_to_Update_InterPlant_Y()

        self.Count_RALs()

        stop_simu = False
        re_eval = False
        diff_nb_RALs = -1
        i = 0
        while i < self.steps and not stop_simu:
            print("Simulation step {0}/{1} (max)".format(i+1, _steps))

            time_detailed=[]
            
            t0 = time.time()
            self.AD.ORDER_RowAs_for_RALs_mean_points()
            time_detailed += [time.time()-t0]

            if (_coerced_X):
                t0 = time.time()
                self.AD.ORDER_RowAs_to_Correct_RALs_X()
                time_detailed += [time.time()-t0]
            else:
                time_detailed += [0]

            if (_coerced_Y):
                t0 = time.time()
                self.AD.ORDER_RowAs_to_Correct_RALs_Y()
                time_detailed += [time.time()-t0]
            else:
                time_detailed += [0]
                
            t0 = time.time()
            self.AD.ORDER_RowAs_for_Moving_RALs_to_active_points()
            time_detailed += [time.time()-t0]
            
            if (self.follow_simulation):
                self.Show_Adjusted_And_RALs_positions(_save=True,
                                                      _save_name=self.simulation_name+"_C_{0}_1".format(i+1))

            t0 = time.time()
            self.AD.ORDER_RowAs_to_Adapt_RALs_sizes()
            time_detailed += [time.time()-t0]
            
            if (self.follow_simulation):
                self.Show_Adjusted_And_RALs_positions(_save=True,
                                                      _save_name=self.simulation_name+"_C_{0}_2".format(i+1))

            t0 = time.time()
            self.AD.ORDER_RowAs_Fill_or_Fuse_RALs()
            time_detailed += [time.time()-t0]
            
            print("After Fill_Or_Fuse_RALs")
            self.AD.ORDER_RowAs_to_Check_irrelevant_neighbours()
            
            if (self.follow_simulation):
                self.Show_Adjusted_And_RALs_positions(_save=True,
                                                      _save_name=self.simulation_name+"_C_{0}_3".format(i+1))
            
            # curved
# =============================================================================
#             t0 = time.time()
#             self.AD.ORDER_RowAs_to_Set_RALs_Neighbours()
#             time_detailed += [time.time()-t0]
# =============================================================================
            print("Before Destroy_Low_Activity_RALs")
            self.AD.ORDER_RowAs_to_Check_irrelevant_neighbours()

            t0 = time.time()
            self.AD.ORDER_RowAs_to_Destroy_Low_Activity_RALs()
            time_detailed += [time.time()-t0]
            
            print("After Destroy_Low_Activity_RALs")
            self.AD.ORDER_RowAs_to_Check_irrelevant_neighbours()
            
            if (self.follow_simulation):
                self.Show_Adjusted_And_RALs_positions(_save=True,
                                                      _save_name=self.simulation_name+"_C_{0}_4".format(i+1))

            # curved
            # update the neighbours : set the neigbours of the recently created RAL
            # and update the neighbours of the neighbours of destroyed RALs
            # Takes one second on one image approximately
            
            # t0 = time.time()
            # self.AD.ORDER_RowAs_to_Set_RALs_Neighbours()
            # time_detailed += [time.time()-t0]

            t0 = time.time()
            self.AD.ORDER_RowAs_to_Update_InterPlant_Y()
            time_detailed += [time.time()-t0]
            
            # Removes some of the rows at first step... Issue when a
            # row is fragmented, estimates that the two rows are too
            # close 
            # if _check_rows_proximity:
            #     t0 = time.time()
            #     self.AD.Check_RowAs_Proximity()
            #     time_detailed += [time.time()-t0]
            
            # t0 = time.time()
            # self.AD.Summarize_RowAs_InterPlant_Y()
            # time_detailed += [time.time()-t0]
            
            self.simu_steps_time_detailed += [time_detailed]
            self.simu_steps_times += [np.sum(time_detailed)]

            self.Count_RALs()

            diff_nb_RALs = self.RALs_recorded_count[-1] - self.RALs_recorded_count[-2]

            if (diff_nb_RALs == 0):
                if not re_eval:
                    self.AD.Summarize_RowAs_InterPlant_Y()
                    re_eval = True
                else:
                    stop_simu = True
            else:
                re_eval = False

            i += 1

        # self.AD.ORDER_RowAs_for_RALs_Surface_Compute()

        if (i == self.steps):
            self.max_steps_reached = True
            print("MAS simulation Finished with max steps reached.")
        else:
            print("MAS simulation Finished")

    def Correct_Adjusted_plant_positions(self):
        """
        Transform the plants position at the string format to integer.
        Also correct the vertical positions relatively to the image ploting origin.
        """
        self.corrected_adjusted_plant_positions = []
        self.real_plant_keys = []
        for adj_pos_string in self.ADJUSTED_img_plant_positions:
            self.corrected_adjusted_plant_positions += [[int(adj_pos_string["rotated_x"]),
                                                        int(adj_pos_string["rotated_y"])]]
            self.real_plant_keys += [str(adj_pos_string["instance_id"])]

    def Count_RALs(self):
        RALs_Count = 0
        for _RowA in self.AD.RowAs:
            RALs_Count += len(_RowA.RALs)
        self.RALs_recorded_count += [RALs_Count]
        print("Nombre de RALs : ", self.RALs_recorded_count)

    def Is_Plant_in_RAL_scanning_zone(self, _plant_pos, _RAL):
        """
        Computes if the position of a labelled plant is within the area of the
        image where RAs are spawn under the RAL command.
        """
        res = False
        if (abs(_plant_pos[0] - _RAL.x) <= _RAL.group_size and
            abs(_plant_pos[1] - _RAL.y) <= _RAL.group_size):
                res = True
        return res

    def Get_RALs_infos(self):
        """
        Returns the dictionnay that will contains the information relative to
        RALs
        """
        self.RALs_dict_infos = {}
        self.RALs_nested_positions=[]
        for _RowA in self.AD.RowAs:
            _row = []
            for _RAL in _RowA.RALs:
                _row.append([int(_RAL.x), int(_RAL.y)])
                self.RALs_dict_infos[str(_RAL.x) + "_" + str(_RAL.y)] = {
                "field_recorded_positions" : _RAL.field_recorded_positions,
                 "recorded_positions" : _RAL.recorded_positions,
                 "detected_plant" : "",
                 "RAL_group_size": _RAL.group_size,
                 "RAL_nb_white_pixels": _RAL.nb_contiguous_white_pixel,
                 "RAL_white_surface": _RAL.white_contigous_surface}
            self.RALs_nested_positions+=[_row]
        print()

    def Compute_Scores(self):
        """
        Computes :
            True positives (labelled plants with a RAL near it)
            False positives (RAL positioned far from a labelled plant)
            False negatives (labelled plant with no RAL near it)

        """
        associated_RAL = 0
        self.nb_real_plants = len(self.corrected_adjusted_plant_positions)
        for i in range(self.nb_real_plants):

            TP_found = False
            for _RowA in self.AD.RowAs:
                for _RAL in _RowA.RALs:
                    if (self.Is_Plant_in_RAL_scanning_zone(self.corrected_adjusted_plant_positions[i], _RAL)):
                        if not TP_found:
                            self.TP += 1
                            TP_found = True
                            self.RALs_dict_infos[str(_RAL.x) + "_" + str(_RAL.y)][
                                    "detected_plant"]=self.real_plant_keys[i]
                            self.real_plant_detected_keys += [self.real_plant_keys[i]]
                        associated_RAL += 1
        
        # self.FN = len(self.ADJUSTED_img_plant_positions) - self.TP
        self.FN = self.nb_real_plants - self.TP
        # self.FP = self.RALs_recorded_count[-1] - associated_RAL
        self.FP = self.RALs_recorded_count[-1] - self.TP
    
    def Show_RALs_Position(self,
                           _ax = None,
                           _color = 'g'):
        """
        Display the Otsu image with overlaying rectangles centered on RALs. The
        size of the rectangle corespond to the area covered by the RAs under the 
        RALs supervision.
        
        _ax (matplotlib.pyplot.axes, optional):
            The axes of an image on which we wish to draw the adjusted 
            position of the plants
        
        _color (optional,list of color references):
            The color of the rectangle representing a RAL.
        """
        
        if (_ax == None):
            fig, ax = plt.subplots(1)
            ax.imshow(self.OTSU_img_array)
        else:
            ax = _ax
        i = 0
        for _RowsA in self.AD.RowAs:
            _color = ["b", "c", "m", "y", "r", "green", "deeppink", "lightblue", "brown", "lightgreen"][i%10]
            for _RAL in _RowsA.RALs:
                #coordinates are the upper left corner of the rectangle
                rect = patches.Rectangle((_RAL.x-_RAL.group_size,
                                          _RAL.y-_RAL.group_size),
                                          2*_RAL.group_size,
                                          2*_RAL.group_size,
                                         linewidth=1,
                                         edgecolor=_color,
                                         facecolor='none')
                ax.add_patch(rect)
            i+=1
        plt.show()
# =============================================================================
#         plt.xlim(170, 270)
#         plt.ylim(1350, 1450)
# =============================================================================

    def Show_Adjusted_Positions(self, _ax = None, _color = "b"):
        """
        Display the adjusted positions of the plants.
        This is considered as the ground truth.

        _ax (matplotlib.pyplot.axes, optional):
            The axes of an image on which we wish to draw the adjusted
            position of the plants

        _color (string):
            color of the circles designating the plants
        """
        if (_ax == None):
            fig, ax = plt.subplots(1)
            ax.set_title("Adjusted positions of the plants")
            ax.imshow(self.OTSU_img_array)
        else:
            ax = _ax

        for [x,y] in self.corrected_adjusted_plant_positions:
            circle = patches.Circle((x,y),  # MODIFIED CURVED
                                    radius = 3,
                                    linewidth = 2,
                                    edgecolor = None,
                                    facecolor = _color)
            ax.add_patch(circle)

    def Show_Adjusted_And_RALs_positions(self,
                                        _colors_recorded = 'g',
                                        _color_adjusted = "r",
                                        _save=False,
                                        _save_name=""):
        
        fig = plt.figure(figsize=(5,5),dpi=300)
        ax = fig.add_subplot(111)
        ax.imshow(self.OTSU_img_array)
        
        self.Show_RALs_Position(_ax = ax,
                                _color = _colors_recorded)
# =============================================================================
#         self.Show_Adjusted_Positions(_ax = ax,
#                                      _color = _color_adjusted)
# =============================================================================
        plt.title(_save_name)
        plt.show()
        if (_save):
            fig.savefig(self.follow_simulation_save_path+"/"+_save_name)
            plt.close()

    def Show_RALs_Deicision_Scores(self):
        """
        Plot the Evolution of the decision score of each RALs in the simulation
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for _RowsA in self.AD.RowAs:
            for _RAL in _RowsA.RALs:
                ax.plot([i for i in range (len(_RAL.recorded_Decision_Score))],
                         _RAL.recorded_Decision_Score, marker = "o")
        ax.set_title("Show RALs decision scores")
    
    def Show_nb_RALs(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([i for i in range (len(self.RALs_recorded_count))],
                         self.RALs_recorded_count, marker = "o")
        ax.set_title("Show number of RALs")

class MetaSimulation(object):
    """
    This class manages the multi agent simulations on a list of images.
    In particular, it concentrates the information needed to make batches of
    tests and compare the results.
    We want to be able to compare the time of the simulations, the confusion
    matrix

    _simu_name (string):
        Name of the meta simulation to be used for results files reference.

    _path_output (string):
        The root directory where the results associated to the Meta simulation
        will be saved.

    _names_input_raw(list):
        _names of the images loaded in the _data_input_raw list

    _data_input_raw (list):
        The list of arrays containing the raw RGB images.
        This would be mostly used for results visualization.

    _data_input_PLANT_FT_PRED (list):
        The list of arrays containing the predicted positions of plants
        organized by rows.
        The lists corresponding to rows contain other lists of length 2 giving
        the predicted position of a plant under the convention
        [image_line, image_column].

    _data_input_OTSU (list):
        The list of arrays containing the OSTU segmented image on which the
        Multi Agent System is working.

    _group_size (int, optional with default value = 5):
        number of pixels layers around the leader on which we instanciate
        reactive agents.

    _group_step (int, optional with default value = 5):
        distance between two consecutive reactive agents.

    _RALs_fuse_factor (float, optional with default value = 0.5):
        The proportion of the inter-plant Y distance under which we decide to
        fuse 2 RALs of a same Row Agent.

    _RALs_fill_factor (float, optional with default value = 1.5):
        The proportion of the inter-plant Y distance above which we decide to
        fill the sapce between 2 RALs of a same Row Agent with new RALs.

    _simulation_step (int, optional with default value = 10):
        Max number of steps for each MAS simulations.

    _data_position_files (list, optional with default value = None):
        The list containing the adjusted positions of the plants coming from
        the csv files. So the positions are still in the string format.

    _field_shape (tuple, optional with default value = (2,2)):
        defines the number of images per rows and columns (first and second
        position respectively) that the drone captured from the field
    """

    def __init__(self,
                 _simu_name,
                 _path_output,
                 _names_input_raw,
                 _data_input_raw,
                 _data_input_PLANT_FT_PRED,
                 _data_input_OTSU,
                 _data_input_DATAFRAME_COORD,
                 _group_size, _group_step,
                 _RALs_fuse_factor, _RALs_fill_factor,
                 _simulation_step = 10,
                 _data_adjusted_position_files = None,
                 _field_shape = (2,2)):

        self.simu_name = _simu_name

        self.path_output = _path_output

        self.names_input_raw = _names_input_raw

        self.data_input_raw = _data_input_raw
        self.nb_images = len(self.data_input_raw)

        self.data_input_PLANT_FT_PRED = _data_input_PLANT_FT_PRED
        self.data_input_OTSU = _data_input_OTSU
        self.data_input_DATAFRAME_COORD = _data_input_DATAFRAME_COORD

        self.group_size = _group_size
        self.group_step = _group_step

        self.RALs_fuse_factor = _RALs_fuse_factor
        self.RALs_fill_factor = _RALs_fill_factor

        self.simulation_step = _simulation_step

        self.data_adjusted_position_files = _data_adjusted_position_files

        self.meta_simulation_results = {}
        self.whole_field_counted_plants = {}
        self.RALs_data = {}
        self.RALs_all_nested_positions=[]
        if (self.data_adjusted_position_files != None):
            self.Initialize_Whole_Field_Counted_Plants()

        self.field_shape = _field_shape

        self.check_data()

    def check_data(self):
        """
        Checks that the input data lists have the same length as the _data_input_raw

        """
        print("len(self.data_input_OTSU) : ", len(self.data_input_OTSU))
        print("len(self.data_input_PLANT_FT_PRED): ", len(self.data_input_PLANT_FT_PRED))
        for _data in [self.data_input_OTSU,
                      self.data_input_PLANT_FT_PRED]:
            assert len(_data) == self.nb_images

        if (self.data_adjusted_position_files != None):
            assert len(self.data_adjusted_position_files) == self.nb_images

    def Get_Field_Assembling_Offsets(self):

        origin_shape = np.array([self.data_input_raw[0].shape[1],
                                  self.data_input_raw[0].shape[0]])
        Otsu_shape = np.array([self.data_input_OTSU[0].shape[1],
                                  self.data_input_OTSU[0].shape[0]])

        p1 = np.array([self.data_input_raw[0].shape[1],
                       0.5 * self.data_input_raw[0].shape[0]])
        p2 = np.array([0.5 * self.data_input_raw[0].shape[1],
                       self.data_input_raw[0].shape[0]])
        pivot = np.array([0.5 * self.data_input_raw[0].shape[1],
                          0.5 * self.data_input_raw[0].shape[0]])

        R = rotation_matrix(np.deg2rad(80))

        print(p1, p2, pivot, R)

        right_offset = rotate_coord(p1, pivot, R) - 0.5*origin_shape
        up_offset = rotate_coord(p2, pivot, R) - 0.5* origin_shape
        print(right_offset, up_offset)

        self.all_offsets = []
        forward=True
        for i in range (self.field_shape[0]):
            if (forward):
                _start = 0
                _stop = self.field_shape[1]
                _step = 1
            else:
                _start = self.field_shape[1]-1
                _stop = -1
                _step = -1

            for j in range (_start, _stop, _step):
                new_offset = i * right_offset + j * up_offset
                self.all_offsets.append([int(new_offset[0]),
                                         int(Otsu_shape[1]-new_offset[1])])

            forward = not forward

        print("all offsets=", self.all_offsets)

    def Launch_Meta_Simu_Labels(self,
                             _coerced_X = False,
                             _coerced_Y = False,
                             _analyse_and_remove_Rows = False,
                             _rows_edges_exploration = True):

        """
        Launch an MAS simulation for each images. The raw images are labelled.
        """

        self.log = []

        self.coerced_X = _coerced_X
        self.coerced_Y = _coerced_Y
        self.analyse_and_remove_Rows = _analyse_and_remove_Rows
        self.rows_edges_exploration = _rows_edges_exploration

# =============================================================================
#         if (self.nb_images > 1):
#             self.Get_Field_Assembling_Offsets()
#         else:
#             self.all_offsets=[[0,0]]
# =============================================================================

        for i in range(self.nb_images):

            print()
            print("Simulation Definition for image {0}/{1}".format(i+1, self.nb_images) )

            try:
                MAS_Simulation = Simulation_MAS(
                                        self.data_input_raw[i],
                                        self.data_input_PLANT_FT_PRED[i],
                                        self.data_input_OTSU[i],
                                        self.data_input_DATAFRAME_COORD[i],
                                        self.group_size, self.group_step,
                                        self.RALs_fuse_factor, self.RALs_fill_factor,
                                        [0,0],
                                        self.data_adjusted_position_files[i],
                                        _follow_simulation = True,
                                        _follow_simulation_save_path = r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\processed\Field_0\GrowthStage_0\Output_Meta_Simulation\Session_1",
                                        _simulation_name = "Test"
                                    )
                MAS_Simulation.Initialize_AD()
                
                MAS_Simulation.Perform_Search_Simulation(self.simulation_step,
                                                         self.coerced_X,
                                                         self.coerced_Y,
                                                         self.analyse_and_remove_Rows,
                                                         self.rows_edges_exploration, 
                                                        )

                MAS_Simulation.Get_RALs_infos()
                self.Add_Simulation_Results(i, MAS_Simulation)
                self.Add_Whole_Field_Results(MAS_Simulation)
                if (MAS_Simulation.max_steps_reached):
                    self.log += ["Simulation for image {0}/{1}, named {2} reached max number of allowed steps".format(
                        i+1, self.nb_images, self.names_input_raw[i])]

            except Exception as ex:
                self.log += ["Simulation for image {0}/{1}, named {2} failed".format(
                        i+1, self.nb_images, self.names_input_raw[i])]
                print("Failure, exception {0} has occured".format(type(ex).__name__))
                print("Exception Arguments : ", ex.args)
                print("Logs : ", self.log)
                raise 

        self.Save_MetaSimulation_Results()
        self.Save_RALs_Infos()
        self.Save_Whole_Field_Results()
        self.Save_RALs_Nested_Positions()
        self.Save_Log()

    def Launch_Meta_Simu_NoLabels(self,
                             _coerced_X = False,
                             _coerced_Y = False,
                             _analyse_and_remove_Rows = False,
                             _rows_edges_exploration = True,
                             ):

        """
        Launch an MAS simulation for each images. The raw images are NOT labelled.
,        """

        self.log = []

        self.coerced_X = _coerced_X
        self.coerced_Y = _coerced_Y
        self.analyse_and_remove_Rows = _analyse_and_remove_Rows
        self.rows_edges_exploration = _rows_edges_exploration

# =============================================================================
#         if (self.nb_images > 1):
#             self.Get_Field_Assembling_Offsets()
#         else:
#             self.all_offsets=[[0,0] for i in range(self.nb_images)]
# =============================================================================

        for i in range(self.nb_images):

            print()
            print("Simulation Definition for image {0}/{1}".format(i+1, self.nb_images))

            try:
                MAS_Simulation = Simulation_MAS(
                                        self.data_input_raw[i],
                                        self.data_input_PLANT_FT_PRED[i],
                                        self.data_input_OTSU[i],
                                        self.group_size, self.group_step,
                                        self.RALs_fuse_factor, self.RALs_fill_factor,
                                        [0,0],
                                        self.data_adjusted_position_files, 
                                        _follow_simulation=True,
                                        _simulation_name = "Test2",
                                        _follow_simulation_save_path = r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\processed\Field_0\GrowthStage_0\Output_Meta_Simulation\Session_1",)
                MAS_Simulation.Initialize_AD()
                
                MAS_Simulation.Perform_Search_Simulation(self.simulation_step,
                                                         self.coerced_X,
                                                         self.coerced_Y,
                                                         self.analyse_and_remove_Rows,
                                                         self.rows_edges_exploration)

                MAS_Simulation.Get_RALs_infos()
                self.Add_Simulation_Results(i, MAS_Simulation)
                if (MAS_Simulation.max_steps_reached):
                    self.log += ["Simulation for image {0}/{1}, named {2} reached max number of allowed steps".format(
                        i+1, self.nb_images, self.names_input_raw[i])]

            except:
                print("Failure")
                self.log += ["Simulation for image {0}/{1}, named {2} failed".format(
                        i+1, self.nb_images, self.names_input_raw[i])]
                raise

        self.Save_MetaSimulation_Results()
        self.Save_RALs_Infos()
        self.Save_RALs_Nested_Positions()
        self.Save_Log()

    def Get_Simulation_Results(self, _MAS_Simulation):

        """
        Gathers the general simulation results
        """

        if (self.data_adjusted_position_files != None):
            print("Computing Scores by comparing to the labellisation...", end = " ")
            _MAS_Simulation.Compute_Scores()
            print("Done")

        data = {"Time_per_steps": _MAS_Simulation.simu_steps_times,
                "Time_per_steps_detailes": _MAS_Simulation.simu_steps_time_detailed,
                "Image_Labelled": _MAS_Simulation.labelled,
                "NB_labelled_plants": _MAS_Simulation.nb_real_plants,
                "NB_RALs" : _MAS_Simulation.RALs_recorded_count[-1],
                "TP" : _MAS_Simulation.TP,
                "FN" : _MAS_Simulation.FN,
                "FP" : _MAS_Simulation.FP,
                "InterPlantDistance": _MAS_Simulation.AD.InterPlant_Y,
                "RAL_Fuse_Factor": _MAS_Simulation.RALs_fuse_factor,
                "RALs_fill_factor": _MAS_Simulation.RALs_fill_factor,
                "RALs_recorded_count": _MAS_Simulation.RALs_recorded_count}

        print(_MAS_Simulation.simu_steps_times)
        print("NB Rals =", _MAS_Simulation.RALs_recorded_count[-1])
        print("Image Labelled = ", _MAS_Simulation.labelled)
        print("NB_labelled_plants", _MAS_Simulation.nb_real_plants)
        print("TP =", _MAS_Simulation.TP)
        print("FN =", _MAS_Simulation.FN)
        print("FP =", _MAS_Simulation.FP)

        return data

    def Initialize_Whole_Field_Counted_Plants(self):
        """
        Initialize the keys of the dictionnary self.whole_field_counted_plants
        """
        for i in range (self.nb_images):
            for adj_pos_string in self.data_adjusted_position_files[i]:
                self.whole_field_counted_plants[str(adj_pos_string["instance_id"])]=0

    def Add_Simulation_Results(self, _image_index, _MAS_Simulation):
        """
        Add the detection results of a MAS simulation to the
        meta_simulation_results dictionary as well as the RALs information.
        """

        data = self.Get_Simulation_Results(_MAS_Simulation)
        self.meta_simulation_results[self.names_input_raw[_image_index]] = data

        self.RALs_data[self.names_input_raw[_image_index]] = _MAS_Simulation.RALs_dict_infos
        self.RALs_all_nested_positions.append(_MAS_Simulation.RALs_nested_positions)

    def Add_Whole_Field_Results(self, _MAS_Simulation):
        """
        Retrieves the real x_y coordinates of the plants that were detected in the
        simulation and fills the dictionary self.whole_field_counted_plants
        """
        for _key in _MAS_Simulation.real_plant_detected_keys:
            self.whole_field_counted_plants[_key] += 1

    def Make_File_Name(self, _base):
        """
        build generic names depending on the options of the simulation
        """
        _name = _base
        if (self.coerced_X):
            _name+= "_cX"
        if (self.coerced_Y):
            _name+= "_cY"
        if (self.analyse_and_remove_Rows):
            _name+="_anrR2"
        if (self.rows_edges_exploration):
            _name+="_REE"
        return _name

    def Save_MetaSimulation_Results(self):
        """
        saves the results of the MAS simulations stored in the
        meta_simulation_results dictionary as a JSON file.
        """
        name = self.Make_File_Name("MetaSimulationResults_Curve_"+self.simu_name)

        file = open(self.path_output+"/"+name+".json", "w")
        json.dump(self.meta_simulation_results, file, indent = 3)
        file.close()

    def Save_RALs_Infos(self):
        """
        saves the results of the MAS simulations stored in the
        meta_simulation_results dictionary as a JSON file.
        """
        name = self.Make_File_Name("RALs_Infos_Curve_"+self.simu_name)
        file = open(self.path_output+"/"+name+".json", "w")
        json.dump(self.RALs_data, file, indent = 2)
        file.close()

    def Save_RALs_Nested_Positions(self):
        """
        saves all the RALs position on the image. It makes one json file per
        image. The json file is in the exact same format as The plant predictions
        on
        """
        name = self.Make_File_Name("RALs_NestedPositions_Curve_"+self.simu_name)
        _path=self.path_output+"/"+name
        gIO.check_make_directory(_path)
        counter = 0
        for _nested_pos in self.RALs_all_nested_positions:
            name = self.names_input_raw[counter]+"NestedPositions"
            file = open(_path+"/"+name+".json", "w")
            json.dump(_nested_pos, file, indent = 2)
            file.close()
            counter+=1

    def Save_Whole_Field_Results(self):
        """
        saves the results of the MAS simulations stored in the
        whole_field_counted_plants dictionary as a JSON file.
        """
        name = self.Make_File_Name("WholeFieldResults_Curve_"+self.simu_name)
        file = open(self.path_output+"/"+name+".json", "w")
        json.dump(self.whole_field_counted_plants, file, indent = 2)
        file.close()

    def Save_Log(self):
        name = self.Make_File_Name("LOG_MetaSimulationResults_Curve_"+self.simu_name)
        gIO.writer(self.path_output, name+".txt", self.log, True, True)
