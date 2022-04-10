import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PA import ReactiveAgent_Leader

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