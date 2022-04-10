import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pandas as pd
import random
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
import time

from RA import Row_Agent
from PA import ReactiveAgent_Leader

os.chdir("../Clustering_Scanner")
import Clustering 

class Director_Agent(object):
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
            #angles = []
            for _RAL in RALs_list[:2] :
                _RAL.Compute_neighbours_distances()
                _RAL.Compute_closest_neighbour_translation()
                
                for _RAL_2 in RALs_list[2:] :
                    if _RAL != _RAL_2 :
                        _RAL_2.Compute_neighbours_distances()
                        _RAL_2.Compute_closest_neighbour_translation()
                            
                        #euclidean_distance = math.sqrt((_RAL_2.x - _RAL.x)**2 + (_RAL_2.y - _RAL.y)**2) #On calcule la translation entre _RAL_2 et _RAL
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