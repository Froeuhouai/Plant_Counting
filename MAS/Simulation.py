import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from DA import Director_Agent

os.chdir("../Utility")
import general_IO as gIO

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
        self.AD = Director_Agent(self.plant_FT_pred_par_crop_rows,
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