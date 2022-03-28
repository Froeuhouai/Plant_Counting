
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from DA import Director_Agent


os.chdir("../Utility")
import general_IO as gIO

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
    """
    
    def __init__(self, _RAW_img_array,
                 _plant_FT_pred_per_crop_rows, _OTSU_img_array, 
                 _group_size = 50, _group_step = 5,
                 _RALs_fuse_factor = 0.5, _RALs_fill_factor = 1.5,
                 _field_offset = [0,0],
                 _ADJUSTED_img_plant_positions = None,
                 _follow_simulation = False,
                 _follow_simulation_save_path = "",
                 _simulation_name = ""):
        
        print("Initializing Simulation class...", end = " ")
        
        self.RAW_img_array = _RAW_img_array
        
        self.plant_FT_pred_par_crop_rows = _plant_FT_pred_per_crop_rows
        
        self.OTSU_img_array = _OTSU_img_array        
        
        self.group_size = _group_size
        self.group_step = _group_step
        
        self.RALs_fuse_factor = _RALs_fuse_factor
        self.RALs_fill_factor = _RALs_fill_factor
        
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
                             self.group_size, self.group_step,
                             self.RALs_fuse_factor, self.RALs_fill_factor,
                             self.field_offset)
        self.AD.Initialize_RowAs()
    
    def Perform_Search_Simulation(self, _steps = 10,
                                   _coerced_X = False,
                                   _coerced_Y = False,
                                   _analyse_and_remove_Rows = False,
                                   _edge_exploration = False):
        print()
        print("Starting Search simulation:")
        self.steps = _steps
        self.max_steps_reached = False
        
        if (_analyse_and_remove_Rows):
            self.AD.Analyse_RowAs_Kmeans()
        
        self.AD.ORDER_RowAs_to_Update_InterPlant_Y()
        self.AD.Summarize_RowAs_InterPlant_Y()
        
        #For Debug purposes
        self.search_simulation = True
        self.growth_simulation = False
        
        if (self.follow_simulation):
            self.Show_Adjusted_And_RALs_positions(_save=True,
                                                  _save_name=self.simulation_name+"_A")
        
        if (_edge_exploration):
            self.AD.ORDER_Rows_for_Edges_Exploration()
            if (self.follow_simulation):
                self.Show_Adjusted_And_RALs_positions(_save=True,
                                                      _save_name=self.simulation_name+"_B")
        
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
            
# =============================================================================
#             t0 = time.time()
#             self.AD.ORDER_RowAs_to_Adapt_RALs_sizes()
#             time_detailed += [time.time()-t0]
#             
#             if (self.follow_simulation):
#                 self.Show_Adjusted_And_RALs_positions(_save=True,
#                                                       _save_name=self.simulation_name+"_C_{0}_2".format(i+1))
# =============================================================================
            
            t0 = time.time()
            self.AD.ORDER_RowAs_Fill_or_Fuse_RALs()
            time_detailed += [time.time()-t0]
            
            if (self.follow_simulation):
                self.Show_Adjusted_And_RALs_positions(_save=True,
                                                      _save_name=self.simulation_name+"_C_{0}_3".format(i+1))
            
            t0 = time.time()
            self.AD.ORDER_RowAs_to_Destroy_Low_Activity_RALs()
            time_detailed += [time.time()-t0]
            
            if (self.follow_simulation):
                self.Show_Adjusted_And_RALs_positions(_save=True,
                                                      _save_name=self.simulation_name+"_C_{0}_4".format(i+1))
            
            t0 = time.time()
            self.AD.Check_Rows_Proximity()
            time_detailed += [time.time()-t0]
            
            t0 = time.time()
            self.AD.ORDER_RowAs_to_Update_InterPlant_Y()
            time_detailed += [time.time()-t0]
            
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
        
        if (i == self.steps):
            self.max_steps_reached = True
            print("MAS simulation Finished with max steps reached.")
        else:
            print("MAS simulation Finished")
    
    def Perform_Growth_Simulation(self, _steps = 10):
        print()
        print("Starting Growth Simulation:")
        self.AD.Switch_From_Search_To_Growth()
        
        if (len(self.RALs_recorded_count)==0):
            self.Count_RALs()
        
        self.steps = _steps
        self.max_steps_reached = False
        
        #For Debug purposes
        self.search_simulation = False
        self.growth_simulation = True
        
        if (self.follow_simulation):
                self.Show_Adjusted_And_RALs_positions(_save=True,
                                                      _save_name=self.simulation_name+"_D")
        
        stop_simu = False
        i = 0
        while i < self.steps and not stop_simu:
            print("Simulation step {0}/{1} (max)".format(i+1, _steps))
            
            self.AD.ORDER_RowAs_to_Adapt_RALs_sizes()
            
# =============================================================================
#             if (self.follow_simulation):
#                 self.Show_Adjusted_And_RALs_positions(_save=True,
#                                                       _save_name=self.simulation_name+"_E_{0}_2".format(i+1))
# =============================================================================
                
            self.AD.ORDER_Check_RALs_Exploration()
            
            if (self.follow_simulation):
                self.Show_Adjusted_And_RALs_positions(_save=True,
                                                      _save_name=self.simulation_name+"_E_{0}_3".format(i+1))
            
            self.AD.ORDER_ROWAs_to_Destroy_Small_RALs()
            self.Count_RALs()
            
            nb_fixed = self.AD.Count_nb_Fixed_RAL()
            
            print("Proportion of fixed RAs:", nb_fixed/self.RALs_recorded_count[-1])
            if (nb_fixed/self.RALs_recorded_count[-1] == 1):
                stop_simu = True
            
            i += 1
        
        #self.AD.ORDER_RowAs_for_RALs_Surface_Compute()
        
        if (i == self.steps):
            self.max_steps_reached = True
            print("Growth simulation Finished with max steps reached.")
        else:
            print("Growth simulation Finished")

#================================== FOR ANALYSES =============================

    def compute_next_square_layer(self, n):
        
        coords=[]
        #y=-n fixed and x moving --> North Border
        for _x in range(-n+1, n):
            coords += [(_x,-n)]
            
        #y=n fixed and x moving --> South Border    
        for _x in range(-n+1, n):
            coords += [(_x,n)]
        
        #x=-n fixed and y moving --> West Border
        for _y in range(-n+1, n):
            coords += [(-n,_y)]
            
        #x=n fixed and y moving --> East Border    
        for _y in range(-n+1, n):
            coords += [(n,_y)]
        
        #NW corner
        coords += [(-n, -n)]
        
        #NE corner
        coords += [(+n, -n)]
        
        #SW corner
        coords += [(-n, +n)]
        
        #SE corner
        coords += [(+n, +n)]
        
        return coords

    def Perform_PA_Growth_Analysis(self, _steps = 10):
            print()
            print("Starting Growth Simulation:")
            #self.AD.Switch_From_Search_To_Growth()
            self.AD.Init_Growth_Analysis()
            
            if (len(self.RALs_recorded_count)==0):
                self.Count_RALs()
            
            self.steps = _steps
            self.max_steps_reached = False
            
            #For Debug purposes
            self.search_simulation = False
            self.growth_simulation = True
            
# =============================================================================
#             if (self.follow_simulation):
#                     self.Show_Adjusted_And_RALs_positions(_save=True,
#                                                           _save_name=self.simulation_name+"_D")
# =============================================================================
            
            stop_simu = False
            i = 0
            while i < self.steps and not stop_simu:
                print("Simulation step {0}/{1} (max)".format(i+1, _steps))
                
                #self.AD.ORDER_RowAs_to_Adapt_RALs_sizes()
                next_coords = self.compute_next_square_layer(i+1)
                #print(next_coords)
                self.AD.ORDER_RowAs_to_Adapt_RALs_sizes_2(next_coords)
                
    # =============================================================================
    #             if (self.follow_simulation):
    #                 self.Show_Adjusted_And_RALs_positions(_save=True,
    #                                                       _save_name=self.simulation_name+"_E_{0}_2".format(i+1))
    # =============================================================================
                    
# =============================================================================
#                 self.AD.ORDER_Check_RALs_Exploration()
#                 
#                 if (self.follow_simulation):
#                     self.Show_Adjusted_And_RALs_positions(_save=True,
#                                                           _save_name=self.simulation_name+"_E_{0}_3".format(i+1))
#                 
#                 self.AD.ORDER_ROWAs_to_Destroy_Small_RALs()
# =============================================================================
                #self.Count_RALs()
                
# =============================================================================
#                 nb_fixed = self.AD.Count_nb_Fixed_RAL()
#                 
#                 print("Proportion of fixed RAs:", nb_fixed/self.RALs_recorded_count[-1])
#                 if (nb_fixed/self.RALs_recorded_count[-1] == 1):
#                     stop_simu = True
# =============================================================================
                
                i += 1
            
            #self.AD.ORDER_RowAs_for_RALs_Surface_Compute()
            
            if (i == self.steps):
                self.max_steps_reached = True
                print("Growth simulation Finished with max steps reached.")
            else:
                print("Growth simulation Finished")
#=============================================================================
    
    def Correct_Adjusted_plant_positions(self):
        """
        Transform the plants position from the string format to integer.
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
            RALs_Count += _RowA.nb_RALs
        self.RALs_recorded_count += [RALs_Count]
    
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
                 "RAL_Border_Distance": [int(_d) for _d in _RAL.Borders_Distance],
                 "Wpx_Count": _RAL.all_wpx_count,
                 "Max_Wpx_Count": _RAL.all_max_wpx}
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
        
        self.FN = len(self.ADJUSTED_img_plant_positions) - self.TP
        self.FP = self.RALs_recorded_count[-1] - associated_RAL
    
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
        
        for _RowsA in self.AD.RowAs:
            
            for _RAL in _RowsA.RALs:
                #coordinates are the upper left corner of the rectangle
                rect = patches.Rectangle((_RAL.x-_RAL.group_size,
                                          _RAL.y-_RAL.group_size),
                                          2*_RAL.group_size,
                                          2*_RAL.group_size,
                                         linewidth=1,
                                         edgecolor=_color,
                                         facecolor='none',
                                         alpha = 0.5)
                ax.add_patch(rect)
    
    def Show_RALs_RAs_Position(self,
                           _ax = None,
                           _colors = ["red", "orange", "purple", "cyan"]):
        """
        Display the Otsu image with overlaying lines representing the RAs of each
        border of each RAL during the growth simulation.
        
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
        
        for _RowsA in self.AD.RowAs:
            
            for _RAL in _RowsA.RALs:
                for i in range(4):
                    x = []
                    y = []
                    for _RA in _RAL.RA_list_card[i]:
                        x += [_RA.global_x]
                        y += [_RA.global_y]
                    plt.plot(x, y,
                             color=_colors[i], alpha = 0.5,
                             linewidth=1, markersize = 1)
        
# =============================================================================
#         plt.xlim(170, 270)
#         plt.ylim(1350, 1450)
# =============================================================================
    
    def Show_Adjusted_Positions(self, _ax = None, _color = "r"):
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
            ax.imshow(self.OTSU_img_array)
        else:
            ax = _ax
        
        for [x,y] in self.corrected_adjusted_plant_positions:
            circle = patches.Circle((x,y),
                                    radius = 10,
                                    facecolor = _color,
                                    alpha = 0.5)
            ax.add_patch(circle)
    
    def Show_Adjusted_And_RALs_positions(self,
                                        _recorded_position_indeces = -1,
                                        _color_recorded = 'g',
                                        _color_adjusted = "r",
                                        _colors_RAs = ["red", "orange", "purple", "cyan"],
                                        _save=False,
                                        _save_name=""):
        
        fig = plt.figure(figsize=(5,5),dpi=300)
        ax = fig.add_subplot(111)
        ax.imshow(self.OTSU_img_array)
        
        if (self.search_simulation and not self.growth_simulation):
            self.Show_RALs_Position(_ax = ax,
                                    _color = _color_recorded)
        
        if (self.growth_simulation and not self.search_simulation):
            self.Show_RALs_RAs_Position(_ax = ax,
                                    _colors = _colors_RAs)
        
        self.Show_Adjusted_Positions(_ax = ax,
                                     _color = _color_adjusted)
        
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
    
    def Show_nb_RALs(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([i for i in range (len(self.RALs_recorded_count))],
                         self.RALs_recorded_count, marker = "o")