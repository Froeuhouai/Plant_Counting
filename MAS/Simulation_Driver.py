import os
import json
import numpy as np
from Simulation import Simulation_MAS

os.chdir("../Utility")
import general_IO as gIO

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
    based on the rotation matrix _R. If _int is set to True, then the coordinates
    are integers.
    _p, _pivot and _R must be numpy arrays
    """
    _r_new_point = np.dot(_R, _p - _pivot) + _pivot
    
    return _r_new_point

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
        
        R = rotation_matrix(np.deg2rad(75))
        
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
                             _rows_edges_exploration = False):

        """
        Launch an MAS simulation for each images. The raw images are labelled.
        """
        
        self.log = []
        
        self.coerced_X = _coerced_X
        self.coerced_Y = _coerced_Y
        self.analyse_and_remove_Rows = _analyse_and_remove_Rows
        self.rows_edges_exploration = _rows_edges_exploration
        
        if (self.nb_images > 1):
            self.Get_Field_Assembling_Offsets()
        else:
            self.all_offsets=[[0,0]]
        
        for i in range(self.nb_images):
            
            print()
            print("Simulation Definition for image {0}/{1}".format(i+1, self.nb_images) )
            
            try:
                MAS_Simulation = Simulation_MAS(
                                        self.data_input_raw[i],
                                        self.data_input_PLANT_FT_PRED[i],
                                        self.data_input_OTSU[i],
                                        self.group_size, self.group_step,
                                        self.RALs_fuse_factor, self.RALs_fill_factor,
                                        self.all_offsets[i],
                                        self.data_adjusted_position_files[i])
                MAS_Simulation.Initialize_AD()
                
                MAS_Simulation.Perform_Search_Simulation(self.simulation_step,
                                                       self.coerced_X,
                                                       self.coerced_Y,
                                                       self.analyse_and_remove_Rows,
                                                       self.rows_edges_exploration)
                
                MAS_Simulation.Get_RALs_infos()
                self.Add_Simulation_Results(i, MAS_Simulation)
                self.Add_Whole_Field_Results(MAS_Simulation)
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
        self.Save_Whole_Field_Results()
        self.Save_RALs_Nested_Positions()
        self.Save_Log()

# ========================================================== ONLY FOR STUDY - TO BE DELETED
    def Launch_GROWTH_Meta_Simu_Labels(self,
                             _coerced_X = False,
                             _coerced_Y = False,
                             _analyse_and_remove_Rows = False,
                             _rows_edges_exploration = False):

        """
        Launch an MAS simulation for each images. The raw images are labelled.
        """
        
        self.log = []
        
        self.coerced_X = _coerced_X
        self.coerced_Y = _coerced_Y
        self.analyse_and_remove_Rows = _analyse_and_remove_Rows
        self.rows_edges_exploration = _rows_edges_exploration
        
        if (self.nb_images > 1):
            self.Get_Field_Assembling_Offsets()
        else:
            self.all_offsets=[[0,0]]
        
        for i in range(self.nb_images):
            
            print()
            print("Simulation Definition for image {0}/{1}".format(i+1, self.nb_images) )
            
            try:
                MAS_Simulation = Simulation_MAS(
                                        self.data_input_raw[i],
                                        self.data_input_PLANT_FT_PRED[i],
                                        self.data_input_OTSU[i],
                                        self.group_size, self.group_step,
                                        self.RALs_fuse_factor, self.RALs_fill_factor,
                                        self.all_offsets[i],
                                        self.data_adjusted_position_files[i])
                MAS_Simulation.Initialize_AD()
                
                MAS_Simulation.Perform_Growth_Simulation(self.simulation_step)
                
                MAS_Simulation.Get_RALs_infos()
                self.Add_Simulation_Results(i, MAS_Simulation)
                self.Add_Whole_Field_Results(MAS_Simulation)
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
        self.Save_Whole_Field_Results()
        self.Save_RALs_Nested_Positions()
        self.Save_Log()

    def Launch_PA_GROWTH_Analysis(self,
                                 _coerced_X = False,
                                 _coerced_Y = False,
                                 _analyse_and_remove_Rows = False,
                                 _rows_edges_exploration = False):

        """
        Launch an MAS simulation for each images. The raw images are labelled.
        """
        
        self.log = []
        
        self.coerced_X = _coerced_X
        self.coerced_Y = _coerced_Y
        self.analyse_and_remove_Rows = _analyse_and_remove_Rows
        self.rows_edges_exploration = _rows_edges_exploration
        
        if (self.nb_images > 1):
            self.Get_Field_Assembling_Offsets()
        else:
            self.all_offsets=[[0,0]]
        
        for i in range(self.nb_images):
            
            print()
            print("Simulation Definition for image {0}/{1}".format(i+1, self.nb_images) )
            
            try:
                MAS_Simulation = Simulation_MAS(
                                        self.data_input_raw[i],
                                        self.data_input_PLANT_FT_PRED[i],
                                        self.data_input_OTSU[i],
                                        self.group_size, self.group_step,
                                        self.RALs_fuse_factor, self.RALs_fill_factor,
                                        self.all_offsets[i],
                                        self.data_adjusted_position_files[i])
                MAS_Simulation.Initialize_AD()
                
               
                MAS_Simulation.Perform_PA_Growth_Analysis(self.simulation_step)
                ########MAS_Simulation.Perform_Growth_Simulation(self.simulation_step)
                
                MAS_Simulation.Get_RALs_infos()
                self.Add_Simulation_Results(i, MAS_Simulation)
                self.Add_Whole_Field_Results(MAS_Simulation)
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
        self.Save_Whole_Field_Results()
        self.Save_RALs_Nested_Positions()
        self.Save_Log()

# ============================================================================

        
    def Launch_Meta_Simu_NoLabels(self,
                             _coerced_X = False,
                             _coerced_Y = False,
                             _analyse_and_remove_Rows = False,
                             _rows_edges_exploration = False):

        """
        Launch an MAS simulation for each images. The raw images are NOT labelled.
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
            print("Simulation Definition for image {0}/{1}".format(i+1, self.nb_images))
            
            try:
                MAS_Simulation = Simulation_MAS(
                                        self.data_input_raw[i],
                                        self.data_input_PLANT_FT_PRED[i],
                                        self.data_input_OTSU[i],
                                        self.group_size, self.group_step,
                                        self.RALs_fuse_factor, self.RALs_fill_factor,
                                        [0,0],
                                        self.data_adjusted_position_files)
                MAS_Simulation.Initialize_AD()
                
                MAS_Simulation.Perform_Simulation(self.simulation_step,
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
        Gathers the generalk simulation results
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
# =============================================================================
#                 [_rx, _ry, x, y] = adj_pos_string.split(",")
#                 self.whole_field_counted_plants[_rx + "_" + _ry]=0
# =============================================================================
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
        name = self.Make_File_Name("MetaSimulationResults_v16_"+self.simu_name)
        
        file = open(self.path_output+"/"+name+".json", "w")
        json.dump(self.meta_simulation_results, file, indent = 3)
        file.close()
    
    def Save_RALs_Infos(self):
        """
        saves the results of the MAS simulations stored in the 
        meta_simulation_results dictionary as a JSON file.
        """
        name = self.Make_File_Name("RALs_Infos_v16_"+self.simu_name)
        file = open(self.path_output+"/"+name+".json", "w")
        json.dump(self.RALs_data, file, indent = 2)
        file.close()
    
    def Save_RALs_Nested_Positions(self):
        """
        saves all the RALs position on the image. It makes one json file per
        image. The json file is in the exact same format as The plant predictions
        on
        """
        name = self.Make_File_Name("RALs_NestedPositions_v16_"+self.simu_name)
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
        name = self.Make_File_Name("WholeFieldResults_v16_"+self.simu_name)
        file = open(self.path_output+"/"+name+".json", "w")
        json.dump(self.whole_field_counted_plants, file, indent = 2)
        file.close()
    
    def Save_Log(self):
        name = self.Make_File_Name("LOG_MetaSimulationResults_v16_"+self.simu_name)
        gIO.writer(self.path_output, name+".txt", self.log, True, True)
        