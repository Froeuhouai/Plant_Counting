import numpy as np
from PA import ReactiveAgent_Leader


class Row_Agent(object):
    """
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
                 _field_offset = [0,0]):
        
        self.plant_FT_pred_in_crop_row = _plant_FT_pred_in_crop_row
        
        self.OTSU_img_array = _OTSU_img_array
        
        self.group_size = _group_size
        self.group_step = _group_step
        
        self.field_offset = _field_offset
        
        self.RALs = []
        self.nb_RALs = 0
        self.nb_Fixed_RALs = 0
        
        self.extensive_init = False
        
        self.Initialize_RALs()
        
        self.Get_Row_Mean_X()


    def Initialize_RALs(self):
        """
        Go through the predicted coordinates of the plants in self.plant_FT_pred_par_crop_rows
        and initialize RALs at these places.
        
        """        
        for _plant_pred in self.plant_FT_pred_in_crop_row:
            RAL = ReactiveAgent_Leader(_x = _plant_pred[0],
                                       _y = _plant_pred[1],
                                       _img_array = self.OTSU_img_array,
                                       _group_size = self.group_size,
                                       _group_step = self.group_step,
                                       _field_offset = self.field_offset)
            
            self.RALs += [RAL]
            self.nb_RALs += 1
        
# =============================================================================
#     def Extensive_Init(self, _filling_step):
#         """
#         Uses the first RAL in the self.RALs list to extensively instanciate
#         RALs between the bottom and the top of the image.
#         """        
#         self.extensive_init = True
#         
#         _RAL_ref_index = 0
#         _RAL_ref = self.RALs[_RAL_ref_index]
#         
#         y_init = _RAL_ref.y
#         while y_init + _filling_step < self.OTSU_img_array.shape[0]:
#             new_RAL = ReactiveAgent_Leader(_x = self.Row_Mean_X,
#                                            _y = int(y_init + _filling_step),
#                                            _img_array = self.OTSU_img_array,
#                                            _group_size = self.group_size,
#                                            _group_step = self.group_step,
#                                            _field_offset = self.field_offset)
#             new_RAL.used_as_filling_bound = False
#             y_init += _filling_step
#             
#             self.RALs += [new_RAL]
#                 
#         
#         y_init = _RAL_ref.y
#         new_RALs = []
#         new_diffs = []
#         while y_init - _filling_step > 0:
#             new_RAL = ReactiveAgent_Leader(_x = self.Row_Mean_X,
#                                            _y = int(y_init + _filling_step),
#                                            _img_array = self.OTSU_img_array,
#                                            _group_size = self.group_size,
#                                            _group_step = self.group_step)
#             new_RAL.used_as_filling_bound = False
#             
#             new_RALs += [new_RAL]
#             new_diffs += [_filling_step]
#             
#             y_init -= _filling_step
#         
#         self.RALs = new_RALs + self.RALs
#         
#         a = np.array([RAL.y for RAL in self.RALs])
#         b = np.argsort(a)
#         self.RALs = list(np.array(self.RALs)[b])
# =============================================================================
    
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
                                           _img_array = self.OTSU_img_array,
                                           _group_size = self.group_size,
                                           _group_step = self.group_step,
                                           _field_offset = self.field_offset)
            new_RAL.used_as_filling_bound = True
            y_init += _filling_step
            
            self.RALs += [new_RAL]
            self.nb_RALs += 1
                
        _RAL_ref_index = 0
        _RAL_ref = self.RALs[_RAL_ref_index]
        y_init = _RAL_ref.y
        new_RALs = []
        new_diffs = []
        while y_init - _filling_step > 0:
            new_RAL = ReactiveAgent_Leader(_x = self.Row_Mean_X,
                                           _y = int(y_init + _filling_step),
                                           _img_array = self.OTSU_img_array,
                                           _group_size = self.group_size,
                                           _group_step = self.group_step)
            new_RAL.used_as_filling_bound = True
            
            new_RALs += [new_RAL]
            self.nb_RALs += 1
            new_diffs += [_filling_step]
            
            y_init -= _filling_step
        
        self.RALs = new_RALs + self.RALs
        
        a = np.array([RAL.y for RAL in self.RALs])
        b = np.argsort(a)
        self.RALs = list(np.array(self.RALs)[b])
    
    def Fuse_RALs(self, _start, _stop):
        """
        _start and _stop are the indeces of the RALs to fuse so that they 
        correspond to the bounderies [_start _stop[
        """
        
# =============================================================================
#         print("Fusing procedure...")
# =============================================================================
        
        fusion_RAL_x = 0
        fusion_RAL_y = 0
        
        for _RAL in self.RALs[_start:_stop+1]:
            fusion_RAL_x += _RAL.x
            fusion_RAL_y += _RAL.y
            
        fusion_RAL = ReactiveAgent_Leader(_x = int(fusion_RAL_x/(_stop+1-_start)),
                                           _y = int(fusion_RAL_y/(_stop+1-_start)),
                                           _img_array = self.OTSU_img_array,
                                           _group_size = self.group_size,
                                           _group_step = self.group_step)
        
        if (self.RALs[_start].used_as_filling_bound and
            self.RALs[_stop].used_as_filling_bound):
                fusion_RAL.used_as_filling_bound = True
        
# =============================================================================
#         print("Fused", self.RALs[_start].y,
#               "and", self.RALs[_stop].y)
# =============================================================================
        newYdist = []
        new_diffs = []
        if (_start - 1 >= 0):
            new_diffs += [abs(fusion_RAL.y-self.RALs[_start-1].y)]
            newYdist = self.InterPlant_Diffs[:_start-1]
        
        tail_newRALs = []
        if (_stop+1<self.nb_RALs):
            new_diffs += [abs(fusion_RAL.y-self.RALs[_stop+1].y)]
            tail_newRALs = self.RALs[_stop+1:]
        
        newYdist += new_diffs
        
        
        if (_stop+1<len(self.InterPlant_Diffs)):
            newYdist += self.InterPlant_Diffs[_stop+1:]
        
        self.InterPlant_Diffs = newYdist
        
        self.RALs = self.RALs[:_start]+[fusion_RAL]+tail_newRALs 
        self.nb_RALs -= 1
    
    def Fill_RALs(self, _RAL_1_index, _RAL_2_index, _filling_step):
        
        if (not self.RALs[_RAL_1_index].used_as_filling_bound or
            not self.RALs[_RAL_2_index].used_as_filling_bound):
# =============================================================================
#             print("Filling procedure...")
# =============================================================================
            y_init = self.RALs[_RAL_1_index].y
            new_RALs = []
            nb_new_RALs = 0
            new_diffs = []
            while y_init + _filling_step < self.RALs[_RAL_2_index].y:
                new_RAL = ReactiveAgent_Leader(_x = self.Row_Mean_X,
                                               _y = int(y_init + _filling_step),
                                               _img_array = self.OTSU_img_array,
                                               _group_size = self.group_size,
                                               _group_step = self.group_step)
                new_RAL.used_as_filling_bound = True
                
                new_RALs += [new_RAL]
                new_diffs += [_filling_step]
                
                y_init += _filling_step
                
                nb_new_RALs += 1
            
            self.RALs[_RAL_1_index].used_as_filling_bound = True
            self.RALs[_RAL_2_index].used_as_filling_bound = True
            
            if (nb_new_RALs > 0):
                new_diffs += [abs(new_RALs[-1].y-self.RALs[_RAL_2_index].y)]
                self.RALs = self.RALs[:_RAL_1_index+1]+new_RALs+self.RALs[_RAL_2_index:]
                
                self.InterPlant_Diffs = self.InterPlant_Diffs[:_RAL_1_index]+ \
                                    new_diffs+ \
                                    self.InterPlant_Diffs[_RAL_2_index:]
                self.nb_RALs += nb_new_RALs
            
    def Fill_or_Fuse_RALs(self, _crit_value, _fuse_factor = 0.5, _fill_factor = 1.5):
        i = 0
        while i < self.nb_RALs-1:
            
# =============================================================================
#             print(self.InterPlant_Diffs[i], _fuse_factor*_crit_value)
# =============================================================================
            min_size = min([self.RALs[i].group_size, self.RALs[i+1].group_size])
            
            if (self.InterPlant_Diffs[i] < _fuse_factor*_crit_value or
                (abs(self.RALs[i].x-self.RALs[i+1].x) < min_size and
                 abs(self.RALs[i].y-self.RALs[i+1].y) < min_size)):
                self.Fuse_RALs(i, i+1)
            
            if (not self.extensive_init):
                if (i<len(self.InterPlant_Diffs)):#in case we fused the last 2 RAL of the crop row
                    if self.InterPlant_Diffs[i] > _fill_factor*_crit_value:
                        self.Fill_RALs(i, i+1, int(1.1*_fuse_factor*_crit_value))
            
            i += 1
        
# =============================================================================
#         print("After fill and fuse procedure over all the crop row, the new RAls list is :", end = ", ")
#         for _RAL in self.RALs:
#             print([_RAL.x, _RAL.y], end=", ")
# =============================================================================

    def Check_RALs_Exploration(self):
        """
        Looks at the exploration status of the RALs: where are the farthest RAs
        under supervision of the RALs. Checks that neighbour RALs do not overlap
        with each other.
        If overlapping is detected (mainly North or South), the RAs of each are
        ordered to withdraw and are fixed.
        """
        
        for i in range (self.nb_RALs-1):
            if (not self.RALs[i].With_Neighbour_Overlap[0] and not self.RALs[i+1].With_Neighbour_Overlap[1]):
                _y_north_border = self.RALs[i].y + self.RALs[i].Borders_Distance[0]
                _y_south_border = self.RALs[i+1].y + self.RALs[i+1].Borders_Distance[1]
                
                if (_y_north_border > _y_south_border):
                    
# =============================================================================
#                     print(i, self.RALs[i].Borders_Distance[0], self.RALs[i].y, _y_north_border)
#                     print (i+1, self.RALs[i+1].Borders_Distance[1], self.RALs[i+1].y, _y_south_border)
# =============================================================================
                    
                    _candidate_north_update = []
                    _candidate_south_update = []
                    
                    for _RA in self.RALs[i].RA_list_card[0]:#North
# =============================================================================
#                         print("North RAs y", _RA.global_y, _y_south_border)
# =============================================================================
                        if (_RA.global_y > _y_south_border):
                            _half_y_overlap = int((_RA.global_y - _y_south_border)*0.5)+1
                            
                            if (_RA.Fixed):
                                _RA.Fixed = False
                            else:
                                self.RALs[i].nb_RAs_Fixed += 1
                                
                            _RA.Update_All_coords(0, -_half_y_overlap)
                            _RA.Fixed = True
                            _candidate_north_update += [_RA.local_y]
                        
                    for _RA in self.RALs[i+1].RA_list_card[1]:#South
# =============================================================================
#                         print("South RAs y", _RA.global_y, _y_north_border)
# =============================================================================
                        if (_RA.global_y < _y_north_border):
                            _half_y_overlap = int((_y_north_border - _RA.global_y)*0.5)+1
                            
                            if (_RA.Fixed):
                                _RA.Fixed = False
                            else:
                                self.RALs[i+1].nb_RAs_Fixed += 1
                                
                            _RA.Update_All_coords(0, _half_y_overlap)
                            _RA.Fixed = True
                            _candidate_south_update += [_RA.local_y]
                    
# =============================================================================
#                     print(_candidate_north_update)
# =============================================================================
                    self.RALs[i].Borders_Distance[0] = max(_candidate_north_update)
                    self.RALs[i].With_Neighbour_Overlap[0] = True
# =============================================================================
#                     print(_candidate_south_update)
# =============================================================================
                    self.RALs[i+1].Borders_Distance[1] = min(_candidate_south_update)
                    self.RALs[i+1].With_Neighbour_Overlap[1] = True
                    
# =============================================================================
#                     print(i, self.RALs[i].Borders_Distance[0], self.RALs[i].y, _y_north_border)
#                     print (i+1, self.RALs[i+1].Borders_Distance[1], self.RALs[i+1].y, _y_south_border)
# =============================================================================
    
    def Get_RALs_mean_points(self):
        for _RAL in self.RALs:
# =============================================================================
#             print("Getting mean active RAs point for RAL", [_RAL.x, _RAL.y])
# =============================================================================
            _RAL.Get_RAs_Mean_Point()
    
    def Get_Row_Mean_X(self):
        RALs_X = []
        
        for _RAL in self.RALs:
            RALs_X += [_RAL.active_RA_Point[0]]
        
# =============================================================================
#         print(RALs_X)
# =============================================================================
        
        self.Row_Mean_X = int(np.mean(RALs_X))
        
    def Get_Inter_Plant_Diffs(self):
        self.InterPlant_Diffs = []
        nb_RALs = len(self.RALs)
        if (nb_RALs > 1):
            for i in range(nb_RALs-1):
                self.InterPlant_Diffs += [abs(self.RALs[i].y - self.RALs[i+1].y)]
                
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
    
    def ORDER_RALs_to_Correct_X(self):
        
        if (len(self.RALs)>0):
            self.Get_Row_Mean_X()
            
            majority_left = self.Is_RALs_majority_on_Left_to_Row_Mean()
        
        for _RAL in self.RALs:
            if (majority_left):
                if (_RAL.active_RA_Point[0] > self.Row_Mean_X):
                    _RAL.active_RA_Point[0] = self.Row_Mean_X
            else:
                if (_RAL.active_RA_Point[0] < self.Row_Mean_X):
                    _RAL.active_RA_Point[0] = self.Row_Mean_X
    
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
    
    def Destroy_RALs(self, _start, _stop, _nb_RALs):
        """
        _start and stop are the indeces of the RALs to destroy so that they 
        correspond to the bounderies [_start _stop[
        """
        if (_stop < _nb_RALs):
            self.RALs = self.RALs[:_start]+self.RALs[_stop:]
        else:
            self.RALs = self.RALs[:_start]
    
    def Destroy_Low_Activity_RALs(self):
        i = 0
        while i < self.nb_RALs:
# =============================================================================
#             print(self.RALs[i].x, self.RALs[i].y, self.RALs[i].recorded_Decision_Score[-1])
# =============================================================================
            if (self.RALs[i].recorded_Decision_Score[-1] < 0.01):
                self.Destroy_RALs(i, i+1, self.nb_RALs)
                self.nb_RALs -= 1
            else:
                i += 1
                
    def Destroy_Small_RALs(self):
        i = 0
        while i < self.nb_RALs:
            if (self.RALs[i].nb_RAs < 0.25*(4*2*self.RALs[i].group_size+1)/self.group_step):
                if (self.RALs[i].Fixed):
                    self.nb_Fixed_RALs -= 1
                self.Destroy_RALs(i, i+1, self.nb_RALs)
                self.nb_RALs -= 1
                
            else:
                i += 1
    
    def Adapt_RALs_group_size(self):
# =============================================================================
#         for _RAL in self.RALs:
#             if (_RAL.recorded_Decision_Score[-1] < 0.2 and
#                 _RAL.group_size > 5*_RAL.group_step):
#                 _RAL.group_size -= 1
#                 _RAL.RAs_square_init()
#             elif (_RAL.recorded_Decision_Score[-1] > 0.8 and
#                   _RAL.group_size < 50*_RAL.group_step):
#                 _RAL.group_size += 1
#                 _RAL.RAs_square_init()
# =============================================================================
        for _RAL in self.RALs:
            if (not _RAL.Fixed):
                _RAL.Manage_RAs_distribution()
                if (_RAL.Fixed):
                    self.nb_Fixed_RALs += 1
                    
#=============================== FOR PA GROWTH ANALYSIS ======================

    def Init_PA_growth_Analysis(self):
        for _RAL in self.RALs:
            _RAL.init_square_Growth()

    def Adapt_RALs_group_size_2(self, _next_coords):
        for _RAL in self.RALs:
            _RAL.square_Growth(_next_coords)

#=============================================================================
    
    def Set_Up_RALs_Growth_Mode(self):
        for _RAL in self.RALs:
            _RAL.RAs_border_init()