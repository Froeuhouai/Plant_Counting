import numpy as np
from PxA import ReactiveAgent

class ReactiveAgent_Leader(object):
    """
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
    def __init__(self, _x, _y, _img_array, _group_size = 50, _group_step = 5,
                 _field_offset = [0,0]):
        
        self.x = int(_x)
        self.y = int(_y)
        self.img_array = _img_array
        
        self.group_size = _group_size
        self.group_step = _group_step
        self.correct_RAL_position()
        
        self.nb_RAs = 0
        self.nb_RAs_card = [0,0,0,0]
        
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

        self.Borders_Distance = [self.group_size,
                                 -self.group_size,
                                 self.group_size,
                                 -self.group_size]
        self.Fixed = False
        self.nb_RAs_Fixed = 0
        self.With_Neighbour_Overlap = [False, False, False, False]
        self.exploration_range = 4
        self.shrinking_range = 2
        
        self.RAs_square_init()
        
        self.Get_RAs_Otsu_Prop()
        self.recorded_Decision_Score = [self.decision_score]
    
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
            else:
                nb_outside_frame_RAs += 1
        
        self.decision_score = nb_true_votes/(self.nb_RAs-nb_outside_frame_RAs)
    
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
            
    def Add_One_Line_of_RAs(self, _x0, _y0, _xy1, _horizontal):
        """
        instanciates RAs on a line
        
        _x0 and _yo are the coordinates of the beginning point
        
        _xy1 is the coordinate of the last point. it should an x if _horizontal
        is true and a y, otherwise
        
        _horizontal (bool) if True the line is horizaontal. It is vertical, otherwise.
        """
        
        if (_horizontal):
            return [ReactiveAgent(
                    self.x, self.y,
                    i, _y0, self.img_array) for i in range(_x0,
                                                           _xy1+self.group_step,
                                                           self.group_step)]
        else:
            return [ReactiveAgent(
                    self.x, self.y,
                    _x0, i, self.img_array) for i in range(_y0,
                                                           _xy1+self.group_step,
                                                           self.group_step)]
    
    def Add_One_Layer_of_RAs(self, _distance_list):
        """
        Add one complete layer of RAs. The layer is square shaped and initialized
        based on _distance_list.
        
        _distance_list is a list of size 4 with distances of the layer respectively
        the center of the RAL. in the order of North, South, East, West.
        """
        
        _layer = []
        
        #Adding North part
        _layer += [self.Add_One_Line_of_RAs(_distance_list[3],#_x0 is the distance to the west
                                            _distance_list[0],#_y0 is the distance to the North
                                            _distance_list[2],#_xy1 is the distance to the East
                                            _horizontal=True)]
        
        #Adding South part
        _layer += [self.Add_One_Line_of_RAs(_distance_list[3],#_x0 is the distance to the west
                                            _distance_list[1],#_y0 is the distance to the South
                                            _distance_list[2],#_xy1 is the distance to the East
                                            _horizontal=True)]
        
        #Adding East part
        _layer += [self.Add_One_Line_of_RAs(_distance_list[2],#_x0 is the distance to the East
                                            _distance_list[1],#_y0 is the distance to the South
                                            _distance_list[0],#_xy1 is the distance to the North
                                            _horizontal=False)]
        
        #Adding West part
        _layer += [self.Add_One_Line_of_RAs(_distance_list[3],#_x0 is the distance to the West
                                            _distance_list[1],#_y0 is the distance to the South
                                            _distance_list[0],#_xy1 is the distance to the North
                                            _horizontal=False)]
        return _layer
    
    def Flatten_Layer(self, _list):
        """
        returns the list of the RAS with out the lists separations of the North,
        South, East and West borders.
        """
        return _list[0]+_list[1]+_list[2]+_list[3]
    
    def RAs_border_init(self):
        """
        Instanciation strategy on the border of the group size.
        Added to accomodate the new growing strategy of the RAL
        """
        
        _grp_s= int(0.3*self.group_size)
        self.RA_list_card = self.Add_One_Layer_of_RAs([_grp_s,-_grp_s,
                                                       _grp_s,-_grp_s])
        
        self.RA_list = self.Flatten_Layer(self.RA_list_card)
        
        for i in range (4):
            self.nb_RAs_card[i] = len(self.RA_list_card[i])
            self.nb_RAs += self.nb_RAs_card[i]
    
    def Manage_RAs_distribution(self):
        """
        Moves the RAs individually
        """
        if (not self.Fixed):
            self.All_Border_Movement()
            
            self.All_Border_Growth()
            
            for i in range (4):
                self.Check_Border_Distance(i)
            
            self.Is_Fixed()
            
            self.RA_list = self.Flatten_Layer(self.RA_list_card)
            
            self.nb_RAs = sum(self.nb_RAs_card)

#========================================= FOR PA GROWTH ANALYSIS ============
    def init_square_Growth(self):
        self.wpx_count = 0
        self.max_wpx = 1
        self.growth_RA = ReactiveAgent(self.x, self.y, 0, 0, self.img_array)
        
        self.growth_RA.Otsu_decision()
        if (self.growth_RA.decision):
            self.wpx_count += 1
        
        self.all_wpx_count = [self.wpx_count]
        self.all_max_wpx = [self.max_wpx]

    def square_Growth(self, _next_coords):
        
        for _coords in _next_coords:
            self.growth_RA.Move_Based_On_RAL(self.x+_coords[0], self.y+_coords[1])
            if (not self.growth_RA.outside_frame):
                self.max_wpx += 1
                self.growth_RA.Otsu_decision()
                if (self.growth_RA.decision):
                    self.wpx_count += 1
            
        self.all_wpx_count += [self.wpx_count]
        self.all_max_wpx += [self.max_wpx]
        

#=============================================================================
        
        
    def All_Border_Movement(self):
        """
        Individual movements or RAs for all 4 borders
        """
        
# =============================================================================
#         print("All_Border_Movement")
# =============================================================================
        self.all_end_score = []
        
        #North
# =============================================================================
#         print("North")
# =============================================================================
        self.Border_movement(0, 0, 1)
        
        #South
# =============================================================================
#         print("South")
# =============================================================================
        self.Border_movement(1, 0, -1)
        
        #East
# =============================================================================
#         print("East")
# =============================================================================
        self.Border_movement(2, 1, 0)
        
        #West
# =============================================================================
#         print("West")
# =============================================================================
        self.Border_movement(3, -1, 0)
    
    def Border_movement(self, _border_index, _x_dir, _y_dir):
        """
        Calls the different steps to let the Reactive Agents inside a border to
        move.

        Parameters
        ----------
        _border_index : INT
            The index of the moving border with 0==North, 1==South, 2==East and
            3==West.
        _x_dir : INT
            An integer representing whether the Reactive agents can move on the
            X axis. The authorized values are -1, 0, and 1. -1 when the Reactive
            agents can move toward the "left"; 1 when the Reactive agents can
            move toward the "right" and 0 when no movements are authorized on
            the axis.
        _y_dir : INT
            An integer representing whether the Reactive agents can move on the
            Y axis. The authorized values are -1, 0, and 1. -1 when the Reactive
            agents can move "downward"; 1 when the Reactive agents can move
            "upward" and 0 when no movements are authorized on the axis.
        Returns
        -------
        None.

        """
        
        nb_RAs_in_Border = len(self.RA_list_card[_border_index])
        _border_scores = self.Individual_Movement_Scores(_border_index, _x_dir, _y_dir, nb_RAs_in_Border)
        
        self.Comparison_To_Fixed_Points(_border_index, _x_dir, _y_dir, _border_scores)
        
        _propagated_score = self.Propagated_Movement_Scores(_border_scores, nb_RAs_in_Border)
        
        end_score = _border_scores + _propagated_score
        self.all_end_score += [end_score]
        
        for i in range (nb_RAs_in_Border):
            if (_border_scores[i] == 0 and _propagated_score[i] == 0):
                self.RA_list_card[_border_index][i].Fixed = True
                self.nb_RAs_Fixed += 1
            
            else:
                self.Check_And_Apply_Movement(_border_index, _x_dir, _y_dir, i, end_score)
            
        
# =============================================================================
#         print("_border_scores", _border_scores)
#         print("_propagated_score", _propagated_score)
#         print("_sum_scores", end_score)
# =============================================================================
    def Check_Border_Distance(self, _border_index):#, _farthest_RAs):
        """
        Keeps track of the RAs (one per border) of each RAL that are the
        farthest away in its exploration.
        """
        #_res = _farthest_RAs
        
        if (_border_index == 0):#North
            _y_list = []
            for _RA in self.RA_list_card[_border_index]:
                _y_list += [_RA.local_y]
            self.Borders_Distance[0] = max(_y_list)
                
        if (_border_index == 1):#South
            _y_list = []
            for _RA in self.RA_list_card[_border_index]:
                _y_list += [_RA.local_y]
            self.Borders_Distance[1] = min(_y_list)
        
        if (_border_index == 2):#East
            _x_list = []
            for _RA in self.RA_list_card[_border_index]:
                _x_list += [_RA.local_x]
            self.Borders_Distance[2] = max(_x_list)
                
        if (_border_index == 3):#West
            _x_list = []
            for _RA in self.RA_list_card[_border_index]:
                _x_list += [_RA.local_x]
            self.Borders_Distance[3] = min(_x_list)

    def Individual_Movement_Scores(self, _border_index, _x_dir, _y_dir, _nb_RAs):
        """
        Compiles how much (in pixels) each Reactive Agent in the border wants
        to move along the direction given by the (_x_dir, _y_dir) vector.

        Parameters
        ----------
        _border_index : INT
            The index of the moving border with 0==North, 1==South, 2==East and
            3==West.
        _x_dir : INT
            An integer representing whether the Reactive agents can move on the
            X axis. The authorized values are -1, 0, and 1. -1 when the Reactive
            agents can move toward the "left"; 1 when the Reactive agents can
            move toward the "right" and 0 when no movements are authorized on
            the axis.
        _y_dir : INT
            An integer representing whether the Reactive agents can move on the
            Y axis. The authorized values are -1, 0, and 1. -1 when the Reactive
            agents can move "downward"; 1 when the Reactive agents can move
            "upward" and 0 when no movements are authorized on the axis.
        _nb_RAs : INT
            The number of Reactive Agents in the current border.

        Returns
        -------
        _border_scores : List
            The number of pixels each Reactive Agent of the border should move 
            for along the (_x_dir,_y_dir) vector.

        """
        _border_scores = np.zeros(_nb_RAs, dtype=np.int_)
        
        for i in range (_nb_RAs):
            _border_scores[i] = self.RA_list_card[_border_index][i].Exploration_Report(_x_dir, _y_dir,
                                                                                      self.exploration_range,
                                                                                      self.shrinking_range)
        return _border_scores
    
    def Propagated_Movement_Scores(self, _border_scores, _nb_RAs):
        """
        We only propagate the exploration score
        """
        
        _propagation_score = np.zeros(_nb_RAs, dtype = np.int_)
        for i in range (_nb_RAs):
            _score = _border_scores[i]
            _reach = abs(_score)
            
            for j in range (1, _reach):
                k = i+j
                if 0 <= k < _nb_RAs:
                    if _score > 0:
                        _propagation_score[k] += _score - j
                        
                k = i-j
                if 0 <= k < _nb_RAs:
                    if _score > 0:
                        _propagation_score[k] += _score - j
                        
        return _propagation_score
    
    def Check_And_Apply_Movement(self, _border_index, _x_dir, _y_dir, _RA_index, _score):
        """
        Directly applies the movement score only if border overlap has not been
        detected with a neighbour.
        otherwise, it applies a correction to the score so that the pixel
        agent do not go beyond the other pixel agents that is the source of the
        overlap.
        
        It also makes sure that a RA do not shrink beyond half the center
        of the RAL
        """
        if (self.With_Neighbour_Overlap[_border_index]):
            if (_border_index == 0):
                if (self.RA_list_card[_border_index][_RA_index].local_y + _y_dir * _score[_RA_index] > self.Borders_Distance[_border_index]):
                    _y_diff = self.RA_list_card[_border_index][_RA_index].local_y + _y_dir * _score[_RA_index] - self.Borders_Distance[_border_index]
                    _score[_RA_index] -= _y_diff
                    
            if (_border_index == 1):

                if (self.RA_list_card[_border_index][_RA_index].local_y + _y_dir * _score[_RA_index] < self.Borders_Distance[_border_index]):
                    _y_diff = self.RA_list_card[_border_index][_RA_index].local_y + _y_dir * _score[_RA_index] - self.Borders_Distance[_border_index]
                    _score[_RA_index] += _y_diff
                    
            if (_border_index == 2):
                if (self.RA_list_card[_border_index][_RA_index].local_x + _x_dir * _score[_RA_index] > self.Borders_Distance[_border_index]):
                    _x_diff = self.RA_list_card[_border_index][_RA_index].local_x + _x_dir * _score[_RA_index] - self.Borders_Distance[_border_index]
                    _score[_RA_index] -= _x_diff
                    
            if (_border_index == 3):
                if (self.RA_list_card[_border_index][_RA_index].local_x + _x_dir * _score[_RA_index] < self.Borders_Distance[_border_index]):
                    _x_diff = self.RA_list_card[_border_index][_RA_index].local_x + _x_dir * _score[_RA_index] - self.Borders_Distance[_border_index]
                    _score[_RA_index] += _x_diff
        
        if (_border_index == 0):
            if (self.RA_list_card[_border_index][_RA_index].local_y + _y_dir * _score[_RA_index] <= 0):
                #print("North, beyond half correction")
                _score[_RA_index] = 0
                    
        if (_border_index == 1):
            if (self.RA_list_card[_border_index][_RA_index].local_y + _y_dir * _score[_RA_index] >= 0):
                #print("South, beyond half correction")
                _score[_RA_index] = 0
                
        if (_border_index == 2):
            if (self.RA_list_card[_border_index][_RA_index].local_x + _x_dir * _score[_RA_index] <= 0):
                #print("East, beyond half correction")
                _score[_RA_index] = 0
                
        if (_border_index == 3):
            if (self.RA_list_card[_border_index][_RA_index].local_x + _x_dir * _score[_RA_index] >= 0):
                #print("West, beyond half correction")
                _score[_RA_index] = 0
        
        
        self.RA_list_card[_border_index][_RA_index].Update_All_coords(_x_dir*_score[_RA_index],
                                                                      _y_dir*_score[_RA_index])
        
    
    def Comparison_To_Fixed_Points(self,
                                   _border_index,
                                   _x_dir, _y_dir,
                                   _border_scores):
        """
        Fixed points are points that have found edges of the target structure.
        We want to use them as references to guide the points that have not
        found anything.
        We will take the movement_scores and force a RA to move in the direction
        of its neighbour if that one is fixed.

        Parameters
        ----------
        _border_index : INT
            The index of the moving border with 0==North, 1==South, 2==East and
            3==West.
        _x_dir : INT
            An integer representing whether the Reactive agents can move on the
            X axis. The authorized values are -1, 0, and 1. -1 when the Reactive
            agents can move toward the "left"; 1 when the Reactive agents can
            move toward the "right" and 0 when no movements are authorized on
            the axis.
        _y_dir : INT
            An integer representing whether the Reactive agents can move on the
            Y axis. The authorized values are -1, 0, and 1. -1 when the Reactive
            agents can move "downward"; 1 when the Reactive agents can move
            "upward" and 0 when no movements are authorized on the axis.
        _border_scores : LIST
            The number of pixels each Reactive Agent of the border should move 
            for along the (_x_dir,_y_dir) vector.

        Returns
        -------
        None.
        """
# =============================================================================
#         print("self.nb_RAs_card[_border_index]", self.nb_RAs_card[_border_index])
# =============================================================================
        for i in range (self.nb_RAs_card[_border_index]):#for every RA
            if (self.RA_list_card[_border_index][i].Fixed or
                self.RA_list_card[_border_index][i].Otsu_decision()):#if RA is fixed or on a white pixel
                for k in [i-1, i+1]:#for left & right neighbours
                    if (0<=k<self.nb_RAs_card[_border_index]):#if such neighbour exists
                        if (_border_scores[k]<=0):#If the neighbour is not exploring
                            if (_x_dir == 0 ):#border movement on the Y axis
                                _y_dir_to_fixed = (self.RA_list_card[_border_index][i].global_y-
                                                   self.RA_list_card[_border_index][k].global_y)
                                #the neighbour is more in the center of th RAL
                                #than the fixed RA
                                if (_y_dir * _y_dir_to_fixed > 0):
                                    _border_scores[k] = min(self.exploration_range, _y_dir * _y_dir_to_fixed)
                                
                            elif(_y_dir == 0):#border movement on the X axis
                                _x_dir_to_fixed = (self.RA_list_card[_border_index][i].global_x-
                                                   self.RA_list_card[_border_index][k].global_x)
                                #the neighbour is more in the center of th RAL
                                #than the fixed RA
                                if (_x_dir * _x_dir_to_fixed > 0):
                                    _border_scores[k] = min(self.exploration_range, _x_dir * _x_dir_to_fixed)
                                
    
    def All_Border_Growth(self):
        """
        Cutting or adding RAs at the extremeties of the borders based on their
        movements computed beforehand (with All_Border_Movement)
        """
        #North - West
        self.Border_Growth(0, 0, 3, -1)
        
        #North - East
        self.Border_Growth(0, -1, 2, -1)
        
        #South - West
        self.Border_Growth(1, 0, 3, 0)
        
        #South - East
        self.Border_Growth(1, -1, 2, 0) 
    
    def Border_Growth(self,
                      _border_index_1, _extreme_index_1,
                      _border_index_2, _extreme_index_2,
                      _limit_size_factor = 0.25):
        
        _limit_size = int(_limit_size_factor * self.group_size)        
            
        if (_border_index_1 == 0):
            if (self.all_end_score[_border_index_1][_extreme_index_1] < 0 and
                self.all_end_score[_border_index_2][_extreme_index_2] < 0):
            
                if (self.nb_RAs_card[_border_index_1] > _limit_size):
                    if (_border_index_2 == 3): #we truncate from the West
                        self.RA_list_card[_border_index_1] = self.RA_list_card[_border_index_1][1:]
                    elif(_border_index_2 == 2): #we truncate from the East
                        self.RA_list_card[_border_index_1] = self.RA_list_card[_border_index_1][:-1]
                    self.nb_RAs_card[_border_index_1] -= 1
                    
                if (self.nb_RAs_card[_border_index_2] > _limit_size):
                    self.RA_list_card[_border_index_2] = self.RA_list_card[_border_index_2][:-1]
                    self.nb_RAs_card[_border_index_2] -= 1
                    
            else:
                _y_RAs = []
                _x_RAs = []
                if self.all_end_score[_border_index_1][_extreme_index_1] > 0: #with North we add at the end of West and East
                    _y_diff = abs(self.RA_list_card[_border_index_1][_extreme_index_1].local_y - \
                                self.RA_list_card[_border_index_2][_extreme_index_2].local_y)
                    _y_RAs = [ReactiveAgent(self.x, self.y,
                                          self.RA_list_card[_border_index_2][_extreme_index_2].local_x,
                                          self.RA_list_card[_border_index_1][_extreme_index_1].local_y - k,
                                          self.img_array) for k in range(0, _y_diff, self.group_step)][::-1]
                    
                    self.nb_RAs_card[_border_index_2] += len(_y_RAs)
                    
                if self.all_end_score[_border_index_2][_extreme_index_2] > 0:
                    
                    _x_diff = abs(self.RA_list_card[_border_index_1][_extreme_index_1].local_x - \
                                    self.RA_list_card[_border_index_2][_extreme_index_2].local_x)
                    
                    if (_border_index_2 == 3): #with West we add at the beginning of North
                        _x_RAs = [ReactiveAgent(self.x, self.y,
                                              self.RA_list_card[_border_index_2][_extreme_index_2].local_x + k,
                                              self.RA_list_card[_border_index_1][_extreme_index_1].local_y,
                                              self.img_array) for k in range(0, _x_diff, self.group_step)]
            
                        self.RA_list_card[_border_index_1] = _x_RAs + self.RA_list_card[_border_index_1]
                        
                                              
                    elif (_border_index_2 == 2): #with East we add at the end of North
                        _x_RAs = [ReactiveAgent(self.x, self.y,
                                              self.RA_list_card[_border_index_2][_extreme_index_2].local_x - k,
                                              self.RA_list_card[_border_index_1][_extreme_index_1].local_y,
                                              self.img_array) for k in range(0, _x_diff, self.group_step)][::-1]
            
                        self.RA_list_card[_border_index_1] += _x_RAs     
                        
                    self.nb_RAs_card[_border_index_1] += len(_x_RAs)
                
                self.RA_list_card[_border_index_2] += _y_RAs
        
        
        if (_border_index_1 == 1):
            if (self.all_end_score[_border_index_1][_extreme_index_1] < 0 and
                self.all_end_score[_border_index_2][_extreme_index_2] < 0):
            
                if (self.nb_RAs_card[_border_index_1] > _limit_size):
                    if (_border_index_2 == 3): #we truncate from the West
                        self.RA_list_card[_border_index_1] = self.RA_list_card[_border_index_1][1:]
                    elif(_border_index_2 == 2): #we truncate fromt he East
                        self.RA_list_card[_border_index_1] = self.RA_list_card[_border_index_1][:-1]
                    self.nb_RAs_card[_border_index_1] -= 1
                    
                if (self.nb_RAs_card[_border_index_2] > _limit_size):
                    self.RA_list_card[_border_index_2] = self.RA_list_card[_border_index_2][1:]
                    self.nb_RAs_card[_border_index_2] -= 1
                    
            else:
                _y_RAs = []
                _x_RAs = []
                if self.all_end_score[_border_index_1][_extreme_index_1] > 0: #with North we add at the end of West and East
                    _y_diff = abs(self.RA_list_card[_border_index_1][_extreme_index_1].local_y - \
                                self.RA_list_card[_border_index_2][_extreme_index_2].local_y)
                    _y_RAs = [ReactiveAgent(self.x, self.y,
                                          self.RA_list_card[_border_index_2][_extreme_index_2].local_x,
                                          self.RA_list_card[_border_index_1][_extreme_index_1].local_y + k,
                                          self.img_array) for k in range(0, _y_diff, self.group_step)]
                    
                    self.nb_RAs_card[_border_index_2] += len(_y_RAs)
                
                if self.all_end_score[_border_index_2][_extreme_index_2] > 0:
                    
                    _x_diff = abs(self.RA_list_card[_border_index_1][_extreme_index_1].local_x - \
                                    self.RA_list_card[_border_index_2][_extreme_index_2].local_x)
                                        
                    if (_border_index_2 == 3): #with West we add at the beginning of South
                        _x_RAs = [ReactiveAgent(self.x, self.y,
                                              self.RA_list_card[_border_index_2][_extreme_index_2].local_x + k,
                                              self.RA_list_card[_border_index_1][_extreme_index_1].local_y,
                                              self.img_array) for k in range(0, _x_diff, self.group_step)]
            
                        self.RA_list_card[_border_index_1] = _x_RAs + self.RA_list_card[_border_index_1]
                        
                                              
                    elif (_border_index_2 == 2): #with East we add at the end of South
                        _x_RAs = [ReactiveAgent(self.x, self.y,
                                              self.RA_list_card[_border_index_2][_extreme_index_2].local_x - k,
                                              self.RA_list_card[_border_index_1][_extreme_index_1].local_y,
                                              self.img_array) for k in range(0, _x_diff, self.group_step)][::-1]
            
                        self.RA_list_card[_border_index_1] += _x_RAs     
                        
                    self.nb_RAs_card[_border_index_1] += len(_x_RAs)
                
                self.RA_list_card[_border_index_2] = _y_RAs + self.RA_list_card[_border_index_2]
        
    def Is_Fixed(self):
        if (self.nb_RAs_Fixed/self.nb_RAs > 0.9):
            self.Fixed = True
