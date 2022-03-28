class ReactiveAgent(object):
    """
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
        
                
        self.Fixed = False
        self.Set_Local_coords(_local_x, _local_y)
        
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
            
    
    def Set_Local_coords(self, _x, _y):
        """
        Sets local_x and local_y
        """
        if (not self.Fixed):
            self.local_x = _x
            self.local_y = _y
    
    def Set_Global_Coord(self, _x, _y):
        """
        Sets global_x and global_y
        """
        if (not self.Fixed):
            self.global_x = _x
            self.global_y = _y
    
    def Update_Global_coords(self, _RAL_x, _RAL_y):
        """
        Update global position based on local and RAL positions
        """
        self.Set_Global_Coord(_RAL_x + self.local_x, _RAL_y + self.local_y)
    
    def Update_All_coords(self, _dir_x, _dir_y):
        """
        Update both local and global coordinates by applying the modificators
        _dir_x and _dir_y.
        _dir_x and _dir_y are the coordinates of the new point relatively to
        the current coordinates.
        """
        self.Set_Local_coords(self.local_x + _dir_x, self.local_y +_dir_y)
        self.Set_Global_Coord(self.global_x + _dir_x, self.global_y + _dir_y)
    
    def Move_Based_On_RAL(self, _RAL_x, _RAL_y):
        """
        Update the position of the RAL based on the order given by the AD (agent
        director).
        _ADO_x (int):
            X coordinate of the target point (column of the image array)
        
        _ADO_y (int):
            Y coordinate of the target point (line of the image array)
        """
        self.Update_Global_coords(_RAL_x, _RAL_y)
        
        self.Is_Inside_Image_Frame()
        
    def Is_Inside_Image_Frame(self):
        
        if (self.global_x < 0 or
            self.global_x >= self.img_array.shape[1] or
            self.global_y < 0 or
            self.global_y >= self.img_array.shape[0]):
            
            self.outside_frame = True
            
        else:
            self.outside_frame = False
    
    def Exploration_Report(self, _x_dir, _y_dir,
                           _exploration_range, _shrinking_range):
        """
        Checks for white pixels in the along the (_x_dir,_y_dir) vector.
        We give priority to exploration (i.e. growth of the Reactive Agent
        Leader). So, if we find only one white pixel in the exploration
        direction, we do not check for white pixels in the opposite direction.

        Parameters
        ----------
        _x_dir : INT
            The first component of the (x,y) vector giving the direction in which
            the Reactive Agent should explore the image to find white pixels.
        _y_dir : INT
            The second component of the (x,y) vector giving the direction in which
            the Reactive Agent should explore the image to find white pixels.
        _exploration_range : INT
            The number of pixels the Reactive Agents should check for in the
            direction of exploration.
        _shrinking_range : INT
            The number of pixels the Reactive Agents should check for in the
            opposite direction of exploration.

        Returns
        -------
        _score : INT
            The number of pixels the Reactive Agent should move for along the
            (_x_dir,_y_dir) vector.

        """       
        
        _exploration_score = 0
        if (not self.outside_frame):
            for _inc in range (1, _exploration_range+1):
                x_dir_global = self.global_x + _inc * _x_dir
                y_dir_global = self.global_y + _inc * _y_dir
            
                if (0<= y_dir_global < self.img_array.shape[0] and
                    0<= x_dir_global < self.img_array.shape[1]):
                    
                    if (self.img_array[y_dir_global, x_dir_global][0] > 220):
                        _exploration_score += 1
                        
        _score = 0
        if (_exploration_score > 0):
            _score = _exploration_score
        elif _exploration_score == 0:
            _shrinking_score = 0
            for _inc in range (-_shrinking_range, 0):
                x_dir_global = self.global_x + _inc * _x_dir
                y_dir_global = self.global_y + _inc * _y_dir
            
                if (0<= y_dir_global < self.img_array.shape[0] and
                    0<= x_dir_global < self.img_array.shape[1]):
                    
                    if (self.img_array[y_dir_global, x_dir_global][0] < 220):
                        _shrinking_score += 1
            _score = -_shrinking_score
        
        return _score