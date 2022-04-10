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