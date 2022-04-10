import math
import numpy as np
import statistics 
from PxA import ReactiveAgent

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