# -*- coding: utf-8 -*-

import os
import sys
from skimage import io
import matplotlib.pyplot as plt

from numpy.lib.function_base import _CORE_DIMENSION_LIST

os.chdir("../Pre_Treatments")
import Process_image_for_FT as PiFT

os.chdir("../Fourier")
import FrequencyAnalysis as FA

os.chdir("../MAS")
import Multi_Images_Simulation as MIS

os.chdir("../Clustering_Scanner")
import Clustering 

os.chdir("../Pre_Treatments")
import Image_erosion

#os.chdir("../MAS") #Avec celui ci ça process jusqu'à la fin du pré traitement au moins
#import Multi_Images_Simulation_v12bis as MIS

if not "D:/Documents/IODAA/Fil Rouge/Plant_Counting" in sys.path:
    sys.path.append("D:/Documents/IODAA/Fil Rouge/Plant_Counting")

# import Pre_Treatments.Process_image_for_FT as PiFT
# import Fourier.FrequencyAnalysis as FA
# import MAS.Multi_Images_Simulation_v12bis as MIS

def CompleteProcess(_path_input_rgb_img, _path_output_root,

                    _labelled_images = False, _path_position_files=None,
                    _rows_real_angle=0,

                    _make_unique_folder_per_session=False, _session=1,
                    _do_Otsu=True, _do_AD=True, do_FA = True, do_erosion = True, do_Clustering = True,
                    _save_AD_score_images=False, _save_BSAS_images=False,
                    _bsas_threshold=1,
                    
                    erosion = 3,
                    dilatation = 3,

                    _bin_div_X=2, _bin_div_Y=4,

                    _RAs_group_size=20, _RAs_group_steps=2, _Simulation_steps=50,
                    _RALs_fuse_factor=0.5, _RALs_fill_factor=1.5):

    _new_session = PiFT.All_Pre_Treatment(_path_input_rgb_img,
                      _path_output_root,
                      _path_position_files,
                      _rows_real_angle,
                      _make_unique_folder_per_session, _session,
                      _do_Otsu, _do_AD,
                      _save_AD_score_images, _save_BSAS_images,
                      _bsas_threshold)
    
    if do_FA :
        print("Starting FA")
        FA.All_Fourier_Analysis(_path_output_root,
                                 _new_session,
                                 _bin_div_X, _bin_div_Y)
        
    if do_erosion : 
        print("Image_erosion")
        for img in os.listdir(r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\Real_images\Field_0\GrowthStage_0\Output\Session_1\Otsu"):
            image = io.imread(r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\Real_images\Field_0\GrowthStage_0\Output\Session_1\Otsu/" + img)
            mask = Image_erosion.calc_mask(image, 2)
            mask_eroded = Image_erosion.erosion(mask,erosion)
            mask_dilated = Image_erosion.dilation(mask_eroded, dilatation)
            plt.imsave(r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\Real_images\Field_0\GrowthStage_0\Output\Session_1\Otsu_Erroded/" + img + ".jpeg", mask_dilated)

    if do_Clustering : #va demander de changer le nom et l'emplacement des fichiers d'Output pour qu'il soit au même format que ceux de la FA
        print("Starting Clustering")
        Clustering.Total_Plant_Position(_path_input_root = r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\Real_images\Field_0\GrowthStage_0\Output\Session_1\Otsu_Erroded",
                                        _path_output_root= r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\Real_images\Field_0\GrowthStage_0\Output_general_clustering",
                                        _set = 2,
                                        session = 1,
                                        growth_stage = 0,
                                        epsilon=80,
                                        min_point=100,
                                        e=0.05,
                                        max_iter=100,
                                        m_p=2,
                                        threshold=500, #1/1000 de la taille de l'image en pixel
                                        _RAs_group_size=20,
                                    )
    
    
    MIS.All_Simulations(_path_input_rgb_img,
                    _path_output_root,
                    _labelled_images,
                    _new_session,
                    _RAs_group_size, _RAs_group_steps, _Simulation_steps,
                    _RALs_fuse_factor, _RALs_fill_factor)
    

if (__name__=="__main__"):
# ========================== FOR NON-LABELLED IMAGES ======================== #
# =============================================================================
#     CompleteProcess(_path_input_rgb_img="../Tutorial/Data/Non-Labelled/Set1",
#                     _path_output_root="../Tutorial/Output_General/Set1",
#                     
#                     _labelled_images = False,
#                     
#                     _make_unique_folder_per_session=False, _session=1,
#                     _do_Otsu=True, _do_AD=True,
#                     _save_AD_score_images=False, _save_BSAS_images=False,
#                     _bsas_threshold=1,
#                     
#                     _bin_div_X=2, _bin_div_Y=4,
#                     
#                     _RAs_group_size=20, _RAs_group_steps=2, _Simulation_steps=50,
#                     _RALs_fuse_factor=0.5, _RALs_fill_factor=1.5)
# =============================================================================
    
# ========================== FOR LABELLED IMAGES ============================ #
    
    #for growth_stage in range(1,3) :
    growth_stage = 0
    CompleteProcess(_path_input_rgb_img=r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\Real_images\Field_0\GrowthStage_{0}\RGB".format(growth_stage),
                    _path_output_root=r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\Real_images\Field_0\GrowthStage_{0}".format(growth_stage),
                    
                    _labelled_images = False,
                    #_path_position_files=r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\Real_images\Field_0\GrowthStage_{0}\Dataset".format(growth_stage),
                    #_path_position_files=r"D:\Datasets\Datasets rangs courbes\Champs_courbes_2\DIR_gt_DIP\processed\Field_0\GrowthStage_{0}\Output\Session_1\Adjusted_Position_Files".format(growth_stage),
                    _path_position_files=None,
                    _rows_real_angle=0,
                    
                    _make_unique_folder_per_session=False, _session=1,
                    _do_Otsu=True, _do_AD=False, do_FA = False, do_erosion = True, do_Clustering = True, # Selon le type de pre-processing choisi il faut changer le path_input_PLANT_FT_PRED dans MIS, peut s'automatiser en passant le type de pre-processing choisi
                    _save_AD_score_images=False, _save_BSAS_images=False,
                    _bsas_threshold=1,
                    
                    erosion = 3,
                    dilatation = 3,
    
                    _bin_div_X=2, _bin_div_Y=4,
                    
                    _RAs_group_size=20, _RAs_group_steps=2, _Simulation_steps=50,
                    _RALs_fuse_factor=0.5, _RALs_fill_factor=1.5)
        
                        
