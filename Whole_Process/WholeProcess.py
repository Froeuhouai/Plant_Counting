# -*- coding: utf-8 -*-

import os
import sys

from numpy.lib.function_base import _CORE_DIMENSION_LIST

# os.chdir("../Pre_Treatments")
# import Process_image_for_FT as PiFT

os.chdir("../MAS")
import Multi_Images_Simulation as MIS

# os.chdir("../MAS")
# import Multi_Images_Simulation_v12bis as MIS

if not "D:/Documents/IODAA/Fil Rouge/Plant_Counting" in sys.path:
    sys.path.append("D:/Documents/IODAA/Fil Rouge/Plant_Counting")

import Pre_Treatments.Process_image_for_FT as PiFT
import Fourier.FrequencyAnalysis as FA
import MAS.Multi_Images_Simulation_v12bis as MIS

def CompleteProcess(_path_input_rgb_img, _path_output_root,

                    _labelled_images = False, _path_position_files=None,
                    _rows_real_angle=0,

                    _make_unique_folder_per_session=False, _session=1,
                    _do_Otsu=True, _do_AD=True,
                    _save_AD_score_images=False, _save_BSAS_images=False,
                    _bsas_threshold=1,

                    _bin_div_X=2, _bin_div_Y=4,

                    _RAs_group_size=20, _RAs_group_steps=2, _Simulation_steps=50,
                    _RALs_fuse_factor=0.5, _RALs_fill_factor=1.5):

    PiFT.All_Pre_Treatment(_path_input_rgb_img,
                      _path_output_root,
                      _path_position_files,
                      _rows_real_angle,
                      _make_unique_folder_per_session, _session,
                      _do_Otsu, _do_AD,
                      _save_AD_score_images, _save_BSAS_images,
                      _bsas_threshold)
    
    
    FA.All_Fourier_Analysis(_path_output_root,
                             _session,
                             _bin_div_X, _bin_div_Y)
    
    
    MIS.All_Simulations(_path_input_rgb_img,
                        _path_output_root,
                        _labelled_images,
                        _session,
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
# =============================================================================
#     CompleteProcess(_path_input_rgb_img="../Tutorial/Data/Labelled/Set3/Processed/Field_0/GrowthStage_0/RGB",
#                     _path_output_root="../Tutorial/Output_General/Set3",
#                     
#                     _labelled_images = True,
#                     _path_position_files="../Tutorial/Data/Labelled/Set3/Processed/Field_0/GrowthStage_0/Dataset",
#                     _rows_real_angle=80,
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

#=============================== FOR EXPERIENCES
    root_data = "C:/Users/eliot/Documents/Education/APT/Stage_Tournesols/Data/Stage_Jules_Marcai/Processed"
    root_output = "C:/Users/eliot/Documents/Source/Education/APT/Plant_Counting_Analysis/Data"
    nb_fields = 5
    nb_gs = 3
    
    for _f in range (nb_fields):
        for _gs in range (nb_gs):
            CompleteProcess(_path_input_rgb_img=root_data+ f"/Field_{_f}/GrowthStage_{_gs}/RGB",
                            _path_output_root=root_output+f"/JM_PA_growth_analysis/Field_{_f}/GrowthStage_{_gs}",
                            
                            _labelled_images = True,
                            _path_position_files=root_data+ f"/Field_{_f}/GrowthStage_{_gs}/Dataset",
                            _rows_real_angle=80,
                            
                            _make_unique_folder_per_session=False, _session=1,
                            _do_Otsu=True, _do_AD=True,
                            _save_AD_score_images=False, _save_BSAS_images=False,
                            _bsas_threshold=1,
                            
                            _bin_div_X=2, _bin_div_Y=4,
                            
                            _RAs_group_size=20, _RAs_group_steps=2, _Simulation_steps=50,
                            _RALs_fuse_factor=0.5, _RALs_fill_factor=1.5)
    