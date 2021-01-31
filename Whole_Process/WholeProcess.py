# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:55:35 2020

@author: eliot
"""
import os
import sys

# os.chdir("../Pre_Treatments")
# import Process_image_for_FT as PiFT

# os.chdir("../Fourier")
# import FrequencyAnalysis as FA

# os.chdir("../MAS")
# import Multi_Images_Simulation_v12bis as MIS

if not "D:/Documents/IODAA/Fil Rouge/Plant_Counting" in sys.path:
    sys.path.append("D:/Documents/IODAA/Fil Rouge/Plant_Counting")

import Pre_Treatments.Process_image_for_FT as PiFT
import Fourier.FrequencyAnalysis as FA
import MAS.Multi_Images_Simulation_v12bis as MIS

def CompleteProcess(_path_input_rgb_img, _path_output_root,
                    
                    _make_unique_folder_per_session=False, _session=1,
                    _do_Otsu=True, _do_AD=True,
                    _save_AD_score_images=False, _save_BSAS_images=False,
                    _bsas_threshold=1,
                    
                    _bin_div_X=2, _bin_div_Y=4,
                    
                    _RAs_group_size=20, _RAs_group_steps=2, _Simulation_steps=50,
                    _RALs_fuse_factor=0.5, _RALs_fill_factor=1.5):
    
    PiFT.All_Pre_Treatment(_path_input_rgb_img,
                      _path_output_root,
                      _make_unique_folder_per_session, _session,
                      _do_Otsu, _do_AD,
                      _save_AD_score_images, _save_BSAS_images,
                      _bsas_threshold)
    
    FA.All_Fourier_Analysis(_path_output_root,
                         _session,
                         _bin_div_X, _bin_div_Y)
    
    
    MIS.All_Simulations(_path_input_rgb_img,
                    _path_output_root,
                    _session,
                    _RAs_group_size, _RAs_group_steps, _Simulation_steps,
                    _RALs_fuse_factor, _RALs_fill_factor)

if (__name__=="__main__"):
    CompleteProcess(_path_input_rgb_img="D:/Documents/IODAA/Fil Rouge/Resultats/2021_1_31_11_48_4/virtual_reality",
                    _path_output_root="D:/Documents/IODAA/Fil Rouge/Resultats/2021_1_31_11_48_4/",
                    _make_unique_folder_per_session=False, _session=1,
                    _do_Otsu=True, _do_AD=True,
                    _save_AD_score_images=False, _save_BSAS_images=False,
                    _bsas_threshold=1,
                    
                    _bin_div_X=2, _bin_div_Y=4,
                    
                    _RAs_group_size=20, _RAs_group_steps=2, _Simulation_steps=50,
                    _RALs_fuse_factor=0.5, _RALs_fill_factor=1.5)
    