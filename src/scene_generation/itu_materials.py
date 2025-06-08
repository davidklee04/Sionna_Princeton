"""
ITU Material Properties Data

This module provides predefined material properties based on
ITU-R Recommendation P.2040-2: "[Effects of building materials and structures
on radiowave propagation above about 100 MHz](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2040-3-202308-I!!PDF-E.pdf)".

TODO: Verify the exact material names using by Mitsuba and Sionna. Especially how they named the low/high frequency glass and ceiling board materials.
TODO: Check what is the exact colors using by Mitsuba/Sionna/Blender/Blosm for the materials.
"""

ITU_MATERIALS = {
    "mat-vacuum": {
        "name": "Vacuum (\u2248Air)",
        "lower_freq_limit": 0.001e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.1216, 0.4667, 0.7059),
    },
    "mat-itu_concrete": {
        "name": "Concrete",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.539479, 0.539479, 0.539480),
    },
    "mat-itu_brick": {
        "name": "Brick",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 40e9,
        "mitsuba_color":(1.0000, 0.4980, 0.0549),
    },
    "mat-itu_plasterboard": {
        "name": "Plasterboard",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(1.0000, 0.7333, 0.4706),
    },
    "mat-itu_wood": {
        "name": "Wood",
        "lower_freq_limit": 0.001e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.043, 0.58, 0.184)
    },
    "mat-itu_glass": {
        "name": "Glass",
        "lower_freq_limit": [0.1e9,220e9],
        "upper_freq_limit": [100e9,450e9],
        "mitsuba_color":(0.5961, 0.8745, 0.5412),
    },
    "mat-itu_ceiling_board": {
        "name": "Ceiling Board",
        "lower_freq_limit": [1e9, 220e9],
        "upper_freq_limit": [100e9, 450e9],
        "mitsuba_color":(1.0000, 0.5961, 0.5882),
    },
    "mat-itu_chipboard": {
        "name": "Chipboard",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(0.7725, 0.6902, 0.8353),
    },
    "mat-itu_plywood": {
        "name": "Plywood",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 40e9,
        "mitsuba_color":(0.5490, 0.3373, 0.2941),
    },
    "mat-itu_marble": {
        "name": "Marble",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 60e9,
        "mitsuba_color": (0.701101, 0.644479, 0.485150),
    },
    "mat-itu_floorboard": {
        "name": "Floorboard",
        "lower_freq_limit": 50e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(0.8902, 0.4667, 0.7608),
    },
    "mat-itu_metal": {
        "name": "Metal",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.219526, 0.219526, 0.254152)
    },
    "mat-itu_very_dry_ground": {
        "name": "Very Dry Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color": (0.4980, 0.4980, 0.4980),
    },
    "mat-itu_medium_dry_ground": {
        "name": "Medium Dry Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color":(0.7804, 0.7804, 0.7804),
    },
    "mat-itu_wet_ground": {
        "name": "Wet Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color": (0.91, 0.569, 0.055)
    },
    "mat-itu_very_dry_ground_P.527": {
        "name": "Very Dry Ground (Extended Range, ITU P.527-3)",
        "lower_freq_limit": 1e4,
        "upper_freq_limit": 3e11,
        "mitsuba_color": (0.4980, 0.4980, 0.4980),
    },
    "mat-itu_medium_dry_ground_P.527": {
        "name": "Medium Dry Ground (Extended Range, ITU P.527-3)",
        "lower_freq_limit": 1e4,
        "upper_freq_limit": 3e11,
        "mitsuba_color":(0.7804, 0.7804, 0.7804),
    },
    "mat-itu_wet_ground_P.527": {
        "name": "Wet Ground (Extended Range, ITU P.527-3)",
        "lower_freq_limit": 1e4,
        "upper_freq_limit": 3e11,
        "mitsuba_color": (0.91, 0.569, 0.055)
    },
}
import pandas as pd
import scene_generation.data 

from importlib_resources import files
from scipy.interpolate import make_smoothing_spline

def get_material(material_name):
    from sionna.rt import RadioMaterial
    material_dict = {
        "mat-itu_wet_ground_P.527":RadioMaterial("itu_wet_ground_P.527",
                                                 frequency_update_callback=wet_ground_ITU_P527_callback),
        "mat-itu_medium_dry_ground_P.527":RadioMaterial("itu_medium_dry_ground_P.527",
                                                        frequency_update_callback=medium_dry_ground_ITU_P527_callback),
        "mat-itu_very_dry_ground_P.527":RadioMaterial("itu_very_dry_ground_P.527",
                                                        frequency_update_callback=very_dry_ground_ITU_P527_callback),
    }
    if material_name not in material_dict.keys():
        return None
    else:
        return material_dict[material_name]
def wet_ground_ITU_P527_callback(f_hz):

    f_mhz = f_hz / 1e6
    if f_mhz < 1e-2 or f_mhz > 3e5:
        return (-1.0, -1.0)

 

    
    # Read and process conductivity, permittivity data extracted from PDF
    with files('scene_generation.data').joinpath("B_wet_ground_con.csv").open("r") as f:
        df_con = pd.read_csv(f, header=None, names=["x", "y"])
    df_con = df_con.sort_values(by="x")
    
    with files('scene_generation.data').joinpath("B_wet_ground_per.csv").open("r") as f:
        df_per = pd.read_csv(f, header=None, names=["x", "y"])
    df_per = df_per.sort_values(by="x")

    # Fit a smoothing spline 
    spl_con = make_smoothing_spline(df_con["x"], df_con["y"])
    spl_per = make_smoothing_spline(df_per["x"], df_per["y"])

    
    relative_permittivity = spl_per(f_mhz)
    conductivity = spl_con(f_mhz)
    
    return (relative_permittivity.item(), conductivity.item())

def medium_dry_ground_ITU_P527_callback(f_hz):

    f_mhz = f_hz / 1e6
    if f_mhz < 1e-2 or f_mhz > 3e5:
        return (-1.0, -1.0)

 

    
    # Read and process conductivity, permittivity data extracted from PDF
    with files('scene_generation.data').joinpath("D_medium_dry_ground_con.csv").open("r") as f:
        df_con = pd.read_csv(f, header=None, names=["x", "y"])
    df_con = df_con.sort_values(by="x")
    
    with files('scene_generation.data').joinpath("D_medium_dry_ground_per.csv").open("r") as f:
        df_per = pd.read_csv(f, header=None, names=["x", "y"])
    df_per = df_per.sort_values(by="x")

    # Fit a smoothing spline 
    spl_con = make_smoothing_spline(df_con["x"], df_con["y"])
    spl_per = make_smoothing_spline(df_per["x"], df_per["y"])

    
    relative_permittivity = spl_per(f_mhz)
    conductivity = spl_con(f_mhz)
    return (relative_permittivity.item(), conductivity.item())

def very_dry_ground_ITU_P527_callback(f_hz):
    f_mhz = f_hz / 1e6
    if f_mhz < 1e-2 or f_mhz > 3e5:
        return (-1.0, -1.0)


    
    # Read and process conductivity, permittivity data extracted from PDF
    with files('scene_generation.data').joinpath("E_very_dry_ground_con.csv").open("r") as f:
        df_con = pd.read_csv(f, header=None, names=["x", "y"])
    df_con = df_con.sort_values(by="x")
    
    with files('scene_generation.data').joinpath("E_very_dry_ground_per.csv").open("r") as f:
        df_per = pd.read_csv(f, header=None, names=["x", "y"])
    df_per = df_per.sort_values(by="x")

    # Fit a smoothing spline 
    spl_con = make_smoothing_spline(df_con["x"], df_con["y"])
    spl_per = make_smoothing_spline(df_per["x"], df_per["y"])

    
    relative_permittivity = spl_per(f_mhz)
    conductivity = spl_con(f_mhz)
    
    return (relative_permittivity.item(), conductivity.item())
