import os, sys, json5, glob
import numpy  as np
import pandas as pd


# ========================================================= #
# ===  analyze_toolkit.py                               === #
# ========================================================= #
def analyze_toolkit():
    postProcess__beam()
    


# ========================================================= #
# ===  postProcess__beam                                === #
# ========================================================= #
def postProcess__beam( params=None, particles=None ):
    
    amu = 931.494   # [MeV]
    cv  = 2.998e8   # [m/s]
    
    # ------------------------------------------------- #
    # --- [1] arguments check                       --- #
    # ------------------------------------------------- #
    if ( params    is None ): sys.exit( "[analyze_toolkit.py] params    == ???" )
    if ( particles is None ): sys.exit( "[analyze_toolkit.py] particles == ???" )
        
    # ------------------------------------------------- #
    # --- [3] additional plots                      --- #
    # ------------------------------------------------- #
    data               = {}
    data["s_refp"]     = refp["s"]
    data["Ek"]         = params["beam.mass.amu"] * amu * ( refp["gamma"] - 1.0 )
    freq               = params["beam.freq.Hz"]  * params["beam.harmonics"]
    data["s_stat"]     = stat["s"]
    data["phase_min"]  = stat["min_t" ] / cv * freq * 180.0
    data["phase_avg"]  = stat["mean_t"] / cv * freq * 180.0
    data["phase_max"]  = stat["max_t" ] / cv * freq * 180.0
    df                 = pd.DataFrame( data )
    
    # ------------------------------------------------- #
    # --- [4] save and return                       --- #
    # ------------------------------------------------- #
    df.to_csv( outFile, index=False )
    return()



    return()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    analyze_toolkit()
