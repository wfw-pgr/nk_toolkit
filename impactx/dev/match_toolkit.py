import os, sys, json5
import numpy  as np
import pandas as pd
import nk_toolkit.impactx.io_toolkit as itk


# ========================================================= #
# ===  match__rfphase.py                               === #
# ========================================================= #
def match__rfphase( refpFile="impactx/diags/ref_particle.0", \
                     partFile="impactx/diags/openPMD/bpm.h5", \
                     recoFile="impactx/diags/records.json"  , \
                     lattFile="impactx/diags/lattice.csv"   , \
                     ext=None, match_reference=True, phi_t=-45.0 ):
    
    cv = 2.99792458e8

    def norm_angle( deg ):
        deg  = np.asarray( deg, dtype=float )
        ret  = ( ( deg + 180.0 ) % 360.0 ) - 180.0
        if ( len( ret ) == 1 ): ret = float( ret )
        return( ret )

    # ------------------------------------------------- #
    # --- [1] load file                             --- #
    # ------------------------------------------------- #
    with open( recoFile, "r" ) as f:
        records = json5.load( f )
    with open( lattFile, "r" ) as f:
        lattice = pd.read_csv( f )
    if ( match_reference ):
        bpm   = itk.get__beamStats()
        t_bpm = bpm["t"]
        s_bpm = bpm["s"]
    else:
        bpm   = itk.get__particles()
        t_bpm = None
        s_bpm = None

    # ------------------------------------------------- #
    # --- [2] calculation                           --- #
    # ------------------------------------------------- #
    s_rf    = lattice.loc[ lattice["type"].isin( ["shortrf","rfcavity"] ), "s_mid" ]
    phi_bpm = 360.0 * records["beam.freq.rf.Hz"] * t_bpm / cv
    phi_rf  = norm_angle( np.interp( s_rf, s_bpm, phi_bpm ) )
    phi_c   = norm_angle( phi_t - phi_rf )
    print( phi_c )
    return( phi_c )


