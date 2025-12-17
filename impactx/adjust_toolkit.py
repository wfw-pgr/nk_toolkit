import os, sys, json5
import numpy  as np
import pandas as pd
import nk_toolkit.impactx.io_toolkit as itk


# ========================================================= #
# ===  adjust__rfphase.py                               === #
# ========================================================= #
def adjust__rfphase( refpFile="impactx/diags/ref_particle.0", \
                     partFile="impactx/diags/openPMD/bpm.h5", \
                     recoFile="impactx/diags/records.json"  , \
                     ext=None, collective=False ):
    
    cv = 2.99792458e8

    # ------------------------------------------------- #
    # --- [1] load file                             --- #
    # ------------------------------------------------- #
    if ( ext is not None ):
        inpFile = os.path.splitext( inpFile )[0] + ext

    with open( recoFile, "r" ) as f:
        records = json5.load( f )
        
    beamStats = itk.get__beamStats()
    if ( collective ):
        particles = itk.get__particles()

    # ------------------------------------------------- #
    # --- [2] calculation                           --- #
    # ------------------------------------------------- #
    tp = beamStats["t"]
    
        

    # ------------------------------------------------- #
    # --- [2] calculation                           --- #
    # ------------------------------------------------- #
    omega   = 2.0*np.pi * freq
    spos    = refp["s"].to_numpy()
    vp      = refp["beta"].to_numpy() * cv
    ds      = spos[1:] - spos[:-1]
    v_avg   = 0.5 * ( vp[:-1] + vp[1:] )
    dt      = ds / v_avg
    t_in    = np.concatenate( ([0.0], np.cumsum(dt)) )

    # ------------------------------------------------- #
    # --- [3] interpolation                         --- #
    # ------------------------------------------------- #
    t_mid = np.interp( s_mid, spos, t_in )
    
    # ------------------------------------------------- #
    # --- [4] phase                                 --- #
    # ------------------------------------------------- #
    def norm_angle( deg ):
        deg  = np.asarray( deg, dtype=float )
        ret  = ( ( deg + 180.0 ) % 360.0 ) - 180.0
        if ( len( ret ) == 1 ): ret = float( ret )
        return( ret )
    phi_o                 = norm_angle( t_mid * omega / np.pi * 180.0 ) 
    phi_c                 = norm_angle( phi_t - phi_o )
    phasedb["phi_o[deg]"] = phi_o
    phasedb["phi_c[deg]"] = phi_c
    phasedb["phi_b[deg]"] = phi_b
    phasedb.to_csv( phaseFile, index=False )
    return( phasedb )


