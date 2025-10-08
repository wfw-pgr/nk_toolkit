import os, sys, json5
import impactx
import numpy   as np
import pandas  as pd

# ========================================================= #
# ===  impactX tracking file                            === #
# ========================================================= #
def main_impactx():

    x_, y_, t_ = 0, 1, 2
    mm, mrad   = 1.0e-3, 1.0e-3
    amu        = 931.494            # [MeV]
    
    sim        = impactx.ImpactX()

    # ------------------------------------------------- #
    # --- [1]  preparation                          --- #
    # ------------------------------------------------- #
    sim.particle_shape                    = 2
    sim.space_charge                      = False
    sim.slice_step_diagnostics            = True
    sim.particle_lost_diagnostics_backend = ".h5"
    
    Ek0           = 40.0                                   # [MeV]
    Em0           = 2.014 * amu                            # [MeV]
    full2rms      = 1.0 / 8.0                              # full -> rms emittance ( 6D wb-> 1/8 )
    alpha_xyt     = np.array( [ 0.0, 0.0, 0.0    ] )
    # alpha_xyt     = np.array( [ 0.1, 0.2, 2.0e-3 ] )
    beta_xyt      = np.array( [ 8.0, 4.0, 442.31 ] )       # [m]
    charge        = 1.0e-12                                # [C]
    npt           = 10000                                  # #.of particles
    epsnxy        = np.array( [ 150.0, 150.0 ] )           
    epsnz         = 28.14                                  # [mm mrad] -- full emittance
    
    # ------------------------------------------------- #
    # --- [2] grid & reference particle             --- #
    # ------------------------------------------------- #
    sim.init_grids()
    ref = sim.particle_container().ref_particle()
    ref.set_charge_qe     ( 1.0 )
    ref.set_mass_MeV      ( Em0 )
    ref.set_kin_energy_MeV( Ek0 )
    
    # ------------------------------------------------- #
    # --- [3] distribution                          --- #
    # ------------------------------------------------- #
    gamma_rel    = ( 1.0 + Ek0/Em0 )
    beta_rel     = np.sqrt( 1.0 - gamma_rel**(-2.0) )
    epsnxy       = epsnxy / ( beta_rel * gamma_rel )
    eps_geom     = np.array( [ epsnxy[x_], epsnxy[y_], epsnz ] )
    eps_geom     = eps_geom * mm * mrad * full2rms
    
    gamma_xyt    = (1.0 + alpha_xyt**2) / ( beta_xyt )
    lambda_q     = np.sqrt( eps_geom / gamma_xyt )
    lambda_p     = np.sqrt( eps_geom /  beta_xyt )
    mu_qp        = alpha_xyt / np.sqrt( beta_xyt * gamma_xyt )
    
    distri       = impactx.distribution.Waterbag(
        lambdaX  = lambda_q[x_], lambdaY  = lambda_q[y_], lambdaT  = lambda_q[t_],
        lambdaPx = lambda_p[x_], lambdaPy = lambda_p[y_], lambdaPt = lambda_p[t_],
        muxpx    = mu_qp[x_]   , muypy    = mu_qp[y_],    mutpt    = mu_qp[t_], 
    )
    sim.add_particles( charge, distri, npt )

    print( " beta_xyt   :: ", beta_xyt )
    print( " eps_geom   :: ", eps_geom )
    print( " sigma_xyt  :: ", lambda_q )
    print( " sigma_pxyt :: ", lambda_p )
    
    # ------------------------------------------------- #
    # --- [4] lattice definition                    --- #
    # ------------------------------------------------- #
    beamlineFile = "../dat/beamline_impactx.json"
    with open( beamlineFile, "r" ) as f:
        beamline = json5.load( f )
    elements = beamline["elements"]

    stack    = []
    bpm      = impactx.elements.BeamMonitor( "bpm", backend="h5" )
    for key,elem in elements.items():
        elem_ = { hkey:val for hkey,val in elem.items() if hkey != "type" }
        if   ( elem["type"] in [ "rfcavity" ]   ):
            bcomp = impactx.elements.RFCavity  ( **elem_ )
        elif ( elem["type"] in [ "rfgap" ] ):
            Rmat = impactx.Map6x6.identity()
            for ii in range(6):
                for ij in range(6):
                    Rmat[ii+1,ij+1] = elem_["R"][ii][ij]
            elem_["R"] = Rmat
            bcomp = impactx.elements.LinearMap ( **elem_ )
        elif ( elem["type"] in [ "quadrupole" ] ):
            bcomp = impactx.elements.ExactQuad ( **elem_ )
        elif ( elem["type"] in [ "drift" ]      ):
            bcomp = impactx.elements.ExactDrift( **elem_ )
        else:
            sys.exit( "[main_impactx.py] unknown element type :: {} ".format( elem["type"] ) )
        stack += [ bcomp ]
    beamline = [ bpm ] + stack + [ bpm ]
        
    # ------------------------------------------------- #
    # --- [5] tracking                              --- #
    # ------------------------------------------------- #
    sim.lattice.extend( beamline )
    sim.track_particles()
        
    # ------------------------------------------------- #
    # --- [6] end                                   --- #
    # ------------------------------------------------- #
    sim.finalize()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    main_impactx()
    
