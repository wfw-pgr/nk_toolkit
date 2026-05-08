import os, sys
import impactx
import numpy   as np
import pandas  as pd

# ========================================================= #
# ===  impactX tracking file                            === #
# ========================================================= #
def main_impactx():

    x_, y_, t_ = 0, 1, 2
    mm, mrad   = 1.0e-3, 1.0e-3
    amu        = 931.494              # [MeV]
    
    sim        = impactx.ImpactX()

    # ------------------------------------------------- #
    # --- [1]  preparation                          --- #
    # ------------------------------------------------- #
    sim.particle_shape                    = 2
    sim.space_charge                      = False
    sim.slice_step_diagnostics            = True
    sim.particle_lost_diagnostics_backend = ".h5"
    
    Ek0        = 10.0          # [MeV]
    Em0        = 1.00 * amu    # [MeV]
    e995_pi_mm = 20.0          # [pi mm mrad]
    alpha_xyt  = np.array( [ 1.0e-3, 1.0e-3, 1.0e-3] )  # 
    beta_xyt   = np.array( [ 2.0, 1.0, 1.0 ] )          # [m]

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
    e995_e1RMS   = 1.0 / 2.807                          # 99.5% emittance -> 1-RMS emittance
    eps_norm     = e995_pi_mm * mm * mrad * e995_e1RMS  # [ m rad ]
    
    gamma_xyt    = (1.0 + alpha_xyt**2) / ( beta_xyt )
    gamma_rel    = ( 1.0 + Ek0/Em0 )
    beta_rel     = np.sqrt( 1.0 - gamma_rel**(-2.0) )
    eps_geom     = eps_norm / ( beta_rel * gamma_rel )
    eps_xyt      = np.array([ eps_geom, eps_geom, eps_geom ])
    lambda_q     = np.sqrt( eps_xyt / gamma_xyt )
    lambda_p     = np.sqrt( eps_xyt /  beta_xyt )
    mu_qp        = alpha_xyt / np.sqrt( beta_xyt * gamma_xyt )
    
    distri       = impactx.distribution.Waterbag(
        lambdaX  = lambda_q[x_], lambdaY  = lambda_q[y_], lambdaT  = lambda_q[t_],
        lambdaPx = lambda_p[x_], lambdaPy = lambda_p[y_], lambdaPt = lambda_p[t_],
        muxpx    = mu_qp[x_]   , muypy    = mu_qp[y_],    mutpt    = mu_qp[t_], 
    )
    charge       = 1.0e-12 # [C]
    npt          = 1000    # #.of particles
    sim.add_particles( charge, distri, npt )
    
    # ------------------------------------------------- #
    # --- [4] lattice definition                    --- #
    # ------------------------------------------------- #
    ns       = 10
    mapsteps = 100
    Ra       = 0.2
    MV       = 1
    ds       = 0.1
    freq     = 1.e7
    phase    = 0.0
    cos_coef = [ 2.0 ]       # [1.0, 0.5] for sin(x)**2
    sin_coef = [ 0.0 ]       # [0. , 0. ] for sin(x)**2
    bpm      = impactx.elements.BeamMonitor( "bpm", backend="h5" )
    dr       = impactx.elements.ExactDrift ( name="dr", ds=ds, \
                                             aperture_x=Ra, aperture_y=Ra, nslice=ns )
    rf       = impactx.elements.RFCavity  ( name="rf", ds=ds, escale=MV/ds/Em0, \
                                            freq=freq, phase=phase, \
                                            cos_coefficients=cos_coef, \
                                            sin_coefficients=sin_coef, \
                                            aperture_x=Ra, aperture_y=Ra, \
                                            mapsteps=mapsteps, nslice=ns )
    beamline = [ bpm, dr, bpm, rf, bpm, dr, bpm ]
        
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
