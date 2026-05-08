import os, sys, json5
import impactx
import numpy   as np

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
    inpFile = "../dat/parameters.json"
    with open( inpFile, "r" ) as f:
        params = json5.load( f )
    sim.particle_shape                    = params["pic.particle_shape"]
    sim.space_charge                      = params["pic.space_charge"]
    sim.slice_step_diagnostics            = params["pic.slice_step_diagnostics"]
    sim.particle_lost_diagnostics_backend = params["pic.output.backend"]

    Ek0 = params["beam.Ek.MeV/u"] * params["beam.u"]
    Em0 = params["beam.mass.amu"] * amu

    # ------------------------------------------------- #
    # --- [2] grid & reference particle             --- #
    # ------------------------------------------------- #
    sim.init_grids()
    ref = sim.particle_container().ref_particle()
    ref.set_charge_qe     ( params["beam.charge"]                      )
    ref.set_mass_MeV      ( Em0 )
    ref.set_kin_energy_MeV( Ek0 )
    
    # ------------------------------------------------- #
    # --- [3] distribution                          --- #
    # ------------------------------------------------- #
    e995_e1RMS   = 1.0 / 2.807
    eps_norm     = 25.0 * mm * mrad * e995_e1RMS         # [m rad]  :: normalized epsilon
    alpha_xyt    = np.array( [ 0.0, 0.0, 0.0 ] )         #     Twiss Parameter :: Alpha
    beta_xyt     = np.array( [ 8.0, 4.0, 4.0 ] )         # [m] Twiss Parameter :: Beta
    gamma_xyt    = ( 1.0 + alpha_xyt**2 ) / ( beta_xyt ) #     Twiss Parameter :: Gamma

    gamma_enter  = ( 1.0 + Ek0/Em0 )                     # Einstein's Gamma
    beta_enter   = ( 1.0 - gamma_enter**(-2) )
    eps_enter    = eps_norm / ( beta_enter * gamma_enter )
    eps_xyt      = np.array( [ eps_enter, eps_enter, eps_enter ] )  # [m rad]
    lambda_q     = np.sqrt( eps_xyt / gamma_xyt )
    lambda_p     = np.sqrt( eps_xyt /  beta_xyt )
    mu_qp        = alpha_xyt / ( beta_xyt * gamma_xyt )
    
    # distri       = impactx.distribution.Gaussian(
    distri       = impactx.distribution.Waterbag(
        lambdaX  = lambda_q[x_],
        lambdaY  = lambda_q[y_],
        lambdaT  = lambda_q[t_],
        lambdaPx = lambda_p[x_],
        lambdaPy = lambda_p[y_],
        lambdaPt = lambda_p[t_],
        muxpx    = mu_qp[x_], 
        muypy    = mu_qp[y_], 
        mutpt    = mu_qp[t_], 
    )
    if   ( params["pic.trackmode"].lower() in [ "linear", "nonlinear" ] ):
        sim.add_particles( params["beam.bunch_charge"], distri, int(params["beam.nparticles"]) )
    elif ( params["pic.trackmode"].lower() in [ "envelope" ] ):
        sim.init_envelope( ref, distri )                      # -- for envelop  tracking -- #
    
    # ------------------------------------------------- #
    # --- [4] lattice definition                    --- #
    # ------------------------------------------------- #
    import lattice_elements
    
    # ------------------------------------------------- #
    # --- [5] tracking                              --- #
    # ------------------------------------------------- #
    sim.lattice.extend( lattice_elements.beamline )
    if   ( params["pic.trackmode"].lower() in [ "linear", "nonlinear" ] ):    # -- for particle tracking -- #
        sim.track_particles()
    elif ( params["pic.trackmode"].lower() in [ "envelope" ] ):               # -- for envelop  tracking -- #
        sim.track_envelope()
        
    # ------------------------------------------------- #
    # --- [6] end                                   --- #
    # ------------------------------------------------- #
    sim.finalize()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    main_impactx()
