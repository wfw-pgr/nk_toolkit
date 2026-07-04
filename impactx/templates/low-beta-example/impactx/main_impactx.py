import json5, time
import impactx
import numpy   as np
import nk_toolkit.impactx.run_toolkit as rtk


# ========================================================= #
# ===  impactX tracking file                            === #
# ========================================================= #

def main_impactx():

    amu        = 931.494          # [MeV]

    # ------------------------------------------------- #
    # --- [1]  load parameters                      --- #
    # ------------------------------------------------- #
    paramsFile = "../dat/parameters.json"
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )

    # ------------------------------------------------- #
    # --- [2]  initialization                       --- #
    # ------------------------------------------------- #
    ts                         = time.perf_counter()
    sim                        = impactx.ImpactX()

    sim.max_level              = params["sim.max_level"]
    sim.n_cell                 = params["sim.n_cell"]
    sim.blocking_factor        = params["sim.blocking_factor"]
    
    sim.particle_shape         = 2
    sim.slice_step_diagnostics = True
    sim.space_charge           = params["mode.space_charge"]
    sim.poisson_solver         = "fft"       # fft or multigrid ( fft(IGF) is only outer side )
    sim.dynamic_size           = True        # True
    sim.prob_relative          = [1.2,1.1]   # ptcl's min-max * prob_relative = pic's sim-size
    #                                        # ( 1.2-1.1 for fft, >3.0 for multigrid )
    sim.particle_lost_diagnostics_backend = ".h5"
    sim.init_grids()
    
    # ------------------------------------------------- #
    # --- [3] reference particle                    --- #
    # ------------------------------------------------- #
    Ek0 = params["beam.Ek.MeV/u"] * params["beam.u.nucleon"]
    Em0 = params["beam.mass.amu"] * amu
    ref = sim.particle_container().ref_particle()
    ref.set_charge_qe     ( params["beam.charge.qe"] )
    ref.set_mass_MeV      ( Em0 )
    ref.set_kin_energy_MeV( Ek0 )
    
    # ------------------------------------------------- #
    # --- [4] distribution                          --- #
    # ------------------------------------------------- #
    distri = rtk.set__waterbag_distribution( alpha   =np.array( params["beam.twiss.alpha"]    ), \
                                             beta    =np.array( params["beam.twiss.beta" ]    ), \
                                             eps_geom=np.array( params["beam.emittance.geom"] ), \
                                             mm_mrad =True, full_emittance=False )
    sim.add_particles( params["beam.charge.C"], distri, int( params["beam.nparticles"]) )

    # ------------------------------------------------- #
    # --- [5] set lattice                           --- #
    # ------------------------------------------------- #
    beamlineFile = "../dat/beamline_impactx.json"
    beamline     = rtk.set__latticeComponents( elements=None, beamlineFile=beamlineFile, \
                                               params  = params, nUse=params["sim.nUse.elements"] )
    sim.lattice.extend( beamline )
    
    # ------------------------------------------------- #
    # --- [6] tracking                              --- #
    # ------------------------------------------------- #
    sim.track_particles()
    sim.finalize()
    te  = time.perf_counter()
    print( "\n Elapsed time ::: {:10.5e} (s)\n".format( te-ts ) )

    # ------------------------------------------------- #
    # --- [7] save in a file                        --- #
    # ------------------------------------------------- #
    rtk.save__run_records( params=params, recoFile="diags/records.json" )
    rtk.save__latticeStructure( beamlineFile=beamlineFile, nUse=params["sim.nUse.elements"], \
                                outFile="diags/lattice.csv" )
    
    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    main_impactx()
    
