import json5
import impactx

# ========================================================= #
# ===  impactX tracking file                            === #
# ========================================================= #
def main_impactx():

    sim = impactx.ImpactX()

    # ------------------------------------------------- #
    # --- [1]  preparation                          --- #
    # ------------------------------------------------- #
    qD                         = +1.0                   # Deuteron [MeV]
    mD                         = 939.494 * 2.014        # Deuteron [MeV]
    Ek0                        = 2.0* 20.0              # kinetic energy :: 20 [MeV/u] * 2
    bunch_charge               = 1.0e-9                 # [C]
    nparticles                 = 10000
    
    sim.particle_shape         = 2
    sim.space_charge           = False
    sim.slice_step_diagnostics = False
    
    # ------------------------------------------------- #
    # --- [2] grid & reference particle             --- #
    # ------------------------------------------------- #
    sim.init_grids()
    ref = sim.particle_container().ref_particle()
    ref.set_charge_qe     ( qD  )
    ref.set_mass_MeV      ( mD  )
    ref.set_kin_energy_MeV( Ek0 )
    
    # ------------------------------------------------- #
    # --- [3] distribution                          --- #
    # ------------------------------------------------- #
    distri       = impactx.distribution.Waterbag(
        lambdaX  = 3.9984884770e-5,
        lambdaY  = 3.9984884770e-5,
        lambdaT  = 1.0e-3,
        lambdaPx = 2.6623538760e-5,
        lambdaPy = 2.6623538760e-5,
        lambdaPt = 2.0e-3,
        muxpx    = -0.846574929020762,
        muypy    = 0.846574929020762,
        mutpt    = 0.0,
    )
    sim.add_particles( bunch_charge, distri, nparticles )   # -- for particle tracking -- #
    # sim.init_envelope(ref, distr)                         # -- for envelop  tracking -- #
    
    # ------------------------------------------------- #
    # --- [4] lattice definition                    --- #
    # ------------------------------------------------- #
    import lattice_elements
    
    # ------------------------------------------------- #
    # --- [5] tracking                              --- #
    # ------------------------------------------------- #
    sim.lattice.extend( lattice_elements.beamline )
    sim.track_particles()                                   # -- for particle tracking -- #
    # sim.track_envelope()                                  # -- for envelop  tracking -- #
    
    # ------------------------------------------------- #
    # --- [6] end                                   --- #
    # ------------------------------------------------- #
    sim.finalize()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    main_impactx()
