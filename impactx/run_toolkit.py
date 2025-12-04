import os, sys, json5
import impactx
import h5py
import numpy   as np
import pandas  as pd

# ========================================================= #
# ===  set__beamlineComponents                          === #
# ========================================================= #

def set__latticeComponents( elements=None, beamlineFile="../dat/beamline_impactx.json",
                            add_bpm=True , logFile="beamline_log.json" ):

    # ------------------------------------------------- #
    # --- [1] load json file                        --- #
    # ------------------------------------------------- #
    if ( beamlineFile is not None ):
        with open( beamlineFile, "r" ) as f:
            elements = json5.load( f )
    if ( elements is None ):
        print( "[impactx_toolkit.py] elements == ??? " )
        sys.exit()
        
    # ------------------------------------------------- #
    # --- [2] set beam components                   --- #
    # ------------------------------------------------- #
    stack  = []
    if ( add_bpm ):
        bpm    = impactx.elements.BeamMonitor( "bpm", backend="h5" )
        stack += [ bpm ]
    for key,elem in elements.items():
        elem_ = { key:val for key,val in elem.items() if ( key != "type" ) }
        
        if   ( elem["type"] in [ "rfcavity" ]   ):
            bcomp = impactx.elements.RFCavity  ( **elem_ )

        elif ( elem["type"] in [ "shortrf"  ]   ):
            bcomp = impactx.elements.ShortRF   ( **elem_ )

        elif ( elem["type"] in [ "rfgap" ] ):
            Rmat = impactx.Map6x6.identity()
            for ii in range(6):
                for ij in range(6):
                    Rmat[ii+1,ij+1] = elem_["R"][ii][ij]
            elem_["R"] = Rmat
            bcomp = impactx.elements.LinearMap ( **elem_ )
            
        elif ( elem["type"] in [ "quadrupole" ]   ):
            # bcomp = impactx.elements.ExactQuad ( **elem_ )
            bcomp = impactx.elements.ChrQuad ( **elem_ )

        elif ( elem["type"] in [ "drift" ]        ):
            bcomp = impactx.elements.ExactDrift( **elem_ )
            
        elif ( elem["type"] in [ "quadrupole.linear" ]  ):
            elem_ = { k:v for k,v in elem_.items() if k not in ["int_order", "mapsteps", "unit"] }
            bcomp = impactx.elements.Quad ( **elem_ )

        elif ( elem["type"] in [ "drift.linear" ] ):
            bcomp = impactx.elements.Drift( **elem_ )

        else:
            sys.exit( "[main_impactx.py] unknown element type :: {} ".format( elem["type"] ) )

        stack += [ bcomp ]
        if ( add_bpm ):
            stack += [ bpm ]

    # ------------------------------------------------- #
    # --- [3] return                                --- #
    # ------------------------------------------------- #
    beamline = stack
    # if ( logFile is not None ):
    #     stack = { str(ik):element.to_dict() for ik,element in enumerate(beamline) }
    #     with open( logFile, "w" ) as f:
    #         json5.dump( stack, f, indent=4 )
    return( beamline )


# ========================================================= #
# ===  set__manualReferenceParticle.py                  === #
# ========================================================= #

def set__manualReferenceParticle( particle_container=None,  # particle_container of impactx
                                  ref_xyt=[0.,0.,0.], ref_pxyt=[0.,0.,0.], 
                                  n_part=2,                 # #.of particles : min. => 2
                                 ):
    x_, y_, t_ = 0, 1, 2
    MeV        = 1.e6
    
    # ------------------------------------------------- #
    # --- [1] set particle distribution             --- #
    # ------------------------------------------------- #
    dx_podv  = amr.PODVector_real_std()
    dy_podv  = amr.PODVector_real_std()
    dt_podv  = amr.PODVector_real_std()
    dpx_podv = amr.PODVector_real_std()
    dpy_podv = amr.PODVector_real_std()
    dpt_podv = amr.PODVector_real_std()
    w_podv   = amr.PODVector_real_std()
    
    for ik in range( n_part ):
        dx_podv.push_back ( ref_xyt[x_]  )
        dy_podv.push_back ( ref_xyt[y_]  )
        dt_podv.push_back ( ref_xyt[t_]  )
        dpx_podv.push_back( ref_pxyt[x_] )
        dpy_podv.push_back( ref_pxyt[y_] )
        dpt_podv.push_back( ref_pxyt[t_] )
        w_podv.push_back  (   1.0        )

    refp   = particle_container.ref_particle()
    qm_eeV = refp.charge_qe / ( refp.mass_MeV * MeV )
    particle_container.add_n_particles( dx_podv , dy_podv , dt_podv ,
                                        dpx_podv, dpy_podv, dpt_podv,
                                        qm_eeV  , w=w_podv )
    return( particle_container )



# ========================================================= #
# ===  set water distribution                           === #
# ========================================================= #

def set__waterbag_distribution( alpha=None, beta=None, eps_geom=None, \
                                mm_mrad=True, full_emittance=False ):
    
    x_, y_, t_   = 0, 1, 2
    mm, mrad     = 1.0e-3, 1.0e-3
    full2rms     = 1.0 / 8.0        # full -> rms emittance ( 6D wb-> 1/8 )
    
    # ------------------------------------------------- #
    # --- [1] unit conversion                       --- #
    # ------------------------------------------------- #
    if ( mm_mrad ):
        eps_geom = eps_geom * mm * mrad
    if ( full_emittance ):
        eps_geom = eps_geom * full2rms

    gamma        = (1.0 + alpha**2) / ( beta )
    lambda_q     = np.sqrt( eps_geom / gamma )
    lambda_p     = np.sqrt( eps_geom /  beta )
    mu_qp        = alpha / np.sqrt( beta * gamma )

    # ------------------------------------------------- #
    # --- [2] definition of distribution            --- #
    # ------------------------------------------------- #
    distri       = impactx.distribution.Waterbag(
        lambdaX  = lambda_q[x_], lambdaY  = lambda_q[y_], lambdaT  = lambda_q[t_],
        lambdaPx = lambda_p[x_], lambdaPy = lambda_p[y_], lambdaPt = lambda_p[t_],
        muxpx    = mu_qp[x_]   , muypy    = mu_qp[y_],    mutpt    = mu_qp[t_], 
    )
    
    print( "\n" + " ===     initial particle distribution    === " )
    print( "  * alpha    :: ", alpha )
    print( "  * beta     :: ", beta  )
    print( "  * gamma    :: ", gamma )
    print( "  * eps_geom :: ", eps_geom  )
    print( "  * sigma    :: ", lambda_q  )
    print( "  * sigma_p  :: ", lambda_p  )
    print( "  * mu_qp    :: ", mu_qp     )
    print( " ============================================ " + "\n" )
    return( distri )

