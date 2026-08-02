import copy, glob, os, shutil, time, json5
import impactx
import numpy         as np
import pandas        as pd
import amrex.space3d as amr
import nk_toolkit.impactx.EBmapElement__RK as ebmrk


# ========================================================= #
# ===  set__beamlineComponents                          === #
# ========================================================= #

def set__latticeComponents( elements=None, beamlineFile="../dat/beamline_impactx.json",
                            add_bpm=True , nUse=None, params ={}, ):

    # ------------------------------------------------- #
    # --- [0] lattice element specifications        --- #
    # ------------------------------------------------- #
    elementSpecifications = {
        "rfcavity" : {
            "class"   : impactx.elements.RFCavity,
            "allowed" : [ "ds", "escale", "freq", "phase", "cos_coefficients", "sin_coefficients",
                          "z", "field_on_axis", "ncoef", "dx", "dy", "rotation",
                          "aperture_x", "aperture_y", "mapsteps", "nslice", "name", ], 
            "options" : "translate.cavity.options",
        }, 
        "shortrf" : {
            "class"   : impactx.elements.ShortRF,
            "allowed" : [ "V", "freq", "phase", "dx", "dy", "rotation", "name" ], 
            "options" : "translate.cavity.options",
        }, 
        "quadrupole" : {
            "class"   : impactx.elements.ChrQuad,
            "allowed" : [ "ds", "k", "unit", "dx", "dy", "rotation",
                          "aperture_x", "aperture_y", "name", "nslice", ],  
            "options" : "translate.quad.options",
        }, 
        "quadrupole.linear" : {
            "class"   : impactx.elements.Quad,
            "allowed" : [ "ds", "k", "dx", "dy", "rotation",
                          "aperture_x", "aperture_y", "name", "nslice", ],  
            "options" : "translate.quad.options",
        },
        "solenoid" : {
            "class"   : impactx.elements.Sol,
            "allowed" : [ "ds", "ks", "dx", "dy", "rotation",
                          "aperture_x", "aperture_y", "name", "nslice", ],
            "options" : "translate.solenoid.options",
        },
        "drift" : {
            "class"   : impactx.elements.ExactDrift,
            "allowed" : [ "ds", "dx", "dy", "rotation",
                          "aperture_x", "aperture_y", "name", "nslice", ],
            "options" : "translate.drift.options",
        }, 
        "drift.linear" : {
            "class"   : impactx.elements.Drift,
            "allowed" : [ "ds", "dx", "dy", "rotation",
                          "aperture_x", "aperture_y", "name", "nslice", ],
            "options" : "translate.drift.options",
        },
        "ebmap.rk" : {
            "class"   : ebmrk.EBmapElement__RK, 
            "allowed" : [ "ds", "name", "nslice", "bfieldfile", "efieldfile", "bfactor", "efactor",
                          "freq", "phase", "phase_sync", "int_method", "aperture_x", "aperture_y",
                          "aperture_cx", "aperture_cy", ],
            "options" : "translate.ebmap.options",
        },
    }
    
    # ------------------------------------------------- #
    # --- [1] load json file                        --- #
    # ------------------------------------------------- #
    if ( beamlineFile is not None ):
        with open( beamlineFile, "r" ) as f:
            elements = json5.load( f )
    if ( elements is None ):
        raise ValueError( "[impactx_toolkit.py] elements == ??? " )
    if ( nUse is not None ):
        keys     = list( elements.keys() )
        nUse_    = min( nUse, len( keys ) )
        elements = { keys[ik]:elements[keys[ik]] for ik in range( nUse_ ) }
        
    # ------------------------------------------------- #
    # --- [2] set beam components                   --- #
    # ------------------------------------------------- #
    stack  = []
    if ( add_bpm ):
        bpm    = impactx.elements.BeamMonitor( "bpm", backend="h5" )
        stack += [ bpm ]
        
    for key,elem in elements.items():

        if   ( elem["type"] in elementSpecifications ):
            specs  = elementSpecifications[ elem["type"] ]
            elem_  = { k:v for k,v in elem.items() if ( k in specs["allowed"] ) }
            elem_  = { **elem_, **( params.get( specs["options"], {} ) ), }
            bcomp  = specs["class"]( **elem_ )
        
        elif ( elem["type"] in [ "rfgap" ] ):
            Rmat = impactx.Map6x6.identity()
            for ii in range(6):
                for ij in range(6):
                    Rmat[ii+1,ij+1] = elem["R"][ii][ij]
            allowed    = [ "R", "dx", "dy", "rotation", "name" ]
            elem_      = { k:v for k,v in elem.items() if ( k in allowed ) }
            elem_["R"] = Rmat
            bcomp      = impactx.elements.LinearMap ( **elem_ )

        else:
            raise ValueError( "[set__latticeComponent]\n" +\
                              "    unknown element type :: {}".format( elem["type"] ) )

        stack += [ bcomp ]
        if ( add_bpm ):
            stack += [ bpm ]

    # ------------------------------------------------- #
    # --- [3] return                                --- #
    # ------------------------------------------------- #
    beamline = stack
    return( beamline )


# ========================================================= #
# ===  set__manualReferenceParticle.py                  === #
# ========================================================= #

def set__manualReferenceParticle( particle_container=None,  # particle_container of impactx
                                  particles=[ [0.,0.,0., 0.,0.,0. ],
                                              [0.,0.,0., 0.,0.,0. ], ] ):
    xp_, yp_, tp_ = 0, 1, 2
    px_, py_, pt_ = 3, 4, 5
    MeV           = 1.e6
    
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
    
    for ik, aparticle in enumerate( particles ):
        dx_podv.push_back ( aparticle[xp_]  )
        dy_podv.push_back ( aparticle[yp_]  )
        dt_podv.push_back ( aparticle[tp_]  )
        dpx_podv.push_back( aparticle[px_] )
        dpy_podv.push_back( aparticle[py_] )
        dpt_podv.push_back( aparticle[pt_] )
        w_podv.push_back  (   1.0        )

    refp   = particle_container.ref_particle()
    qm_eeV = refp.charge_qe / ( refp.mass_MeV * MeV )
    particle_container.add_n_particles( dx_podv , dy_podv , dt_podv ,
                                        dpx_podv, dpy_podv, dpt_podv,
                                        qm_eeV  , w=w_podv )
    return( particle_container )



# ========================================================= #
# ===  set waterbag distribution                        === #
# ========================================================= #

def set__waterbag_distribution( alpha=None, beta=None, eps_geom=None, \
                                mm_mrad=True, full_emittance=False, verbose=True ):
    
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

    if ( verbose ) :
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



# ========================================================= #
# ===  save__run_records                                === #
# ========================================================= #

def save__run_records( params=None, keys=None, recoFile="diags/records.json", verbose=True ):

    amu                    = 931.494

    # ------------------------------------------------- #
    # --- [1] arguments / calculation               --- #
    # ------------------------------------------------- #
    if ( params is None ): raise ValueError( "[save__run_records] params == ???" )
    if ( keys   is None ):
        keys = [ "beam.charge.qe"   , "beam.charge.C"  , "beam.nparticles", \
                 "beam.u.nucleon"   , "beam.mass.amu"  , "beam.Ek.MeV/u"  , \
                 "beam.freq.Hz"     , "beam.freq.rf.Hz", "beam.harmonics" , \
                 "beam.twiss.alpha" , "beam.twiss.beta", "beam.emittance.geom", \
                 "beam.Em0.MeV"     , "beam.Ek0.MeV"   , "sim.nUse.elements", \
                ]
    params["beam.Em0.MeV"]    = params["beam.mass.amu"] * amu
    params["beam.Ek0.MeV"]    = params["beam.Ek.MeV/u"] * params["beam.u.nucleon"]
    params["beam.freq.rf.Hz"] = params["beam.freq.Hz"]  * params["beam.harmonics"]
    
    # ------------------------------------------------- #
    # --- [2] save in a file / return               --- #
    # ------------------------------------------------- #
    records = { key:params[key] for key in keys }
    with open( recoFile, "w" ) as f:
        json5.dump( records, f, indent=2 )
        if ( verbose ):
            print( " records output :: {} \n".format( recoFile ) )
    return( records )


# ========================================================= #
# ===  save__latticeStructure                           === #
# ========================================================= #

def save__latticeStructure( elements=None, beamlineFile=None, nUse=None, add_bpm=True, \
                            outFile="diags/lattice.csv", labelFile="diags/lattice_label.csv" ):

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( elements is None ):
        if ( beamlineFile is None ):
            raise ValueError( "[save__latticesStructure]  beamlineFile or elements== ???" )
        else:
            with open( beamlineFile, "r" ) as f:
                elements = json5.load( f )
    if ( nUse is not None ):
        keys     = list( elements.keys() )
        nUse_    = min( nUse, len( keys ) )
        elements = { keys[ik]:elements[keys[ik]] for ik in range( nUse_ ) }
        
    # ------------------------------------------------- #
    # --- [2] expand elements by nslice             --- #
    # ------------------------------------------------- #
    expanded = []
    for _, elem in elements.items():
        name   =        elem.get( "name", ""      )
        etype  =        elem.get( "type", ""      )
        ds     = float( elem.get( "ds"    , 0.0 ) )
        nslice =   int( elem.get( "nslice", 1   ) )

        if nslice < 1:
            nslice = 1
        ds_slice = ds / nslice
        for i in range( nslice ):
            expanded.append( {
                "name"   : name     ,
                "type"   : etype    ,
                "ds"     : ds_slice ,
                "slice"  :  i+1     ,
                "nslice" : nslice   ,
            } )
    lattice = pd.DataFrame(expanded)
                
    # ------------------------------------------------- #
    # --- [3] cumulative s                          --- #
    # ------------------------------------------------- #
    ds = lattice["ds"].to_numpy()
    lattice["s_in"]  = np.cumsum(np.insert(ds, 0, 0.0))[:-1]
    lattice["s_out"] = np.cumsum(ds)
    lattice["s_mid"] = 0.5 * (lattice["s_in"] + lattice["s_out"])

    # ------------------------------------------------- #
    # --- [4] save in a file                        --- #
    # ------------------------------------------------- #
    with open( outFile, "w" ) as f:
        lattice.to_csv( f, float_format="%.8e" )

    # ------------------------------------------------- #
    # --- [5] save labels                           --- #
    # ------------------------------------------------- #
    result = []
    if ( add_bpm ):
        result.append( "bpm" )

    for row in expanded:
        result.append( row["name"] )
        if ( add_bpm and row["slice"] == row["nslice"] ):
            result.append( "bpm" )
    df            = pd.DataFrame( { "name":result } )
    df.index      = range(1, len(df)+1 )
    df.index.name = "id"
    df.to_csv( labelFile, index=True )



# ========================================================= #
# ===  execute__impactx                                 === #
# ========================================================= #

def execute__impactx( params=None, paramsFile="../dat/parameters.json",
                      elements=None, beamlineFile="../dat/beamline_impactx.json",
                      workDir=".", runMode=None, add_bpm=None, clearDiags=True,
                      saveRecords=True, saveLattice=True, verbose=None ):
    """Execute ImpactX tracking or envelope calculation."""

    amu     = 931.494   # [MeV]
    workDir = os.path.abspath( workDir )
    cwd     = os.getcwd()

    # ------------------------------------------------- #
    # --- [1] load input                            --- #
    # ------------------------------------------------- #
    try:
        os.chdir( workDir )

        if ( params is None ):
            with open( paramsFile, "r" ) as fk:
                params_ = json5.load( fk )
        else:
            params_     = copy.deepcopy( params )

        if ( elements is None ):
            with open( beamlineFile, "r" ) as fk:
                elements_ = json5.load( fk )
        else:
            elements_     = copy.deepcopy( elements )

        if ( runMode is None ):
            runMode     = params_["mode.run"]
        if ( add_bpm is None ):
            add_bpm     = params_["sim.add_bpm"]
        if ( verbose is None ):
            verbose     = int( params_["sim.verbose"] )
        else:
            verbose     = int( verbose )
        runMode = runMode.lower()

        # ------------------------------------------------- #
        # --- [2] initialize ImpactX                    --- #
        # ------------------------------------------------- #
        if ( clearDiags and os.path.isdir( "diags" ) ):
            shutil.rmtree( "diags" )

        startTime                  = time.perf_counter()
        sim                        = impactx.ImpactX()
        sim.max_level              = params_["sim.max_level"]
        sim.n_cell                 = params_["sim.n_cell"]
        sim.blocking_factor        = params_["sim.blocking_factor"]
        sim.particle_shape         = params_["sim.particle_shape"]
        sim.slice_step_diagnostics = params_["sim.slice_step_diagnostics"]
        sim.space_charge           = params_["mode.space_charge"]
        sim.poisson_solver         = params_["sim.poisson_solver"]
        sim.dynamic_size           = params_["sim.dynamic_size"]
        sim.prob_relative          = params_["sim.prob_relative"]
        sim.particle_lost_diagnostics_backend = params_["sim.particle_lost_diagnostics_backend"]
        sim.verbose                = verbose
        sim.mlmg_verbosity         = verbose          # MLMG std log
        sim.tiny_profiler          = bool( verbose )  # profiler 
        sim.init_grids()

        # ------------------------------------------------- #
        # --- [3] reference particle                   --- #
        # ------------------------------------------------- #
        Ek0  = params_["beam.Ek.MeV/u"] * params_["beam.u.nucleon"]
        Em0  = params_["beam.mass.amu"] * amu
        refp = sim.beam.ref
        refp.set_charge_qe     ( params_["beam.charge.qe"] )
        refp.set_mass_MeV      ( Em0 )
        refp.set_kin_energy_MeV( Ek0 )

        # ------------------------------------------------- #
        # --- [4] initial distribution                  --- #
        # ------------------------------------------------- #
        distri = set__waterbag_distribution( alpha=np.array( params_["beam.twiss.alpha"] ),
                                             beta =np.array( params_["beam.twiss.beta"] ),
                                             eps_geom=np.array( params_["beam.emittance.geom"] ),
                                             mm_mrad=True, full_emittance=False,
                                             verbose=verbose )
        bunch_charge_C           = params_["beam.current.A"] / params_["beam.freq.Hz"]
        params_["beam.charge.C"] = bunch_charge_C
        if ( runMode == "tracking" ):
            sim.add_particles( bunch_charge_C, distri,int( params_["beam.nparticles"] ) )
        elif ( runMode == "envelope" ):
            sc_flag = params_["mode.space_charge"]
            if   ( sc_flag is False or sc_flag is None ):
                sim.init_envelope( refp, distri )
            elif ( str( sc_flag ).upper() == "3D" ):
                sim.init_envelope( refp, distri, bunch_charge_C )
            elif ( str( sc_flag ).upper() == "2D" ):
                sim.init_envelope( refp, distri, float( params_["beam.current.A"] ) )
            else:
                raise ValueError( "Unknown space-charge mode: {}".format( sc_flag ) )
        else:
            raise ValueError( "Unknown ImpactX run mode: {}".format( runMode ) )

        # ------------------------------------------------- #
        # --- [5] lattice and execution                 --- #
        # ------------------------------------------------- #
        beamline = set__latticeComponents( elements=elements_, beamlineFile=None, add_bpm=add_bpm,
                                           nUse=params_["sim.nUse.elements"], params=params_ )
        sim.lattice.extend( beamline )

        if   ( runMode == "tracking" ):
            sim.track_particles()
        elif ( runMode == "envelope" ):
            sim.track_envelope()
        sim.finalize()
        elapsedS = time.perf_counter() - startTime

        # ------------------------------------------------- #
        # --- [6] output                                --- #
        # ------------------------------------------------- #
        if ( saveRecords ):
            save__run_records( params=params_, recoFile="diags/records.json", verbose=verbose )
        if ( saveLattice ):
            save__latticeStructure( elements=elements_, nUse=params_["sim.nUse.elements"],
                                    outFile="diags/lattice.csv", add_bpm=add_bpm, 
                                    labelFile="diags/lattice_label.csv" )
        if ( verbose ):
            print( "\n Elapsed time ::: {:10.5e} (s)\n".format( elapsedS ) )

        statFile = sorted( glob.glob( "diags/reduced_beam_characteristics.*" ) )[0]
        refpFile = sorted( glob.glob( "diags/ref_particle.*" ) )[0]
        ret      = { "params":params_, "elements":elements_, "mode":runMode,
                     "elapsedS":elapsedS, "statFile":os.path.abspath( statFile ),
                     "refpFile":os.path.abspath( refpFile ) }
    finally:
        os.chdir( cwd )
    return( ret )
