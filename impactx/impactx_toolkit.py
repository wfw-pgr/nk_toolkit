import os, sys, tqdm, glob, json5
import impactx
import h5py
import numpy   as np
import pandas  as pd
import pyvista as pv
import scipy   as sp
import amrex.space3d                   as amr
import matplotlib.pyplot               as plt
import matplotlib.patches              as patches
import nk_toolkit.plot.load__config    as lcf
import nk_toolkit.plot.gplot1D         as gp1
import nk_toolkit.math.fourier_toolkit as ftk


# ========================================================= #
# === impactx_toolkit.py                                === #
# ========================================================= #
#
#  * load__impactHDF5
#  * convert__hdf2vtk
#  * compute__fourierCoefficients
#  * adjust__RFcavityPhase
#  * plot__lattice
# 
# ========================================================= #


# ========================================================= #
# === load__impactHDF5.py                               === #
# ========================================================= #

def load__impactHDF5( inpFile=None, pids=None, steps=None, random_choice=None, 
                      redefine_pid=True, redefine_step=True, step_start=0 ):

    default_step_start = 1
    
    # ------------------------------------------------- #
    # --- [1] load HDF5 file                        --- #
    # ------------------------------------------------- #
    stack = []
    with h5py.File( inpFile, "r" ) as f:
        isteps = sorted( [ int( key ) for key in f["data"].keys() ] )
        if   ( steps is None ):
            pass
        elif ( isinstance( steps, ( list, tuple, np.ndarray ) ) ):
            isteps = sorted( set(isteps) & set( steps ) )
        elif ( isinstance( steps, ( int, float ) ) ):
            index  = np.linspace( 0,len(isteps)-1, int(steps), dtype=int )
            isteps = [ isteps[ik] for ik in index ]
        for step in isteps:
            try:
                key, df    = str(step), {}
                df["pid"]  = f["data"][key]["particles"]["beam"]["id"][:]
                df["xp"]   = f["data"][key]["particles"]["beam"]["position"]["x"][:] 
                df["yp"]   = f["data"][key]["particles"]["beam"]["position"]["y"][:] 
                df["tp"]   = f["data"][key]["particles"]["beam"]["position"]["t"][:] 
                df["px"]   = f["data"][key]["particles"]["beam"]["momentum"]["x"][:] 
                df["py"]   = f["data"][key]["particles"]["beam"]["momentum"]["y"][:] 
                df["pt"]   = f["data"][key]["particles"]["beam"]["momentum"]["t"][:]
                kstep      = step + ( step_start - default_step_start )
                df["step"] = np.full( df["pid"].shape, kstep, dtype=int )
                stack     += [ pd.DataFrame( df ) ]
            except TypeError:
                print( "[load__impactHDF5.py] detected TypeError at step == {}.. continue. ".format( step ) )
                
    ret = pd.concat( stack, ignore_index=True )
    
    # ------------------------------------------------- #
    # --- [2] return                                --- #
    # ------------------------------------------------- #
    if ( redefine_pid  ):
        ret["pid"]  = pd.factorize( ret["pid"]  )[0] + step_start
    if ( redefine_step ):
        ret["step"] = pd.factorize( ret["step"] )[0] + step_start
    if ( random_choice is not None ):
        npart = len( set( ret["pid"] ) )
        if ( random_choice > npart ):
            raise ValueError( f"random_choice ({random_choice}) > number of particles ({npart})")
        pids  = np.random.choice( np.arange(1,npart+1), size=random_choice, replace=False )
    if ( pids  is not None ):
        ret   = ret[ ret["pid"].isin( pids ) ]
    # if ( steps is not None ):
    #     ret   = ret[ ret["step"].isin( steps ) ]
        
    return( ret )



# ========================================================= #
# ===  convert__hdf2vtk.py                              === #
# ========================================================= #

def convert__hdf2vtk( hdf5File=None, outFile=None, \
                      pids=None, steps=None, random_choice=None ):

    # ------------------------------------------------- #
    # --- [1] arguments check                       --- #
    # ------------------------------------------------- #
    if ( os.path.exists( hdf5File ) ):
        Data = load__impactHDF5( inpFile=hdf5File, random_choice=random_choice, \
                                 pids=pids, steps=steps )
    else:
        raise FileNotFoundError( f"[ERROR] HDF5 File not Found :: {hdf5File}" )
    if ( outFile is None ):
        raise TypeError( f"[ERROR] outFile == {outFile} ???" )
    else:
        ext = os.path.splitext( outFile )[1]

    # ------------------------------------------------- #
    # --- [2] save as vtk poly data                 --- #
    # ------------------------------------------------- #
    steps = sorted( Data["step"].unique() )

    for ik,step in enumerate(steps):
        # -- points coordinate make -- #
        df     = Data[ Data["step"] == step ]
        df     = df[ np.isfinite(df["xp"]) & \
                     np.isfinite(df["yp"]) & \
                     np.isfinite(df["tp"]) ]
        if ( df.shape[0] == 0 ):
            print( "[impactx_toolkit.py] [WARNING] no appropriate point data :: ik={0}, step={1}".format( ik, step ) )
            continue
        df["dt"] = df["tp"] - df["tp"].mean()
        coords   = df[ ["xp", "yp", "dt"] ].to_numpy()
        cloud    = pv.PolyData( coords )
        # -- momentum & pid -- #
        cloud.point_data["pid"]      = df["pid"].to_numpy()
        cloud.point_data["x"]        = df["xp" ].to_numpy()
        cloud.point_data["y"]        = df["yp" ].to_numpy()
        cloud.point_data["t"]        = df["tp" ].to_numpy()
        cloud.point_data["dt"]       = df["dt" ].to_numpy()
        cloud.point_data["momentum"] = df[ ["px", "py", "pz"] ].to_numpy()
    
        # -- save file -- #
        houtFile = outFile.replace( ext, "-{0:06}".format(ik+1) + ext )
        cloud.save( houtFile )
        print( "[convert__hdf2vtk.py] outFile :: {} ".format( houtFile ) )
    return()
        


# ========================================================= #
# ===  postProcessed__beam                              === #
# ========================================================= #

def postProcessed__beam( refpFile=None, statFile=None, paramsFile="dat/parameters.json", \
                         outFile="impactx/diags/postProcessed_beam.csv" ):
    
    amu = 931.494   # [MeV]
    cv  = 2.998e8
    
    # ------------------------------------------------- #
    # --- [1] arguments check                       --- #
    # ------------------------------------------------- #
    if ( refpFile is None ):
        flist = glob.glob( "impactx/diags/ref_particle.*" )
        if ( len( flist ) == 1 ):
            refpFile = flist[0]
        else:
            sys.exit( "[plot__additionals] no refpFile " )
    if ( statFile is None ):
        flist = glob.glob( "impactx/diags/reduced_beam_characteristics.*" )
        if ( len( flist ) == 1 ):
            statFile = flist[0]
        else:
            sys.exit( "[plot__additionals] no statFile " )
        
    # ------------------------------------------------- #
    # --- [2] load file                             --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    refp = pd.read_csv( refpFile, sep=r"\s+" )
    stat = pd.read_csv( statFile, sep=r"\s+" )

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


# ========================================================= #
# ===  compute__fourierCoefficients                     === #
# ========================================================= #
def compute__fourierCoefficients( xp=None, fx=None, nMode=None, normalize=True, \
                                  pngFile=None, coefFile=None, tolerance=1.e-10, a0_as_2a0=True ):
    
    ret      = ftk.get__fourierCoeffs( xp, fx, nMode=nMode, tolerance=tolerance )
    func     = ret["reconstruction"]
    if ( normalize ):
        volt = sp.integrate.simpson( fx, x=xp )
        # fx   = fx / volt
        ret["cos"] = ret["cos"] / volt
        ret["sin"] = ret["sin"] / volt
        # norm       = np.linalg.norm( fx )
        # ret["cos"] = ret["cos"] / norm
        # ret["sin"] = ret["sin"] / norm
    if ( a0_as_2a0 ):
        ret["cos"][0] = 2.0 * ret["cos"][0]
    data     = np.concatenate( [ ret["cos"][:,np.newaxis], ret["sin"][:,np.newaxis] ], axis=1 )
    if (  pngFile is not None ):
        ftk.display__fourierExpansion( xp, fx, func=func, pngFile=pngFile )
    if ( coefFile is not None ):
        with open( coefFile, "w" ) as f:
            sc = ",".join( [ "{:16.8e}".format( val ) for val in ret["cos"] ] )
            ss = ",".join( [ "{:16.8e}".format( val ) for val in ret["sin"] ] )
            st = "[ {} ]\n".format(sc) + "[ {} ]\n".format(ss)
            f.write( st )
            # -- old -- #
            # np.savetxt( f, data, fmt="%15.8e" )
            # print( "[impactx_toolkit.py] outfile :: {} ".format( coefFile ) )
            # -- old -- #
    return( data )


# ========================================================= #
# ===  adjust__RFcavityPhase.py                         === #
# ========================================================= #

def adjust__RFcavityPhase( paramsFile="dat/beamline.json", \
                           outFile="dat/adjust__RFcavityPhase.dat" ):
    
    cv     = 299792458.0
    ns     = 1.e-9

    # ------------------------------------------------- #
    # --- [1] preparation                           --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    bl       = params["beamline"]
    omega    = 2.0*np.pi * params["freq"]
    phi_t    = params["phase_tobe"]
    Em0, Ek  = params["Em0"], params["Ek0"]
    nElems   = len( bl )
    
    # ------------------------------------------------- #
    # --- [2] use functions                         --- #
    # ------------------------------------------------- #
    def beta_from_Ek(Ek):
        gamma = 1.0 + Ek/Em0
        beta  = np.sqrt( 1.0 - 1.0/gamma**2 )
        return( beta )

    def dt_drift( L, Ek ):
        beta = beta_from_Ek( Ek )
        dt   = L / ( cv * beta )
        return( dt )

    def dt_cavity( L, V, Ek ):
        Ek_out = Ek + V
        v1     = beta_from_Ek(Ek    ) * cv
        v2     = beta_from_Ek(Ek_out) * cv
        dt     = L * (1.0/v1 + 1.0/v2) / 2.0   # 台形近似
        return( dt )

    def norm_angle( deg ):
        deg = deg % 360.0
        if ( deg > 180.0 ): deg = deg - 360.0
        return( deg )

    # ------------------------------------------------- #
    # --- [3] main loop                             --- #
    # ------------------------------------------------- #
    t    = 0.0
    rows = []
    for i,el in enumerate( bl ):
        t_in  = t
        if ( el["type"] in ["cavity"] ):
            dt   = dt_cavity( el["L"], el["V"], Ek )
            V    = el["V"]
            Ek  += V
        elif ( el["type"] in ["drift"] ):
            dt  = dt_drift( el["L"], Ek )
            V   = 0.0
        t     += dt
        t_out  = t
        t_mid  = 0.5*( t_in + t_out )
        phi_o  = norm_angle( omega*t_mid /np.pi*180.0 )
        phi_c  = norm_angle( phi_t - phi_o )
        phi_t  = norm_angle( phi_t )
        rows  += [ { "type"       : el["type"],
                     "L[m]"       : el["L"],
                     "V[MV]"      : V,
                     "Ek[MeV]"    : Ek-V,
                     "dt[ns]"     : dt/ns,
                     "t_in[ns]"   : t_in/ns,
                     "t_out[ns]"  : t_out/ns,
                     "t_mid[ns]"  : t_mid/ns,
                     "phi_o[deg]" : phi_o,
                     "phi_c[deg]" : phi_c,
                     "phi_t[deg]" : phi_t,
                    } ]
        
    # ------------------------------------------------- #
    # --- [4] return                                --- #
    # ------------------------------------------------- #
    df             = pd.DataFrame(rows)
    ds             = df["L[m]"]
    s_in           = np.concatenate( ( [0.0], np.cumsum(ds)[:-1] ) )
    s_out          = np.cumsum(ds)
    s_mid          = 0.5*( s_in + s_out )
    df["s_in[m]"]  = s_in
    df["s_mid[m]"] = s_mid
    df["s_out[m]"] = s_out
    df.to_csv( outFile )


    
# ========================================================= #
# ===  adjust__refpPhase.py                             === #
# ========================================================= #
def adjust__refpPhase( inpFile="impactx/diags/ref_particle.0", \
                       phaseFile="dat/rfphase.csv", ext=None, freq=None, phi_t=0.0 ):
    
    cv = 299792458.0

    # ------------------------------------------------- #
    # --- [1] load file                             --- #
    # ------------------------------------------------- #
    if ( freq is None ):
        sys.exit( "[adjust_refpPhase.py] freq == ??? " )    
    if ( ext is not None ):
        inpFile = os.path.splitext( inpFile )[0] + ext
    refp        = pd.read_csv( inpFile, sep=r"\s+", engine="python" )
    if ( os.path.exists( phaseFile ) ):
        phasedb = pd.read_csv( phaseFile )
        s_mid   = phasedb["s_mid[m]"].to_numpy()
        phi_b   = phasedb["phi_c[deg]"].to_numpy()
        phi_t   = phasedb["phi_t[deg]"].to_numpy()
    else:
        phi_b   = None
    

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



# ========================================================= #
# ===  get__energy.py                                   === #
# ========================================================= #

def get__energy( paramsFile="dat/parameters.json" ):

    amu = 931.494

    # ------------------------------------------------- #
    # --- [1] functions                             --- #
    # ------------------------------------------------- #
    def step_mapping( step_bpm ):
        return( (step_bpm-1)*2 )
    
    # ------------------------------------------------- #
    # --- [2] load data                             --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    bpm   = load__impactHDF5( inpFile=params["file.bpm"] )
    ref   = pd.read_csv( params["file.ref"], sep=r"\s+" )

    # ------------------------------------------------- #
    # --- [3] get energy                            --- #
    # ------------------------------------------------- #
    Em0       = params["beam.mass.amu"] * amu
    Ek0       = params["beam.Ek.MeV/u"] * params["beam.u"]
    Et0       = Em0 + Ek0
    p0c       = np.sqrt( Et0**2 - Em0**2 )
    gamma     = ref.loc[ step_mapping( bpm["step"] ), "gamma" ].to_numpy()
    Ek_ref    = ( gamma - 1.0 ) * Em0
    Ek        = Ek_ref + p0c * bpm["pt"].to_numpy()
    return( Ek )



# ========================================================= #
# ===  get__particles.py                                === #
# ========================================================= #

def get__particles( paramsFile="dat/parameters.json", bpmFile=None, refFile=None, \
                    steps=None, pids=None ):

    amu = 931.494
    cv  = 2.99792458e8
    
    # ------------------------------------------------- #
    # --- [1] functions                             --- #
    # ------------------------------------------------- #
    def step_mapping( step_bpm ):
        return( (step_bpm-1)*2 )
    
    # ------------------------------------------------- #
    # --- [2] load data                             --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    if ( bpmFile is None ):
        bpmFile = params["file.bpm"]
    if ( refFile is None ):
        refFile = params["file.ref"]
    bpm         = load__impactHDF5( inpFile=bpmFile, pids=pids, \
                                    steps=steps, redefine_step=False, \
                                    step_start=0 ).reset_index( drop=True )
    ref         = pd.read_csv( refFile, sep=r"\s+" )

    # ------------------------------------------------- #
    # --- [3] concatenate ref / particle data       --- #
    # ------------------------------------------------- #
    ref_df  = ref.loc[ bpm["step"], : ]
    slist   = [ "s","beta","gamma", "x","y","z","t", "px","py","pz","pt" ]
    renames = { s:"ref_"+s for s in slist }
    ref_df  = ( ref_df[ slist ] ).rename( columns=renames ).reset_index( drop=True )
    bpm     = pd.concat( [ bpm, ref_df ], axis=1 )

    # ------------------------------------------------- #
    # --- [4] get energy /                          --- #
    # ------------------------------------------------- #
    Em0        = params["beam.mass.amu"] * amu
    Ek0        = params["beam.Ek.MeV/u"] * params["beam.u"]
    Et0        = Em0 + Ek0
    p0c        = np.sqrt( Et0**2 - Em0**2 )
    Ek_ref     = ( bpm["ref_gamma"] - 1.0 ) * Em0
    bpm["dEk"] =          p0c * bpm["pt"].to_numpy()
    bpm["Ek"]  = Ek_ref + p0c * bpm["pt"].to_numpy()
    bpm["dt"]  = bpm["tp"]  / cv
    return( bpm )



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
# ===  translate__ExactQuad_to_ExactDrift               === #
# ========================================================= #

def translate__ExactQuad_to_ExactDrift( inpFile="dat/beamline_impactx.json",
                                        outFile="dat/beamline_noExactQuad.json" ):

    # ------------------------------------------------- #
    # --- [1] import beamline file                  --- #
    # ------------------------------------------------- #
    with open( inpFile, "r" ) as f:
        beamline = json5.load( f )
    sequence = beamline["sequence"]
    elements = beamline["elements"]

    # ------------------------------------------------- #
    # --- [2] translate ( ExactQuad -> ExactDrift ) --- #
    # ------------------------------------------------- #
    elements_ = {}
    for key,elem in elements.items():
        if ( elem["type"].lower() == "quadrupole" ):
            elem = { "type":"drift", "name":elem["name"], "ds":elem["ds"] }
        elements_[key] = elem
            
    # ------------------------------------------------- #
    # --- [3] export virtual beamline               --- #
    # ------------------------------------------------- #
    beamline_ = { "elements":elements_, "sequence":sequence }
    if ( outFile is not None ):
        with open( outFile, "w" ) as f:
            json5.dump( beamline_, f, indent=4 )
            print( "[translate__ExactQuad_to_ExactDrift] output :: {} ".format( outFile ) )
    return( beamline_ )


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


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    # # ------------------------------------------------- #
    # # --- [1] load HDF5 file                        --- #
    # # ------------------------------------------------- #
    # inpFile = "test/bpm.h5"
    # Data    = load__impactHDF5( inpFile=inpFile )
    # print( Data )

    # # ------------------------------------------------- #
    # # --- [2] plot reference particle               --- #
    # # ------------------------------------------------- #
    # inpFile = "test/ref_particle.0.0"
    # plot__refparticle( inpFile=inpFile )
    
    # # ------------------------------------------------- #
    # # --- [3] plot statistics                       --- #
    # # ------------------------------------------------- #
    # inpFile = "test/reduced_beam_characteristics.0.0"
    # plot__statistics( inpFile=inpFile )

    # # ------------------------------------------------- #
    # # --- [4] plot trajectories                     --- #
    # # ------------------------------------------------- #
    # hdf5File = "test/bpm.h5"
    # refpFile = "test/ref_particle.0.0"
    # plot__trajectories( hdf5File=hdf5File, refpFile=refpFile, random_choice=100 )
    
    # # ------------------------------------------------- #
    # # --- [5] convert to paraview vtk               --- #
    # # ------------------------------------------------- #
    # hdf5File = "test/bpm.h5"
    # outFile  = "png/bpm.vtp"
    # ret      = convert__hdf2vtk( hdf5File=hdf5File, outFile=outFile )

    # # ------------------------------------------------- #
    # # --- [6] compute  fourier coeffs               --- #
    # # ------------------------------------------------- #
    # coefFile = "test/fourier_expansion.dat"
    # pngFile  = "test/fourier_expansion.png"
    # xp       = np.linspace( 0.0, 1.0, 101 )
    # fx       = np.sin( xp*np.pi )**2.0
    # nMode    = None
    # compute__fourierCoefficients( xp=xp, fx=fx, nMode=nMode, \
    #                               pngFile=pngFile, coefFile=coefFile )

    # # ------------------------------------------------- #
    # # --- [7] adjust phase shift                    --- #
    # # ------------------------------------------------- #
    # adjust__RFcavityPhase( paramsFile="test/beamline.json", \
    #                        outFile="test/adjust__RFcavityPhase.dat" )
    

    # # ------------------------------------------------- #
    # # --- [8] plot__lattice                         --- #
    # # ------------------------------------------------- #
    # latticeFile = "dat/beamline_impactx.json"
    # pngFile     = "lattice.png"
    # plot__lattice( latticeFile=latticeFile, pngFile=pngFile )

    # ------------------------------------------------- #
    # --- [9] get__particles                        --- #
    # ------------------------------------------------- #
    particles = get__particles( paramsFile="dat/parameters.json" )
    
    
