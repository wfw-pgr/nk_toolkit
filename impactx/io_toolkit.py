import os, sys, json5, h5py
import numpy  as np
import pandas as pd

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
# ===  get__particles.py                                === #
# ========================================================= #

def get__particles( params=None,
                    paramsFile="dat/parameters.json", \
                    bpmFile="impactx/diags/openPMD/bpm.h5", \
                    refFile="impactx/diags/ref_particle.0", \
                    steps=None, pids=None ):

    amu = 931.494
    cv  = 2.99792458e8

    def norm_angle( deg ):
        deg  = np.asarray( deg, dtype=float )
        ret  = ( ( deg + 180.0 ) % 360.0 ) - 180.0
        if ( np.ndim(ret) == 0 ): ret = float( ret )
        return( ret )
    
    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    if ( params is None ):
        with open( paramsFile, "r" ) as f:
            params = json5.load( f )
    bpm = load__impactHDF5( inpFile=bpmFile, pids=pids, \
                            steps=steps, redefine_step=False, \
                            step_start=0 ).reset_index( drop=True )
    ref = pd.read_csv( refFile, sep=r"\s+" )

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
    rf_freq     = params["beam.freq.Hz"]  * params["beam.harmonics"]
    Em0         = params["beam.mass.amu"] * amu
    Ek0         = params["beam.Ek.MeV/u"] * params["beam.u.nucleon"]
    Et0         = Em0 + Ek0
    p0c         = np.sqrt( Et0**2 - Em0**2 )
    Ek_ref      = ( bpm["ref_gamma"] - 1.0 ) * Em0
    bpm["dEk"]  = p0c * bpm["pt"]
    bpm["Ek"]   = Ek_ref + bpm["dEk"]
    bpm["dt"]   = bpm["tp"]  / cv
    bpm["dphi"] = norm_angle( (                   bpm["dt"] ) * rf_freq * 360.0 )
    bpm["phi"]  = norm_angle( ( bpm["ref_t"]/cv + bpm["dt"] ) * rf_freq * 360.0 )
    return( bpm )


# ========================================================= #
# ===  get__beamStats                                   === #
# ========================================================= #

def get__beamStats( statFile="impactx/diags/reduced_beam_characteristics.0", \
                    refpFile="impactx/diags/ref_particle.0", ext=None  ):
    
    if ( ext is not None ):
        statFile = os.path.splitext()[0] + ext
        refpFile = os.path.splitext()[0] + ext
    
    refp = pd.read_csv( refpFile, sep=r"\s+" )
    stat = pd.read_csv( statFile, sep=r"\s+" )
    ret  = pd.concat( [ refp.set_index("step"), stat.set_index("step") ], axis=1 ).reset_index()
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
        
