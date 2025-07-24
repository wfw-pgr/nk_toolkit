import os, sys
import h5py
import numpy   as np
import pandas  as pd
import pyvista as pv
import nk_toolkit.plot.load__config as lcf
import nk_toolkit.plot.gplot1D      as gp1


# ========================================================= #
# === impactx_toolkit.py                                === #
# ========================================================= #
#
#  * load__impactHDF5
#  * plot__refparticle
#  * plot__statistics
#  * convert__hdf2vtk
#
# 
# ========================================================= #


# ========================================================= #
# === load__impactHDF5.py                               === #
# ========================================================= #

def load__impactHDF5( inpFile=None, pids=None, steps=None, random_choice=None, 
                      redefine_pid=True, redefine_step=True ):

    # ------------------------------------------------- #
    # --- [1] load HDF5 file                        --- #
    # ------------------------------------------------- #
    stack = []
    with h5py.File( inpFile, "r" ) as f:
        isteps = sorted( [ int( key ) for key in f["data"].keys() ] )
        for step in isteps:
            key, df    = str(step), {}
            df["pid"]  = f["data"][key]["particles"]["beam"]["id"][:]
            df["xp"]   = f["data"][key]["particles"]["beam"]["position"]["x"][:]
            df["yp"]   = f["data"][key]["particles"]["beam"]["position"]["y"][:]
            df["tp"]   = f["data"][key]["particles"]["beam"]["position"]["t"][:]
            df["px"]   = f["data"][key]["particles"]["beam"]["momentum"]["x"][:]
            df["py"]   = f["data"][key]["particles"]["beam"]["momentum"]["y"][:]
            df["pz"]   = f["data"][key]["particles"]["beam"]["momentum"]["t"][:]
            df["step"] = np.full( df["pid"].shape, step, dtype=int )
            stack     += [ pd.DataFrame( df ) ]
    ret = pd.concat( stack, ignore_index=True )
    
    # ------------------------------------------------- #
    # --- [2] return                                --- #
    # ------------------------------------------------- #
    if ( redefine_pid  ):
        ret["pid"]  = pd.factorize( ret["pid"]  )[0] + 1
    if ( redefine_step ):
        ret["step"] = pd.factorize( ret["step"] )[0] + 1
    if ( random_choice is not None ):
        npart = len( set( ret["pid"] ) )
        if ( random_choice > npart ):
            raise ValueError( f"random_choice ({random_choice}) > number of particles ({npart})")
        pids  = np.random.choice( np.arange(1,npart+1), size=random_choice, replace=False )
    if ( pids  is not None ):
        ret   = ret[ ret["pid"].isin( pids ) ]
    if ( steps is not None ):
        ret   = ret[ ret["step"].isin( steps ) ]
    return( ret )



# ========================================================= #
# ===  plot__refparticle.py                             === #
# ========================================================= #
def plot__refparticle( inpFile=None, pngDir="png/" ):

    ylabels  = { "beta"       : r"$\beta$"        ,
                 "gamma"      : r"$\gamma$"       ,
                 "beta_gamma" : r"$\beta \gamma$" ,
                 "x"          : r"$x$ [m]"        ,
                 "y"          : r"$y$ [m]"        ,
                 "z"          : r"$z$ [m]"        ,
                 "t"          : r"$ct$ [m]"       ,
                 "px"         : r"$p_x$ [rad]"    ,
                 "py"         : r"$p_y$ [rad]"    ,
                 "pz"         : r"$p_z$ [rad]"    ,
                 "pt"         : r"$p_t$ [rad]"    ,
                }
    THRESHOLD = 1e32
    
    # ------------------------------------------------- #
    # --- [1] load .stat file                       --- #
    # ------------------------------------------------- #
    data    = pd.read_csv( inpFile, sep=r"\s+" )
    data    = data.where( data.abs() <= THRESHOLD, np.nan )
    xAxis   = data["s"]
    
    # ------------------------------------------------- #
    # --- [2] plot config                           --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [10.5,3.5],
        "figure.position"    : [ 0.10, 0.15, 0.97, 0.93 ],
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "s [m]",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 2.0,
        "legend.fontsize"    : 8.0, 
    }
    config = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [3] xyt range ( mean, min, max ) graph    --- #
    # ------------------------------------------------- #
    for key,hylabel in ylabels.items():
        config_  = {
            "figure.pngFile" : os.path.join( pngDir, "refp__s-{0}.png".format( key ) ), 
            "ax1.y.label"    : hylabel,
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=xAxis, yAxis=data[key] )
        fig.set__axis()
        fig.save__figure()


# ========================================================= #
# ===  plot__statistics.py                              === #
# ========================================================= #
def plot__statistics( inpFile=None, pngDir="png/"  ):

    ylabels_ = { "x-range"        : "x [m]"                 ,
                 "y-range"        : "y [m]"                 ,
                 "t-range"        : "t [m]"                 ,
                 "px-range"       : "px"                    ,
                 "py-range"       : "py"                    ,
                 "pt-range"       : "pt"                    ,
                 "sig_xyt"        : "RMS beam size [m]"     , 
                 "sig_pxyt"       : "Momentum"              ,
                 "alpha_xyt"      : r"$\alpha$"             ,
                 "beta_xyt"       : r"$\beta$ [m]"          ,
                 "dispersion_xy"  : r"$Dispersion$"         ,
                 "dispersion_pxy" : r"$Dispersion_p$"       ,
                 "emittance_xyt"  : r"$\epsilon$ [m rad]"   ,
                 "emittance_xytn" : r"$\epsilon_n$ [m  rad]",
                 "charge_C"       : "Charge [C]"            , 
                }
    THRESHOLD = 1.e30
    
    # ------------------------------------------------- #
    # --- [1] load .stat file                       --- #
    # ------------------------------------------------- #
    data    = pd.read_csv( inpFile, sep=r"\s+" )
    data    = data.where( data.abs() <= THRESHOLD, np.nan )
    ylabels = { key:key for key in data.keys() }
    ylabels = { **ylabels, **ylabels_ }
    
    # ------------------------------------------------- #
    # --- [2] plot config                           --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [10.5,3.5],
        "figure.position"    : [ 0.10, 0.15, 0.97, 0.93 ],
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "s [m]",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 2.0,
        "legend.fontsize"    : 8.0, 
    }
    config = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [3] xyt range ( mean, min, max ) graph    --- #
    # ------------------------------------------------- #
    for xyt in ["x","y","t", "px","py","pt"]:
        config_  = {
            "figure.pngFile" : os.path.join( pngDir, "refp__s-{0}-range.png".format( xyt ) ), 
            "ax1.y.label"    : ylabels["{}-range".format( xyt )],
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=data["s"], yAxis=data[xyt+"_mean"], label="Mean" )
        fig.add__plot( xAxis=data["s"], yAxis=data[xyt+"_min"] , label="Min"  )
        fig.add__plot( xAxis=data["s"], yAxis=data[xyt+"_max"] , label="Max"  )
        fig.set__axis()
        fig.set__legend()
        fig.save__figure()

    # ------------------------------------------------- #
    # --- [4] statistics graph                      --- #
    # ------------------------------------------------- #
    for stat in [ "sig_{}", "sig_p{}", "alpha_{}", "beta_{}", \
                  "emittance_{}", "emittance_{}n" ]:
        gname    = stat.format( "xyt" )
        config_  = {
            "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}.png".format( gname ) ), 
            "ax1.y.label"    : ylabels[gname],
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("x")], label=stat.format("x") )
        fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("y")], label=stat.format("y")  )
        fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("t")], label=stat.format("t")  )
        fig.set__axis()
        fig.set__legend()
        fig.save__figure()

    # ------------------------------------------------- #
    # --- [5] dispersion                            --- #
    # ------------------------------------------------- #
    for stat in [ "dispersion_{}", "dispersion_p{}", ]:
        gname    = stat.format( "xy" )
        config_  = {
            "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}.png".format( gname ) ), 
            "ax1.y.label"    : ylabels[gname],
        }
        config = { **config, **config_ }
        fig    = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("x")], label=stat.format("x") )
        fig.add__plot( xAxis=data["s"], yAxis=data[stat.format("y")], label=stat.format("y")  )
        fig.set__axis()
        fig.set__legend()
        fig.save__figure()

    # ------------------------------------------------- #
    # --- [5] charge C                              --- #
    # ------------------------------------------------- #
    config_  = {
        "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}.png".format( "charge_C" ) ), 
        "ax1.y.label"    : ylabels["charge_C"],
    }
    config = { **config, **config_ }
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=data["s"], yAxis=data["charge_C"] )
    fig.set__axis()
    fig.save__figure()


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
    print( "steps == ", steps )

    for ik,step in enumerate(steps):
        # -- points coordinate make -- #
        df     = Data[ Data["step"] == step ]
        df     = df[ np.isfinite(df["xp"]) & \
                     np.isfinite(df["yp"]) & \
                     np.isfinite(df["tp"]) ]
        if ( df.shape[0] == 0 ):
            print( "[impactx_toolkit.py] [WARNING] no appropriate point data :: ik={0}, step={1}".format( ik, step ) )
            continue
        coords = df[ ["xp", "yp", "tp"] ].to_numpy()
        cloud  = pv.PolyData( coords )
        # -- momentum & pid -- #
        cloud.point_data["pid"]      = df["pid"].to_numpy()
        cloud.point_data["x"]        = df["xp" ].to_numpy()
        cloud.point_data["y"]        = df["yp" ].to_numpy()
        cloud.point_data["t"]        = df["tp" ].to_numpy()
        cloud.point_data["momentum"] = df[ ["px", "py", "pz"] ].to_numpy()
    
        # -- save file -- #
        houtFile = outFile.replace( ext, "-{0:06}".format(ik+1) + ext )
        cloud.save( houtFile )
        print( "[convert__hdf2vtk.py] outFile :: {} ".format( houtFile ) )
    return()
        



# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1] load HDF5 file                        --- #
    # ------------------------------------------------- #
    inpFile = "test/bpm.h5"
    Data    = load__impactHDF5( inpFile=inpFile )
    print( Data )
    sys.exit()

    # ------------------------------------------------- #
    # --- [2] plot reference particle               --- #
    # ------------------------------------------------- #
    inpFile = "test/ref_particle.0.0"
    plot__refparticle( inpFile=inpFile )
    
    # ------------------------------------------------- #
    # --- [3] plot statistics                       --- #
    # ------------------------------------------------- #
    inpFile = "test/reduced_beam_characteristics.0.0"
    plot__statistics( inpFile=inpFile )

    # ------------------------------------------------- #
    # --- [4] convert to paraview vtk               --- #
    # ------------------------------------------------- #
    hdf5File = "test/bpm.h5"
    outFile  = "png/bpm.vtp"
    ret      = convert__hdf2vtk( hdf5File=hdf5File, outFile=outFile )
