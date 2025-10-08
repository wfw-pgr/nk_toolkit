import os, sys, tqdm, glob, json5
import h5py
import numpy   as np
import pandas  as pd
import pyvista as pv
import scipy   as sp
import matplotlib.pyplot               as plt
import nk_toolkit.plot.load__config    as lcf
import nk_toolkit.plot.gplot1D         as gp1
import nk_toolkit.math.fourier_toolkit as ftk


# ========================================================= #
# === impactx_toolkit.py                                === #
# ========================================================= #
#
#  * load__impactHDF5
#  * plot__refparticle
#  * plot__statistics
#  * plot__trajectory
#  * convert__hdf2vtk
#  * compute__fourierCoefficients
#  * adjust__RFcavityPhase
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
            try:
                key, df    = str(step), {}
                df["pid"]  = f["data"][key]["particles"]["beam"]["id"][:]
                df["xp"]   = f["data"][key]["particles"]["beam"]["position"]["x"][:] 
                df["yp"]   = f["data"][key]["particles"]["beam"]["position"]["y"][:] 
                df["tp"]   = f["data"][key]["particles"]["beam"]["position"]["t"][:] 
                df["px"]   = f["data"][key]["particles"]["beam"]["momentum"]["x"][:] 
                df["py"]   = f["data"][key]["particles"]["beam"]["momentum"]["y"][:] 
                df["pt"]   = f["data"][key]["particles"]["beam"]["momentum"]["t"][:] 
                df["step"] = np.full( df["pid"].shape, step, dtype=int )
                stack     += [ pd.DataFrame( df ) ]
            except TypeError:
                print( "[load__impactHDF5.py] detected TypeError at step == {}.. continue. ".format( step ) )
                
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
def plot__refparticle( inpFile=None, pngDir="png/", plot_conf=None  ):

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
    data       = pd.read_csv( inpFile, sep=r"\s+" )
    data       = data.where( data.abs() <= THRESHOLD, np.nan )
    xAxis      = data["s"]
    
    # ------------------------------------------------- #
    # --- [2] plot config                           --- #
    # ------------------------------------------------- #
    config  = lcf.load__config()
    config_ = {
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
    if ( plot_conf is not None ):
        config = { **config, **plot_conf }        

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
def plot__statistics( inpFile=None, pngDir="png/", plot_conf=None  ):

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

    nparticle_threshold = 0.10
    if ( nparticle_threshold ):
        fraction = ( data["charge_C"] ) / ( data["charge_C"][0] )
        data     = data.where( fraction >= nparticle_threshold, np.nan )
    
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
    if ( plot_conf is not None ):
        config = { **config, **plot_conf }        


    # ------------------------------------------------- #
    # --- [3] xyt range ( mean, min, max ) graph    --- #
    # ------------------------------------------------- #
    for xyt in ["x","y","t", "px","py","pt"]:
        config_  = {
            "figure.pngFile" : os.path.join( pngDir, "stat__s-{0}-range.png".format( xyt ) ), 
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
# ===  plot__trajectories.py                            === #
# ========================================================= #

def plot__trajectories( hdf5File=None, refpFile=None, pids=None, random_choice=None, \
                        cmap="plasma", nColors=128, pngDir="png/", plot_conf=None ):

    ylabels  = { "xp":r"$x$ [mm]"  , "yp":r"$y$ [mm]"  , "tp":r"$t$ [mm]", \
                 "px":r"$p_x$ [mrad]", "py":r"$p_y$ [mrad]", "pz":r"$p_z$ [mrad]" }
    
    # ------------------------------------------------- #
    # --- [1] load HDF5 file                        --- #
    # ------------------------------------------------- #
    pinfo    = load__impactHDF5( inpFile=hdf5File, \
                                 pids=pids, random_choice=random_choice )
    rinfo    = pd.read_csv( refpFile, sep=r"\s+"  )
    ref_s    = rinfo["s"]
    colors   = plt.get_cmap( cmap, nColors )
    if ( pids is None ):
        pids = np.array( pinfo["pid"].unique(), dtype=np.int64 )
    
    # ------------------------------------------------- #
    # --- [2] configuration                         --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [10.5,3.5],
        "figure.position"    : [ 0.10, 0.12, 0.97, 0.95 ],
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "s [m]",
        "plot.marker"        : "none",
        "plot.linestyle"     : "-", 
        "plot.markersize"    : 0.2,
    }
    config   = { **config, **config_ }
    if ( plot_conf is not None ):
        config = { **config, **plot_conf }        

    pngFile  = os.path.join( pngDir, "traj__s-{}.png" )

    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    for obj in [ "xp","yp","tp","px","py","pz" ]:
        fig = gp1.gplot1D( config=config, pngFile=pngFile.format( obj ) )
        config["ax1.y.label"] = ylabels[obj]
        for ik,pid in enumerate(tqdm.tqdm(pids)):
            traj = ( pinfo[ pinfo["pid"] == pid ] )[obj].values
            if ( len(traj) == 0 ): continue
            xAxis  = ref_s[:(len(traj))]
            hcolor = colors( ik%nColors )
            fig.add__plot( xAxis=xAxis, yAxis=traj, color=hcolor )
        fig.set__axis()
        fig.save__figure()
    return()


# ========================================================= #
# ===  plot__postProcessed                              === #
# ========================================================= #

def plot__postProcessed( paramsFile="dat/parameters.json", \
                         postFile="impactx/diags/postProcessed_beam.csv", plot_conf=None ):

    # ------------------------------------------------- #
    # --- [1] load post processed / set variables   --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    data       = pd.read_csv( postFile )
    plots      = {
        "post__s-Ek":{ "xAxis":data["s_refp"], "yAxis":data["Ek"], "ylabel":"Energy (MeV)" },
    }
    xmin, xmax = np.min( data["s_refp"] ), np.max( data["s_refp"] )
    
    # ------------------------------------------------- #
    # --- [2] plot config                           --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [10.5,3.5],
        "figure.position"    : [ 0.10, 0.15, 0.97, 0.93 ],
        "ax1.x.range"        : { "auto":True, "min": xmin, "max":xmax, "num":11 },
        "ax1.y.range"        : { "auto":True, "min":  0.0, "max": 1.0, "num":11 },
        "ax1.x.label"        : "s [m]",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 2.0,
        "legend.fontsize"    : 8.0, 
    }
    config = { **config, **config_ }
    if ( plot_conf is not None ):
        config = { **config, **plot_conf }

    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    for key,contents in plots.items():
        fig = gp1.gplot1D( config=config, pngFile="png/{}.png".format( key ) )
        fig.add__plot( **contents )
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
    freq               = params["beam.freq"] # * params["beam.harmonics"]
    data["s_stat"]     = stat["s"]
    data["phase_min"]  = stat["t_min" ] / cv * freq * 180.0
    data["phase_avg"]  = stat["t_mean"] / cv * freq * 180.0
    data["phase_max"]  = stat["t_max" ] / cv * freq * 180.0
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

    # ------------------------------------------------- #
    # --- [6] compute  fourier coeffs               --- #
    # ------------------------------------------------- #
    coefFile = "test/fourier_expansion.dat"
    pngFile  = "test/fourier_expansion.png"
    xp       = np.linspace( 0.0, 1.0, 101 )
    fx       = np.sin( xp*np.pi )**2.0
    nMode    = None
    compute__fourierCoefficients( xp=xp, fx=fx, nMode=nMode, \
                                  pngFile=pngFile, coefFile=coefFile )

    # ------------------------------------------------- #
    # --- [7] adjust phase shift                    --- #
    # ------------------------------------------------- #
    adjust__RFcavityPhase( paramsFile="test/beamline.json", \
                           outFile="test/adjust__RFcavityPhase.dat" )
    
