import os, sys, json5, h5py
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.patches as patches
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1
import nk_toolkit.impactx.io_toolkit  as itk

# ========================================================= #
# ===  plot__lattice                                    === #
# ========================================================= #

def plot__lattice( latticeFile=None, ax=None, \
                   height=1.0, y0=0.0, pngFile=None, label=False ):

    qf_plot = { "color":"royalblue", "alpha":0.8 }
    qd_plot = { "color":"tomato"   , "alpha":0.8 }
    rf_plot = { "color":"orange"   , "alpha":0.8 }
    dr_plot = { "color":"grey"     , "alpha":0.8 }
    
    # ------------------------------------------------- #
    # --- [1] load lattice file                     --- #
    # ------------------------------------------------- #
    with open( latticeFile, "r" ) as f:
        elements = json5.load( f )
    lattice = pd.DataFrame.from_dict( elements, orient="index" )
    lattice = lattice[ [ "name", "type", "ds", "k" ] ]
    ds      = lattice.loc[ :,"ds" ].to_numpy()
    lattice["s_in"]  = np.cumsum( np.insert( ds, 0, 0.0 ) )[:-1]
    lattice["s_out"] = np.cumsum( ds )
    
    # ------------------------------------------------- #
    # --- [2] prepare figure / axis                 --- #
    # ------------------------------------------------- #
    if ( ax is None ):
        fig,ax   = plt.subplots( figsize=(8,2) )
        given_ax = False
    else:
        fig      = ax.figure
        given_ax = True

    # ------------------------------------------------- #
    # --- [3] draw lattice elements                 --- #
    # ------------------------------------------------- #
    for _, row in lattice.iterrows():
        s0,s1  = row["s_in"], row["s_out"]
        etype  = str( row["type"] ).lower()
        name   = str( row["name"] )
        width  = s1 - s0
        k      = row["k"]
        
        # -- [3-1] QF/QD -- #
        if   ( etype in ["quadrupole", "quadrupole.linear" ] ):
            if ( row["k"] > 0 ):
                ax.add_patch( patches.Rectangle( (s0,y0), width, height, **qf_plot ) )
            else:
                ax.add_patch( patches.Rectangle( (s0,y0-height), width, height, **qd_plot ) )
        # -- [3-2] RFcavity -- #
        elif ( etype in ["rfcavity", "rfgap" ] ):
            ax.add_patch( patches.Polygon( [ [s0,y0], [s1,y0+0.5*height], [s1,y0-0.5*height] ],
                                           closed=True, **rf_plot ) )
        # -- [3-3] drift  -- #
        elif ( etype in ["drift","drift.linear"] ):
            ax.plot( [s0,s1], [y0,y0], **dr_plot )
        # -- [3-4] その他（黒線） -- #
        else:
            ax.plot([s0, s1], [y0, y0], color="black", lw=1 )
            
        # --- ラベルを中央に配置 --- #
        if ( label ):
            ax.text( (s0+s1)/2, y0+1.1*height*np.sign(y0+0.1),
                     name, ha="center", va="bottom", fontsize=7 )

    # ------------------------------------------------- #
    # --- [4] axis settings                         --- #
    # ------------------------------------------------- #
    if ( not( given_ax ) ):
        ax.set_xlim( lattice["s_in"].min(), lattice["s_out"].max() )
        ax.set_ylim( -1.5*height, 1.5*height )
        ax.set_xlabel("s [m]")
        ax.set_yticks([])
        ax.grid( False )

        legend_handles = [
            patches.Patch( color="royalblue", label="QF"),
            patches.Patch( color="tomato"   , label="QD"),
            patches.Patch( color="orange"   , label="RFcavity"),
            plt.Line2D([0], [0], color="gray", lw=2, label="Drift")
        ]
        plt.tight_layout()
        ax.legend( handles=legend_handles, loc="upper right", ncol=4, fontsize=6 )
        
        if ( pngFile is not None ):
            ax.axis( "off" )
            ax.set_facecolor("none")
            plt.savefig( pngFile, dpi=300 )
            plt.close()
        else:
            plt.show()

    return( lattice )


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
                 "sigma_xyt"      : "RMS beam size [m]"     , 
                 "sigma_pxyt"     : "Momentum"              ,
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
        fig.add__plot( xAxis=data["s"], yAxis=data["mean_"+xyt], label="Mean" )
        fig.add__plot( xAxis=data["s"], yAxis=data["min_"+xyt] , label="Min"  )
        fig.add__plot( xAxis=data["s"], yAxis=data["max_"+xyt] , label="Max"  )
        fig.set__axis()
        fig.set__legend()
        fig.save__figure()

    # ------------------------------------------------- #
    # --- [4] statistics graph                      --- #
    # ------------------------------------------------- #
    for stat in [ "sigma_{}", "sigma_p{}", "alpha_{}", "beta_{}", \
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
# ===  plot__postprocessed                              === #
# ========================================================= #

def plot__postprocessed( postFile="impactx/diags/posts.csv", plot_conf=None ):

    # ------------------------------------------------- #
    # --- [1] load post processed / set variables   --- #
    # ------------------------------------------------- #
    data       = pd.read_csv( postFile )
    plots      = {
        "post__s-Ek_ref": [ { "xAxis":data["s"], "yAxis":data["Ek_ref"], \
                              "label":"kinetic", "ylabel":"Energy (MeV)" } ],
        "post__s-Ek"    : [ { "xAxis":data["s"], "yAxis":data["Ek_min"], \
                              "label":r"$min(E_k)$", "ylabel":"Energy (MeV)" }, \
                            { "xAxis":data["s"], "yAxis":data["Ek_avg"], \
                              "label":r"$avg(E_k)$", "ylabel":"Energy (MeV)" }, \
                            { "xAxis":data["s"], "yAxis":data["Ek_max"], \
                              "label":r"$max(E_k)$", "ylabel":"Energy (MeV)" }, \
                           ], \
        "post__s-dphi"  : [ { "xAxis":data["s"], "yAxis":data["dphi_min"], "label":"min.", },
                            { "xAxis":data["s"], "yAxis":data["dphi_avg"], "label":"avg.", },
                            { "xAxis":data["s"], "yAxis":data["dphi_max"], "label":"max.", 
                              "ylabel":r"$\Delta \phi \mathrm{(deg)}$" } ],
        "post__s-dp_p0" : [ { "xAxis":data["s"], "yAxis":data["dp/p_min"], "label":"min.", },
                            { "xAxis":data["s"], "yAxis":data["dp/p_avg"], "label":"avg.", },
                            { "xAxis":data["s"], "yAxis":data["dp/p_max"], "label":"max.", \
                              "ylabel":r"$\Delta p/p \ \mathrm{(\%)}$" } ],
        "post__s-dE_E0" : [ { "xAxis":data["s"], "yAxis":data["dE/E_min"], "label":"min.", },
                            { "xAxis":data["s"], "yAxis":data["dE/E_avg"], "label":"avg.", },
                            { "xAxis":data["s"], "yAxis":data["dE/E_max"], "label":"max.", \
                              "ylabel":r"$\Delta E/E \ \mathrm{(\%)}$" } ],
        "post__s-sigma_phi"     : [ { "xAxis":data["s"], "yAxis":data["dphi_rms"], "label":"RMS",
                                      "ylabel":r"$\sigma_{\phi} \ \mathrm{(deg)}$" } ],
        "post__s-sigma_dp_p"    : [ { "xAxis":data["s"], "yAxis":data["dp/p_rms"], "label":"RMS",
                                      "ylabel":r"$\sigma_{\Delta p/p} \ \mathrm{(\%)}$" } ],
        "post__s-sigma_dE_E0"   : [ { "xAxis":data["s"], "yAxis":data["dE/E_rms"], "label":"RMS",
                                      "ylabel":r"$\sigma_{\Delta E/E} \ \mathrm{(\%)}$" } ],
        "post__s-transmission"  : [ { "xAxis":data["s"], "yAxis":data["transmission"], \
                                      "label":"Transmission", 
                                      "ylabel":"Transmission (%)", "yRange":[ 0.0, 120.0 ] } ],
        "post__s-max_sigma_xy"  : [ { "xAxis":data["s"], "yAxis":data["max/sigma_x"], "label":"x",
                                      "ylabel":r"$max(|x|)/\sigma_x, max(|y|)/\sigma_y$"}, \
                                    { "xAxis":data["s"], "yAxis":data["max/sigma_y"], "label":"y"}],
        "post__s-max_sigma_t"   : [ { "xAxis":data["s"], "yAxis":data["max/sigma_t"], "label":"t",
                                      "ylabel":r"$max(|t|)/\sigma_t$" } ], 
        "post__s-max_sigma_dphi": [ { "xAxis":data["s"], "yAxis":data["max/sigma_dphi"],
                                      "label":r"$\Delta \phi$",
                                      "ylabel":r"$max(|\Delta \phi|)/\sigma_{\Delta \phi}$" } ],
        "post__s-corr__xp-yp"   : [ { "xAxis":data["s"], "yAxis":data["xp-yp"]*100.0, \
                                      "label":r"$<x,y>$", \
                                      "ylabel":r"$cov(x,y)/\sigma_x \sigma_y \ (\%)$" } ],
        "post__s-corr__xp-tp"   : [ { "xAxis":data["s"], "yAxis":data["xp-tp"]*100.0, \
                                      "label":r"$<x,t>$", \
                                      "ylabel":r"$cov(x,t)/\sigma_x \sigma_t \ (\%)$" } ],
        "post__s-corr__xp-px"   : [ { "xAxis":data["s"], "yAxis":data["xp-px"]*100.0, \
                                      "label":r"$<x,p_x>$", \
                                      "ylabel":r"$cov(x,p_x)/\sigma_x \sigma_{p_x}$ \ (\%)" } ],
        "post__s-corr__xp-py"   : [ { "xAxis":data["s"], "yAxis":data["xp-py"]*100.0, \
                                      "label":r"$<x,p_y>$", \
                                      "ylabel":r"$cov(x,p_y)/\sigma_x \sigma_{p_y}$ \ (\%)" } ],
        "post__s-corr__xp-pt"   : [ { "xAxis":data["s"], "yAxis":data["xp-pt"]*100.0, \
                                      "label":r"$<x,p_t>$", \
                                      "ylabel":r"$cov(x,p_t)/\sigma_x \sigma_{p_t}$ \ (\%)" } ],
        "post__s-corr__tp-pt"   : [ { "xAxis":data["s"], "yAxis":data["tp-pt"]*100.0, \
                                      "label":r"$<t,p_t>$", \
                                      "ylabel":r"$cov(t,p_t)/\sigma_t \sigma_{p_t}$ \ (\%)" } ],

    }
    xmin, xmax = np.min( data["s"] ), np.max( data["s"] )
    
    # ------------------------------------------------- #
    # --- [2] plot config                           --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [9.0,3.0],
        "figure.position"    : [ 0.10, 0.15, 0.97, 0.93 ],
        "ax1.x.range"        : { "auto":True, "min": xmin, "max":xmax, "num":8 },
        "ax1.y.range"        : { "auto":True, "min":  0.0, "max": 1.0, "num":6 },
        "ax1.x.label"        : "s (m)",
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
        for content in contents:
            fig.add__plot( **content )
        fig.set__legend()
        fig.save__figure()


# ========================================================= #
# ===  plot__poincare                                   === #
# ========================================================= #

def plot__poincare( bpms=None, step=None ):

    mm, mrad  = 1.0e-3, 1.0e-3
    cv, nsec  = 2.99792458e8, 1.0e-9
    
    if ( bpms is None ): sys.exit( "[plot_poincare] bpms == ???" )
    if ( step is None ): step = sorted( list( set(bpms["step"]) ) )[0]
    if ( type(bpms) is str ):
        bpms = h5py.File( bpms, "r" )

    # ------------------------------------------------- #
    # --- [1] prepare variables                     --- #
    # ------------------------------------------------- #
    if ( type(bpms) is pd.DataFrame ):
        bpms_        = bpms.loc[ bpms["step"] == step ].copy()
        bpms_["xp"]  = bpms_["xp"] / mm
        bpms_["yp"]  = bpms_["yp"] / mm
        bpms_["tp"]  = bpms_["tp"] / mm
        bpms_["px"]  = bpms_["px"] / mrad
        bpms_["py"]  = bpms_["py"] / mrad
        bpms_["pt"]  = bpms_["pt"] / mrad
        bpms_["dt"]  = bpms_["dt"] / nsec  # [ns]
        bpms_["dEk"] = bpms_["dEk"]        # [MeV]

    axpMax = np.max( [ np.abs( bpms_["xp"].min() ), np.abs( bpms_["xp"].max() ) ] )
    aypMax = np.max( [ np.abs( bpms_["yp"].min() ), np.abs( bpms_["yp"].max() ) ] )
    atpMax = np.max( [ np.abs( bpms_["tp"].min() ), np.abs( bpms_["tp"].max() ) ] )
    apxMax = np.max( [ np.abs( bpms_["px"].min() ), np.abs( bpms_["px"].max() ) ] )
    apyMax = np.max( [ np.abs( bpms_["py"].min() ), np.abs( bpms_["py"].max() ) ] )
    aptMax = np.max( [ np.abs( bpms_["pt"].min() ), np.abs( bpms_["pt"].max() ) ] )
    adtMax = np.max( [ np.abs( bpms_["dt"].min() ), np.abs( bpms_["dt"].max() ) ] )
    adEMax = np.max( [ np.abs( bpms_["dEk"].min() ), np.abs( bpms_["dEk"].max() ) ] )
    
    # ------------------------------------------------- #
    # --- [2] plot config                           --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    def_conf = {
        "figure.size"        : [4.5,4.5],
        "figure.position"    : [ 0.18, 0.18, 0.92, 0.92 ],
        "ax1.x.minor.nticks" : 1, 
        "plot.linestyle"     : "none", 
        "plot.marker"        : "o",
        "plot.markersize"    : 0.2,
        "legend.fontsize"    : 9.0, 
    }
    def_conf = { **config, **def_conf }

    config_ = {
        "xp-px" : {
            "figure.pngFile"     : "png/poincare__xp-px_step{:06}.png".format( step ), 
            "ax1.x.range"        : { "auto":False, "min": -axpMax, "max":axpMax, "num":11 },
            "ax1.y.range"        : { "auto":False, "min": -apxMax, "max":apxMax, "num":11 },
            "ax1.x.label"        : "x (mm)",
            "ax1.y.label"        : "x' (mrad)",
        },
        "yp-py" : {
            "figure.pngFile"     : "png/poincare__yp-py_step{:06}.png".format( step ), 
            "ax1.x.range"        : { "auto":False, "min": -aypMax, "max":+aypMax, "num":11 },
            "ax1.y.range"        : { "auto":False, "min": -apyMax, "max":+apyMax, "num":11 },
            "ax1.x.label"        : "y (mm)",
            "ax1.y.label"        : "y' (mrad)",
        },
        "tp-pt" : {
            "figure.pngFile"     : "png/poincare__tp-pt_step{:06}.png".format( step ), 
            "ax1.x.range"        : { "auto":False, "min": -atpMax, "max":+atpMax, "num":11 },
            "ax1.y.range"        : { "auto":False, "min": -aptMax, "max":+aptMax, "num":11 },
            "ax1.x.label"        : "t (mm)",
            "ax1.y.label"        : "t' (mrad)",
        },
        "dt-dE" : {
            "figure.pngFile"     : "png/poincare__dt-dE_step{:06}.png".format( step ), 
            "ax1.x.range"        : { "auto":False, "min": -adtMax, "max":+adtMax, "num":11 },
            "ax1.y.range"        : { "auto":False, "min": -adEMax, "max":+adEMax, "num":11 },
            "ax1.x.label"        : "dt (ns)",
            "ax1.y.label"        : "dE (MeV)",
        },
    }
    # ------------------------------------------------- #
    # --- [3] plots settings                        --- #
    # ------------------------------------------------- #
    plots      = {
        "xp-px": [ { "xAxis":bpms_["xp"], "yAxis":bpms_["px"] , "density":True }, ], \
        "yp-py": [ { "xAxis":bpms_["yp"], "yAxis":bpms_["py"] , "density":True }, ], \
        "tp-pt": [ { "xAxis":bpms_["tp"], "yAxis":bpms_["pt"] , "density":True }, ], \
        "dt-dE": [ { "xAxis":bpms_["dt"], "yAxis":bpms_["dEk"], "density":True }, ], \
    }
    
    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    for key,contents in plots.items():
        config = { **def_conf, **config_[key] }
        fig = gp1.gplot1D( config=config )
        for content in contents:
            fig.add__scatter( **content )
        fig.set__axis()
        fig.save__figure()


# ========================================================= #
# ===  plot__stats                                      === #
# ========================================================= #

def plot__stats( stat=None, plot_conf=None ):

    mm, mrad = 1.e-3, 1.e-3
    
    # ------------------------------------------------- #
    # --- [1] load post processed / set variables   --- #
    # ------------------------------------------------- #
    plots      = {
        "stat__s-xRange" : [ { "xAxis":stat["s"], "yAxis":stat["min_x"]/mm , "label":"min(x)", \
                               "ylabel":r"$x\ \mathrm{(mm)}$" },
                             { "xAxis":stat["s"], "yAxis":stat["mean_x"]/mm, "label":"mean(x)" },
                             { "xAxis":stat["s"], "yAxis":stat["max_x"]/mm, "label":"max(x)" }, 
                            ],
        "stat__s-yRange" : [ { "xAxis":stat["s"], "yAxis":stat["min_y"]/mm , "label":"min(y)", \
                               "ylabel":r"$y\ \mathrm{(mm)}$" },
                             { "xAxis":stat["s"], "yAxis":stat["max_y"]/mm, "label":"max(y)" }, 
                             { "xAxis":stat["s"], "yAxis":stat["mean_y"]/mm, "label":"mean(y)" },
                            ],
        "stat__s-tRange" : [ { "xAxis":stat["s"], "yAxis":stat["min_t"]/mm , "label":"min(t)", \
                               "ylabel":r"$t\ \mathrm{(mm)}$" },
                             { "xAxis":stat["s"], "yAxis":stat["max_t"]/mm, "label":"max(t)" }, 
                             { "xAxis":stat["s"], "yAxis":stat["mean_t"]/mm, "label":"mean(t)" },
                            ], 
        "stat__s-sigma_xy": [ { "xAxis":stat["s"], "yAxis":stat["sigma_x"]/mm , "label":r"$\sigma_x$",\
                                "ylabel":r"$\sigma \ \mathrm{(mm)}$" },
                              { "xAxis":stat["s"], "yAxis":stat["sigma_y"]/mm , "label":r"$\sigma_y$",}
                             ], 
        "stat__s-sigma_t" : [ { "xAxis":stat["s"], "yAxis":stat["sigma_t"]/mm , "label":r"$\sigma_t$",\
                               "ylabel":r"$\sigma \ \mathrm{(mm)}$" },
                            ], 
        "stat__s-emit_xyt": [ { "xAxis":stat["s"], "yAxis":stat["emittance_x"]/mm/mrad , \
                                "label":r"$\epsilon_x$",\
                                "ylabel":r"$\epsilon \ \mathrm{(mm \ mrad)}$" },
                              { "xAxis":stat["s"], "yAxis":stat["emittance_y"]/mm/mrad , \
                                "label":r"$\epsilon_y$", }, 
                              { "xAxis":stat["s"], "yAxis":stat["emittance_t"]/mm/mrad , \
                                "label":r"$\epsilon_t$", }, 
                            ],
        "stat__s-emit_xytn": [ { "xAxis":stat["s"], "yAxis":stat["emittance_xn"]/mm/mrad , \
                                 "label":r"$\epsilon_x^{norm}$",\
                                 "ylabel":r"$\epsilon^{norm} \ \mathrm{(mm \ mrad)}$" },          
                               { "xAxis":stat["s"], "yAxis":stat["emittance_yn"]/mm/mrad , \
                                 "label":r"$\epsilon_y^{norm}$", }, 
                               { "xAxis":stat["s"], "yAxis":stat["emittance_tn"]/mm/mrad , \
                                 "label":r"$\epsilon_t^{norm}$", }
                              ],
        "stat__s-beta_xy": [ { "xAxis":stat["s"], "yAxis":stat["beta_x"], \
                               "label":r"$\beta_x$",\
                               "ylabel":r"$\beta \ \mathrm{(m)}$" },
                             { "xAxis":stat["s"], "yAxis":stat["beta_y"], \
                               "label":r"$\beta_y$", }
                            ],
        "stat__s-beta_t"  : [ { "xAxis":stat["s"], "yAxis":stat["beta_t"], \
                                "label":r"$\beta_t$",\
                                "ylabel":r"$\beta \ \mathrm{(m)}$" },
                             ],
        "stat__s-alpha_xyt": [ { "xAxis":stat["s"], "yAxis":stat["alpha_x"], \
                                 "label":r"$\alpha_x$",\
                                 "ylabel":r"$\alpha$" },
                               { "xAxis":stat["s"], "yAxis":stat["alpha_y"], \
                                 "label":r"$\alpha_y$", },
                               { "xAxis":stat["s"], "yAxis":stat["alpha_t"], \
                                 "label":r"$\alpha_t$", }, 
                              ],
    }
    
    
    # ------------------------------------------------- #
    # --- [2] plot config                           --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [9.0,3.0],
        "figure.position"    : [ 0.11, 0.16, 0.96, 0.92 ],
        "ax1.x.range"        : { "auto":True, "min":  0.0, "max": 1.0, "num":8 },
        "ax1.y.range"        : { "auto":True, "min":  0.0, "max": 1.0, "num":6 },
        "ax1.x.label"        : "s (m)",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 2.0,
        "legend.fontsize"    : 8.0, 
    }
    config = { **config, **config_ }
    if ( plot_conf is not None ):
        if ( type( plot_conf ) is str ):
            with open( plot_conf, "r" ) as f:
                plot_conf = json5.load( f )

    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    for key,contents in plots.items():
        if ( plot_conf is not None ):
            config = { **config, **plot_conf[key]["config"] }
        fig = gp1.gplot1D( config=config, pngFile="png/{}.png".format( key ) )
        for content in contents:
            fig.add__plot( **content )
        fig.set__legend()
        fig.save__figure()



# # ========================================================= #
# # ===  plot__poincare2                                  === #
# # ========================================================= #

# def plot__poincare2( bpmsFile="impactx/diags/openPMD/bpm.h5", step=0 ):

#     mm, mrad  = 1.0e-3, 1.0e-3
#     cv, nsec  = 2.99792458e8, 1.0e-9

#     # ------------------------------------------------- #
#     # --- [1] plot config                           --- #
#     # ------------------------------------------------- #
#     config   = lcf.load__config()
#     def_conf = {
#         "figure.size"        : [4.5,4.5],
#         "figure.position"    : [ 0.18, 0.18, 0.92, 0.92 ],
#         "ax1.x.minor.nticks" : 1, 
#         "plot.linestyle"     : "none", 
#         "plot.marker"        : "o",
#         "plot.markersize"    : 0.2,
#         "legend.fontsize"    : 9.0, 
#     }
#     def_conf = { **config, **def_conf }

#     config_ = {
#         "xp-px" : {
#             "figure.pngFile"     : "png/poincare__xp-px_step{:06}.png".format( step ), 
#             "ax1.x.range"        : { "auto":False, "min": -axpMax, "max":axpMax, "num":11 },
#             "ax1.y.range"        : { "auto":False, "min": -apxMax, "max":apxMax, "num":11 },
#             "ax1.x.label"        : "x (mm)",
#             "ax1.y.label"        : "x' (mrad)",
#         },
#         "yp-py" : {
#             "figure.pngFile"     : "png/poincare__yp-py_step{:06}.png".format( step ), 
#             "ax1.x.range"        : { "auto":False, "min": -aypMax, "max":+aypMax, "num":11 },
#             "ax1.y.range"        : { "auto":False, "min": -apyMax, "max":+apyMax, "num":11 },
#             "ax1.x.label"        : "y (mm)",
#             "ax1.y.label"        : "y' (mrad)",
#         },
#         "tp-pt" : {
#             "figure.pngFile"     : "png/poincare__tp-pt_step{:06}.png".format( step ), 
#             "ax1.x.range"        : { "auto":False, "min": -atpMax, "max":+atpMax, "num":11 },
#             "ax1.y.range"        : { "auto":False, "min": -aptMax, "max":+aptMax, "num":11 },
#             "ax1.x.label"        : "t (mm)",
#             "ax1.y.label"        : "t' (mrad)",
#         },
#     }
    
#     # ------------------------------------------------- #
#     # --- [2] plot poinncare                        --- #
#     # ------------------------------------------------- #
#     with h5py.File( bpmsFile, "r" ) as bpms:
#         for chunk in itk.load__bpms( bpms=bpms, step=step ):
#             chunk["xp"] = chunk["xp"] / mm
#             chunk["yp"] = chunk["yp"] / mm
#             chunk["tp"] = chunk["tp"] / mm
#             chunk["px"] = chunk["px"] / mrad
#             chunk["py"] = chunk["py"] / mrad
#             chunk["pt"] = chunk["pt"] / mrad
#             axpMax = np.max( [ np.abs( np.min( chunk["xp"] ) ), np.abs( np.max( chunk["xp"] ) ) ] )
#             aypMax = np.max( [ np.abs( np.min( chunk["yp"] ) ), np.abs( np.max( chunk["yp"] ) ) ] )
#             atpMax = np.max( [ np.abs( np.min( chunk["tp"] ) ), np.abs( np.max( chunk["tp"] ) ) ] )
#             apxMax = np.max( [ np.abs( np.min( chunk["px"] ) ), np.abs( np.max( chunk["px"] ) ) ] )
#             apyMax = np.max( [ np.abs( np.min( chunk["py"] ) ), np.abs( np.max( chunk["py"] ) ) ] )
#             aptMax = np.max( [ np.abs( np.min( chunk["pt"] ) ), np.abs( np.max( chunk["pt"] ) ) ] )

#         for key,contents in plots.items():
#             if ( plot_conf is not None ):
#                 config = { **config, **plot_conf[key]["config"] }
#         fig = gp1.gplot1D( config=config, pngFile="png/{}.png".format( key ) )
#         for content in contents:
#             fig.add__plot( **content )
#         fig.set__legend()
#         fig.save__figure()

#         for key 
#         config = { **def_conf, **config_[key] }
#         fig = gp1.gplot1D( config=config )
#         for content in contents:
#             fig.add__scatter( **content )
#         fig.set__axis()
#         fig.save__figure()

