import os, sys, json5
import pandas as pd
import numpy  as np
import nk_toolkit.madx.load__tfs as ltf
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1


# ========================================================= #
# ===  postProcess__track.py                            === #
# ========================================================= #

def postProcess__track( paramsFile="dat/parameters.json" ):
    """evaluate twiss parameters from track results...."""

    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    
    # ------------------------------------------------- #
    # --- [1] calculate                             --- #
    # ------------------------------------------------- #
    twiss  = calculate__twiss ( paramsFile=paramsFile )
    energy = calculate__energy( paramsFile=paramsFile )
    
    # ------------------------------------------------- #
    # --- [2] save in csv and return                --- #
    # ------------------------------------------------- #
    twiss.to_csv( params["post.track.outFile"], index=False )
    print( "[postProcess__track.py]  output :: {}".format( params["post.track.outFile"] ) )
    return( twiss )


# ========================================================= #
# ===  calculate__twiss.py                              === #
# ========================================================= #

def calculate__twiss( paramsFile="dat/parameters.json" ):
    """evaluate twiss parameters from track results...."""
    
    # ------------------------------------------------- #
    # --- [1] load files                            --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    df = ltf.load__tfs( tfsFile=params["post.track.inpFile"] )["df"]
        
    # ------------------------------------------------- #
    # --- [2] make position list                    --- #
    # ------------------------------------------------- #
    df["S"] = df["S"].apply( lambda x: significant_digit( x, digit=params["post.digit"] ) )
    sarr    = sorted( list( set( df["S"] ) ) )
    
    # ------------------------------------------------- #
    # --- [3] evaluation                            --- #
    # ------------------------------------------------- #
    stack   = []
    for ik,sv in enumerate( sarr ):
        sgroup  = df[ df["S"] == sv ]
        xp      = sgroup["X"] .values
        px      = sgroup["PX"].values
        yp      = sgroup["Y"] .values
        py      = sgroup["PY"].values
        xpxcov  = np.cov( xp,px )
        xp2avg  = xpxcov[0,0]
        px2avg  = xpxcov[1,1]
        xpxavg  = xpxcov[0,1]
        ypycov  = np.cov( yp,py )
        yp2avg  = ypycov[0,0]
        py2avg  = ypycov[1,1]
        ypyavg  = ypycov[0,1]
        emit_x  = np.sqrt( xp2avg * px2avg - xpxavg**2 )
        emit_y  = np.sqrt( yp2avg * py2avg - ypyavg**2 )
        beta_x  =          xp2avg / emit_x
        beta_y  =          yp2avg / emit_y
        gamma_x =          px2avg / emit_x
        gamma_y =          py2avg / emit_y
        alpha_x = (-1.0) * xpxavg / emit_x
        alpha_y = (-1.0) * ypyavg / emit_y
        stack += [ np.array( [ sv,\
                               emit_x, beta_x, alpha_x, gamma_x,\
                               emit_y, beta_y, alpha_y, gamma_y ] ) [np.newaxis,:] ]
    stack   = np.concatenate( stack, axis=0 )
    columns = [ "S", \
                "emit_x", "beta_x", "alpha_x", "gamma_x", \
                "emit_y", "beta_y", "alpha_y", "gamma_y" ]
    twiss   = pd.DataFrame( stack, columns=columns )
    return( twiss )


# ========================================================= #
# ===  significant-digit                                === #
# ========================================================= #
def significant_digit( x, digit=8 ):
    "return significant digit of x"
    if   ( pd.isna(x) ):
        return x
    elif ( x == 0     ):
        return 0
    else:
        return round( x, digit-int( np.floor(np.log10(abs(x)))) - 1 )


# ========================================================= #
# ===  calculate__energy.py                             === #
# ========================================================= #
def calculate__energy( paramsFile="dat/parameters.json" ):

    MeV2GeV  = 1.e6 / 1.e9
    
    # ------------------------------------------------- #
    # --- [1] load files                            --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    track    = ( ltf.load__tfs( tfsFile=params["post.track.inpFile"] ) )["df"]
    track    = track[ track["NUMBER"]==1 ]
    track    = track.fillna( 0.0 )

    # ------------------------------------------------- #
    # --- [2] calculate                             --- #
    # ------------------------------------------------- #
    sL       = track["S"].values                           # [m]
    E0ref    = params["umass"] * params["mass/u"]          # [MeV]
    Ekref    = params["init.Ek"]                           # [MeV]
    pcref    = np.sqrt( Ekref**2 + 2.0*Ekref*E0ref )       # [MeV]
    dE       = track["PT"].values * pcref                  # [MeV]
    Ek       = Ekref + dE
    
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [4.5,4.5],
        "figure.pngFile"     : "png/s_Ekinetic.png", 
        "figure.position"    : [ 0.18, 0.18, 0.94, 0.94 ],
        "ax1.y.normalize"    : 1.0e0, 
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":80.0, "num":9 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":100.0, "num":11 },
        "ax1.x.label"        : r"$s$  (m)",
        "ax1.y.label"        : r"$E_k$ (MeV)",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 3.0,
        "legend.fontsize"    : 9.0, 
    }
    config = { **config, **config_ }
    
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=sL, yAxis=Ek, label="kinetic [MeV]" )
    fig.set__axis()
    fig.save__figure()

    
    
# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    
    inpFile = "madx/out/track.tfsone"
    outFile = "dat/track.csv"
    track   = postProcess__track( inpFile=inpFile, outFile=outFile )
