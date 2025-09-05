import sys
import numpy as np
import matplotlib.pyplot as plt
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1

# ========================================================= #
# ===  get__fourierCoeffs.py                            === #
# ========================================================= #

def get__fourierCoeffs( xp, fx, nMode=None, resample=None, tolerance=None, window_function=None ):

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    xp = np.asarray( xp )
    fx = np.asarray( fx )
    if ( xp.ndim != 1 ) or ( fx.ndim != 1 ) or ( xp.shape[0] != fx.shape[0] ):
        raise ValueError("xp and fx must be 1D arrays of same length.")

    # ------------------------------------------------- #
    # --- [2] L and resampling                      --- #
    # ------------------------------------------------- #
    x0 = float( xp[ 0] )
    xN = float( xp[-1] )
    L  = xN - x0
    if ( resample is not None ):
        Nx  = resample
    else:
        Nx  = xp.shape[0]
    xpu = x0 + np.linspace( 0.0, L, Nx, endpoint=False )
    fxu = np.interp( xpu, xp, fx )
    if ( L <= 0 ):
        raise ValueError( " [ERROR] L < 0 ( xp[-1] > xp[0] )" )

    # ------------------------------------------------- #
    # --- [3] window function                       --- #
    # ------------------------------------------------- #
    if ( window_function is not None ):
        if ( window_function == "Hann" ):
            wf  = 0.5 - 0.5*np.cos( (xpu-x0)/L* 2.0*np.pi )
            fxu = wf * fxu
        else:
            sys.exit( " window_function == ?? " )

    # ------------------------------------------------- #
    # --- [3] execute fft                           --- #
    # ------------------------------------------------- #
    Fx      = np.fft.rfft( fxu )
    Fx      = Fx / float(Nx)
    modeMax = Fx.shape[0] - 1
    if ( nMode is None ):
        nMode = modeMax + 1
    
    # ------------------------------------------------- #
    # --- [4] store coefficients                    --- #
    # ------------------------------------------------- #
    aj, bj  = np.zeros( nMode, dtype=float ), np.zeros( nMode, dtype=float )
    aj[0]   = float( Fx[0].real )
    for ik in range( 1, nMode ):
        if   ( ik  < modeMax ):
            aj[ik] =  2.0 * Fx[ik].real
            bj[ik] = -2.0 * Fx[ik].imag
        elif ( ik == modeMax ):
            aj[ik] =  Fx[ik].real
            bj[ik] =  0.0
        else:
            aj[ik] =  0.0
            bj[ik] =  0.0
    if ( tolerance is not None ):
        aj[ np.abs(aj) < tolerance ] = 0.0
        bj[ np.abs(bj) < tolerance ] = 0.0

    # ------------------------------------------------- #
    # --- [5] reconstruction                        --- #
    # ------------------------------------------------- #
    reconstruction = reconstruct__fromFourierCoeffs( aj, bj, period=L, x0=x0 )
    
    ret = { "cos":aj, "sin":bj, "reconstruction":reconstruction }
    return( ret )


# ========================================================= #
# === reconstruct__fromFourierCoeffs                    === #
# ========================================================= #
def reconstruct__fromFourierCoeffs( aj, bj, period=1.0, x0=0.0 ):

    aj = np.asarray( aj, dtype=float )
    bj = np.asarray( bj, dtype=float )
    if (aj.shape != bj.shape ):
        raise ValueError( "[reconstruct__fromFourierCoeffs] aj and bj must have same length ")
    if ( period <= 0.0 ):
        raise ValueError( "[reconstruct__fromFourierCoeffs] period > 0.0")

    def fourierExpanded( xn ):
        xn_arr = np.asarray( xn )
        x_norm = 2.0*np.pi / float(period) * ( xn_arr - x0 )
        ret    = np.full( xn_arr.shape, aj[0], dtype=float )
        for jk in range( 1, aj.size ):
            ret += aj[jk] * np.cos(jk*x_norm) + bj[jk] * np.sin(jk*x_norm)
        return( ret )

    return( fourierExpanded )


# ========================================================= #
# === display__fourierExpansion                         === #
# ========================================================= #
def display__fourierExpansion( xp, fx, func=None, pngFile="reconst.png" ):

    # ------------------------------------------------- #
    # --- [1] fourier expansion                     --- #
    # ------------------------------------------------- #
    if ( func is None ):
        func  = ( get__fourierCoeffs( xp=xp, fx=fx ) )["reconstruction"]
    rec  = func( xp )

    # ------------------------------------------------- #
    # --- [2] config                                --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [4.5,4.5],
        "figure.position"    : [ 0.16, 0.16, 0.94, 0.94 ],
        "figure.pngFile"     : pngFile, 
        "ax1.y.normalize"    : 1.0e0,
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "x",
        "ax1.y.label"        : "f(x)",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 3.0,
        "legend.fontsize"    : 9.0, 
    }
    config = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [3] display                               --- #
    # ------------------------------------------------- #
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=xp, yAxis=fx , label="original"      , marker="o"   , linestyle="none" )
    fig.add__plot( xAxis=xp, yAxis=rec, label="reconstruction", marker="none", linestyle="-"    )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1] coeff check                           --- #
    # ------------------------------------------------- #
    xp  = np.linspace( 0.0, 2.0*np.pi, 100 )
    fx  = 0.1*np.sin( xp*2.0 ) + 0.3*np.cos( xp*3.0 )
    tol = 1.e-10
    wf  = None
    ret = get__fourierCoeffs( xp=xp, fx=fx, nMode=10, resample=100, \
                              tolerance=tol, window_function=wf )
    print()
    print( "     :: " + " ".join( [ "{:<8}"  .format(val) for val in range( len( ret["cos"] ))]))
    print( "cos  :: " + " ".join( [ "{:8.2e}".format(val) for val in ret["cos"] ] ) )
    print( "sin  :: " + " ".join( [ "{:8.2e}".format(val) for val in ret["sin"] ] ) )
    print()
    
    # ------------------------------------------------- #
    # --- [2] reconstruction                        --- #
    # ------------------------------------------------- #
    xp   = np.linspace( 0.0, 1.0, 101 )
    fx   = ( np.sin( xp * np.pi ) )**2
    ret  = display__fourierExpansion( xp=xp, fx=fx )

    
    
    
