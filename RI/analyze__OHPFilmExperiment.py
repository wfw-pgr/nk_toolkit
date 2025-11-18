import os, sys, cv2
import numpy as np
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1
import nk_toolkit.plot.gplot2D        as gp2


norm__16bit   = 65535.0
xfilmsize_px  = 300          # (pixel)
yfilmsize_px  = 300          # (pixel)

# ========================================================= #
# ===  analyze__OHPFilmExperiment                       === #
# ========================================================= #

def analyze__OHPFilmExperiment( inpFile=None, Lx=15.0, Ly=15.0, method="gauss-fit-projection" ):
    
    edges   = detect__OHPsheetEdge( inpFile=inpFile, \
                                    bbcheckFile="png/image_bbcheck.png",
                                    croppedFile="png/image_cropped.png" )
    shift   = measure__beamshift  ( image=edges["image_cropped"], Lx=Lx, Ly=Ly, \
                                    method=method )
    plot1   = plot__fittedGaussian( xAxis=shift["xAxis"], xproj=shift["proj_x"], \
                                    xdiff=shift["xdiff"], xpeak=shift["xpeak"], \
                                    xfit =shift["xfit"], \
                                    yAxis=shift["yAxis"], yproj=shift["proj_y"], \
                                    ydiff=shift["ydiff"], ypeak=shift["ypeak"],\
                                    yfit =shift["yfit"], \
                                    Lx=Lx, Ly=Ly )
    plot2   = plot__images( bbcheck=edges["image_bbcheck"], cropped=edges["image_cropped"],\
                            Lx=Lx, Ly=Ly )
    return()
    

# ========================================================= #
# ===  detect OHP sheet's edge point and crop it        === #
# ========================================================= #

def detect__OHPsheetEdge( inpFile=None, bbcheckFile=None, croppedFile=None ):


    # ------------------------------------------------- #
    # --- [1] load images                           --- #
    # ------------------------------------------------- #
    img16         = cv2.imread( inpFile, cv2.IMREAD_UNCHANGED )
    blur          = cv2.GaussianBlur( img16, (11,11), 5 )

    # ------------------------------------------------- #
    # --- [2] to 1D profile                         --- #
    # ------------------------------------------------- #
    ypeak, xpeak  = np.unravel_index( np.argmax( blur ), shape=blur.shape )
    xrange_tight  = [ int(xpeak-xfilmsize_px/5), int(xpeak+xfilmsize_px/5) ]
    yrange_tight  = [ int(ypeak-yfilmsize_px/5), int(ypeak+yfilmsize_px/5) ]
    xrange_broad  = [ int(xpeak-xfilmsize_px*2), int(xpeak+xfilmsize_px*2) ]
    yrange_broad  = [ int(ypeak-yfilmsize_px*2), int(ypeak+yfilmsize_px*2) ]
    xprofiles     = blur[ yrange_tight[0]:yrange_tight[1], xrange_broad[0]:xrange_broad[1] ]
    yprofiles     = blur[ yrange_broad[0]:yrange_broad[1], xrange_tight[0]:xrange_tight[1] ]


    # ------------------------------------------------- #
    # --- [3] search max. derivative point          --- #
    # ------------------------------------------------- #
    xstack = []
    for ik in range(xprofiles.shape[0]):
        prof      = xprofiles[ik,:] / ( np.max( xprofiles[ik,:] ) - np.min( xprofiles[ik,:] ) )
        diff      = np.diff( prof )
        xL        = int( np.argmax( diff ) + xrange_broad[0] )
        xR        = int( np.argmin( diff ) + xrange_broad[0] )
        xstack   += [ [ xL, xR ] ]
    ystack = []
    for ik in range(yprofiles.shape[1]):
        prof      = yprofiles[:,ik] / ( np.max( yprofiles[:,ik] ) - np.min( yprofiles[:,ik] ) )
        diff      = np.diff( prof )
        yL        = int( np.argmax( diff ) + yrange_broad[0] )
        yR        = int( np.argmin( diff ) + yrange_broad[0] )
        ystack   += [ [ yL, yR ] ]

    # ------------------------------------------------- #
    # --- [4] edge/center point under ristriction   --- #
    # ------------------------------------------------- #
    xedges    = np.median( np.array( xstack ), axis=0 )
    yedges    = np.median( np.array( ystack ), axis=0 )
    xcenter   = round( np.average( xedges ) )
    ycenter   = round( np.average( yedges ) )
    xLCR      = [ int(xcenter-xfilmsize_px/2), int(xcenter), int(xcenter+xfilmsize_px/2) ]
    yBCT      = [ int(ycenter-yfilmsize_px/2), int(ycenter), int(ycenter+yfilmsize_px/2) ]
    img8_bw   = cv2.normalize( img16, None, 0, 255, cv2.NORM_MINMAX).astype( np.uint8 )
    img8_bw   = cv2.bitwise_not( img8_bw )
    img       = cv2.cvtColor( img8_bw, cv2.COLOR_GRAY2BGR )
    img       = cv2.rectangle( img, (xLCR[0],yBCT[0]), (xLCR[2],yBCT[2]), \
                               (255,0,0), 2 )
    img         = cv2.line( img, (xLCR[1],yBCT[0]), (xLCR[1],yBCT[2]), (255,0,0), 2 )
    img_bbcheck = cv2.line( img, (xLCR[0],yBCT[1]), (xLCR[2],yBCT[1]), (255,0,0), 2 )
    img_cropped = img16[ yBCT[0]:yBCT[2], xLCR[0]:xLCR[2] ]
    
    # ------------------------------------------------- #
    # --- [5] save and return                       --- #
    # ------------------------------------------------- #
    if ( bbcheckFile is not None ):
        cv2.imwrite( bbcheckFile, img_bbcheck )
    if ( croppedFile  is not None ):
        cv2.imwrite( croppedFile, img_cropped )
    ret = { "xLeft"   :xLCR[0], "xCenter":xLCR[1], "xRight":xLCR[2],
            "yBottom" :yBCT[0], "yCenter":yBCT[1], "yTop"  :yBCT[2],
            "image_bw":img8_bw, "image_bbcheck":img_bbcheck, "image_cropped":img_cropped }
    return( ret )


# ========================================================= #
# ===  measure__beamshift                               === #
# ========================================================= #

def measure__beamshift( image=None, Lx=None, Ly=None, method="find-maximum-in-2D" ):

    def gaussFunc( xin, a0, a1, a2, a3 ):
        return( a0*np.exp( -( xin-a1 )**2 / (2*a2**2) ) + a3 )
    
    if ( image is None ): sys.exit( "[detect__OHPsheetEdge.py] image == ???" )
    if ( Lx    is None ): sys.exit( "[detect__OHPsheetEdge.py] Lx    == ???" )
    if ( Ly    is None ): sys.exit( "[detect__OHPsheetEdge.py] Ly    == ???" )

    Ny   , Nx    = image.shape[0], image.shape[1]
    ycent, xcent = Ny/2          , Nx/2
    Ly_px, Lx_px = Ly/Ny         , Lx/Nx

    if   ( method == "find-maximum-in-2D" ):
        blur        = cv2.GaussianBlur( image, (5,5), 3 )
        ypeak,xpeak = np.unravel_index( np.argmax( blur ), shape=blur.shape )
        ydiff,xdiff = Ly_px*( ypeak - ycent ), Lx_px*( xpeak - xcent )
        ret         = { "xpeak":xpeak, "ypeak":ypeak, "xdiff":xdiff, "ydiff":ydiff, }
        
    elif ( method == "gauss-fit-projection" ):
        # -- vertical / horizontal projection -- #
        proj_y = image.mean( axis=1 ) / ( norm__16bit )
        proj_x = image.mean( axis=0 ) / ( norm__16bit )
        
        # -- gauss fitting -- #
        import scipy.optimize
        xAxis, yAxis  = np.linspace( -Lx/2, +Lx/2, Nx ), np.linspace( -Ly/2, +Ly/2, Ny )
        a0_x , a0_y   = np.max(proj_x)-np.min( proj_x ), np.max(proj_y)-np.min( proj_y )
        a1_x , a1_y   = xAxis[ np.argmax(proj_x) ]     , yAxis[ np.argmax(proj_y) ]
        a2_x , a2_y   = 0.2 * Lx, 0.2 * Ly
        a3_x , a3_y   = np.min( proj_x ), np.min( proj_y )
        poptx, _      = scipy.optimize.curve_fit( gaussFunc, xAxis, proj_x, \
                                                 p0=[ a0_x, a1_x, a2_x, a3_x] )
        popty, _      = scipy.optimize.curve_fit( gaussFunc, yAxis, proj_y, \
                                                 p0=[ a0_y, a1_y, a2_y, a3_y] )
        xdiff,sigma_x = poptx[1], poptx[2]
        ydiff,sigma_y = popty[1], popty[2]
        xfit ,yfit    = gaussFunc( xAxis, *poptx ), gaussFunc( yAxis, *popty )
        xpeak,ypeak   = round(xcent+xdiff/Lx_px), round(ycent+ydiff/Ly_px)
        ret           = { "xpeak":xpeak, "ypeak":ypeak, "xdiff":xdiff, "ydiff":ydiff, \
                          "proj_x":proj_x, "proj_y":proj_y, "xAxis":xAxis, "yAxis":yAxis, \
                          "sigma_x":sigma_x, "sigma_y":sigma_y, "xfit":xfit, "yfit":yfit, \
                         }
            
    else:
        print( "[detect__OHPsheetEdge.py] unknown method :: {}".format( method ) )
        sys.exit()
    return( ret )


# ========================================================= #
# ===  plot graph of the distribution                   === #
# ========================================================= #

def plot__fittedGaussian( xAxis=None, xproj=None, xpeak=None, xdiff=None, xfit=None, Lx=None, \
                          yAxis=None, yproj=None, ypeak=None, ydiff=None, yfit=None, Ly=None, \
                         ):

    # ------------------------------------------------- #
    # --- [1] config                                --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [7.0,2.5],
        "figure.position"    : [ 0.08, 0.20, 0.98, 0.96 ],
        "ax1.y.range"        : { "auto":False, "min":   0.0, "max": 1.0, "num": 6 },
        "ax1.y.label"        : "Intensity (a.u.)",
        "ax1.x.minor.nticks" : 5, 
        "ax1.y.minor.nticks" : 5, 
        "plot.marker"        : "none",
        "legend.fontsize"    : 12.0, 
        "legend.location"    : "upper right", 
    }
    config = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [2] x-profile                             --- #
    # ------------------------------------------------- #
    config_ = { "ax1.x.range": { "auto":False, "min":  -8.0, "max":+8.0, "num":17 },
                "ax1.x.label": "x (mm)",
               }
    config  = { **config, **config_ }
    text1  = "peak  at = ({0:.2},{1:.2}) (mm)".format( xdiff, ydiff )
    text2  = "pixel at = ({0},{1:}) (px)"     .format( xpeak, ypeak )
    fig    = gp1.gplot1D( config=config, pngFile="png/fittedGaussian_x.png" )
    fig.add__plot   ( xAxis=xAxis, yAxis=xfit , color="grey", label="Fitted", linestyle="--" )
    fig.add__plot   ( xAxis=xAxis, yAxis=xproj, color="C1"  , label="Exp."  , linestyle="-"  )
    fig.add__cursor ( xAxis=xdiff, label="Beam shift" )
    fig.add__text   ( xpos=0.10, ypos=0.88, text=text1, fontsize=12.0 )
    fig.add__text   ( xpos=0.10, ypos=0.76, text=text2, fontsize=12.0 )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [2] y-profile                             --- #
    # ------------------------------------------------- #
    config_ = { "ax1.x.range": { "auto":False, "min":  -8.0, "max":+8.0, "num":17 },
                "ax1.x.label": "y (mm)",
               }
    config  = { **config, **config_ }
    fig     = gp1.gplot1D( config=config, pngFile="png/fittedGaussian_y.png" )
    fig.add__plot   ( xAxis=yAxis, yAxis=yfit , color="grey", label="Fitted", linestyle="--" )
    fig.add__plot   ( xAxis=yAxis, yAxis=yproj, color="C1"  , label="Exp."  , linestyle="-"  )
    fig.add__cursor ( xAxis=ydiff, label="Beam shift" )
    fig.add__text   ( xpos=0.10, ypos=0.88, text=text1, fontsize=12.0 )
    fig.add__text   ( xpos=0.10, ypos=0.76, text=text2, fontsize=12.0 )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()

    return()


# ========================================================= #
# === plot cropped image using matplotlib               === #
# ========================================================= #

def plot__images( bbcheck=None, cropped=None, \
                  Lx=None, Ly=None ):
    
    bbcheck = bbcheck / 255
    cropped = cropped / norm__16bit

    # ------------------------------------------------- #
    # --- [1] config settings                       --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [4.5,4.5],
        "figure.position"    : [ 0.16, 0.16, 0.88, 0.88 ],
        "cmp.level"          : { "auto":False, "min": 0.0, "max":1.0, "num":64 },
        "cmp.colortable"     : "jet",
        "clb.title"          : "Intensity (a.u.)"
    }
    config   = { **config, **config_ }
    
    # ------------------------------------------------- #
    # --- [2] plot cropped ( px )                   --- #
    # ------------------------------------------------- #
    config_ = {
        "ax1.x.label"        : "x (px)",
        "ax1.y.label"        : "y (px)",
        "ax1.x.range"        : { "auto":False, "min": 0.0, "max":300.0, "num":7 },
        "ax1.y.range"        : { "auto":False, "min": 0.0, "max":300.0, "num":7 },
    }
    config   = { **config, **config_ }
    xAxis_px = np.arange( cropped.shape[1]+1 )
    yAxis_px = np.arange( cropped.shape[0]+1 )
    fig      = gp2.gplot2D( config=config, pngFile="png/image_cropped_px.png" )
    fig.add__cMap( xAxis=xAxis_px, yAxis=yAxis_px, cMap=cropped, cmpmode="pcolor" )
    fig.set__axis()
    fig.ax1.invert_yaxis()
    fig.set__colorbar()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [3] plot cropped ( mm )                   --- #
    # ------------------------------------------------- #
    config_ = {
        "ax1.x.label"        : "x (mm)",
        "ax1.y.label"        : "y (mm)",
        "ax1.x.range"        : { "auto":False, "min": -8.0, "max":8.0, "num":9 },
        "ax1.y.range"        : { "auto":False, "min": -8.0, "max":8.0, "num":9 },
    }
    config   = { **config, **config_ }
    xAxis_mm = np.linspace( -Lx/2, +Lx/2, cropped.shape[1]+1 )
    yAxis_mm = np.linspace( -Ly/2, +Ly/2, cropped.shape[0]+1 )
    fig      = gp2.gplot2D( config=config, pngFile="png/image_cropped_mm.png" )
    fig.add__cMap( xAxis=xAxis_mm, yAxis=yAxis_mm, cMap=cropped, cmpmode="pcolor" )
    fig.set__axis()
    fig.ax1.invert_yaxis()
    fig.set__colorbar()
    fig.save__figure()


    
# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1] input file                            --- #
    # ------------------------------------------------- #
    # inpFile = "img/ohp-01_20251112-145755.tif"
    inpFile = "img/ohp-02_20251112-162159.tif"
    # inpFile = "img/ohp-03_20251113-154053.tif"

    # ------------------------------------------------- #
    # --- [2] settings                              --- #
    # ------------------------------------------------- #
    Lx, Ly  = 15.0, 15.0
    method  = "gauss-fit-projection"


    # ------------------------------------------------- #
    # --- [3] call analyzer                         --- #
    # ------------------------------------------------- #
    analyze__OHPFilmExperiment( inpFile=inpFile, method=method, Lx=Lx, Ly=Ly )
    


