import os, re, json5, tqdm, glob
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import matplotlib.colors as mcolors
import PIL.Image         as pimg
import nk_toolkit.plot.load__config as lcf
import nk_toolkit.plot.gplot1D      as gp1


# ========================================================= #
# ===  load phase-space data                            === #
# ========================================================= #
def _load__phaseSpaceData( parameterFile=None ):
    with open( parameterFile, "r", encoding="utf-8" ) as inp:
        parameterText = re.sub( r",\s*([}\]])", r"\1", inp.read() )
        params        = json5.loads( parameterText )

    phaseSpaceFile = params["output"].get( "phaseSpaceFile", "out/phaseSpace.csv" )
    if ( not os.path.isfile( phaseSpaceFile ) ):
        raise FileNotFoundError( f"Phase-space data is not found: {phaseSpaceFile}" )
    return( params, pd.read_csv( phaseSpaceFile ) )


# ========================================================= #
# ===  plot__phaseSpacebyElement                        === #
# ========================================================= #
def plot__phaseSpacebyElement( parameterFile="inp/parameters.json" ):

    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    params, phaseSpaceData = _load__phaseSpaceData( parameterFile=parameterFile )
    output         = params["output"]
    elementData    = phaseSpaceData[phaseSpaceData["record_type"] == "element"]
    particleData   = phaseSpaceData[phaseSpaceData["record_type"] == "particle"]
    separatrixData = phaseSpaceData[phaseSpaceData["record_type"] == "separatrix"]
    elementId      = elementData["element_id"].to_numpy( dtype=int )

    # ------------------------------------------------- #
    # --- [2] select elements                       --- #
    # ------------------------------------------------- #
    cellnums = output.get( "snapshot.cellNums", None )
    if   ( cellnums is None ):
        plotIdList = elementId.tolist()
    elif ( isinstance( cellnums ) is list ):
        plotIdList = []
        for cellValue in cellnums:
            plotId = elementId[-1] if ( int( cellValue ) < 0 ) \
                else min( int( cellValue ), int( elementId[-1] ) )
            if ( plotId not in plotIdList ):
                plotIdList.append( plotId )
    else:
        raise ValueError( f"Unknown snapshot.cellNums : {snapshot.cellNums} => ( null or list )" )

    pngPrefix      = output.get( "snapshot.pngPrefix", "png/snapshot/cell" )
    plotEnergyYlim = output.get( "snapshot.ylim"     , None )
    pngDir         = os.path.dirname( pngPrefix )
    os.makedirs( pngDir, exist_ok=True )

    # ------------------------------------------------- #
    # --- [3] plot each element                     --- #
    # ------------------------------------------------- #
    pngFileList = []
    for plotId in tqdm.tqdm( plotIdList ):
        elementAtPlot    = elementData[elementData["element_id"] == plotId].iloc[0]
        particleAtPlot   = particleData[particleData["element_id"] == plotId]
        separatrixAtPlot = separatrixData[separatrixData["element_id"] == plotId]
        
        pngFile = f"{pngPrefix}_{plotId:04d}.png"
        yRange  = { "auto":plotEnergyYlim is None, "min":0.0,
                    "max":1.0 if ( plotEnergyYlim is None ) else plotEnergyYlim, "num":11 }
        config = {
            **lcf.load__config(),
            "figure.size"        : [ 6.0, 6.0 ],
            "figure.pngFile"     : pngFile,
            "figure.position"    : [ 0.14, 0.14, 0.92, 0.92 ],
            "ax1.x.range"        : { "auto":False, "min":-180.0, "max":180.0, "num":9 },
            "ax1.y.range"        : yRange,
            "ax1.x.label"        : r"RF phase $\phi$ [deg.]",
            "ax1.y.label"        : r"Energy $K$ [MeV/u]",
            "grid.major.sw"      : True,
            "grid.major.alpha"   : 0.3,
            "legend.fontsize"    : 8,
        }

        fig = gp1.gplot1D( config=config )
        fig.add__plot( xAxis=np.rad2deg( particleAtPlot["phase_rad"] ),
                       yAxis=particleAtPlot["energy_MeV_u"], label="particles",
                       linestyle="None", marker=".", markersize=2.0 )

        segmentIdList = separatrixAtPlot["segment_id"].dropna().unique()
        for iSegment, segmentId in enumerate( segmentIdList ):
            segmentData = separatrixAtPlot[separatrixAtPlot["segment_id"] == segmentId]
            sepLabel    = "separatrix" if ( iSegment == 0 ) else "_nolegend_"

            fig.add__plot( xAxis=np.rad2deg( segmentData["phase_rad"] ),
                           yAxis=segmentData["upper_energy_MeV_u"], label=sepLabel,
                           color="lightgrey", linestyle="-", linewidth=1.0, marker="None" )
            fig.add__plot( xAxis=np.rad2deg( segmentData["phase_rad"] ),
                           yAxis=segmentData["lower_energy_MeV_u"], label="_nolegend_",
                           color="lightgrey", linestyle="-", linewidth=1.0, marker="None" )

        fig.add__plot( xAxis=np.rad2deg( elementAtPlot["sync_phase_rad"] ),
                       yAxis=elementAtPlot["sync_energy_MeV_u"], label="sync.",
                       color="crimson", linestyle="None", marker="o", markersize=4.0 )

        fig.ax1.set_title(
            f"cell #={plotId};    z={elementAtPlot['position_m']:.5f} m\n"
            f"Esync={elementAtPlot['sync_energy_MeV_u']:.5f} MeV/u;    "
            f"capture={elementAtPlot['capture_efficiency_percent']:.2f} %",
            fontsize=10 )
        fig.set__legend( loc="upper left", fontsize=8 )
        fig.save__figure( silent=True )
        pngFileList.append( pngFile )
    
    print( f"[plot by element] output dir :: {pngDir}" )
    print( f"[plot by element] nFigures   :: {len( pngFileList )}" )
    return( pngFileList )


# ========================================================= #
# ===  plot__phaseSpacebyEnergy                         === #
# ========================================================= #
def plot__phaseSpacebyEnergy( parameterFile="inp/parameters.json" ):

    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    params, phaseSpaceData = _load__phaseSpaceData( parameterFile=parameterFile )
    output          = params["output"]
    elementData     = phaseSpaceData[phaseSpaceData["record_type"] == "element"]
    particleData    = phaseSpaceData[phaseSpaceData["record_type"] == "particle"]
    syncEnergy      = elementData["sync_energy_MeV_u"].to_numpy()

    energyRangeList = output.get( "energyRange.ranges_MeV_u",
                                  [[ float( syncEnergy[0] ), float( syncEnergy[-1] ) ]] )
    pngPrefix       = output.get( "energyRange.pngPrefix", "png/energyRange/energyRange" )
    pngDir          = os.path.dirname( pngPrefix )
    os.makedirs( pngDir, exist_ok=True )

    # ------------------------------------------------- #
    # --- [2] plot each energy range                --- #
    # ------------------------------------------------- #
    pngFileList = []
    for iRange, energyRange in enumerate( tqdm.tqdm( energyRangeList ) ):
        if ( len( energyRange ) != 2 ):
            raise ValueError( "Energy range must be [Emin, Emax]." )

        energyMin = float( energyRange[0] )
        energyMax = float( energyRange[1] )
        if ( energyMax <= energyMin ):
            raise ValueError( "Emax must be larger than Emin." )

        selectedElement = elementData[
            ( elementData["sync_energy_MeV_u"] >= energyMin )
            & ( elementData["sync_energy_MeV_u"] <= energyMax )]
        selectedId = selectedElement["element_id"].to_numpy( dtype=int )

        if ( selectedId.size == 0 ):
            print( f"[WARNING] no element in {energyMin:g}-{energyMax:g} MeV/u." )
            continue

        colorMin  = float( selectedId[0] )
        colorMax  = max( float( selectedId[-1] ), colorMin + 1.0 )
        colorNorm = mcolors.Normalize( vmin=colorMin, vmax=colorMax )
        colorMap  = plt.get_cmap( "jet" )

        pngFile = f"{pngPrefix}_{iRange:02d}.png"
        config  = {
            **lcf.load__config(),
            "figure.size"        : [ 7.0, 7.0 ],
            "figure.pngFile"     : pngFile,
            "figure.position"    : [ 0.14, 0.14, 0.90, 0.92 ],
            "ax1.x.range"        : { "auto":False, "min":-180.0, "max":180.0, "num":9 },
            "ax1.y.range"        : { "auto":True , "min":0.0, "max":energyMax, "num":11 },
            "ax1.x.label"        : r"RF phase $\phi$ [deg.]",
            "ax1.y.label"        : r"Energy $K$ [MeV/u]",
            "grid.major.sw"      : True,
            "grid.major.alpha"   : 0.3,
        }

        colorMin  = float( selectedId[0] )
        colorMax  = max( float( selectedId[-1] ), colorMin + 1.0 )
        colorNorm = mcolors.Normalize( vmin=colorMin, vmax=colorMax )
        colorMap  = plt.get_cmap( "jet" )
        fig       = gp1.gplot1D( config=config )

        for plotId in selectedId:
            particleAtPlot = particleData[particleData["element_id"] == plotId]
            fig.add__plot( xAxis=np.rad2deg( particleAtPlot["phase_rad"] ),
                           yAxis=particleAtPlot["energy_MeV_u"], label="_nolegend_",
                           color=colorMap( colorNorm( plotId ) ), alpha=0.50,
                           linestyle="None", marker=".", markersize=1.5 )

        colorBarData = cm.ScalarMappable( norm=colorNorm, cmap=colorMap )
        colorBarData.set_array( [] )
        colorBar = fig.fig.colorbar( colorBarData, ax=fig.ax1 )
        colorBar.set_label( "cell #", fontsize=config["ax1.y.major.fontsize"] )
        colorBar.ax.tick_params( labelsize=config["ax1.y.major.fontsize"] )

        fig.ax1.set_title( f"K-phi phase space: {energyMin:g}-{energyMax:g} MeV/u;    "
                           f"cell #={selectedId[0]}-{selectedId[-1]}",
                           fontsize=config["figure.font.size"]  )
        fig.save__figure( silent=True )
        pngFileList.append( pngFile )

    print( f"[plot by energy] output dir :: {pngDir}" )
    print( f"[plot by energy] nFigures   :: {len( pngFileList )}" )
    return( pngFileList )


# ========================================================= #
# ===  make__phaseSpaceAnimation                        === #
# ========================================================= #
def make__phaseSpaceAnimation( parameterFile=None, pngFileList=None, gifFile=None,
                               duration_ms=120, loop=2 ):

    if ( pngFileList is None or len( pngFileList ) == 0 or ( gifFile is None ) ):
        with open( parameterFile, "r" ) as f:
            params      = json5.load( f )
    if ( pngFileList is None or len( pngFileList ) == 0 ):
        pngFilePath = params["output"]["snapshot.pngPrefix"] + "*"
        pngFileList = sorted( glob.glob( pngFilePath ) )
    if ( gifFile is None ):
        gifFile = params["output"]["snapshot.gifFile"]

    imageList = [ pimg.open( pngFile ).convert( "RGB" ) for pngFile in pngFileList ]
    os.makedirs( os.path.dirname( gifFile ) or ".", exist_ok=True )
    imageList[0].save( gifFile, save_all=True, append_images=imageList[1:],
                       duration=duration_ms, loop=loop, disposal=2 )
    for image in tqdm.tqdm( imageList ):
        image.close()
    print( f"[animation] nFrames    :: {len( pngFileList )}" )
    print( f"[animation] output gif :: {gifFile}" )
    return( gifFile )


# ========================================================= #
# ===  Execution of Program                             === #
# ========================================================= #
if ( __name__=="__main__" ):
    parameterFile = "inp/parameters.json"
    plot__phaseSpacebyElement( parameterFile=parameterFile )
    plot__phaseSpacebyEnergy ( parameterFile=parameterFile )
    make__phaseSpaceAnimation(  )
