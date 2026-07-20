import json5, glob, os, shutil
import invoke
import pyt.track__longitudinal1D        as tl1
import pyt.plot__longitudinal1D         as pl1
import pyt.analyze__TTFbyEnergyAperture as att


# ========================================================= #
# ===  run longitudinal tracking                        === #
# ========================================================= #
@invoke.task
def track( ctx, parameterFile="inp/parameters.json" ):
    """Run analysis."""
    tl1.track__longitudinal1D( parameterFile=parameterFile )


# ========================================================= #
# ===  pre-analysis of ttf from efield map              === #
# ========================================================= #
@invoke.task
def ttf( ctx, parameterFile="inp/parameters.json" ):
    """TTF analysis from eh_DTL.#01"""
    with open( parameterFile, "r" ) as f:
        params       = json5.load( f )
        ttf_analysis = params.get( "ttf-analysis", {} )
        lattice      = params.get( "lattice"     , {} )
        rf           = params.get( "rf"          , {} )
        
    inpFile       = ttf_analysis.get( "efieldFile"      , "inp/eh_DTL.#01"       )
    csvFile       = ttf_analysis.get( "csvFile"         , "out/ttf_table.csv"    )
    outFile       = ttf_analysis.get( "summaryFile"     , "out/ttf_summary.json" )
    aperture_m    = 2.0* lattice.get( "apertureRadius_m", 0.050                  )
    frequency_MHz =           rf.get( "frequency_MHz"   , 36.5                   )
    ret    = att.analyze__TTFbyEnergyAperture( inpFile=inpFile, outFile=outFile, csvFile=csvFile,
                                               frequency_MHz=frequency_MHz,aperture_m=aperture_m,
                                               coordUnit="cm", dataOrder="z-fastest",
                                               minimumAperturePoints=2, mirrorZ=True )

    
# ========================================================= #
# ===  post process (plot)                              === #
# ========================================================= #
@invoke.task
def post( context, parameterFile="inp/parameters.json", snapshot=False, energyRange=False, ):
    """Post-process plotting."""
    
    if ( not snapshot and not energyRange ):
        print( "[post] Specify --snapshot and/or --energyRange." )
        return()
    if ( snapshot ):
        pl1.plot__phaseSpacebyElement( parameterFile=parameterFile )
        pl1.make__phaseSpaceAnimation( parameterFile=parameterFile )
    if ( energyRange ):
        pl1.plot__phaseSpacebyEnergy ( parameterFile=parameterFile )
        
        
        
# ========================================================= #
# ===  clean output files                               === #
# ========================================================= #
@invoke.task
def clean( ctx ):
    """Remove output files from previous runs."""
    patterns = [ "png/*", "out/*" ]
    for pattern in patterns:
        for path in glob.glob( pattern ):
            if os.path.isfile(path):
                print( f"Removing file {path}" )
                os.remove(path)
            elif os.path.isdir(path):
                print( f"Removing directory {path}" )
                shutil.rmtree(path)


# ========================================================= #
# ===  all = clean + ttf + track                        === #
# ========================================================= #
@invoke.task(pre=[clean, ttf, track,
                  invoke.call( post, snapshot=True, energyRange=True ) ] ) 
def all(ctx):
    """Run all steps: clean, ttf, track, post"""
    pass
