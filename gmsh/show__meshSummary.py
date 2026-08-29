import os
import gmsh
import numpy as np


# ========================================================= #
# ===  show__meshSummary.py                             === #
# ========================================================= #
def show__meshSummary( mshFile=None, elapsedTime_s=None, barWidth=45 ):

    # ------------------------------------------------- #
    # --- [1] load mesh                             --- #
    # ------------------------------------------------- #
    if ( not os.path.exists( mshFile ) ):
        raise FileNotFoundError( "Cannot find mesh file :: {}".format( mshFile ) )

    gmsh.initialize()
    gmsh.option.setNumber( "General.Terminal", 0 )
    gmsh.open( mshFile )
    elementTypes, elementTags, _ = gmsh.model.mesh.getElements( dim=3 )

    tetraBlocks = []
    for elementType,tags in zip( elementTypes, elementTags ):
        elementName = gmsh.model.mesh.getElementProperties( elementType )[0]
        if ( elementName.startswith( "Tetrahedron" ) ):
            tetraBlocks += [ np.asarray( tags, dtype=np.uint64 ) ]
    if ( len( tetraBlocks ) == 0 ):
        gmsh.finalize()
        raise ValueError( "No tetrahedral elements were found :: {}".format( mshFile ) )

    tetraTags     = np.concatenate( tetraBlocks )
    totalMeshSize = len( tetraTags )
    gamma = np.asarray( gmsh.model.mesh.getElementQualities(
        tetraTags, qualityName="gamma" ), dtype=float )
    gmsh.finalize()
    gamma = np.clip( gamma, 0.0, 1.0 )

    # ------------------------------------------------- #
    # --- [2] quality population                    --- #
    # ------------------------------------------------- #
    histogram, _ = np.histogram( gamma, bins=np.linspace( 0.0, 1.0, 11 ) )
    population    = histogram.astype( float ) / totalMeshSize * 100.0
    maxPopulation = population.max()

    # ------------------------------------------------- #
    # --- [3] formatting                           --- #
    # ------------------------------------------------- #
    elapsedSec  = int( round( elapsedTime_s ) )
    hours       = elapsedSec // 3600
    minutes     = ( elapsedSec % 3600 ) // 60
    seconds     = elapsedSec % 60
    elapsedText = "{} h {} min {} sec".format( hours, minutes, seconds )

    exponent = int( np.floor( np.log10( totalMeshSize ) ) ) if ( totalMeshSize > 0 ) else 0
    mantissa = totalMeshSize / 10.0**exponent if ( totalMeshSize > 0 ) else 0.0
    meshText = "{:,}  ( {:.1f} x 10^{} )".format( totalMeshSize, mantissa, exponent )

    # ------------------------------------------------- #
    # --- [4] summary output                       --- #
    # ------------------------------------------------- #
    line = "-" * 70
    print( "\n{}".format( " mesh summary ".center( 70, "-" ) ) )
    print( "\nelapsed time      : {}".format( elapsedText ) )
    print( "total mesh size   : {}".format( meshText ) )
    print( "quality metric    : Gmsh gamma  ( 0 % = degenerate, 100 % = ideal )" )
    print( "bar scale         : relative to largest population bin" )
    print( "\nmesh quality population" )
    for index,populationPct in enumerate( population ):
        lower  = index * 10
        upper  = ( index + 1 ) * 10
        filled = int( round( populationPct / maxPopulation * barWidth ) )
        if ( populationPct > 0.0 and filled == 0 ):
            filled = 1
        bar = "#" * filled + " " * ( barWidth - filled )
        print( "{:>3} - {:>3} % : [{}] {:>6.2f} %".format(
            lower, upper, bar, populationPct ) )
    print( "\n{}\n".format( line ) )

    return( { "total_mesh_size":totalMeshSize, "quality_name":"gamma",
              "quality_min":float( gamma.min() ), "quality_mean":float( gamma.mean() ),
              "histogram":histogram, "population_pct":population } )
