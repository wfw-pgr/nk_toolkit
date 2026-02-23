import numpy as np

# ========================================================= #
# ===  map__csvOnMesh.py                                === #
# ========================================================= #

def map__csvOnMesh( mshFile=None, outFile=None, cellDataFiles=[] ):
    
    # ------------------------------------------------- #
    # --- [1] Arguments                             --- #
    # ------------------------------------------------- #
    if ( mshFile is None ): sys.exit( "[map__csvOnMesh.py] mshFile == ???" )
    if ( outFile is None ):
        extention = mshFile.split(".")
        outFile   = mshFile.replace( "."+extention, ".vtu" )

    # ------------------------------------------------- #
    # --- [2] cellDataFiles                         --- #
    # ------------------------------------------------- #
    import nkMeshRoutines.convert__withMeshIO as cwm
    cwm.convert__withMeshIO( mshFile=mshFile, outFile=outFile, cellDataFiles=cellDataFiles, replaceData=False )

    # ------------------------------------------------- #
    # --- [3] add material property                 --- #
    # ------------------------------------------------- #
    import nkMeshRoutines.assign__materialProperty as amp
    propertyFile = "dat/materialProperty.conf"
    amp.assign__materialProperty( inpFile=outFile, outFile=outFile, propertyFile=propertyFile )

    # ------------------------------------------------- #
    # --- [4] calculate volume integral             --- #
    # ------------------------------------------------- #
    import nkMeshRoutines.calculate__volumeIntegral as cvi
    target       = "Dose[J/m^3/source]"
    physNum_list = [ 301, 302, 303, 304, 305, 306, 307, 308, 309 ]
    ints         = []
    for ik,physNum in enumerate( physNum_list ):
        ret = cvi.calculate__volumeIntegral( inpFile=outFile, target=target, physNum=physNum )
        print( " (physNum,Heat,Volume,AvgHeat) == ( {0}, {1[0]}, {1[1]}, {1[2]} )".format( physNum, ret ) )
        ints += [ ret ]
    ints   = np.array( ints    )
    volInt = np.sum( ints[:,0] )
    volTot = np.sum( ints[:,1] )
    volAvg = np.sum( ints[:,2] )
    print( "-"*70 )
    print( "--- volume integral summary ---" )
    print( "-"*70 )
    for ik,physNum in enumerate( physNum_list ):
        print( " (physNum,Heat,Volume,AvgHeat) == ( {0}, {1[0]:.6e}, {1[1]:.6e}, {1[2]:.6e} )".format( physNum, ints[ik] ) )
    print( "-"*70 )
    print( " [[Total]] " )
    print( "  volInt :: {} ".format( volInt ) )
    print( "  volTot :: {} ".format( volTot ) )
    print( "  volAvg :: {} ".format( volAvg ) )
    print( "-"*70 )

    # ------------------------------------------------- #
    # --- [5] save volume integral                  --- #
    # ------------------------------------------------- #
    Data = np.concatenate( [ (np.array(physNum_list ))[:,np.newaxis], ints[:,:] ], axis=1 )
    import nkUtilities.save__pointFile as spf
    outFile   = "dat/volume_integral.dat"
    names     = ["physNum","volInt", "volTot", "volAvg"]
    spf.save__pointFile( outFile=outFile, Data=Data, names=names )
    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    mshFile       = "msh/model.msh"
    outFile       = "msh/heatload.vtu"
    cellDataFiles = [ "out/heatload.csv" ]
    map__csvOnMesh( mshFile=mshFile, outFile=outFile, cellDataFiles=cellDataFiles )
    
    
