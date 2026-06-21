import sys
import numpy as np

# ========================================================= #
# ===  save point File with prescribed header           === #
# ========================================================= #

def save__pointFile( Data =None, names=None, ndim     =None , outFile="out.dat", \
                     shape=None, size =None, DataOrder="ijk", fmt    ="%15.8e" , \
                     silent=False, no_header=False ):
    # ------------------------------------------------- #
    # --- [1] Arguments Preparation                 --- #
    # ------------------------------------------------- #
    #  -- [1-1] no Data error raise                 --  #
    if ( Data    is None ): sys.exit( "[save__pointFile] Data    == ???" )
    Data_ = np.copy( Data )

    #  -- [1-2] Data Dimension & shape Check        --  #
    if ( ndim    is None ): ndim    = Data_.ndim
    

    # ------------------------------------------------- #
    # --- [2] if Data dimension == 1 :: 1D Data     --- #
    # ------------------------------------------------- #
    if ( ndim == 1 ):
        if ( names is None ): names   = [ "x1" ]
        size    = [ Data_.size ]
        shape   = Data_.shape

    # ------------------------------------------------- #
    # --- [3] if Data dimension == 2 :: point       --- #
    # ------------------------------------------------- #
    if ( ndim == 2 ):
        size        = Data_.shape
        if ( shape is None ): # -- assume Data is point -- #
            shape       = Data_.shape
        if ( names is None ):
            names       = [ "x{0}".format(i+1) for i in range( size[1] ) ]
            
    # ------------------------------------------------- #
    # --- [3] if Data dimension  > 2 :: structured  --- #
    # ------------------------------------------------- #
    if ( ndim  > 2 ):
        if ( DataOrder == "ijk" ):
            shape       = Data.shape
            nComponents = Data.shape[-1]
            nData       = ( Data[...,0] ).size
            Data_       = np.reshape( Data_, (nData,nComponents) )
            size        = Data_.shape
        if ( DataOrder == "kji" ):
            shape       = Data.shape
            nComponents = Data.shape[0]
            nData       = ( Data[0,...] ).size
            Data_       = np.transpose( np.reshape( Data_, (nComponents,nData) ) )
            size        = Data_.shape
        if ( names is None ):
            names       = [ "x{0}".format(i+1) for i in range( nComponents ) ]
            
    # ------------------------------------------------- #
    # --- [4] save Data with header                 --- #
    # ------------------------------------------------- #
    with open( outFile, "w" ) as f:
        if ( not(silent) ): print( names )
        if ( not( no_header ) ):
            f.write( "# " + " ".join( names ) + "\n" )
            f.write( "# " + " ".join( [ str(i) for i in size ] ) + "\n" )
            if ( shape is not None ):
                f.write( "# " + " ".join( [str(i) for i in shape] ) + "\n" )
        np.savetxt( f, Data_, fmt=fmt )

    # ------------------------------------------------- #
    # --- [5] return Nothing                        --- #
    # ------------------------------------------------- #
    if ( not( silent ) ):
        print( "[save__pointFile.py] output :: {0} ".format( outFile ) )
    return()


# ======================================== #
# ===  実行部                          === #
# ======================================== #
if ( __name__=="__main__" ):

    outFile ="output.dat"
    import nkUtilities.equiSpaceGrid as esg
    LI, LJ, LK  = 11, 21, 31
    x1MinMaxNum = [0.0,   3.0, LI]
    x2MinMaxNum = [0.0,  10.0, LJ]
    x3MinMaxNum = [0.0, 100.0, LK]

    returnType  = "structured"

    # -- 1D test -- #
    # Data        = esg.equiSpaceGrid( x1MinMaxNum=x1MinMaxNum, returnType="structured"  )
    # Data        = np.ravel( Data )

    # -- 2D test -- #
    # Data        = esg.equiSpaceGrid( x1MinMaxNum=x1MinMaxNum, x2MinMaxNum=x2MinMaxNum, \
    #                                  returnType="structured"  )
    
    # -- 3D test -- #
    Data        = esg.equiSpaceGrid( x1MinMaxNum=x1MinMaxNum, x2MinMaxNum=x2MinMaxNum, \
                                     x3MinMaxNum=x3MinMaxNum, returnType=returnType, )
    
    print( Data.shape )
    save__pointFile( outFile=outFile, Data=Data )
