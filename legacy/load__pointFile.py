import os, sys
import numpy   as np


# ========================================================= #
# ===  load point file with prescribed header           === #
# ========================================================= #

def load__pointFile( inpFile=None, returnType="point", shape=None, \
                     order="C", readHeader=True, skiprows=None, dtype=None, digit=None ):

    # ------------------------------------------------- #
    # --- [1] Arguments                             --- #
    # ------------------------------------------------- #
    names, size = None, None
    if ( inpFile  is None ): sys.exit( "[load__pointFile] inpFile   == ??? " )
    extention   = ( inpFile.split( "." ) )[-1]

    # ------------------------------------------------- #
    # --- [2] load csv                              --- #
    # ------------------------------------------------- #
    if ( extention.lower() == "csv" ):
        import pandas as pd
        Data    = pd.read_csv( inpFile )
        names   = Data.columns
        if   ( returnType.lower() in [ "point", "structured" ] ):
            ret = Data.values
        elif ( returnType.lower() in [ "dict" ] ):
            ret = { name:Data[name].values for name in names }
        return( ret )
    
    # ------------------------------------------------- #
    # --- [2] load Data with header                 --- #
    # ------------------------------------------------- #
    if ( type(skiprows) is int ):
        readHeader = False
    else:
        skiprows = 0
    if ( readHeader ):
        with open( inpFile, "r" ) as f:
            line1 = ( f.readline() ).strip()
            line2 = ( f.readline() ).strip()
            line3 = ( f.readline() ).strip()
        if   ( ( len(line1) == 0 ) or ( len(line2) == 0 ) or ( len(line3) == 0 ) ):
            readHeader = False
        elif ( ( line1[0] != "#" ) or ( line2[0] != "#" ) or ( line3[0] != "#" ) ):
            readHeader = False

    with open( inpFile, "r" ) as f:
        Data  = np.loadtxt( f, skiprows=skiprows )
    if ( dtype is not None ):
        Data  = np.array( Data, dtype=dtype )
        
    # ------------------------------------------------- #
    # --- [3] names & size                          --- #
    # ------------------------------------------------- #
    if ( readHeader ):
        ext1  = ( ( ( line1.strip() ).strip( "#" ) ).strip() ).split()
        ext2  = ( ( ( line2.strip() ).strip( "#" ) ).strip() ).split()
        ext3  = ( ( ( line3.strip() ).strip( "#" ) ).strip() ).split()
        
        try:
            names = [ str(s) for s in ext1 ]
        except:
            names = [ "x{0}".format(ik) for ik in range( nComponents ) ]
            
        try:
            size  = [ int(i) for i in ext2 ]
        except:
            size  = Data.shape
                    
        try:
            shape = [ int(i) for i in ext3 ]
        except:
            if ( returnType.lower() == "structured" ):
                print( "[load__pointFile.py] [WARNING] returnType == structured, but, header specification is somehow wrong. " )
                print( "[load__pointFile.py] line1 :: {0} ".format( line1 ) )
                print( "[load__pointFile.py] line2 :: {0} ".format( line2 ) )
                print( "[load__pointFile.py] line3 :: {0} ".format( line3 ) )
            shape = Data.shape
    else:
        size        = Data.shape
        shape       = Data.shape
        if ( Data.ndim == 1 ):
            nComponents = 1
        else:
            nComponents = Data.shape[1]
        names = [ "x{0}".format(ik) for ik in range( nComponents ) ]

    # ------------------------------------------------- #
    # --- [4] digit                                 --- #
    # ------------------------------------------------- #
    if   ( digit is not None ):
        if ( type(digit) in [ int, float ] ):
            digit = [ int(digit) ] * nComponents
        if ( type(digit) in [ list, tuple, np.ndarray ] ):
            for ik in range( nComponents ):
                if   ( digit[ik] is None ):
                    pass
                else:
                    pass
                    # Data[...,ik] = np.round( Data[...,ik], decimals=int(digit[ik]) )
            
    # ------------------------------------------------- #
    # --- [4] return                                --- #
    # ------------------------------------------------- #
    if   ( returnType.lower() == "point" ):
        ret   = Data
    elif ( returnType.lower() == "dict"  ):
        ret   = {}
        ret["size"] = size
        for ik,name in enumerate( names ):
            ret[name] = np.ravel( Data[:,ik] )
    elif ( returnType.lower() == "structured"  ):
        if ( shape is None ):
            sys.exit( "[load__pointFile] returnType == structured :: but No Shape information [Error]" )
        if ( np.prod( shape ) != np.size( Data ) ):
            print( "[save__pointFile] shape and size are inconsistent [ERROR] " )
            print( "[save__pointFile] shape :: {0} ".format( shape      ) )
            print( "[save__pointFile] size  :: {0} ".format( Data.shape ) )
            sys.exit()
        ret   = np.reshape( Data, shape, order=order )
    elif ( returnType.lower() == "info" ):
        ret = { "names":names, "shape":shape, "size":size }
    else:
        sys.exit( "[load__pointFile] returnType = ( point, dict )" )
    return( ret )





# ======================================== #
# ===  実行部                          === #
# ======================================== #
if ( __name__=="__main__" ):

    # import nkUtilities.equiSpaceGrid as esg
    # x1MinMaxNum = [ 0.0, 1.0, 11 ]
    # x2MinMaxNum = [ 0.0, 1.0, 11 ]
    # x3MinMaxNum = [ 0.0, 1.0, 11 ]
    # Data        = esg.equiSpaceGrid( x1MinMaxNum=x1MinMaxNum, x2MinMaxNum=x2MinMaxNum, \
    #                                  x3MinMaxNum=x3MinMaxNum, returnType = "structured" )
    # import nkUtilities.save__pointFile as spf
    # outFile   = "test/out.dat"
    # spf.save__pointFile( outFile=outFile, Data=Data )

    
    inpFile = "test/out.dat"
    
    Data = load__pointFile( inpFile=inpFile, returnType="structured", skiprows=0, digit=3 )
    print( Data )
    # print( Data[0,0,:,0] )
