import os, sys, re
import numpy as np

# ========================================================= #
# ===  retrieveData__afterStatement.py                  === #
# ========================================================= #

def retrieveData__afterStatement( inpFile=None, outFile=None, expr_from=None, expr_to=None, \
                                  sendline=1, fmt="%15.8e", names=None ):
    
    x_, y_, z_ = 0, 1, 2

    # ------------------------------------------------- #
    # --- [1] arguments check                       --- #
    # ------------------------------------------------- #
    if ( inpFile   is None ): sys.exit( "[retrieveData__afterStatement.py] inpFile   == ???" )
    if ( expr_from is None ): sys.exit( "[retrieveData__afterStatement.py] expr_from == ???" )

    # ------------------------------------------------- #
    # --- [2] open file and read lines              --- #
    # ------------------------------------------------- #
    with open( inpFile, "r" ) as f:
        lines = f.readlines()
    
    # ------------------------------------------------- #
    # --- [3] retrive start & end line              --- #
    # ------------------------------------------------- #
    found = False
    for ik,line in enumerate(lines):
        if ( found is False ):
            match = re.match( expr_from, line )
            if ( match ):
                iFrom = ik+sendline
                iTo   = len( lines )
                found = True
        elif ( expr_to is not None ):
            match = re.match( expr_to  , line )
            if ( match ):
                iTo   = ik+1
                break
        else:
            pass   # to last line of the file.
    if ( found is False ):
        print( "[retrieveData__afterStatement.py] Cannot find reg-exp {} in this file... ".format( expr_from ) )
        sys.exit()

    # ------------------------------------------------- #
    # --- [4] extract data                          --- #
    # ------------------------------------------------- #
    Data = np.loadtxt( lines[ iFrom:iTo ] )

    # ------------------------------------------------- #
    # --- [5] save in a file & return               --- #
    # ------------------------------------------------- #
    if ( outFile is not None ):
        with open( outFile, "w" ) as f:
            if ( names is None ):
                names = [ "x{0}".format(ik+1) for ik in range( Data.shape[1] ) ]
            f.write( "# " + " ".join( names ) + "\n" )
            f.write( "# " + " ".join( [ str(val) for val in Data.shape ] ) + "\n" )
            f.write( "# " + " ".join( [ str(val) for val in Data.shape ] ) + "\n" )
            np.savetxt( f, Data, fmt=fmt )
            print( "[retrieveData__afterStatement.py] save in a file :: {}".format( outFile ) )
    return( Data )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    inpFile   = "test/retrieveData__afterStatement/sample.inp"
    outFile   = "test/retrieveData__afterStatement/sample.out"
    expr_from = r"#\s*e-lower"
    expr_to   = r"^\s*$"
    ret       = retrieveData__afterStatement( inpFile=inpFile, outFile=outFile, \
                                              expr_from=expr_from, expr_to=expr_to )

