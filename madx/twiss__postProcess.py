import numpy as np

# ========================================================= #
# ===  twiss__postProcess.py                            === #
# ========================================================= #

def twiss__postProcess( twiss=None, csvFile=None ):

    # ------------------------------------------------- #
    # --- [1] parameters                            --- #
    # ------------------------------------------------- #
    if ( twiss is None ):
        sys.exit( "[calculate__beamsize.py] twiss == ???" )

    # ------------------------------------------------- #
    # --- [2] calculate beam size                   --- #
    # ------------------------------------------------- #
    emitx, emity        = float( twiss["EX"] ), float( twiss["EY"] )
    betax, betay        = twiss["df"]["BETX"] , twiss["df"]["BETY"]
    twiss["df"]["SIGX"] = np.sqrt( betax * emitx )
    twiss["df"]["SIGY"] = np.sqrt( betay * emity )
    
    # ------------------------------------------------- #
    # --- [3] save post data                        --- #
    # ------------------------------------------------- #
    if ( csvFile is not None ):
        twiss["df"].to_csv( csvFile, index=False )
        print( "[twiss__postProcess.py]  csvFile = {}".format( csvFile ) )
    return( twiss )

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #
if ( __name__=="__main__" ):
    twiss__postProcess()
    
