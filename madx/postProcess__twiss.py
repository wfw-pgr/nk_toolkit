import os, sys, json5
import pandas                    as pd
import numpy                     as np
import nk_toolkit.madx.load__tfs as ltf

# ========================================================= #
# ===  postProcess__twiss.py                            === #
# ========================================================= #

def postProcess__twiss( paramsFile="dat/parameters.json" ):

    # ------------------------------------------------- #
    # --- [1] parameters                            --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    twiss  = ltf.load__tfs( tfsFile=params["post.twiss.inpFile"] )
        
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
    if ( params["post.twiss.outFile"] is not None ):
        twiss["df"].to_csv( params["post.twiss.outFile"], index=False )
        print( "[postProcess__twiss.py]  output :: {}".format( params["post.twiss.outFile"] ) )
    return( twiss )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #
if ( __name__=="__main__" ):

    paramsFile = "dat/parameters.json"
    ret     = postProcess__twiss( paramsFile=paramsFile )
    
