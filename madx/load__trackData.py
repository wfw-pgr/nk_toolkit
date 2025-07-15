import os, sys, re
import numpy  as np
import pandas as pd

# ========================================================= #
# ===  load__trackData                                  === #
# ========================================================= #

def load__trackData( trackFile="madx/out/track.tfsone" ):

    # ------------------------------------------------- #
    # --- [1] load particle info                    --- #
    # ------------------------------------------------- #
    comment_marks = ( "@", "$" )
    with open( trackFile, "r" ) as f:
        lines = [ line for line in f if not( line.strip().startswith( comment_marks ) ) ]
    line   = lines.pop(0)
    smatch = re.match( r"^\*(.+)$", line )
    if ( smatch ):
        columns = ( smatch.group(1) ).strip().split()
    else:
        sys.exit( "no match... no columns ERROR." )

    # ------------------------------------------------- #
    # --- [2] store dataframe by segment            --- #
    # ------------------------------------------------- #
    ret, key, sbuff = {}, None, []
    for line in lines:
        if ( line.startswith( "#segment" ) ):
            if ( key and sbuff ):
                ret[key] = pd.DataFrame( sbuff, columns=columns )
                ret[key] = ret[key].apply( pd.to_numeric )
                sbuff    = []
            key      = line.strip().split()[-1]
        else:
            sbuff   += [ line.split() ]
    #  -- last element -- #
    if ( key and sbuff ):
        ret[key] = pd.DataFrame( sbuff, columns=columns )
        ret[key] = ret[key].apply( pd.to_numeric )
    return( ret )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    
    trackFile = "test/load__trackData/track.tfsone"
    Data      = load__trackData( trackFile=trackFile )
    print( Data )
