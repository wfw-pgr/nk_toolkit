import os, sys, re, json5

# ========================================================= #
# ===  load config.json to plot settings                === #
# ========================================================= #

def load__config( config=None ):

    dirname = os.path.dirname( os.path.abspath( __file__ ) )

    # ------------------------------------------------- #
    # --- [1] load config file                      --- #
    # ------------------------------------------------- #
    if ( config is None ):
        config  = os.path.join( dirname, "config.json" )
        with open( config, "r" ) as f:
            const = json5.load( f )
    return( const )
    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #
if ( __name__=="__main__" ):

    config = load__config()
    print( config )
    
