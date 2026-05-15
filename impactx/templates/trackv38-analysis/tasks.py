import json5, glob, os, shutil
import invoke
import nk_toolkit.impactx.trackv38_toolkit as ttk


# ========================================================= #
# ===  plot                                             === #
# ========================================================= #
@invoke.task
def plot( ctx, \
          stat=False, poincare=False, all=False, ext=None, pcnfFile="dat/visualize.json" ):
    """Run plot script."""
    
    with open( pcnfFile, "r" ) as f:
        plot_conf = json5.load( f )
    ttk.visualize__main( stat=stat, poincare=poincare, plot_conf=plot_conf )

    

# ========================================================= #
# ===  compare                                          === #
# ========================================================= #
@invoke.task
def compare( ctx, \
             pcnfFile="dat/visualize.json" ):
    """compare script."""
    
    with open( pcnfFile, "r" ) as f:
        plot_conf = json5.load( f )
    trackFile    = plot_conf["files"]["statistics"]
    impactxFile  = plot_conf["files"]["impactx_stat"]
    ttk.compare__impactx_vs_trackv38( trackFile=trackFile, impactxFile=impactxFile, \
                                      plot_conf=plot_conf )

    



# ========================================================= #
# ===  clean output files                               === #
# ========================================================= #
@invoke.task
def clean( ctx ):
    """Remove output files from previous runs."""
    patterns = [ "png/*" ]
    for pattern in patterns:
        for path in glob.glob( pattern ):
            if os.path.isfile(path):
                print( f"Removing file {path}" )
                os.remove(path)
            elif os.path.isdir(path):
                print( f"Removing directory {path}" )
                shutil.rmtree(path)


