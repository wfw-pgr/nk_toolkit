import os, glob, subprocess, shutil, json5
import invoke
import nk_toolkit.RI.track__RIactivity         as tri
import nk_toolkit.RI.integrate__RIprodReaction as irr


# ========================================================= #
# ===  run track__RIactivity.py                         === #
# ========================================================= #
@invoke.task
def track( ctx, settingsFile="dat/settings-trackRI.json" ):
    """Run the track__RIactivity.py"""
    tri.track__RIactivity( settingsFile=settingsFile )


# ========================================================= #
# ===  run integrate__RIprodReaction.py                 === #
# ========================================================= #
@invoke.task
def integrate( ctx, settingsFile="dat/RIprod_Ra226gn.json" ):
    """Run the integrate__RIprodReaction.py"""
    irr.integrate__RIprodReaction( settingsFile=settingsFile )

    
# ========================================================= #
# ===  clean output files                               === #
# ========================================================= #
@invoke.task
def clean( ctx ):
    """Remove output files from previous runs."""
    patterns = [ "png/*.png", \
                 "dat/results__*.json", "dat/summary__*.dat" , "dat/dYield_vs_energy__*.dat", \
                 "dat/summary__*.json", "dat/results__*.csv" , \
                ]
    for pattern in patterns:
        for path in glob.glob( pattern ):
            if os.path.isfile(path):
                print( f"Removing file {path}" )
                os.remove(path)
            elif os.path.isdir(path):
                print( f"Removing directory {path}" )
                shutil.rmtree(path)
            

# ========================================================= #
# ===  all = clean + run + post                         === #
# ========================================================= #
@invoke.task(pre=[clean, integrate, track])
def all(ctx):
    """Run all steps: clean, run"""
    pass
