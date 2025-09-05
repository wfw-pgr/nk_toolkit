import os, glob, subprocess, shutil, json5
import invoke
import nk_toolkit.RI.track__RIactivity as tri


# ========================================================= #
# ===  run track__RIactivity.py                         === #
# ========================================================= #
@invoke.task
def run( ctx, settingFile="dat/settings.json" ):
    """Run the track__RIactivity.py"""
    tri.track__RIactivity( settingFile=settingFile )
    
    
# ========================================================= #
# ===  clean output files                               === #
# ========================================================= #
@invoke.task
def clean( ctx ):
    """Remove output files from previous runs."""
    patterns = [ "png/*.png", "dat/results.dat", ]
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
@invoke.task(pre=[clean, run])
def all(ctx):
    """Run all steps: clean, run"""
    pass
