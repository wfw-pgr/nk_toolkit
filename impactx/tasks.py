import os, glob, subprocess
import invoke
import importlib.util

# ========================================================= #
# ===  execute impactx                                  === #
# ========================================================= #
@invoke.task
def impactx( ctx, logFile="impactx.log" ):
    """Run the ImpactX simulation."""
    cwd = os.getcwd()
    try:
        os.chdir( "impactx/" )
        spec = importlib.util.spec_from_file_location( "main_impactx", "impactx/main_impactx.py" )
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module( mod )
        mod.main_impactx()
    finally:
        os.chdir( cwd )

        
# ========================================================= #
# ===  clean output files                               === #
# ========================================================= #
@invoke.task
def clean( ctx ):
    """Remove output files from previous runs."""
    patterns = [ "impactx/*.h5", "impactx/*.bp" ]
    for pattern in patterns:
        for file in glob.glob( pattern ):
            print(f"Removing {file}")
            os.remove(file)

            
# ========================================================= #
# ===  post analysis                                    === #
# ========================================================= #
@invoke.task
def post( ctx ):
    """Run post-analysis script."""
    cwd = os.getcwd()
    try:
        os.chdir( "impactx/" )
        import post_analysis
        post_analysis.post_analysis()
    finally:
        os.chdir( cwd )


# ========================================================= #
# ===  all = clean + impactx + post                      === #
# ========================================================= #
@invoke.task(pre=[clean, impactx, post])
def all(ctx):
    """Run all steps: clean, impact, post."""
    pass
