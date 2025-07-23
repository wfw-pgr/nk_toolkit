import os, glob
import invoke


# ========================================================= #
# ===  execute impactx                                  === #
# ========================================================= #
@invoke.task
def impactx( ctx ):
    """Run the ImpactX simulation."""
    cwd = os.getcwd()
    try:
        os.chdir( "impactx/" )
        import main_impactx
        main_impactx.main_impactx()
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
# ===  all = clean + impact + post                      === #
# ========================================================= #
@invoke.task(pre=[clean, impact, post])
def all(ctx):
    """Run all steps: clean, impact, post."""
    pass
