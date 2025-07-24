import os, glob, subprocess
import invoke
import nk_toolkit.impactx.impactx_toolkit as itk

# ========================================================= #
# ===  execute impactx                                  === #
# ========================================================= #
@invoke.task
def run( ctx, logFile="impactx.log" ):
    """Run the ImpactX simulation."""
    cwd = os.getcwd()
    cmd = "python main_impactx.py"
    try:
        os.chdir( "impactx/" )
        with open("impactx.log", "w") as log:
            process = subprocess.Popen( cmd.split(), \
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, \
                text=True, bufsize=1 )
            for line in process.stdout:
                print( line, end="" )  # terminal stdout
                log.write( line )      # save in log file
        process.wait()
    finally:
        os.chdir( cwd )

        
# ========================================================= #
# ===  clean output files                               === #
# ========================================================= #
@invoke.task
def clean( ctx ):
    """Remove output files from previous runs."""
    patterns = [ "impactx/diags.old.*", "impactx/diags", "impactx/__pychache__" ]
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
    # ------------------------------- #
    # --- [1] reference particle  --- #
    # ------------------------------- #
    refpFile  = "impactx/diags/ref_particle.0.0"
    itk.plot__refparticle( inpFile=refpFile )

    # ------------------------------- #
    # --- [2] statistics          --- #
    # ------------------------------- #
    statFile  = "impactx/diags/reduced_beam_characteristics.0.0"
    itk.plot__statistics( inpFile=statFile )

    # ------------------------------- #
    # --- [3] trajectory          --- #
    # ------------------------------- #
    hdf5File      = "impactx/diags/openPMD/bpm.h5"
    refpFile      = "impactx/diags/ref_particle.0.0"
    random_choice = 300
    itk.plot__trajectories( hdf5File=hdf5File, refpFile=refpFile, \
                            random_choice=random_choice )

    # ------------------------------- #
    # --- [4] convert to vtk      --- #
    # ------------------------------- #
    hdf5File = "impactx/diags/openPMD/bpm.h5"
    outFile  = "png/bpm.vtp"
    itk.convert__hdf2vtk( hdf5File=hdf5File, outFile=outFile )
    return()


# ========================================================= #
# ===  all = clean + run + post                         === #
# ========================================================= #
@invoke.task(pre=[clean, run, post])
def all(ctx):
    """Run all steps: clean, impact, post."""
    pass
