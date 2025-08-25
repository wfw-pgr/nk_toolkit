import os, glob, subprocess, shutil, json5
import invoke
import nk_toolkit.impactx.impactx_toolkit          as itk
import nk_toolkit.impactx.translate__track2impactx as t2i

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
# ===  initilize & prepare                              === #
# ========================================================= #
@invoke.task
def prepare( ctx ):
    """initialize and prepare the ImpactX simulation."""
    paramsFile = "dat/parameters.json"
    ret        = t2i.translate__track2impactx( paramsFile=paramsFile )

    
# ========================================================= #
# ===  clean output files                               === #
# ========================================================= #
@invoke.task
def clean( ctx ):
    """Remove output files from previous runs."""
    patterns = [ "impactx/diags.old.*", "impactx/diags", "impactx/__pycache__",\
                 "png/*.png" ]
    for pattern in patterns:
        for path in glob.glob( pattern ):
            if os.path.isfile(path):
                print( f"Removing file {path}" )
                os.remove(path)
            elif os.path.isdir(path):
                print( f"Removing directory {path}" )
                shutil.rmtree(path)
            
# ========================================================= #
# ===  post analysis                                    === #
# ========================================================= #
@invoke.task
def post( ctx, paramsFile="dat/parameters.json", \
          refp=True, stat=True, trajectory=False, vtk=False, postProcessed=True, \
          ext=".0.0" ):
    """Run post-analysis script."""

    if ( paramsFile is not None ):
        with open( paramsFile, "r" ) as f:
            params = json5.load( f )
    else:
        params = { "plot.conf.refp":None, \
                   "plot.conf.stat":None, \
                   "plot.conf.traj":None  }
        
    # ------------------------------- #
    # --- [1] reference particle  --- #
    # ------------------------------- #
    if ( refp ):
        refpFile  = "impactx/diags/ref_particle" + ext
        itk.plot__refparticle( inpFile=refpFile, plot_conf=params["plot.conf.refp"] )

    # ------------------------------- #
    # --- [2] statistics          --- #
    # ------------------------------- #
    if ( stat ):
        statFile  = "impactx/diags/reduced_beam_characteristics" + ext
        itk.plot__statistics( inpFile=statFile, plot_conf=params["plot.conf.stat"]  )

    # ------------------------------- #
    # --- [3] trajectory          --- #
    # ------------------------------- #
    if ( trajectory ):
        hdf5File      = "impactx/diags/openPMD/bpm.h5"
        refpFile      = "impactx/diags/ref_particle" + ext
        random_choice = 300
        itk.plot__trajectories( hdf5File=hdf5File, refpFile=refpFile, \
                                random_choice=random_choice, plot_conf=params["plot.conf.traj"] )

    # ------------------------------- #
    # --- [4] convert to vtk      --- #
    # ------------------------------- #
    if ( vtk ):
        hdf5File = "impactx/diags/openPMD/bpm.h5"
        outFile  = "png/bpm.vtp"
        itk.convert__hdf2vtk( hdf5File=hdf5File, outFile=outFile )

    # ------------------------------------------------- #
    # --- [5] additional analysis                   --- #
    # ------------------------------------------------- #
    if ( postProcessed ):
        refpFile   = None   # auto 
        statFile   = None
        paramsFile = "dat/parameters.json"
        plot_conf  = None
        itk.postProcessed__beam( refpFile=refpFile, statFile=statFile, paramsFile=paramsFile )
        itk.plot__postProcessed( paramsFile=paramsFile, plot_conf=plot_conf )
    return()


# ========================================================= #
# ===  all = clean + run + post                         === #
# ========================================================= #
@invoke.task(pre=[clean, run, post])
def all(ctx):
    """Run all steps: clean, impact, post."""
    pass
