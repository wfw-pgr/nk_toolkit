import os, re, glob, subprocess, shutil, json5, pathlib
import invoke
import nk_toolkit.impactx.plot_toolkit    as ptk
import nk_toolkit.impactx.analyze_toolkit as atk
import nk_toolkit.impactx.convert_toolkit as ctk



# ========================================================= #
# ===  execute impactx                                  === #
# ========================================================= #
@invoke.task
def run( ctx, logFile="impactx.log" ):
    """Run the ImpactX simulation."""
    # -- run -- #
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
    # -- file copy -- #
    copyfiles = [ "dat/parameters.json", "dat/beamline_impactx.json" ]
    for copyfile in copyfiles:
        shutil.copy( copyfile, "impactx/diags/" )
    

        
        
# ========================================================= #
# ===  translate track -> impactx                       === #
# ========================================================= #
@invoke.task
def track2impactx( ctx ):
    """initialize and prepare the ImpactX simulation."""
    ctk.translate__track2impactx( paramsFile   = "dat/parameters.json", \
                                  trackFile    = "track/sclinac.dat" , \
                                  impactxBLFile= "dat/beamline_impactx.json", \
                                  trackBLFile  = "dat/beamline_track.json", \
                                  phaseFile    = "dat/rfphase.csv" )

    
    
# ========================================================= #
# ===  clean output files                               === #
# ========================================================= #
@invoke.task
def clean( ctx ):
    """Remove output files from previous runs."""
    patterns = [ "impactx/diags.old.*", "impactx/diags", "impactx/__pycache__",\
                 "impactx/impactx.log", "png/*" ]
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
def post( ctx, \
          refp=False, stat=False, post=False, poincare=False, \
          trajectory=False, all=False, ext=None, pcnfFile="dat/visualize.json" ):
    """Run post-analysis script."""

    # ------------------------------------------------- #
    # --- [1] load config                           --- #
    # ------------------------------------------------- #
    with open( pcnfFile, "r" ) as f:
        plot_conf = json5.load( f )

    # ------------------------------------------------- #
    # --- [2] modify file extension                 --- #
    # ------------------------------------------------- #
    if ( ext is None ):
        ext  = os.environ.get( "IMPACTX_REFP_EXTENSION", ".0.0" )
    plot_conf["files"]["refp"] = re.sub( r"(\.\d+)+$", "", plot_conf["files"]["refp"] ) + ext
    plot_conf["files"]["stat"] = re.sub( r"(\.\d+)+$", "", plot_conf["files"]["stat"] ) + ext

    if ( all ):
        refp, stat, post, poincare, trajectory = True, True, True, True, True
    
    # ------------------------------------------------- #
    # --- [3] call general visualization routine    --- #
    # ------------------------------------------------- #
    vis = ptk.visualize__main( refp=refp, stat=stat, poincare=poincare, post=post, \
                               trajectory=trajectory, plot_conf=plot_conf )

    # ------------------------------------------------- #
    # --- [4] merge png files                       --- #
    # ------------------------------------------------- #
    if ( refp ):
        fileList = [ "s-x", "s-y", "s-z", "s-t", "s-px", "s-py", "s-pz", "s-pt",
                     "s-beta", "s-gamma", "s-beta_gamma", ]
        cmd      = [ "magick", "montage" ] + [ f"png/refp/{afile}.png" for afile in fileList ] \
            + [ "-tile", "2x", "-geometry", "+0+0", "png/refp/refp-merged.png" ]
        subprocess.run( cmd, check=True )
        print( "[ magick append @tasks.py ] output :: png/refp/refp-merged.png" )
    if ( stat ):
        fileList = [ "s-xRange", "s-yRange", "s-tRange", "s-sigma_xy", "s-sigma_t", "s-alpha_xyt",
                     "s-emit_xyt", "s-emit_xytn", "s-beta_xy", "s-beta_t", ]
        cmd      = [ "magick", "montage" ] + [ f"png/stat/{afile}.png" for afile in fileList ] \
            + [ "-tile", "2x", "-geometry", "+0+0", "png/stat/stat-merged.png" ]
        subprocess.run( cmd, check=True )
        print( "[ magick append @tasks.py ] output :: png/stat/stat-merged.png" )
    if ( post ):
        fileList = [ "s-Ek_ref", "s-Ek", "s-dphi", "s-dp_p0", "s-dE_E0","s-sigma_phi",
                     "s-sigma_dp_p", "s-sigma_dE_E0", "s-transmission",
                     "s-max_sigma_xy", "s-max_sigma_t", "s-max_sigma_dphi",
                     "s-corr__xp-yp", "s-corr__xp-tp", "s-corr__xp-px", "s-corr__xp-py",
                     "s-corr__xp-pt", "s-corr__tp-pt", ]
        cmd      = [ "magick", "montage" ] + [ f"png/post/{afile}.png" for afile in fileList ] \
            + [ "-tile", "2x", "-geometry", "+0+0", "png/post/post-merged.png" ]
        subprocess.run( cmd, check=True )
        print( "[ magick append @tasks.py ] output :: png/post/post-merged.png" )
    if ( trajectory ):
        fileList = [ "s-xp", "s-yp", "s-tp" ]
        cmd      = [ "magick", "montage" ] \
            + [ f"png/trajectory/{afile}.png" for afile in fileList ] \
            + [ "-tile", "1x", "-geometry", "+0+0", "png/trajectory/trajectory-merged.png" ]
        subprocess.run( cmd, check=True )
        print( "[ magick append @tasks.py ] output :: png/trajectory/trajectory-merged.png" )
    if ( poincare ):
        plotNames = [ "xp-px", "yp-py", "dt-dE" ]

        pdir      = pathlib.Path( "png/poincare/" )
        indices   = sorted( [ afile.stem.rsplit("_", 1)[1]
                              for afile in pdir.glob(f"{plotNames[0]}_*.png") ], key=int, )
        if ( len( indices ) <= 5 ):
            fileList  = []
            for idx in indices:
                row        = [ pdir / f"{plotName}_{idx}.png" for plotName in plotNames ]
                missing    = [ str(afile) for afile in row if not afile.exists() ]
                if missing:
                    raise FileNotFoundError(f"missing poincare plot(s): {missing}")
                fileList  += [ str(afile) for afile in row ]
            cmd      = [ "magick", "montage" ] + fileList \
                + [ "-tile", "3x", "-geometry", "+0+0", "png/poincare/poincare-merged.png" ]
            subprocess.run( cmd, check=True )
            print( "[ magick append @tasks.py ] output :: png/poincare/poincare-merged.png" )

        else:
            animeList = []
            
            for idx in indices:
                row     = [ pdir / f"{plotName}_{idx}.png" for plotName in plotNames ]
                missing = [ str( afile ) for afile in row if not afile.exists() ]
                if ( missing ):
                    raise FileNotFoundError( f"missing poincare plot(s): {missing}" )
                outFile = pdir / f"merged_{idx}.png"
                cmd     = [ "magick", "montage" ] \
                    + [ str( afile ) for afile in row ] \
                    + [ "-tile", "3x", "-geometry", "+0+0", str( outFile ) ]
                subprocess.run( cmd, check=True )
                animeList += [ str( outFile ) ]
                
            cmd = [ "magick", "-delay", "30", "-loop", "0" ] \
                + animeList \
                + [ "png/poincare/poincare.gif" ]
            subprocess.run( cmd, check=True )
            print( "[ magick gif @tasks.py ] output :: png/poincare/poincare.gif" )

    



    
# ========================================================= #
# ===  all = clean + run + post                         === #
# ========================================================= #
@invoke.task(pre=[clean, run, post])
def all(ctx):
    """Run all steps: clean, impact, post."""
    pass



# ========================================================= #
# ===  calc twiss parameter from track.dat              === #
# ========================================================= #
@invoke.task()
def calctwiss( ctx, trackFile="track/track.dat" ):
    """calculate twiss parameter for impactx from track.dat"""
    ret = ctk.calculate__twissFromTrackv38( trackFile=trackFile, )
    print( "\n ------------- \n" )
    print( ret )
    print( "\n ------------- \n" )
    return( ret )



# ========================================================= #
# ===  convertDTLmap                                    === #
# ========================================================= #
@invoke.task()
def convertDTLmap( ctx ):
    """Convert DTL's map ( eh_DTL.#nn, eh_SOL.#nn -> efield.dat, bfield.dat )"""
    ef,bf = ctk.translate__TrackDTLmap2impactx( eh_DTL="track/lb/eh_DTL.#01", \
                                                eh_PMQ="track/lb/eh_SOL.#64", \
                                                efieldFile="dat/efield.csv" , \
                                                bfieldFile="dat/bfield.csv"   )
