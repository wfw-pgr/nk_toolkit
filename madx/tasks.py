import os, sys, glob
import invoke
import cpymad.madx
import nk_toolkit.madx.load__tfs        as ltf
import nk_toolkit.madx.plotly__beamline as pbl

# ========================================================= #
# ===  madx command                                     === #
# ========================================================= #
@invoke.task( help={ 'file': '.madx file == input file.'} )
def madx( c, file="main.madx" ):
    """execute madx calculation."""
    # ------------------------------------------------- #
    # --- [1] file check                            --- #
    # ------------------------------------------------- #
    if not( os.path.exists(file) ):
        filelist = glob.glob( "./*.madx" )
        if ( len(filelist) == 1 ):
            file = filelist[0]
            print( f" following .madx file will be used... :: {file}" )
        else:
            sys.exit(" cannot find input file :: {}".format( file ))
    else:
        print( f"\n --- input : {file} --- \n" )
    # ------------------------------------------------- #
    # --- [2] execute madx                          --- #
    # ------------------------------------------------- #
    m = cpymad.madx.Madx()
    try:
        m.call( file )
        print( "\n --- End of Execution --- \n" )
    except:
        print( "Error" )
    return()


# ========================================================= #
# ===  plot command                                     === #
# ========================================================= #
@invoke.task
def plot( c, surveyFile="out/survey.tfs", twissFile="out/twiss.tfs", html="out/plot.html" ):
    """plot analysis, like plot."""
    # ------------------------------------------------- #
    # --- [1] file check                            --- #
    # ------------------------------------------------- #
    if not( os.path.exists(surveyFile) ):
        sys.exit(" cannot find survey file :: {:<30}".format( surveyFile ) )
    else:
        print( f" --- surveyFile : {surveyFile}" )
    if not( os.path.exists(twissFile) ):
        sys.exit(" cannot find survey file :: {:<30}".format( twissFile ) )
    else:
        print( f" ---  twissFile : {twissFile}" )
    # ------------------------------------------------- #
    # --- [2] execute plot analysis                 --- #
    # ------------------------------------------------- #
    survey     = ltf.load__tfs( tfsFile=surveyFile )
    twiss      = ltf.load__tfs( tfsFile= twissFile )
    pbl.plotly__beamline( survey=survey, twiss=twiss, html=html )
    return()
        
    
# ========================================================= #
# ===  clean command                                    === #
# ========================================================= #
@invoke.task
def clean(c):
    """clean madx outputs."""    
    extensions = [".twiss", ".track", ".table", ".log", ".out"]
    deleted    = 0
    for file in os.listdir("."):
        if any( file.endswith( ext ) for ext in extensions ):
            os.remove(file)
            deleted += 1
    print(f" --- delte : {deleted} files --- " )
    return()


