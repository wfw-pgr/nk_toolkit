import os, sys, subprocess, time, glob, json5, shutil
import invoke
import meshio, jinja2
import numpy                                   as np
import datetime                                as dt
import nkUtilities.show__section               as sct
import nk_toolkit.phits.materials__fromJSON    as mfj
import nk_toolkit.phits.tetra_toolkit          as ttk
import nk_toolkit.gmsh.mesh__solidworksSTEP    as mss
import nk_toolkit.legacy.json__formulaParser   as jso
import nk_toolkit.RI.track__RIactivity         as tri
import nk_toolkit.RI.integrate__RIprodReaction as irr


# ========================================================= #
# ===  meshing STEP file from solidworks output         === #
# ========================================================= #
@invoke.task(
    help={
        "stpFile"    : ".stp file ( solidworks )",
        "mshFile"    : ".msh file ( gmsh )",
        "bdfFile"    : ".bdf file ( phits )",
        "configFile" : ".json file",
        "phits_mesh" : "Enable PHITS mesh mode ( true/false or --phits-mesh/--no-phits-mesh )",
    }
)
def mesh( context, stpFile="msh/model.stp", \
          configFile="dat/mesh.json", mshFile="msh/model.msh", matFile="dat/materials.json", \
          bdfFile="msh/model.bdf", materialPhitsFile="inp/materials.phits.j2", phits_mesh=False ):
    """
    run mesh__solidworksSTEP( stpFile=.., mshFile=.., bdfFile=.., configFile=.., phits_mesh=.., )
    """
    mss.mesh__solidworksSTEP( stpFile=stpFile, configFile=configFile, matFile=matFile, \
                              mshFile=mshFile, bdfFile=bdfFile, \
                              materialPhitsFile=materialPhitsFile, phits_mesh=phits_mesh )


    
# ========================================================= #
# ===  build PHITS input files                          === #
# ========================================================= #
@invoke.task(
    help={
        "targetFile"        : "main file to be built. ( inp/main.phits.j2 )", \
        "paramsFile"        : "json parameter file. ( dat/parameters.json )", \
        "materialFile"      : "materials.json ( dat/materials.json )", \
        "materialPhitsFile" : "materials file for phits ( inp/materials.phits.j2 )", \
        "meshFile"          : "mesh config file for gmsh ( dat/mesh.json )", \
        "phits_mesh"        : "Enable gmsh-phits mode ( T/F or --phits-mesh/--no-phits-mesh )", \
        "executeFile"       : "execute file for PHITs ( inp/execute.phits.inp )", \
    }
)
def build( ctx, \
           targetFile  ="inp/main.phits.j2" , paramsFile ="dat/parameters.json", \
           materialFile="dat/materials.json", materialPhitsFile="inp/materials.phits.j2", \
           meshFile    ="dat/mesh.json"     , phits_mesh=False, \
           bdfFile     ="msh/model.bdf", \
           executeFile ="inp/execute.phits.inp", \
          ) -> None:

    
    rootDir     = "./"

    # ------------------------------------------------- #
    # --- [1] file existence check                  --- #
    # ------------------------------------------------- #
    if ( os.path.exists( targetFile ) is False ):
        raise FileNotFoundError( "[tasks.py] targetFile = ?? :: {}".format( targetFile ) )
    if ( os.path.exists( paramsFile ) is False ):
        raise FileNotFoundError( "[tasks.py] paramsFile = ?? :: {}".format( paramsFile ) )
    if ( os.path.exists( materialFile ) is False ):
        raise FileNotFoundError( "[tasks.py] materialFile = ?? :: {}".format( materialFile ) )
    
    # ------------------------------------------------- #
    # --- [2] precompile PHITS input files          --- #
    # ------------------------------------------------- #
    sct.show__section( "Conversion :: .phits.j2  >> .phits.inp file.", length=71 )
    if ( phits_mesh ):
        with open( meshFile, "r" ) as f:
            meshconfig = json5.load( f )
            if ( "options" in meshconfig ):
                options = meshconfig.pop( "options" )
        matKeys        = [ item["material"] for item in meshconfig.values() ]
        if ( bdfFile is not None ):
            rmesh          = meshio.read( bdfFile )
            unq,idx        = np.unique( rmesh.cell_data["nastran:ref"], return_index=True )
            physNums_order = unq[ np.argsort( idx ) ]
            matKeys        = [ matKeys[ ik-1 ] for ik in physNums_order ]
        material_dn = mfj.materials__fromJSON( matFile=materialFile, \
                                               outFile=materialPhitsFile, \
                                               keys=matKeys, tetra_auto_mat=True )
    else:
        material_dn = mfj.materials__fromJSON( matFile=materialFile )

    # ------------------------------------------------- #
    # --- [3] load json                             --- #
    # ------------------------------------------------- #
    prm = jso.json__formulaParser( inpFile=paramsFile, table=material_dn, \
                                   variable_mark="@" )
    
    # ------------------------------------------------- #
    # --- [4] jinja2 rendering                      --- #
    # ------------------------------------------------- #
    environ  = jinja2.Environment( loader=jinja2.FileSystemLoader( rootDir ),
                                   undefined=jinja2.StrictUndefined,  # 未定義変数は即エラー
                                   autoescape=False, keep_trailing_newline=True )
    template = environ.get_template( targetFile )
    rendered = template.render( prm=prm )
    
    # ------------------------------------------------- #
    # --- [5] save as execute file                  --- #
    # ------------------------------------------------- #
    with open( executeFile, "w", encoding="utf-8" ) as f:
        f.write( rendered )
    print( f" [OK] rendered:   {targetFile} -> {executeFile} " )
    return()

    
# ========================================================= #
# ===  run PHITS calculation                            === #
# ========================================================= #
@invoke.task(
    help={
        "phits_cmd" : "phits command, depending on the environment. ( --phits-cmd='phits.sh' )",
        "exeFile"   : "phits execution file, built by above command ( inp/execute.phits.inp )",
    }

)
def run( ctx, phits_cmd="phits.sh", exeFile="inp/execute.phits.inp" ):
    
    # ------------------------------------------------- #
    # --- [1] run command                           --- #
    # ------------------------------------------------- #
    phits_cmd = "{0} {1}".format( phits_cmd, exeFile )
    print( "\nrun command :: {}\n".format( phits_cmd ) )
    
    # ------------------------------------------------- #
    # --- [2] run PHITS calculation                 --- #
    # ------------------------------------------------- #
    sct.show__section( "PHITS calculation Begin", length=71 )
    stime   = time.time()
    ret     = subprocess.run( phits_cmd, shell=True )
    etime   = time.time()
    elapsed = etime - stime
    hms     = dt.datetime.strftime( dt.datetime.utcfromtimestamp( elapsed ),'%H:%M:%S' )
    print( "\n" + "[tasks.py] elapsed time :: {} ".format( hms ) + "\n" )


    
# ========================================================= #
# ===  post-process of the calculation                  === #
# ========================================================= #
@invoke.task(
    help={
        "factor"     : "factor of cell2point conversion for Femtet.",
        "phits_mesh" : "Enable gmsh-phits mode ( T/F or --phits-mesh/--no-phits-mesh )", \
    }
)
def post( ctx, factor=1.0, phits_mesh=False ):
    
    # ------------------------------------------------- #
    # --- [1] post execution commands               --- #
    # ------------------------------------------------- #
    command1 = "for f in `ls out/*.eps`; do gs -dSAFER -dEPSCrop "\
        "-sDEVICE=pdfwrite -o ${f%.eps}_%d.pdf ${f};done"
    command2 = "magick mogrify -background white -alpha off -density 400 "\
        "-resize 50%x50% -path png -format png out/*.pdf"
    subprocess.run( command1, shell=True )
    subprocess.run( command2, shell=True )

    # ------------------------------------------------- #
    # --- [2] map csv on vtu                        --- #
    # ------------------------------------------------- #
    if ( phits_mesh ):
        mshFile       = "msh/model.msh"
        cellDataFiles = [ "out/heatload.csv" ]
        ret           = ttk.map__csvOnMesh( mshFile=mshFile, cellDataFiles=cellDataFiles )

    # ------------------------------------------------- #
    # --- [3] redistribute cell data on points      --- #
    # ------------------------------------------------- #
    if ( phits_mesh ):
        key     = "Dose[J/m^3/source]"
        shape   = "tetra"
        mshFile = "msh/model.msh"
        csvFile = "out/heatload.csv"
        hNums   = [ ik for ik in range( 1,8+1 ) ]
        for hNum in hNums:
            outFile = "dat/heatload_phys-{}.csv".format( hNum )
            ttk.redistribute__cell2point_forFemtet( mshFile=mshFile, csvFile=csvFile, \
                                                    key=key, shape=shape, factor=factor, \
                                                    outFile=outFile, target_physNum=hNum )
            print( "output :: {}".format( outFile ) )
    return()
    


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
# ===  build run post-process :: phits calculation      === #
# ========================================================= #
@invoke.task
def all( ctx, \
         inpFile="inp/main_phits.inp", materialFile="inp/materials.json", \
         stpFile="msh/model.stp", phits_mesh=False, \
         configFile="dat/mesh.json", mshFile="msh/model.msh", bdfFile="msh/model.bdf", \
         exeFile="inp/execute.phits.inp", phits_cmd="phits.sh" ):
    if ( phits_mesh ):
        mesh( ctx, stpFile=stpFile, \
              configFile=configFile, mshFile=mshFile, bdfFile=bdfFile, phits_mesh=phits_mesh )
    build( ctx, inpFile=inpFile, materialFile=materialFile, exeFile=exeFile )
    run  ( ctx, phits_cmd=phits_cmd, exeFile=exeFile )
    post ( ctx )
    integrate( ctx )
    track( ctx )
    return()
