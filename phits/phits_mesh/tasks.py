import invoke
import os, sys, subprocess, time, json5
import datetime                              as dt
import nkUtilities.show__section             as sct
import nkUtilities.precompile__parameterFile as ppf
import nkUtilities.command__postProcess      as cpp
import nk_toolkit.phits.materials__fromJSON  as mfj

import os, sys, json5, jinja2, pathlib, invoke
import nkUtilities.show__section as sct
import nk_toolkit.phits.materials__fromJSON  as mfj


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
    import nk_toolkit.gmsh.mesh__solidworksSTEP as mss
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
        matKeys     = [ item["material"] for item in meshconfig.values() ]
        material_dn = mfj.materials__fromJSON( matFile=materialFile, bdfFile=bdfFile, \
                                               outFile=materialPhitsFile, \
                                               keys=matKeys, tetra_auto_mat=True )
    else:
        material_dn = mfj.materials__fromJSON( matFile=materialFile )

    # ------------------------------------------------- #
    # --- [3] load json                             --- #
    # ------------------------------------------------- #
    import nkUtilities.json__formulaParser as jso
    params = jso.json__formulaParser( inpFile=paramsFile, table=material_dn, \
                                      variable_mark="@" )
    
    # ------------------------------------------------- #
    # --- [4] jinja2 rendering                      --- #
    # ------------------------------------------------- #
    environ  = jinja2.Environment( loader=jinja2.FileSystemLoader( rootDir ),
                                   undefined=jinja2.StrictUndefined,  # 未定義変数は即エラー
                                   autoescape=False, keep_trailing_newline=True )
    template = environ.get_template( targetFile )
    rendered = template.render( params=params )

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
@invoke.task
def post( ctx ):
    
    # ------------------------------------------------- #
    # --- [1] post execution commands               --- #
    # ------------------------------------------------- #
    # command1 = "for f in `ls out/*.eps`; do gs -dSAFER -dEPSCrop "\
    #     "-sDEVICE=pdfwrite -o ${f%.eps}_%d.pdf ${f};done"
    # command2 = "mogrify -background white -alpha off -density 400 "\
    #     "-resize 50%x50% -path png -format png out/*.pdf"
    # subprocess.run( command1, shell=True )
    # subprocess.run( command2, shell=True )

    # ------------------------------------------------- #
    # --- [2] map csv on vtu                        --- #
    # ------------------------------------------------- #
    import nk_toolkit.phits.tetra_toolkit as ttk
    mshFile       = "msh/model.msh"
    cellDataFiles = [ "out/heatload.csv" ]
    ret           = ttk.map__csvOnMesh( mshFile=mshFile, cellDataFiles=cellDataFiles )
    return()
    


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
    return()
