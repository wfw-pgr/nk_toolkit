#!/usr/bin/env python3
import os, sys, json5, jinja2, pathlib, invoke
import nkUtilities.show__section as sct
import nk_toolkit.phits.materials__fromJSON  as mfj


# # ========================================================= #
# # ===  build PHITS input files                          === #
# # ========================================================= #
# @invoke.task(
#     help={
#         "inpFile"                : "main file to be built. ( inp/main.phits.inp )", \
#         "materialFile"           : "materials.json ( dat/materials.json )", \
#         "materialPhitsInputFile" : "materials_phits.inp ( inp/materials_phits.inp )", \
#         "exeFile"                : "execute file for PHITs ( inp/execute_phits.inp )", \
#         "phits_mesh"             : "Enable PHITs mesh mode ( True/False or --phits-mesh/--no-phits-mesh )", \
#     }
# )
# def build( ctx, \
#            inpFile="inp/main_phits.inp", \
#            materialFile="dat/materials.json", \
#            materialPhitsInputFile="inp/materials_phits.inp", \
#            exeFile="inp/execute_phits.inp", phits_mesh=False ):
    
#     # ------------------------------------------------- #
#     # --- [1] file existence check                  --- #
#     # ------------------------------------------------- #
#     if ( os.path.exists( inpFile ) is False ):
#         raise FileNotFoundError( "[tasks.py] inpFile = ?? :: {}".format( inpFile ) )
#     if ( os.path.exists( materialFile ) is False ):
#         raise FileNotFoundError( "[tasks.py] materialFile = ?? :: {}".format( materialFile ) )
    
#     # ------------------------------------------------- #
#     # --- [2] precompile PHITS input files          --- #
#     # ------------------------------------------------- #
#     sct.show__section( "Conversion :: _phits.inp >> .inp File", length=71 )
#     if ( phits_mesh ):
#         configFile = "dat/mesh.json"
#         with open( configFile, "r" ) as f:
#             config = json5.load( f )
#         matKeys     = [ item["material"] for item in config.values() ]
#         material_dn = mfj.materials__fromJSON( matFile=materialFile, outFile=materialPhitsInputFile, \
#                                                keys=matKeys, tetra_auto_mat=True )
#     else:
#         material_dn = mfj.materials__fromJSON( matFile=materialFile )
    
    

# ========================================================= #
# ===  render.py                                        === #
# ========================================================= #

def build( ctx, \
           targetFile  ="inp/main.phits.j2" , paramsFile ="dat/parameters.json", \
           materialFile="dat/materials.json", materialPhitsInputFile="inp/materials.phits.j2", \
           meshFile    = "dat/mesh.json"    , phits_mesh=False, \
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
        matKeys     = [ item["material"] for item in meshconfig.values() ]
        material_dn = mfj.materials__fromJSON( matFile=materialFile, \
                                               outFile=materialPhitsInputFile, \
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
    

if __name__ == "__main__":
    ctx=None
    build( ctx )
