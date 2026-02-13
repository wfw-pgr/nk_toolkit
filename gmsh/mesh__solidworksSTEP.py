import os, sys, re, gmsh, json5, numbers
import numpy as np
import nk_toolkit.gmsh.assign__meshsize as ams

# ========================================================= #
# ===  mesh__solidworksSTEP.py                          === #
# ========================================================= #

def mesh__solidworksSTEP( stpFile=None, configFile="mesh.json" ):

    # ------------------------------------------------- #
    # --- [1] initialize                            --- #
    # ------------------------------------------------- #
    gmsh.initialize()
    gmsh.option.setNumber( "General.Terminal"         , 1 )
    gmsh.option.setNumber( "Mesh.Algorithm"           , 5 )
    gmsh.option.setNumber( "Mesh.Algorithm3D"         , 4 )
    gmsh.option.setNumber( "Geometry.OCCImportLabels" , 1 )
    gmsh.model.add( "model" )

    # ------------------------------------------------- #
    # --- [2] import models                         --- #
    # ------------------------------------------------- #
    if ( stpFile is None ):
        raise FileNotFoundError( "Cannot find file :: {}".format( stpFile ) )
    else:
        if ( not( os.path.exists( stpFile ) ) ):
            raise FileNotFoundError( "Cannot find file :: {}".format( stpFile ) )

    dimtags = gmsh.model.occ.importShapes( stpFile )
    gmsh.model.occ.synchronize()

    names   = [ (gmsh.model.getEntityName( dim,tag ) ).split("/")[-1] for dim,tag in dimtags ]
    numDict = { name:[] for name in names }
    for name,dimtag in zip( names, dimtags ):
        numDict[name]  += [ dimtag ]
    entities = { name:[ dimtag[1] for dimtag in numDict[name] ] for name in names }

    print( "\n ================    Import    ================" )
    for key,dimtags in numDict.items():
        print( "   * {0} :: {1}".format( key, dimtags ) )
    print( "==============================================" )
    print()
    
    # ------------------------------------------------- #
    # --- [3] mesh config                           --- #
    # ------------------------------------------------- #
    with open( configFile, "r" ) as f:
        config = json5.load( f )
    for key,item in config.items():
        if ( key in names ):
            item["entities"] = entities[key]
        else:
            print( "[mesh__solidworksSTEP.py] cannot find  {}".format( key ) )
    
    # ------------------------------------------------- #
    # --- [4] meshing                               --- #
    # ------------------------------------------------- #
    ams.assign__meshsize( config=config )
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
    gmsh.write( "test/model.msh" )
    gmsh.finalize()


# ========================================================= #
# ===   実行部                                          === #
# ========================================================= #

if ( __name__=="__main__" ):

    stpFile    = "test/全溶接コンバータ_全体アセンブリ.STEP"
    configFile = "mesh_forSTEP.json"
    mesh__solidworksSTEP( stpFile=stpFile, configFile=configFile )
