import os, sys, re, gmsh, json5, numbers
import numpy as np
import nk_toolkit.gmsh.assign__meshsize     as ams
import nk_toolkit.phits.convert__gmsh2phits as g2p
import nk_toolkit.phits.materials__fromJSON as mfj

# ========================================================= #
# ===  mesh__solidworksSTEP.py                          === #
# ========================================================= #

def mesh__solidworksSTEP( stpFile="msh/model.stp", configFile="dat/mesh.json", \
                          mshFile="msh/model.msh", bdfFile="msh/model.bdf", phits_mesh=False, \
                          matFile="dat/materials.json", \
                          materialPhitsFile="inp/materials.phits.j2", scale_unit="mm" ):

    # ------------------------------------------------- #
    # --- [1] mesh config                           --- #
    # ------------------------------------------------- #
    with open( configFile, "r" ) as f:
        config    = json5.load( f )
        if ( "options" in config ):
            options = config.pop( "options" )
        else:
            options = {}
    if   ( scale_unit == "m"  ):
        scale_unit = 1.0
    elif ( scale_unit == "cm" ):
        scale_unit = 1.0e-2
    elif ( scale_unit == "mm" ):
        scale_unit = 1.0e-3
    elif ( type( scale_unit ) is float ):
        pass
    else:
        raise TypeError( "[mesh__solidworksStep.py] scale_unit = ??  " )
            
    # ------------------------------------------------- #
    # --- [2] initialize                            --- #
    # ------------------------------------------------- #
    gmsh.initialize()
    gmsh.option.setNumber( "General.Terminal"         , 1    )
    gmsh.option.setNumber( "Mesh.Algorithm"           , 6    )
    gmsh.option.setNumber( "Mesh.Algorithm3D"         , 1   )
    gmsh.option.setNumber( "Mesh.Optimize"            , 1    )
    gmsh.option.setNumber( "Mesh.OptimizeNetgen"      , 1    )
    gmsh.option.setNumber( "Mesh.Smoothing"           , 3    )
    gmsh.option.setNumber( "Geometry.OCCImportLabels" , 1    )
    for key in options.keys():
        gmsh.option.setNumber( key, options[key] )
    gmsh.model.add( "model" )
            
    # ------------------------------------------------- #
    # --- [3] import models                         --- #
    # ------------------------------------------------- #
    if ( not( os.path.exists( stpFile ) ) ):
        raise FileNotFoundError( "Cannot find file :: {}".format( stpFile ) )
    if ( mshFile is None ):
        mshFile = ( os.path.splitext( stpFile ) )[0] + ".msh"

    dimtags = gmsh.model.occ.importShapes( stpFile )
    gmsh.model.occ.synchronize()
    names   = [ (gmsh.model.getEntityName( dim,tag ) ).split("/")[-1] for dim,tag in dimtags ]
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    if ( scale_unit != 1.0 ):
        all_ents = gmsh.model.getEntities( dim=3 )
        gmsh.model.occ.dilate( all_ents, 0,0,0, scale_unit,scale_unit,scale_unit  )
        gmsh.model.occ.synchronize()
    dimtags = gmsh.model.getEntities( dim=3 )
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
    gmsh.write( mshFile )
    gmsh.finalize()

    # ------------------------------------------------- #
    # --- [5] gmsh -> phits (.bdf)                  --- #
    # ------------------------------------------------- #
    if ( phits_mesh ):
        if ( os.path.exists( matFile ) ):
            with open( matFile, "r" ) as f:
                materials = json5.load( f )
        else:
            raise FileNotFoundError( "[mesh__solidworksSTEP.py] matFile={} ??".format( matFile ) )
        
        for key,item in config.items():
            if ( not( "density" in item ) ):
                if ( "material" in item ):
                    if ( item["material"] in materials ):
                        item["density"] = materials[ item["material"] ]["Density"]
                    else:
                        raise KeyError( "[mesh__solidworksSTTEP.py] Cannot find key :: {}"\
                                        .format(item["material"] ) )
                else:
                    raise KeyError( "[mesh__solidworksSTTEP.py] Cannot find material in key :: {}"\
                                    .format( key ) )
        matKeys = []
        for key,item in config.items():
            if ( key in names ):
                matKeys += [ item["material"] ]
                
        g2p.convert__gmsh2phits( mshFile=mshFile, bdfFile=bdfFile, \
                                 config=config )
        import meshio
        rmesh          = meshio.read( bdfFile )
        unq,idx        = np.unique( rmesh.cell_data["nastran:ref"], return_index=True )
        physNums_order = unq[ np.argsort( idx ) ]
        matKeys        = [ matKeys[ ik-1 ] for ik in physNums_order ]
        mfj.materials__fromJSON( matFile=matFile, outFile=materialPhitsFile, \
                                 keys=matKeys, tetra_auto_mat=True )

        
# ========================================================= #
# ===   実行部                                          === #
# ========================================================= #

if ( __name__=="__main__" ):

    stpFile    = "test/test.stp"
    configFile = "test/mesh.json"
    phits_mesh = False
    mesh__solidworksSTEP( stpFile=stpFile, configFile=configFile, phits_mesh=phits_mesh )
