import os, sys, re, gmsh, json5, numbers
import numpy as np

# ========================================================= #
# ===  assing mesh size                                 === #
# ========================================================= #

def assign__meshsize( config =None, configFile="mesh.json", \
                      uniform=None, default_geotype="volume" ):
    
    # ------------------------------------------------- #
    # --- [1] uninform mesh settings                --- #
    # ------------------------------------------------- #
    if ( uniform is not None ):
        if ( uniform == 0.0 ):
            xyzMinMax = gmsh.model.getBoundingBox( -1, -1 )
            xL        = ( xyzMinMax[3] - xyzMinMax[0] )
            yL        = ( xyzMinMax[4] - xyzMinMax[1] )
            zL        = ( xyzMinMax[5] - xyzMinMax[2] )
            typLength = np.max( np.array( [ xL, yL, zL ] ) )
            uniform   = typLength / 50.0
        if ( isinstance( uniform, numbers.Real ) ):
            gmsh.option.setNumber( "Mesh.CharacteristicLengthMin", uniform )
            gmsh.option.setNumber( "Mesh.CharacteristicLengthMax", uniform )
        else:
            raise( TypeError( "uniform is not real." ) )
        return( uniform )

    
    # ------------------------------------------------- #
    # --- [2] mesh settings                         --- #
    # ------------------------------------------------- #
    if ( config is None ):
        if ( os.path.exists( configFile ) ):
            with open( configFile, "r" ) as f:
                config = json5.load( f )
        else:
            raise FileNotFoundError( "Cannot Find File :: {}".format( configFile ) )


    # ------------------------------------------------- #
    # --- [3] check unknown geometries              --- #
    # ------------------------------------------------- #
    default_dim = ( ["points","line","surface","volume"] ).index( default_geotype )
    found_ents  = sorted( gmsh.model.occ.getEntities( dim=default_dim ) )
    given_ents  = ( np.concatenate( [ val["entities"] for key,val in config.items() ] ) ).tolist()
    given_ents  = sorted( [ ( default_dim,ent ) for ent in given_ents ] )
    missing     = set( found_ents ) - set( given_ents )
    extra       = set( given_ents ) - set( found_ents )
    print()
    print( "  -- found_ents :: {}".format( found_ents ) )
    print( "  -- given_ents :: {}".format( given_ents ) )
    print( "  -- missing    :: {}".format( missing    ) )
    print( "  -- extra      :: {}".format( extra      ) )
    print()
    if ( len( missing ) > 0 ):
        raise ValueError( "Entity info. are missing...  :: {}".format( missing ) )
    if ( len( extra   ) > 0 ):
        raise ValueError( "Entity info. are too many... :: {}".format( extra   ) )
    
    # ------------------------------------------------- #
    # --- [4] assign mesh size                      --- #
    # ------------------------------------------------- #
    matNames    = list( dict.fromkeys([ val.get("matNum",key) for key,val in config.items() ]) )
    matNumTable = { names:(matNum+1) for matNum,names in enumerate(matNames) }
    meshsizes   = [ tab["meshsize"] for key,tab in config.items() ]
    fieldlist   = []
    for key,content in config.items():
        if ( not( "type" in content ) ):
            content["type"] = "volume"
        geotype    = content.get( "type"   , default_geotype )
        name       = content.get( "matName", key         )
        dim        = ( ["points","line","surface","volume"] ).index( geotype )
        ret        = gmsh.model.addPhysicalGroup( dim, content["entities"], \
                                                  tag=int(matNumTable[name]), name=name )
        dimtags_   = [ (dim,ent) for ent in content["entities"] ]
        ret        = assign__meshsize_on_each_dimtags( dimtags =dimtags_, \
                                                       meshsize=content["meshsize"] )
        fieldlist += [ ret ]

        
    # ------------------------------------------------- #
    # --- [5] define total field                    --- #
    # ------------------------------------------------- #
    totalfield = gmsh.model.mesh.field.add( "Min" )
    gmsh.model.mesh.field.setNumbers( totalfield, "FieldsList", fieldlist )
    gmsh.model.mesh.field.setAsBackgroundMesh( totalfield )

    gmsh.option.setNumber( "Mesh.CharacteristicLengthMin", min( meshsizes ) )
    gmsh.option.setNumber( "Mesh.CharacteristicLengthMax", max( meshsizes ) )

    
    # ------------------------------------------------- #
    # --- [6] return                                --- #
    # ------------------------------------------------- #
    ret = [ fieldlist, totalfield ]
    return( ret )



# ========================================================= #
# ===  assigne meshsize onto each dimtags               === #
# ========================================================= #
def assign__meshsize_on_each_dimtags( dimtags=None, meshsize=None, target="volu" ):
    
    # ------------------------------------------------- #
    # --- [1] Arguments                             --- #
    # ------------------------------------------------- #
    if ( dimtags  is None ): sys.exit( "[assign__meshsize_on_each_volume] dimtags  == ??? " )
    if ( meshsize is None ): sys.exit( "[assign__meshsize_on_each_volume] meshsize == ??? " )
    if ( ( dimtags[0] )[0] == 2 ): target = "surf"
        
    
    # ------------------------------------------------- #
    # --- [2] define MathEval Field                 --- #
    # ------------------------------------------------- #
    mathEval  = "{0}".format( meshsize )
    fieldmath = gmsh.model.mesh.field.add( "MathEval" )
    gmsh.model.mesh.field.setString( fieldmath, "F", mathEval )
    
    # ------------------------------------------------- #
    # --- [3] define Restrict Field                 --- #
    # ------------------------------------------------- #
    if   ( target == "volu" ):
        dimtags_v = dimtags
        dimtags_s = gmsh.model.getBoundary( dimtags_v )
        dimtags_l = gmsh.model.getBoundary( dimtags_s, combined=False, oriented=False )
        faces     = [ int( dimtag[1] ) for dimtag in dimtags_s ]
        edges     = [ int( dimtag[1] ) for dimtag in dimtags_l ]
        fieldrest = gmsh.model.mesh.field.add( "Restrict" )
        gmsh.model.mesh.field.setNumber ( fieldrest, "IField"   , fieldmath )
        gmsh.model.mesh.field.setNumbers( fieldrest, "SurfacesList", faces     )
        gmsh.model.mesh.field.setNumbers( fieldrest, "CurvesList", edges     )
        regions   = [ int( dimtag[1] ) for dimtag in dimtags_v ]
        gmsh.model.mesh.field.setNumbers( fieldrest, "VolumesList", regions )
        # -- future change ??  InField, SurfacesList, CurvesList, VolumesList ?? -- #
        
    elif ( target == "surf" ):
        dimtags_s = dimtags
        dimtags_l = gmsh.model.getBoundary( dimtags_s, combined=False, oriented=False )
        faces     = [ int( dimtag[1] ) for dimtag in dimtags_s ]
        edges     = [ int( dimtag[1] ) for dimtag in dimtags_l ]
        fieldrest = gmsh.model.mesh.field.add( "Restrict" )
        gmsh.model.mesh.field.setNumber ( fieldrest, "IField"   , fieldmath )
        gmsh.model.mesh.field.setNumbers( fieldrest, "SurfacesList", faces     )
        gmsh.model.mesh.field.setNumbers( fieldrest, "CurvesList", edges     )
        # -- future change ??  InField, SurfacesList, CurvesList, VolumesList ?? -- #
    else:
        print( "[assign__meshsize_on_each_volume] ONLY volu & surf is implemented..." )
        sys.exit()
    return( fieldrest )



# ========================================================= #
# ===   実行部                                          === #
# ========================================================= #

if ( __name__=="__main__" ):

    gmsh.initialize()
    gmsh.option.setNumber( "General.Terminal", 1 )
    gmsh.model.add( "model" )

    box1 = gmsh.model.occ.addBox( -0.5, -0.5, -0.5, \
                                  +1.0, +1.0, +1.0 )
    box2 = gmsh.model.occ.addBox( -0.0, -0.0, -0.0, \
                                  +1.0, +1.0, +1.0 )
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    dimtags = { "cube01":[(3,1)], "cube02":[(3,2)], "cube03":[(3,3)] }
    for key,dimtag in dimtags.items():
        gmsh.model.setEntityName( dimtag[0][0], dimtag[0][1], key )
        
    configFile = "mesh.json"
    assign__meshsize( configFile=configFile )
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
    gmsh.write( "test/model.msh" )
    gmsh.finalize()
