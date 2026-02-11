import os, sys, re, gmsh, json5, numbers
import numpy as np

# ========================================================= #
# ===  assing mesh size                                 === #
# ========================================================= #

def assign__meshsize( jsonFile="dat/mesh.json", uniform=None ):

    dim_   , ent_    = 0, 1
    ptsDim , lineDim = 0, 1
    surfDim, voluDim = 2, 3

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
    with open( jsonFile, "r" ) as f:
        settings = json5.load( f )
    
    # ------------------------------------------------- #
    # --- [3] assign mesh size                      --- #
    # ------------------------------------------------- #
    fieldlist = []
    for key,content in settings.items():
        dt         = content["dimtags"]
        ms         = content["meshsize"]
        ret        = assign__meshsize_on_each_dimtags( dimtags=dt, meshsize=ms )
        fieldlist += [ ret ]

    # ------------------------------------------------- #
    # --- [4] define total field                    --- #
    # ------------------------------------------------- #
    totalfield = gmsh.model.mesh.field.add( "Min" )
    gmsh.model.mesh.field.setNumbers( totalfield, "FieldsList", fieldlist )
    gmsh.model.mesh.field.setAsBackgroundMesh( totalfield )

    gmsh.option.setNumber( "Mesh.CharacteristicLengthMin", 0.005 )
    gmsh.option.setNumber( "Mesh.CharacteristicLengthMax", 1.0 )

    # ------------------------------------------------- #
    # --- [5] return                                --- #
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

    gmsh.model.occ.addBox( -0.5, -0.5, -0.5, \
                           +1.0, +1.0, +1.0 )
    gmsh.model.occ.addBox( -0.0, -0.0, -0.0, \
                           +1.0, +1.0, +1.0 )
    
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    dimtags = { "cube01":[(3,1)], "cube02":[(3,2)], "cube03":[(3,3)] }

    jsonFile = "mesh.json"
    assign__meshsize( jsonFile=jsonFile )
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
    gmsh.write( "model.msh" )
    gmsh.finalize()
