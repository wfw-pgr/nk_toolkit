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
                          matFile="dat/materials.json", duplicates="", \
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
    gmsh.option.setNumber( "Mesh.Algorithm3D"         , 1    )
    gmsh.option.setNumber( "Mesh.Optimize"            , 1    )
    gmsh.option.setNumber( "Mesh.OptimizeNetgen"      , 1    )
    gmsh.option.setNumber( "Mesh.Smoothing"           , 3    )
    gmsh.option.setNumber( "Geometry.OCCImportLabels" , 1    )
    gmsh.option.setNumber( "Geometry.OCCBooleanPreserveNumbering", 1 )
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

    if ( scale_unit != 1.0 ):
        all_ents = gmsh.model.getEntities( dim=3 )
        gmsh.model.occ.dilate( all_ents, 0,0,0, scale_unit,scale_unit,scale_unit  )
        gmsh.model.occ.synchronize()
        
    # names   = [ (gmsh.model.getEntityName( dim,tag ) ).split("/")[-1] for dim,tag in dimtags ]
    # gmsh.model.occ.removeAllDuplicates()
    # gmsh.model.occ.synchronize()
    
    if   ( duplicates in [ "cut-newer" ] ):
        names, numDict, entities = cut__duplicatedObjects( config=config, dimtags=dimtags, priority="newer" )
    elif ( duplicates in [ "cut-older" ] ):
        names, numDict, entities = cut__duplicatedObjects( config=config, dimtags=dimtags, priority="older" )
    else:
        names, numDict, entities = collect__entitiesByName()
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
        
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
                        g/cm3_to_kg_m3  = 1.0e3
                        item["density"] = abs( materials[ item["material"] ]["Density"] ) * g_cm3_to_kg_m3
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
# ===  get entity name                                  === #
# ========================================================= #
def get__entityName( dimtag ):
    dim, tag = dimtag
    name     = gmsh.model.getEntityName( dim, tag )
    name     = name.split("/")[-1]
    if ( name == "" ):
        name = "volume_{}".format( tag )
    return( name )


# ========================================================= #
# ===  collect entities by name                         === #
# ========================================================= #
def collect__entitiesByName( dimtags=None ):
    if ( dimtags is None ):
        dimtags = gmsh.model.getEntities( dim=3 )
    dimtags = [ dimtag for dimtag in dimtags if dimtag[0] == 3 ]

    numDict = {}
    for dimtag in dimtags:
        name = get__entityName( dimtag )
        if ( not( name in numDict ) ):
            numDict[name] = []
        numDict[name] += [ dimtag ]

    names    = list( numDict.keys() )
    entities = { name:[ dimtag[1] for dimtag in numDict[name] ] for name in names }

    return( names, numDict, entities )


# ========================================================= #
# ===  cleanup verySmallVolumes                         === #
# ========================================================= #
def cleanup__verySmallVolumes( dimtags, volume_tol=0.0 ):
    """
    - 微小体積の要素を削除
    1. try で体積計算に失敗 => 無効な dimtag  => 削除
    2. volume_tol 以下 => 微小体積として無視 (1e-10以下など)
    """
    ret = []
    for dim, tag in dimtags:
        if ( dim != 3 ):
            continue
        try:
            vol = gmsh.model.occ.getMass( dim, tag )
        except Exception:
            continue
        if ( vol > volume_tol ):
            ret += [ ( dim, tag ) ]
    return ret



# ========================================================= #
# ===  重なるツールのみ選択、返却                       === #
# ========================================================= #
def select__overlappingTools( objects, tools, bbox_cache=None, tol=1.0e-6 ):
    """
    objects と bounding box が重なる tools のみ抽出。
    """
    # ------------------------------------------------- #
    # --- [1] functions                             --- #
    # ------------------------------------------------- #
    def _get__bbox( dimTags ):
        bbs  = [ gmsh.model.occ.getBoundingBox(dim, tag) for dim, tag in dimTags ]
        xmin = min(bb[0] for bb in bbs)
        ymin = min(bb[1] for bb in bbs)
        zmin = min(bb[2] for bb in bbs)
        xmax = max(bb[3] for bb in bbs)
        ymax = max(bb[4] for bb in bbs)
        zmax = max(bb[5] for bb in bbs)
        return( xmin, ymin, zmin, xmax, ymax, zmax )

    def _is__bboxOverlap( bb1, bb2, tol=1.0e-6 ):
        return( ( bb1[0] <= bb2[3] + tol ) and ( bb2[0] <= bb1[3] + tol ) and
                ( bb1[1] <= bb2[4] + tol ) and ( bb2[1] <= bb1[4] + tol ) and
                ( bb1[2] <= bb2[5] + tol ) and ( bb2[2] <= bb1[5] + tol ) )

    # ------------------------------------------------- #
    # --- [2] select                                --- #
    # ------------------------------------------------- #
    if ( bbox_cache is None ):
        bbox_cache = {}
        
    obj_bb   = _get__bbox( objects )
    selected = []
    for tool in tools:
        
        if ( tool not in bbox_cache ):
            bbox_cache[tool] = _get__bbox( [tool] )
        tool_bb = bbox_cache[tool]
        if ( _is__bboxOverlap( obj_bb, tool_bb, tol=tol ) ):
            selected.append( tool )
    return( selected )


# ========================================================= #
# ===  cut duplicated objects                           === #
# ========================================================= #
def cut__duplicatedObjects( config=None, dimtags=None, volume_tol=0.0, priority="newer" ):
    
    # ------------------------------------------------- #
    # --- [1] arguments check                       --- #
    # ------------------------------------------------- #
    if ( config is None ):
        raise ValueError( "[cut__duplicatedObjects] config is None." )

    # ------------------------------------------------- #
    # --- [2] preparation                           --- #
    # ------------------------------------------------- #
    gmsh.model.occ.synchronize()
    names0, numDict0, entities0 = collect__entitiesByName( dimtags=dimtags )

    # mesh.json に書かれている順序を優先順位として使う
    if   ( priority == "newer" ):
        orderedNames = [ key for key in config.keys() if key in numDict0 ][::-1]
    elif ( priority == "older" ):
        orderedNames = [ key for key in config.keys() if key in numDict0 ]
    else:
        raise ValueError( "[make__solidworksSTEP.py] priority == {} ??".format( priority ) )
        
    # STEP にはあるが mesh.json にないもの
    unusedNames  = [ key for key in names0 if not( key in orderedNames ) ]
    if ( len( unusedNames ) > 0 ):
        print( "[cut__duplicatedObjects] warning :: imported but not in config" )
        for key in unusedNames:
            print( "   - {}".format( key ) )


    # # ------------------------------------------------- #
    # # --- [3] loop                                  --- #
    # # ------------------------------------------------- #
    # tools   = []
    # newDict = {}
    # for name in orderedNames:
    #     objects = cleanup__verySmallVolumes( numDict0[name], volume_tol=volume_tol )

    #     if ( len( objects ) == 0 ):
    #         print( "[cut__duplicatedObjects] skip empty object :: {}".format( name ) )
    #         continue
    #     if ( len( tools ) > 0 ):
    #         outDimTags, outMap = gmsh.model.occ.cut( objects, tools, \
    #                                                  removeObject=True, removeTool=False )
    #         gmsh.model.occ.synchronize()
    #         objects = cleanup__verySmallVolumes( outDimTags, volume_tol=volume_tol )

    #     # Boolean 後の tag に部品名を再付与
    #     for dim, tag in objects:
    #         gmsh.model.setEntityName( dim, tag, name )
    #     newDict[name] = objects

    #     # 以降の part を削る tool として登録
    #     tools += objects


    # ------------------------------------------------- #
    # --- [3] loop                                  --- #
    # ------------------------------------------------- #
    tools      = []
    newDict    = {}
    bbox_cache = {}
    for name in orderedNames:
        objects = cleanup__verySmallVolumes( numDict0[name], volume_tol=volume_tol )

        if ( len( objects ) == 0 ):
            print( "[cut__duplicatedObjects] skip empty object :: {}".format( name ) )
            continue
        if ( len( tools ) > 0 ):
            activeTools = select__overlappingTools( objects   =objects, tools = tools,
                                                    bbox_cache=bbox_cache, tol= 1.0e-5 )
            print( "[cut__duplicatedObjects] {} :: tools {} -> {}"\
                   .format( name, len(tools), len(activeTools) ) )

            if ( len(activeTools) > 0 ):
                outDimTags, outMap = gmsh.model.occ.cut( objects, activeTools,
                                                         removeObject= True, removeTool=False )
                objects = cleanup__verySmallVolumes( outDimTags, volume_tol=volume_tol )
        newDict[name] = objects
        # -- 以降の part を削る tool として登録 -- #
        tools += objects

    # -- Boolean 後にまとめて synchronize -- #
    gmsh.model.occ.synchronize()
                
    # -- Boolean 後の tag に部品名を再付与 -- #
    for name, objects in newDict.items():
        for dim, tag in objects:
            gmsh.model.setEntityName(dim, tag, name)

    # ------------------------------------------------- #
    # --- [4] return                                --- #
    # ------------------------------------------------- #
    gmsh.model.occ.synchronize()
    names    = list( newDict.keys() )
    entities = { name:[ dimtag[1] for dimtag in newDict[name] ] for name in names }

    return( names, newDict, entities )


        
# ========================================================= #
# ===   実行部                                          === #
# ========================================================= #

if ( __name__=="__main__" ):

    stpFile    = "test/test.stp"
    configFile = "test/mesh.json"
    phits_mesh = False
    mesh__solidworksSTEP( stpFile=stpFile, configFile=configFile, phits_mesh=phits_mesh )
