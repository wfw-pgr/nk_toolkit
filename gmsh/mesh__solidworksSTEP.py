import json, os, time
import gmsh, json5
import numpy as np
import nk_toolkit.gmsh.assign__meshsize     as ams
import nk_toolkit.gmsh.show__meshSummary    as sms
import nk_toolkit.io.show__activity         as sha
import nk_toolkit.phits.convert__gmsh2phits as g2p
import nk_toolkit.phits.materials__fromJSON as mfj

# ========================================================= #
# ===  mesh__solidworksSTEP.py                          === #
# ========================================================= #

def mesh__solidworksSTEP( stpFile="msh/model.stp", configFile="dat/mesh.json",
                          mshFile="msh/model.msh", bdfFile="msh/model.bdf", phits_mesh=False,
                          matFile="dat/materials.json", duplicates="fragment-newer",
                          materialPhitsFile="inp/materials.phits.j2", scale_unit="mm",
                          global_duplicates=True, geometry_only=False, timingFile=None,
                          logFile=None ):

    totalStart = time.perf_counter()
    timings    = []
    scaleLabel = scale_unit
    totalStage = 3 if ( geometry_only ) else 4 + int( phits_mesh )

    def _record__timing( stage=None, start=None, **details ):
        elapsed = time.perf_counter() - start
        record  = { "stage":stage, "elapsed_s":round( elapsed, 6 ), **details }
        timings.append( record )
        return( elapsed )

    def _show__heading( title=None, mark="-" ):
        width      = 60 if ( mark == "=" ) else 52
        titleWidth = width - 8
        bar        = "# " + mark*width + " #"
        titleLine  = "# {0} {1:^{2}s} {0} #".format( mark*3, title, titleWidth )
        print( "\n{}\n{}\n{}\n".format( bar, titleLine, bar ), flush=True )

    def _show__stage( number=None, title=None ):
        _show__heading( title="Stage {}/{}: {}".format( number, totalStage, title ) )

    def _save__gmshLog():
        messages = gmsh.logger.get()
        if ( logFile is not None ):
            with open( logFile, "w", encoding="utf-8" ) as outFile:
                outFile.write( "\n".join( messages ) + "\n" )
        gmsh.logger.stop()
        return( len( [ message for message in messages if "Warning" in message ] ) )

    _show__heading( title="Gmsh mesh begin", mark="=" )
    print( "Gmsh により STEP geometry から tetrahedral mesh を生成します。\n" )
    print( "Input STEP:       {}".format( stpFile ) )
    print( "Mesh config:      {}".format( configFile ) )
    print( "Material config:  {}".format( matFile ) )
    print( "Mesh output:      {}".format( mshFile ) )
    print( "Boolean strategy: {}".format( duplicates ) )
    print( "Scale unit:       {}".format( scaleLabel ) )
    print( "PHITS mesh:       {}".format( phits_mesh ) )
    print( "Geometry only:    {}".format( geometry_only ) )
    print( "Timing:           {}".format( timingFile ) )
    print( "Log:              {}".format( logFile ) )

    # ------------------------------------------------- #
    # --- [1] mesh config                           --- #
    # ------------------------------------------------- #
    _show__stage( number=1, title="load configuration and initialize" )
    stageStart = time.perf_counter()
    with open( configFile, "r" ) as inpFile:
        config    = json5.load( inpFile )
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
    elapsed = _record__timing( stage="load_config", start=stageStart )
    print( "Loaded {} parts and {} Gmsh options. ({:.3f} s)".format(
        len( config ), len( options ), elapsed ) )
            
    # ------------------------------------------------- #
    # --- [2] initialize                            --- #
    # ------------------------------------------------- #
    stageStart = time.perf_counter()
    gmsh.initialize()
    gmsh.option.setNumber( "General.Terminal"         , 0    )
    gmsh.option.setNumber( "General.Verbosity"        , 5    )
    gmsh.logger.start()
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
    _record__timing( stage="initialize_gmsh", start=stageStart )
            
    # ------------------------------------------------- #
    # --- [3] import models                         --- #
    # ------------------------------------------------- #
    _show__stage( number=2, title="import geometry" )
    if ( not( os.path.exists( stpFile ) ) ):
        raise FileNotFoundError( "Cannot find file :: {}".format( stpFile ) )
    if ( mshFile is None ):
        mshFile = ( os.path.splitext( stpFile ) )[0] + ".msh"

    stageStart = time.perf_counter()
    dimtags = gmsh.model.occ.importShapes( stpFile )
    volumeCount = len( [ item for item in dimtags if item[0] == 3 ] )
    elapsed = _record__timing( stage="import_shapes", start=stageStart,
                               imported_volumes=volumeCount )
    stageStart = time.perf_counter()
    gmsh.model.occ.synchronize()
    _record__timing( stage="synchronize_import", start=stageStart )

    if ( scale_unit != 1.0 ):
        stageStart = time.perf_counter()
        all_ents = gmsh.model.getEntities( dim=3 )
        gmsh.model.occ.dilate( all_ents, 0,0,0, scale_unit,scale_unit,scale_unit  )
        gmsh.model.occ.synchronize()
        _record__timing( stage="scale_and_synchronize", start=stageStart,
                         volume_count=len( all_ents ) )
    print( "Imported {} volumes. ({:.3f} s)".format( volumeCount, elapsed ) )
        
    # gmsh.model.occ.removeAllDuplicates()
    # gmsh.model.occ.synchronize()
    
    _show__stage( number=3, title="resolve overlapping volumes" )
    booleanStart = time.perf_counter()
    fragmented = False
    if   ( duplicates in [ "fragment-newer" ] ):
        names, numDict, entities = fragment__duplicatedObjects(
            config=config, dimtags=dimtags, priority="newer", timings=timings )
        fragmented = True
    elif ( duplicates in [ "fragment-older" ] ):
        names, numDict, entities = fragment__duplicatedObjects(
            config=config, dimtags=dimtags, priority="older", timings=timings )
        fragmented = True
    elif ( duplicates in [ "cut-newer" ] ):
        names, numDict, entities = cut__duplicatedObjects(
            config=config, dimtags=dimtags, priority="newer", timings=timings )
    elif ( duplicates in [ "cut-older" ] ):
        names, numDict, entities = cut__duplicatedObjects(
            config=config, dimtags=dimtags, priority="older", timings=timings )
    else:
        names, numDict, entities = collect__entitiesByName()

    if ( global_duplicates and not( fragmented ) ):
        stageStart = time.perf_counter()
        with sha.show__activity( label="Resolving global duplicates" ):
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
        _record__timing( stage="remove_all_duplicates", start=stageStart,
                         volume_count=len( gmsh.model.getEntities( dim=3 ) ) )
        
    missingNames = sorted( set( config.keys() ) - set( names ) )
    extraNames   = sorted( set( names ) - set( config.keys() ) )
    for key,item in config.items():
        if ( key in names ):
            item["entities"] = entities[key]
    print( "Resolved {} configured parts into {} volumes. ({:.3f} s)".format(
        len( names ), sum( len( dimtags ) for dimtags in numDict.values() ),
        time.perf_counter() - booleanStart ) )
    print( "Consistency: missing parts = {} | extra parts = {}".format(
        len( missingNames ), len( extraNames ) ) )
    print( "Missing parts: {}".format(
        ", ".join( missingNames ) if ( len( missingNames ) > 0 ) else "none" ) )
    print( "Extra parts:   {}".format(
        ", ".join( extraNames ) if ( len( extraNames ) > 0 ) else "none" ) )
                
    # ------------------------------------------------- #
    # --- [4] meshing                               --- #
    # ------------------------------------------------- #
    stageStart = time.perf_counter()
    ams.assign__meshsize( config=config )
    gmsh.model.occ.synchronize()
    _record__timing( stage="assign_mesh_size", start=stageStart )

    if ( geometry_only ):
        warningCount = _save__gmshLog()
        gmsh.finalize()
        timings.append( { "stage":"total", "elapsed_s":round(
            time.perf_counter() - totalStart, 6 ) } )
        if ( timingFile is not None ):
            with open( timingFile, "w", encoding="utf-8" ) as outFile:
                json.dump( timings, outFile, indent=2 )
        _show__heading( title="completed", mark="=" )
        print( "Geometry diagnostics completed." )
        print( "Elapsed:  {:.3f} s".format( time.perf_counter() - totalStart ) )
        print( "Gmsh warnings: {} ( details: {} )".format( warningCount, logFile ) )
        print( "\nOutputs:" )
        print( "- {}".format( timingFile ) )
        print( "- {}".format( logFile ) )
        return( timings )

    _show__stage( number=4, title="generate tetrahedral mesh" )
    stageStart = time.perf_counter()
    with sha.show__activity( label="Meshing tetrahedra" ):
        gmsh.model.mesh.generate( 3 )
    meshElapsed = _record__timing( stage="generate_mesh_3d", start=stageStart )
    stageStart = time.perf_counter()
    gmsh.write( mshFile )
    _record__timing( stage="write_msh", start=stageStart )
    warningCount = _save__gmshLog()
    gmsh.finalize()

    meshSummary = sms.show__meshSummary( mshFile=mshFile, elapsedTime_s=meshElapsed )
    for item in timings:
        if ( item["stage"] == "generate_mesh_3d" ):
            item["mesh_elements"] = meshSummary["total_mesh_size"]
            break

    # ------------------------------------------------- #
    # --- [5] gmsh -> phits (.bdf)                  --- #
    # ------------------------------------------------- #
    if ( phits_mesh ):
        _show__stage( number=5, title="export PHITS mesh" )
        if ( os.path.exists( matFile ) ):
            with open( matFile, "r" ) as inpFile:
                materials = json5.load( inpFile )
        else:
            raise FileNotFoundError(
                "[mesh__solidworksSTEP.py] matFile={} ??".format( matFile ) )

        activeItems      = { key:item for key,item in config.items() if key in names }
        missingMatKeys   = sorted( key for key,item in activeItems.items()
                                   if "material" not in item )
        usedMaterials    = { item["material"] for item in activeItems.values()
                             if "material" in item }
        missingMaterials = sorted( usedMaterials - set( materials.keys() ) )
        extraMaterials   = sorted( set( materials.keys() ) - usedMaterials )
        print( "Material consistency: missing keys = {} | missing definitions = {} | "
               "extra/unused = {}".format(
                   len( missingMatKeys ), len( missingMaterials ), len( extraMaterials ) ) )
        print( "Parts missing material keys: {}".format(
            ", ".join( missingMatKeys ) if ( len( missingMatKeys ) > 0 ) else "none" ) )
        print( "Missing material definitions: {}".format(
            ", ".join( missingMaterials ) if ( len( missingMaterials ) > 0 ) else "none" ) )
        print( "Extra/unused materials: {}".format(
            ", ".join( extraMaterials ) if ( len( extraMaterials ) > 0 ) else "none" ) )

        for key,item in config.items():
            if ( not( "density" in item ) ):
                if ( "material" in item ):
                    if ( item["material"] in materials ):
                        item["density"] = materials[ item["material"] ]["Density"]
                    else:
                        raise KeyError( "[mesh__solidworksSTTEP.py] Cannot find key :: {}"\
                                        .format(item["material"] ) )
                else:
                    raise KeyError( "[mesh__solidworksSTTEP.py] Cannot find material "
                                    "in key :: {}".format( key ) )
        matKeys = []
        for key,item in config.items():
            if ( key in names ):
                matKeys += [ item["material"] ]

        stageStart = time.perf_counter()
        with sha.show__activity( label="Converting PHITS mesh" ):
            g2p.convert__gmsh2phits( mshFile=mshFile, bdfFile=bdfFile, config=config )
            import meshio
            rmesh          = meshio.read( bdfFile )
            unq,idx        = np.unique( rmesh.cell_data["nastran:ref"], return_index=True )
            physNums_order = unq[ np.argsort( idx ) ]
            matKeys        = [ matKeys[ ik-1 ] for ik in physNums_order ]
            mfj.materials__fromJSON( matFile=matFile, outFile=materialPhitsFile,
                                     keys=matKeys, tetra_auto_mat=True )
        _record__timing( stage="convert_phits_bdf", start=stageStart )
        print( "Generated PHITS mesh and material definitions." )

    timings.append( { "stage":"total", "elapsed_s":round(
        time.perf_counter() - totalStart, 6 ) } )
    if ( timingFile is not None ):
        with open( timingFile, "w", encoding="utf-8" ) as outFile:
            json.dump( timings, outFile, indent=2 )
    _show__heading( title="completed", mark="=" )
    print( "Gmsh mesh generation completed." )
    print( "Gmsh warnings: {} ( details: {} )".format( warningCount, logFile ) )
    print( "\nOutputs:" )
    print( "- {}".format( mshFile ) )
    if ( phits_mesh ):
        print( "- {}".format( bdfFile ) )
        print( "- {}".format( materialPhitsFile ) )
    print( "- {}".format( timingFile ) )
    print( "- {}".format( logFile ) )
    return( timings )


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
# ===  fragment duplicated objects                      === #
# ========================================================= #
def fragment__duplicatedObjects( config=None, dimtags=None, priority="newer", timings=None ):

    # ------------------------------------------------- #
    # --- [1] preparation                           --- #
    # ------------------------------------------------- #
    if ( config is None ):
        raise ValueError( "[fragment__duplicatedObjects] config is None." )
    if ( priority not in [ "newer", "older" ] ):
        raise ValueError(
            "[fragment__duplicatedObjects] priority == {} ??".format( priority ) )
    if ( dimtags is None ):
        dimtags = gmsh.model.getEntities( dim=3 )
    dimtags = [ dimtag for dimtag in dimtags if dimtag[0] == 3 ]
    gmsh.model.occ.synchronize()

    sourceNames = { dimtag:get__entityName( dimtag ) for dimtag in dimtags }
    configOrder = { name:index for index, name in enumerate( config.keys() ) }
    unusedNames = sorted( set( sourceNames.values() ) - set( configOrder.keys() ) )
    if ( len( unusedNames ) > 0 ):
        print( "Warning: imported parts are absent from config: {}".format(
            ", ".join( unusedNames ) ) )

    # ------------------------------------------------- #
    # --- [2] single BooleanFragments               --- #
    # ------------------------------------------------- #
    fragmentStart = time.perf_counter()
    with sha.show__activity( label="Resolving Boolean fragments" ):
        outDimtags, outMap = gmsh.model.occ.fragment(
            dimtags[:1], dimtags[1:], removeObject=True, removeTool=True )
        gmsh.model.occ.synchronize()

    sourceMap = {}
    for source, mappedDimtags in zip( dimtags, outMap ):
        for dimtag in mappedDimtags:
            if ( dimtag[0] == 3 ):
                sourceMap.setdefault( dimtag, [] ).append( sourceNames[source] )

    # ------------------------------------------------- #
    # --- [3] priority ownership                    --- #
    # ------------------------------------------------- #
    newDict = {}
    shared  = 0
    for dimtag in outDimtags:
        if ( dimtag[0] != 3 ):
            continue
        candidates = list( dict.fromkeys( sourceMap[dimtag] ) )
        knownNames = [ name for name in candidates if name in configOrder ]
        if ( len( knownNames ) == 0 ):
            owner = candidates[0]
        elif ( priority == "newer" ):
            owner = max( knownNames, key=lambda name:configOrder[name] )
        elif ( priority == "older" ):
            owner = min( knownNames, key=lambda name:configOrder[name] )
        shared += int( len( candidates ) > 1 )
        newDict.setdefault( owner, [] ).append( dimtag )
        gmsh.model.setEntityName( dimtag[0], dimtag[1], owner )
    gmsh.model.occ.synchronize()

    elapsed = time.perf_counter() - fragmentStart
    if ( timings is not None ):
        timings.append( { "stage":"boolean_fragment", "input_objects":len( dimtags ),
                          "output_objects":len( sourceMap ), "shared_objects":shared,
                          "elapsed_s":round( elapsed, 6 ) } )
    names    = [ name for name in config.keys() if name in newDict ]
    names   += [ name for name in newDict.keys() if name not in configOrder ]
    newDict  = { name:newDict[name] for name in names }
    entities = { name:[ dimtag[1] for dimtag in newDict[name] ] for name in names }
    return( names, newDict, entities )


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
def cut__duplicatedObjects( config=None, dimtags=None, volume_tol=0.0, priority="newer",
                            timings=None ):
    
    # ------------------------------------------------- #
    # --- [1] arguments check                       --- #
    # ------------------------------------------------- #
    if ( config is None ):
        raise ValueError( "[cut__duplicatedObjects] config is None." )

    # ------------------------------------------------- #
    # --- [2] preparation                           --- #
    # ------------------------------------------------- #
    gmsh.model.occ.synchronize()
    names0, numDict0, _ = collect__entitiesByName( dimtags=dimtags )

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
        print( "Warning: imported parts are absent from config: {}".format(
            ", ".join( unusedNames ) ) )


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
    totalParts = len( orderedNames )
    for partIndex,name in enumerate( orderedNames, 1 ):
        objects = cleanup__verySmallVolumes( numDict0[name], volume_tol=volume_tol )

        if ( len( objects ) == 0 ):
            print( "Warning: skipped empty part: {}".format( name ) )
            continue
        if ( len( tools ) > 0 ):
            activeTools = select__overlappingTools( objects   =objects, tools = tools,
                                                    bbox_cache=bbox_cache, tol= 1.0e-5 )
            if ( len(activeTools) > 0 ):
                cutStart = time.perf_counter()
                label = "Resolving Boolean cuts [{}/{}] {}".format(
                    partIndex, totalParts, name )
                with sha.show__activity( label=label ):
                    outDimTags, _ = gmsh.model.occ.cut(
                        objects, activeTools, removeObject=True, removeTool=False )
                objects = cleanup__verySmallVolumes( outDimTags, volume_tol=volume_tol )
                if ( timings is not None ):
                    timings.append( { "stage":"boolean_cut", "part":name,
                                      "input_objects":len( numDict0[name] ),
                                      "active_tools":len( activeTools ),
                                      "output_objects":len( objects ),
                                      "elapsed_s":round(
                                          time.perf_counter() - cutStart, 6 ) } )
        newDict[name] = objects
        # -- 以降の part を削る tool として登録 -- #
        tools += objects

    # -- Boolean 後にまとめて synchronize -- #
    syncStart = time.perf_counter()
    gmsh.model.occ.synchronize()
    if ( timings is not None ):
        timings.append( { "stage":"synchronize_cuts", "elapsed_s":round(
            time.perf_counter() - syncStart, 6 ) } )
                
    # -- Boolean 後の tag に部品名を再付与 -- #
    for name, objects in newDict.items():
        for dim, tag in objects:
            gmsh.model.setEntityName(dim, tag, name)

    # ------------------------------------------------- #
    # --- [4] return                                --- #
    # ------------------------------------------------- #
    syncStart = time.perf_counter()
    gmsh.model.occ.synchronize()
    if ( timings is not None ):
        timings.append( { "stage":"synchronize_names", "elapsed_s":round(
            time.perf_counter() - syncStart, 6 ) } )
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
