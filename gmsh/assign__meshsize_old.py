# ========================================================= #
# ===  assign mesh size ( main routine )                === #
# ========================================================= #
def assign__meshsize( meshFile=None, physFile=None, logFile=None, \
                      dimtags=None, uniform=None, target=None ):

    dim_   , ent_    = 0, 1
    ptsDim , lineDim = 0, 1
    surfDim, voluDim = 2, 3

    # ------------------------------------------------- #
    # --- [1] Arguments                             --- #
    # ------------------------------------------------- #
    if ( uniform is not None ):
        if ( uniform == 0.0 ):
            xyzMinMax = gmsh.model.getBoundingBox( -1, -1 )
            xL        = ( xyzMinMax[3] - xyzMinMax[0] )
            yL        = ( xyzMinMax[4] - xyzMinMax[1] )
            zL        = ( xyzMinMax[5] - xyzMinMax[2] )
            typLength = np.max( np.array( [ xL, yL, zL ] ) )
            uniform   = typLength * 0.02
        gmsh.option.setNumber( "Mesh.CharacteristicLengthMin", uniform )
        gmsh.option.setNumber( "Mesh.CharacteristicLengthMax", uniform )
        return()
    if ( meshFile is None ): sys.exit( "[assign__meshsize.py] meshFile == ???" )
    if ( physFile is None ): sys.exit( "[assign__meshsize.py] physFile == ???" )
    if ( dimtags  is None ): sys.exit( "[assign__meshsize.py] dimtags  == ???" )
    if ( target   is None ):
        max_dims = max( [ dt[dim_] for dimtag in dimtags.values() for dt in dimtag ] )
        target   = ( ["pts","line","surf","volu"] )[max_dims]
    
    # ------------------------------------------------- #
    # --- [2] obtain table & possible dimtags keys  --- #
    # ------------------------------------------------- #
    meshconfig = lkt.load__keyedTable( inpFile=meshFile )
    physconfig = lkt.load__keyedTable( inpFile=physFile )
    itarget    = ( ["pts","line","surf","volu"] ).index( target )
    physKeys     = list( physconfig.keys() )
    meshKeys     = list( meshconfig.keys() )
    aldtKeys     = []
    resolveDict  = {}
    for physKey in physKeys:
        dimtags_keys_loc  = ( physconfig[physKey] )["dimtags_keys"]
        aldtKeys         += dimtags_keys_loc
        stack             = { dimtag_key:physKey for dimtag_key in dimtags_keys_loc }
        resolveDict       = { **resolveDict, **stack }

    # ------------------------------------------------- #
    # --- [3] store mesh / phys info as lists       --- #
    # ------------------------------------------------- #
    dtagKeys     = []
    entitiesList = []
    physNumsList = []
    entitiesDict = {}
    physNumsDict = {}
    for aldtKey in aldtKeys:
        if ( aldtKey in dimtags ):
            n_dimtag = len( dimtags[aldtKey] )
            if   ( n_dimtag >= 2 ):
                dtagKeys     += [ aldtKey+".{0}".format(ik+1) for ik in range( n_dimtag ) ]
                entitiesList += [ dimtag[ent_] for dimtag in dimtags[aldtKey] \
                                  if ( dimtag[dim_] == itarget ) ]
                physNumsList += [ physconfig[ resolveDict[aldtKey] ]["physNum"] \
                                  for ik in range( n_dimtag ) ]
                for ik in range( n_dimtag ):
                    key_loc               = aldtKey+".{0}".format(ik+1)
                    resolveDict[key_loc]  = resolveDict[aldtKey]
                    entitiesDict[key_loc] = ( dimtags[aldtKey] )[ik][ent_]
                    physNumsDict[key_loc] = physconfig[ resolveDict[aldtKey] ]["physNum"]
                # resolveDict.pop( aldtKey ) # to erase old key
                
            elif ( n_dimtag == 1 ):
                dtagKeys     += [ aldtKey ]
                if ( dimtags[aldtKey][0][dim_] == itarget ):
                    entitiesList += [ dimtags[aldtKey][0][ent_] ]
                physNumsList += [ physconfig[ resolveDict[aldtKey] ]["physNum"] ]
                entitiesDict[aldtKey] = dimtags[aldtKey][0][ent_]
                physNumsDict[aldtKey] = physconfig[ resolveDict[aldtKey] ]["physNum"]
            else:
                print( "[assign__meshsize.py] empty dimtags @ key = {0}".format( aldtKey ) )

    # ------------------------------------------------- #
    # --- [4] convert dictionary for mesh config    --- #
    # ------------------------------------------------- #
    mc           = meshconfig
    meshTypeDict = { str(mc[key]["physNum"]):mc[key]["meshType"]    for key in meshKeys }
    resolut1Dict = { str(mc[key]["physNum"]):mc[key]["resolution1"] for key in meshKeys }
    resolut2Dict = { str(mc[key]["physNum"]):mc[key]["resolution2"] for key in meshKeys }
    evaluateDict = { str(mc[key]["physNum"]):mc[key]["evaluation"]  for key in meshKeys }
    
    # ------------------------------------------------- #
    # --- [5] make physNum <=> entityNum table      --- #
    # ------------------------------------------------- #
    ptsPhys, linePhys, surfPhys, voluPhys = {}, {}, {}, {}
    physNameDict = {}
    for dtagKey in dtagKeys:
        physType             = str( physconfig[ resolveDict[dtagKey] ]["type"]    )
        s_phys               = str( physconfig[ resolveDict[dtagKey] ]["physNum"] )
        physNameDict[s_phys] = str( physconfig[ resolveDict[dtagKey] ]["key"]     )
        if ( physType.lower() == "pts" ):
            if ( s_phys in  ptsPhys ):
                ptsPhys[s_phys]  += [ entitiesDict[dtagKey] ]
            else:
                ptsPhys[s_phys]   = [ entitiesDict[dtagKey] ]
        if ( physType.lower() == "line" ):
            if ( s_phys in linePhys ):
                linePhys[s_phys] += [ entitiesDict[dtagKey] ]
            else:
                linePhys[s_phys]  = [ entitiesDict[dtagKey] ]
        if ( physType.lower() == "surf" ):
            if ( s_phys in surfPhys ):
                surfPhys[s_phys] += [ entitiesDict[dtagKey] ]
            else:
                surfPhys[s_phys]  = [ entitiesDict[dtagKey] ]
        if ( physType.lower() == "volu" ):
            if ( s_phys in voluPhys ):
                voluPhys[s_phys] += [ entitiesDict[dtagKey] ]
            else:
                voluPhys[s_phys]  = [ entitiesDict[dtagKey] ]

    # ------------------------------------------------- #
    # --- [6] physical grouping                     --- #
    # ------------------------------------------------- #
    for s_phys in list(  ptsPhys.keys() ):
        gmsh.model.addPhysicalGroup(  ptsDim,  ptsPhys[str(s_phys)], tag=int(s_phys) )
        gmsh.model.setPhysicalName (  ptsDim, int(s_phys), physNameDict[s_phys]      )
    for s_phys in list( linePhys.keys() ):
        gmsh.model.addPhysicalGroup( lineDim, linePhys[str(s_phys)], tag=int(s_phys) )
        gmsh.model.setPhysicalName ( lineDim, int(s_phys), physNameDict[s_phys]      )
    for s_phys in list( surfPhys.keys() ):
        gmsh.model.addPhysicalGroup( surfDim, surfPhys[str(s_phys)], tag=int(s_phys) )
        gmsh.model.setPhysicalName ( surfDim, int(s_phys), physNameDict[s_phys]      )
    for s_phys in list( voluPhys.keys() ):
        gmsh.model.addPhysicalGroup( voluDim, voluPhys[str(s_phys)], tag=int(s_phys) )
        gmsh.model.setPhysicalName ( voluDim, int(s_phys), physNameDict[s_phys]      )
        
    # ------------------------------------------------- #
    # --- [7] make list for every dimtags's keys    --- #
    # ------------------------------------------------- #
    meshTypes    = [ meshTypeDict[ str(physNum) ] for physNum in physNumsList ]
    resolute1    = [ resolut1Dict[ str(physNum) ] for physNum in physNumsList ]
    resolute2    = [ resolut2Dict[ str(physNum) ] for physNum in physNumsList ]
    mathEvals    = [ evaluateDict[ str(physNum) ] for physNum in physNumsList ]

    # ------------------------------------------------- #
    # --- [8] resolution (Min,Max) treatment        --- #
    # ------------------------------------------------- #
    resolute1    = [ None if type(val) is str else val for val in resolute1 ]
    resolute2    = [ None if type(val) is str else val for val in resolute2 ]
    minMeshSize  = min( [ val for val in resolute1 if val is not None ] + [ val for val in resolute2 if val is not None ] )
    maxMeshSize  = max( [ val for val in resolute1 if val is not None ] + [ val for val in resolute2 if val is not None ] )
    print( "[assign__meshsize.py] Min. of MeshSize   :: {0} ".format( minMeshSize ) )
    print( "[assign__meshsize.py] Max. of MeshSize   :: {0} ".format( maxMeshSize ) )
    print()
    
    # ------------------------------------------------- #
    # --- [9] check entity numbers                  --- #
    # ------------------------------------------------- #
    itarget   = ( ["pts","line","surf","volu"] ).index( target )
    allEntities = gmsh.model.getEntities(itarget)
    allEntities = [ int(dimtag[1]) for dimtag in allEntities ]
    missing     = sorted( list( set( entitiesList ) - set( allEntities  ) ) )
    remains     = sorted( list( set( allEntities  ) - set( entitiesList ) ) )
    print( "[assign__meshsize.py] listed entity nums :: {0} ".format( sorted( entitiesList ) ) )
    print( "[assign__meshsize.py] all Entities       :: {0} ".format( sorted( allEntities  ) ) )
    print( "[assign__meshsize.py] remains            :: {0} ".format( sorted( remains      ) ) )
    print()
    print( "[assign__meshsize.py] Mesh Configuration " )
    print( "  key :: type  min  max  mathEval" )
    for ik, physNum in enumerate(physNumsList):
        print( "  {0} :: {1}  {2}  {3}  {4}".format( physNum, meshTypes[ik], resolute1[ik], \
                                                     resolute2[ik], mathEvals[ik] ) )
    print()
    if ( logFile is not None ):
        with open( logFile, "w" ) as f:
            f.write( "[assign__meshsize.py] listed entity nums :: {0} \n".format( sorted( entitiesList ) ) )
            f.write( "[assign__meshsize.py] all Entities       :: {0} \n".format( sorted( allEntities  ) ) )
            f.write( "[assign__meshsize.py] remains            :: {0} \n".format( sorted( remains      ) ) )
            f.write( "[assign__meshsize.py] Mesh Configuration \n" )
            f.write( "  key :: type  min  max  mathEval\n" )
            for ik, physNum in enumerate(physNumsList):
                f.write( "  {0} :: {1}  {2}  {3}  {4}\n".format( physNum, meshTypes[ik], resolute1[ik], \
                                                                 resolute2[ik], mathEvals[ik] ) )
    
    # ------------------------------------------------- #
    # --- [10] error message for missing entities   --- #
    # ------------------------------------------------- #
    if ( len( missing ) > 0 ):
        print( "[assign__meshsize.py] missing            :: {0} ".format( missing  ) )
        print( "[assign__meshsize.py] aborting, saving   :: current.msh "       )
        gmsh.option.setNumber( "General.Verbosity"           ,           3 )
        gmsh.option.setNumber( "Mesh.CharacteristicLengthMin", minMeshSize )
        gmsh.option.setNumber( "Mesh.CharacteristicLengthMax", maxMeshSize )
        gmsh.model.mesh.generate(3)
        gmsh.write( "current.msh" )
        print( "[assign__meshsize.py] missing Entity Error STOP " )
        sys.exit()
        
    # ------------------------------------------------- #
    # --- [11] error message for remains entities   --- #
    # ------------------------------------------------- #
    if ( len( remains ) > 0 ):
        print( "[assign__meshsize.py] remains            :: {0}  ".format( remains  ) )
        print( "[assign__meshsize.py] generate uniform mesh ??? >> (y/n)", end="" )
        typing = ( ( input() ).strip() ).lower()
        if ( typing == "y" ):
            gmsh.option.setNumber( "General.Verbosity"           ,           3 )
            gmsh.option.setNumber( "Mesh.CharacteristicLengthMin", minMeshSize )
            gmsh.option.setNumber( "Mesh.CharacteristicLengthMax", maxMeshSize )
            gmsh.model.mesh.generate(3)
            gmsh.write( "current.msh" )
        print( "[assign__meshsize.py] continue ???              >> (y/n)", end="" )
        typing = ( ( input() ).strip() ).lower()
        if ( typing == "y" ):
            pass
        else:
            print( "[assign__meshsize.py] remains Entity Error STOP " )
            sys.exit()
                
    # ------------------------------------------------- #
    # --- [12] define each mesh field               --- #
    # ------------------------------------------------- #
    fieldlist = []
    for ik,vl in enumerate( entitiesList ):
        ms  = [ resolute1[ik], resolute2[ik] ]
        ret = assign__meshsize_on_each_volume( volume_num=vl, meshsize=ms, target=target, \
                                               meshType  =meshTypes[ik], \
                                               mathEval  =mathEvals[ik] )
        fieldlist.append( ret[1] )

    # ------------------------------------------------- #
    # --- [13] define total field                   --- #
    # ------------------------------------------------- #
    totalfield = gmsh.model.mesh.field.add( "Min" )
    gmsh.model.mesh.field.setNumbers( totalfield, "FieldsList", fieldlist )
    gmsh.model.mesh.field.setAsBackgroundMesh( totalfield )

    # ------------------------------------------------- #
    # --- [14] define Min Max size                  --- #
    # ------------------------------------------------- #
    gmsh.option.setNumber( "Mesh.CharacteristicLengthMin", minMeshSize )
    gmsh.option.setNumber( "Mesh.CharacteristicLengthMax", maxMeshSize )
    
    # ------------------------------------------------- #
    # --- [15] return                               --- #
    # ------------------------------------------------- #
    ret = { "meshsize_list":resolute1, "entitiesList":entitiesList, \
            "field_list":fieldlist }
    return( ret )
    

