import copy, os, shutil, sys, time
import json5
import numpy          as np
import pandas         as pd
import scipy.optimize as opt
import nk_toolkit.impactx.io_toolkit  as itk
import nk_toolkit.impactx.run_toolkit as rtk


# ========================================================= #
# ===  optimize__quadFromEnvelope                       === #
# ========================================================= #

def optimize__quadFromEnvelope( inpFile   ="dat/matching.json",
                                impactxDir="impactx/",
                                paramsFile="dat/parameters.json" ):
    """Match ImpactX lattice by changing QM factors and incident Twiss parameters."""

    # ========================================================= #
    # ===  [1] internal functions                           === #
    # ========================================================= #

    # ------------------------------------------------- #
    # --- [1-2] select matching section             --- #
    # ------------------------------------------------- #
    def _selectSection( elements=None, startElement=None, endElement=None, nUse=None ):
        elemKeys = list( elements.keys() )
        if ( nUse is not None ):
            elemKeys = elemKeys[:min( int( nUse ), len( elemKeys ) )]

        def _findIndex( target=None, defaultIndex=None ):
            if ( target is None ):
                return( defaultIndex )
            for elemIndex, elemKey in enumerate( elemKeys ):
                if ( target in [ elemKey, elements[elemKey]["name"] ] ):
                    return( elemIndex )
            raise ValueError( "Unknown section element: {}".format( target ) )

        startIndex = _findIndex( target=startElement, defaultIndex=0 )
        endIndex   = _findIndex( target=endElement, defaultIndex=len( elemKeys )-1 )
        if ( startIndex > endIndex ):
            raise ValueError( "section.startElement must precede section.endElement." )

        simKeys     = elemKeys[:endIndex + 1]
        activeKeys  = elemKeys[startIndex:endIndex + 1]
        simElements = { elemKey:copy.deepcopy( elements[elemKey] ) for elemKey in simKeys }
        return( simElements, activeKeys, startIndex )

    # ------------------------------------------------- #
    # --- [1-3] build optimization variables        --- #
    # ------------------------------------------------- #
    def _buildVariables( params=None, elements=None, activeKeys=None ):
        variables   = []
        variableCfg = config["variables"]
        usedQmNames = set()

        def _expandQmID( qmID=None ):
            qmList  = []
            for item in qmID:
                item = str( item ).strip()
                if ( "-" in item ):
                    startId, endId = item.split( "-", 1 )
                    startId       = int( startId )
                    endId         = int(   endId )
                    if ( startId > endId ):
                        raise ValueError( "Invalid qmID range: {}".format( item ) )
                    qmList += [ "qm{}".format( qmId ) for qmId in range( startId, endId + 1 ) ]
                else:
                    qmList.append( "qm{}".format( int( item ) ) )
            return( qmList )

        def _appendVariable( name=None, kind=None, settings=None,
                             target=None, defaultValue=1.0 ):
            if ( not( settings["enabled"] ) ):
                return
            initial = settings["initial"]
            if ( initial is None ):
                initial = defaultValue
            variables.append( { "name"   :name, "kind":kind, "target":target,
                                "initial":float( initial ), "min":float( settings["min"] ),
                                "max"    :float( settings["max"] ), } )

        _appendVariable( name="quadAll", kind="quadAll",
                         settings=variableCfg["quadAll"] )
        for fdName, settings in variableCfg["quadFD"].items():
            if ( fdName not in [ "QF", "QD" ] ):
                raise ValueError( "quadFD accepts only QF and QD." )
            _appendVariable( name="quadFD.{}".format( fdName ), kind="quadFD",
                             settings=settings, target=fdName )
        for settings in variableCfg["quadEach"].values():
            if ( not( settings["enabled"] ) ):
                continue
            qmList = _expandQmID( qmID=settings["qmID"] )
            for qmName in qmList:
                if ( qmName in usedQmNames ):
                    raise ValueError( "QM is already assigned : {}".format( qmName ) )
                usedQmNames.add( qmName )
                _appendVariable( name="quadEach.{}".format( qmName ), kind="quadEach",
                                 settings=settings, target=qmName )
        twissMap = { "alphaX":("alpha",0), "alphaY":("alpha",1), "alphaT":("alpha",2),
                     "betaX" :("beta" ,0), "betaY" :("beta" ,1), "betaT" :("beta" ,2), }
        for twissName, settings in variableCfg["twiss"].items():
            if ( twissName not in twissMap ):
                raise ValueError( "Unknown Twiss variable: {}".format( twissName ) )
            twissType, twissIndex = twissMap[twissName]
            if ( twissType == "alpha" ):
                # alpha: absolute value
                defaultValue = params["beam.twiss.alpha"][twissIndex]
            elif ( twissType == "beta" ):
                # beta: scale factor relative to parameters.json
                defaultValue = 1.0
            else:
                raise ValueError( "Unknown Twiss type: {}".format( twissType ) )
            _appendVariable( name="twiss.{}".format( twissName ), kind="twiss",
                             settings=settings, target=twissName, defaultValue=defaultValue )
            
        activeNames = []
        for elemKey in activeKeys:
            activeNames += [ elemKey, elements[elemKey]["name"] ]
        for variable in variables:
            if ( variable["kind"] == "quadEach" and variable["target"] not in activeNames ):
                raise ValueError( "QM variable is outside matching section: {}".format( variable["target"] ) )
        return( variables )

    # ------------------------------------------------- #
    # --- [1-4] apply QM and Twiss variables        --- #
    # ------------------------------------------------- #
    def _applyVariables( vector=None, params=None, elements=None,
                         variables=None, activeKeys=None ):
        params_   = copy.deepcopy( params )
        elements_ = copy.deepcopy( elements )
        factors   = { "quadAll":1.0, "quadFD":{ "QF":1.0, "QD":1.0 }, "quadEach":{} }
        twissMap  = { "alphaX":("alpha",0), "alphaY":("alpha",1), "alphaT":("alpha",2),
                      "betaX" :("beta" ,0), "betaY" :("beta" ,1), "betaT" :("beta" ,2), }

        for varIndex, variable in enumerate( variables ):
            value = float( vector[varIndex] )
            if   ( variable["kind"] == "quadAll" ):
                factors["quadAll"] = value
            elif ( variable["kind"] == "quadFD" ):
                factors["quadFD"][variable["target"]] = value
            elif ( variable["kind"] == "quadEach" ):
                factors["quadEach"][variable["target"]] = value
            elif ( variable["kind"] == "twiss" ):
                twissType, twissIndex = twissMap[variable["target"]]
                if ( twissType == "alpha" ):
                    # alpha: absolute value
                    params_["beam.twiss.alpha"][twissIndex] = value
                elif ( twissType == "beta" ):
                    # beta: scale factor relative to the original parameters.json
                    beta0 = float( params["beam.twiss.beta"][twissIndex] )
                    params_["beam.twiss.beta"][twissIndex] = beta0 * value

                else:
                    raise ValueError( "Unknown Twiss type: {}".format( twissType ) )

        for elemKey in activeKeys:
            elem = elements_[elemKey]
            if ( elem["type"] not in [ "quadrupole", "quadrupole.linear" ] ):
                continue

            elemName = elem["name"]
            k0       = float( elements[elemKey]["k"] )
            factor   = factors["quadAll"]
            if   ( k0 > 0.0 ):
                factor *= factors["quadFD"]["QF"]
            elif ( k0 < 0.0 ):
                factor *= factors["quadFD"]["QD"]

            if ( elemKey in factors["quadEach"] ):
                factor *= factors["quadEach"][elemKey]
            if ( elemName in factors["quadEach"] ):
                factor *= factors["quadEach"][elemName]
            elem["k"] = k0 * factor

        return( params_, elements_ )

    # ------------------------------------------------- #
    # --- [1-6] make statistics                    --- #
    # ------------------------------------------------- #
    def _makeStats( rawStat=None, simElements=None, activeKeys=None, startIndex=0 ):
        elemKeys = list( simElements.keys() )
        stat     = rawStat.sort_values( "step" ).drop_duplicates( subset=["step"], keep="last" )
        stat     = stat.reset_index( drop=True )

        expectedRows = 1
        for elemKey in elemKeys:
            nSlice = int( simElements[elemKey]["nslice"] ) \
                if ( "nslice" in simElements[elemKey] ) else 1
            expectedRows += max( 1, nSlice )

        if ( len( stat ) == expectedRows ):
            rowIndexList, rowIndex = [0], 0
            for elemKey in elemKeys:
                nSlice = int( simElements[elemKey]["nslice"] ) \
                    if ( "nslice" in simElements[elemKey] ) else 1
                rowIndex += max( 1, nSlice )
                rowIndexList.append( rowIndex )
            records = stat.iloc[rowIndexList].copy().reset_index( drop=True )
        else:
            targetS, sPos = [ 0.0 ], 0.0
            for elemKey in elemKeys:
                elem = simElements[elemKey]
                if ( elem["type"] == "shortrf" ):
                    ds = 0.0
                else:
                    ds = float( elem["ds"] )
                sPos += ds
                targetS.append( sPos )

            rowList = []
            for targetS_ in targetS:
                distance = np.abs( stat["s"].to_numpy() - targetS_ )
                rowIndex = int( np.where( distance == np.min( distance ) )[0][-1] )
                rowList.append( stat.iloc[rowIndex] )
            records = pd.DataFrame( rowList ).reset_index( drop=True )

        records["location"] = [ "__beamlineStart__" ] + elemKeys
        if ( "charge_C" in records.columns ):
            initialCharge = float( records["charge_C"].iloc[0] )
            if ( initialCharge == 0.0 ):
                raise ValueError( "Initial beam charge is zero." )
            records["transmission"] = records["charge_C"] / initialCharge

        records.attrs["sectionStartRow"] = startIndex
        records.attrs["sectionEndRow"]   = startIndex + len( activeKeys )
        return( records )

    # ------------------------------------------------- #
    # --- [1-7] execute ImpactX                     --- #
    # ------------------------------------------------- #
    def _evaluateImpactX( vector=None, keepStat=False ):
        """ Evaluate ImpactX simulation """
        params_, elements_ = _applyVariables( vector=vector, params=params, elements=simElements,
                                              variables=variables, activeKeys=activeKeys )
        runResult = rtk.execute__impactx( params=params_, elements=elements_, workDir=impactxDir,
                                          runMode=config["matching"]["mode"], clearDiags=True,
                                          add_bpm=False, saveRecords=False,
                                          saveLattice=False, verbose=False )
        stat      = itk.get__beamStats  ( statFile=runResult["statFile"],
                                          refpFile=runResult["refpFile"] )
        records   = _makeStats          ( rawStat=stat, simElements=elements_,
                                          activeKeys=activeKeys, startIndex=startIndex )
        if ( keepStat ):
            records.to_csv( config["files"]["statFile"], index=False )
        return( records )

    # ------------------------------------------------- #
    # --- [1-8] select objective rows              --- #
    # ------------------------------------------------- #
    def _selectRows( records=None, location="end" ):
        startRow = records.attrs["sectionStartRow"]
        endRow   = records.attrs["sectionEndRow"]
        if   ( location == "start" ):
            return( records.iloc[[startRow]] )
        elif ( location == "end" ):
            return( records.iloc[[endRow]] )
        elif ( location == "all" ):
            return( records.iloc[startRow + 1:endRow + 1] )

        for elemKey in activeKeys:
            if ( location in [ elemKey, simElements[elemKey]["name"] ] ):
                return( records[records["location"] == elemKey] )
        raise ValueError( "Unknown objective location: {}".format( location ) )

    # ------------------------------------------------- #
    # --- [1-9] evaluate expression and aggregate  --- #
    # ------------------------------------------------- #
    def _evaluateExpression( rows=None, expression=None ):
        localVars = { "np":np }
        for column in rows.columns:
            if ( pd.api.types.is_numeric_dtype( rows[column] ) ):
                localVars[column] = rows[column].to_numpy()
        values = eval( expression, { "__builtins__":{} }, localVars )
        return( np.atleast_1d( np.asarray( values, dtype=float ) ) )

    def _aggregate( values=None, method="mean" ):
        if   ( method == "mean" ):
            return( float( np.mean( values ) ) )
        elif ( method == "max" ):
            return( float( np.max( values ) ) )
        elif ( method == "sum" ):
            return( float( np.sum( values ) ) )
        raise ValueError( "Unknown aggregate method: {}".format( method ) )

    # ------------------------------------------------- #
    # --- [1-10] build residual vector             --- #
    # ------------------------------------------------- #
    def _compressResidual( values=None, method="mean" ):
        if   ( method == "mean" ):
            return( values / np.sqrt( len( values ) ) )
        elif ( method == "sum" ):
            return( values )
        elif ( method == "max" ):
            index = int( np.argmax( np.abs( values ) ) )
            return( values[[index]] )
        raise ValueError( "Unknown aggregate method: {}".format( method ) )

    def _evaluateResiduals( records=None, vector=None, initial=None,
                            includeRegularization=True ):
        
        for objective in config["objectives"]:
            if ( not( objective["enabled"] ) ):
                continue

            if ( objective["type"] == "periodicSigma" ):
                locationList = objective["locations"]
                residualPair = []
                
            for locIndex in range( 1, len( locationList ) ):
                rowPrev = _selectRows( records=records, location=locationList[locIndex-1] ).iloc[0]
                rowCurr = _selectRows( records=records, location=locationList[locIndex] ).iloc[0]

            sigmaPrev = _sigmaMatrix( row=rowPrev )
            sigmaCurr = _sigmaMatrix( row=rowCurr )
            diagPrev  = np.abs( np.diag( sigmaPrev ) )
            scaleMat  = np.sqrt( np.outer( diagPrev, diagPrev ) )
            scaleMat  = np.maximum( scaleMat, 1.0e-30 )
            
            residualPair.append( ( sigmaCurr - sigmaPrev ).ravel() / scaleMat.ravel() )

            residual = np.concatenate( residualPair )
            residualList.append( np.sqrt( float( objective["weight"] ) ) * residual / np.sqrt( len( residual ) ) )
            continue
            
            rows      = _selectRows( records=records, location=objective["location"] )
            values    = _evaluateExpression( rows=rows, expression=objective["expr"] )
            objType   = objective["type"]
            scale     = float( objective["scale"] )
            weight    = float( objective["weight"] )
            aggregate = objective["aggregate"]

            if   ( objType == "target" ):
                normalized = ( values - float( objective["target"] ) ) / scale

            elif ( objType == "range" ):
                lower = float( objective["min"] )
                upper = float( objective["max"] )
                violation  = np.where( values < lower, values - lower,
                                       np.where( values > upper, values - upper, 0.0 ) )
                normalized = violation / scale

            elif ( objType == "minimize" ):
                normalized = values / scale

            elif ( objType == "maximize" ):
            elif ( objType == "maximize" ):
                if ( "target" not in objective ):
                    raise ValueError( "maximize requires target for least_squares/SVD: {}"
                                      .format( objective["name"] ) )
                normalized = ( values - float( objective["target"] ) ) / scale            
            else:
                raise ValueError( "Unknown least_squares objective: {}".format( objType ) )

            residual = _compressResidual( values=normalized, method=aggregate )
            residualList.append( np.sqrt( weight ) * residual )

        regularization = config["regularization"]
        if ( includeRegularization and regularization["enabled"] ):
            lower  = np.array( [ variable["min"] for variable in variables ] )
            upper  = np.array( [ variable["max"] for variable in variables ] )
            span   = np.maximum( upper - lower, 1.0e-12 )
            weight = float( regularization["weight"] )
            residualList.append( np.sqrt( weight ) * ( vector - initial ) / span )

        return( np.concatenate( residualList ) )
    
    # ------------------------------------------------- #
    # --- [1-10] build sigma matrix                --- #
    # ------------------------------------------------- #
    def _sigmaMatrix( row=None ):
        sigma     = np.zeros( (6,6) )
        planeList = [ ("x",0,1), ("y",2,3), ("t",4,5) ]
        for planeName, coordIndex, momIndex in planeList:
            alpha = float( row["alpha_{}".format( planeName )] )
            beta  = float( row["beta_{}".format( planeName )] )
            emit  = float( row["emittance_{}".format( planeName )] )
            gamma = ( 1.0 + alpha**2 ) / beta
            sigma[coordIndex,coordIndex] = beta  * emit
            sigma[coordIndex,momIndex]   = -alpha * emit
            sigma[momIndex,coordIndex]   = -alpha * emit
            sigma[momIndex,momIndex]     = gamma * emit
        return( sigma )

    # ------------------------------------------------- #
    # --- [1-11] build SVD variable modes           --- #
    # ------------------------------------------------- #
    def _vectorToLatent( vector=None, lower=None, upper=None ):
        midpoint = 0.5 * ( lower + upper )
        halfspan = 0.5 * ( upper - lower )
        scaled   = ( vector - midpoint ) / halfspan
        scaled   = np.clip( scaled, -1.0 + 1.0e-12, 1.0 - 1.0e-12 )
        return( np.arctanh( scaled ) )
    

    def _latentToVector( latent=None, lower=None, upper=None ):
        midpoint = 0.5 * ( lower + upper )
        halfspan = 0.5 * ( upper - lower )
        return( midpoint + halfspan * np.tanh( latent ) )
    

    def _buildSvdBasis( initial=None, lower=None, upper=None ):
        svdCfg        = config["diagnostics"]["svd"]
        latentInitial = _vectorToLatent(
            vector=initial, lower=lower, upper=upper
        )
        step         = float( svdCfg["step"] )
        jacobianList = []
        
        for varIndex in range( len( variables ) ):
            latentPlus  = latentInitial.copy()
            latentMinus = latentInitial.copy()
            
            latentPlus[varIndex]  += step
            latentMinus[varIndex] -= step
            
            vectorPlus = _latentToVector(
                latent=latentPlus, lower=lower, upper=upper
            )
            vectorMinus = _latentToVector(
                latent=latentMinus, lower=lower, upper=upper
            )
            
            recordsPlus  = _evaluateImpactX(
                vector=vectorPlus, keepStat=False
            )
            recordsMinus = _evaluateImpactX(
                vector=vectorMinus, keepStat=False
            )
            
            residualPlus = _evaluateResiduals(
                records=recordsPlus, vector=vectorPlus, initial=initial,
                includeRegularization=False
            )
            residualMinus = _evaluateResiduals(
                records=recordsMinus, vector=vectorMinus, initial=initial,
                includeRegularization=False
            )
            
            jacobianList.append(
                ( residualPlus - residualMinus ) / ( 2.0 * step )
            )
            
        jacobian = np.column_stack( jacobianList )
            
        _, singularValue, rightModeT = np.linalg.svd(
            jacobian, full_matrices=False
        )
        
    if ( len( singularValue ) == 0 or singularValue[0] <= 0.0 ):
        raise ValueError( "SVD Jacobian has no finite sensitivity." )

    relativeValue = singularValue / singularValue[0]

    if ( svdCfg["nModes"] is None ):
        modeIndex = np.where(
            relativeValue >= float( svdCfg["relativeCutoff"] )
        )[0]
    else:
        nModes    = min( int( svdCfg["nModes"] ), len( singularValue ) )
        modeIndex = np.arange( nModes, dtype=int )
        
    if ( len( modeIndex ) == 0 ):
        modeIndex = np.array( [0], dtype=int )

    jacobianData = pd.DataFrame(
        jacobian, columns=[ variable["name"] for variable in variables ]
    )
    jacobianData.insert(
        0, "residual", np.arange( len( jacobian ) )
    )
    jacobianData.to_csv(
        config["files"]["jacobianFile"], index=False
    )
    
    svdData = {
        "mode"          : np.arange( 1, len( singularValue ) + 1 ),
        "singularValue" : singularValue,
        "relativeValue" : relativeValue,
        "retained"      : np.isin(
            np.arange( len( singularValue ) ), modeIndex
        ),
    }
    
    for varIndex, variable in enumerate( variables ):
        svdData[variable["name"]] = rightModeT[:,varIndex]
        
    pd.DataFrame( svdData ).to_csv(
        config["files"]["svdFile"], index=False
    )

    return(
        {
            "latentInitial" : latentInitial,
            "rightMode"     : rightModeT[modeIndex,:].T,
            "singularValue" : singularValue,
            "relativeValue" : relativeValue,
            "modeIndex"     : modeIndex,
            "nEval"         : 2 * len( variables ),
        }
    )

    # ------------------------------------------------- #
    # --- [1-11] evaluate objectives               --- #
    # ------------------------------------------------- #
    def _evaluateObjectives( records=None ):
        total  = 0.0
        detail = {}
        for objective in config["objectives"]:
            if ( not( objective["enabled"] ) ):
                continue

            name      = objective["name"]
            objType   = objective["type"]
            weight    = float( objective["weight"] )
            aggregate = objective["aggregate"]
            metric    = {}

            if ( objType == "periodicSigma" ):
                locationList = objective["locations"]
                if ( len( locationList ) < 2 ):
                    raise ValueError( "periodicSigma requires two or more locations." )

                pairPenalty = []
                for locIndex in range( 1, len( locationList ) ):
                    rowPrev   = _selectRows( records=records,
                                             location=locationList[locIndex-1] ).iloc[0]
                    rowCurr   = _selectRows( records=records,
                                             location=locationList[locIndex]   ).iloc[0]
                    sigmaPrev = _sigmaMatrix( row=rowPrev )
                    sigmaCurr = _sigmaMatrix( row=rowCurr )
                    diagPrev  = np.abs( np.diag( sigmaPrev ) )
                    scaleMat  = np.sqrt( np.outer( diagPrev, diagPrev ) )
                    scaleMat  = np.maximum( scaleMat, 1.0e-30 )
                    pairPenalty.append( np.mean( ( ( sigmaCurr-sigmaPrev ) / scaleMat )**2 ) )

                basePenalty   = _aggregate( values=np.array( pairPenalty ), method=aggregate )
                penalty       = weight * basePenalty
                metric["value"]        = np.sqrt( basePenalty )
                metric["target"]       = 0.0
                metric["residual"]     = np.sqrt( basePenalty )
                metric["normResidual"] = np.sqrt( basePenalty )
            else:
                rows   = _selectRows( records=records, location=objective["location"] )
                values = _evaluateExpression( rows=rows, expression=objective["expr"] )
                scale  = float( objective["scale"] )
                if ( scale <= 0.0 ):
                    raise ValueError( "objective scale must be positive: {}".format( name ) )

                metric["value"] = _aggregate( values=values, method=aggregate )
                if   ( objType == "target" ):
                    target     = float( objective["target"] )
                    normalized = ( values - target ) / scale
                    basePenalty = _aggregate( values=normalized**2, method=aggregate )
                    penalty     = weight * basePenalty
                    metric["target"]       = target
                    metric["residual"]     = metric["value"] - target
                    metric["normResidual"] = float( np.max( np.abs( normalized ) ) )
                elif ( objType == "range" ):
                    lower      = float( objective["min"] )
                    upper      = float( objective["max"] )
                    violation  = np.maximum( lower-values, 0.0 ) \
                               + np.maximum( values-upper, 0.0 )
                    normalized = violation / scale
                    basePenalty = _aggregate( values=normalized**2, method=aggregate )
                    penalty     = weight * basePenalty
                    metric["min"]          = lower
                    metric["max"]          = upper
                    metric["residual"]     = _aggregate( values=violation, method=aggregate )
                    metric["normResidual"] = float( np.max( normalized ) )
                elif ( objType == "minimize" ):
                    penalty = weight * _aggregate( values=values/scale, method=aggregate )
                elif ( objType == "maximize" ):
                    penalty = -weight * _aggregate( values=values/scale, method=aggregate )
                else:
                    raise ValueError( "Unknown objective type: {}".format( objType ) )

            metric["penalty"] = float( penalty )
            detail[name]       = metric
            total             += float( penalty )
        return( total, detail )

    # ------------------------------------------------- #
    # --- [1-12] run optimizer                      --- #
    # ------------------------------------------------- #
    def _runOptimizer():

        # ------------------------------------------------- #
        # --- [1] optimizer settings                    --- #
        # ------------------------------------------------- #
        history         = []
        iterCount       = 0
        evalCount       = 0
        bestObjective   = np.inf
        bestMaxResidual = np.nan
        bestVector      = None
        bestEvaluation  = None
        prevObjective   = np.nan

        optimizerCfg = config["optimizer"]
        initial = np.array(
            [ variable["initial"] for variable in variables ], dtype=float
        )
        lower = np.array(
            [ variable["min"] for variable in variables ], dtype=float
        )
        upper = np.array(
            [ variable["max"] for variable in variables ], dtype=float
        )
        bounds = list( zip( lower, upper ) )
        
        for x0, ( xmin, xmax ), variable in zip( initial, bounds, variables ):
            if not ( xmin <= x0 <= xmax ):
                raise ValueError(
                    "{} initial={} is outside [{}, {}]"
                    .format( variable["name"], x0, xmin, xmax )
                )
            
        svdCfg  = config["diagnostics"]["svd"]
        svdInfo = None
        
        if ( svdCfg["enabled"] ):
            svdInfo = _buildSvdBasis(
                initial=initial, lower=lower, upper=upper
            )
            
        if ( svdCfg["enabled"] and svdCfg["useModes"] ):
            nModes              = len( svdInfo["modeIndex"] )
            modeBound           = float( svdCfg["modeBound"] )
            optimizationInitial = np.zeros( nModes )
            optimizationBounds  = [ ( -modeBound, modeBound ) ] * nModes
            
            def _decodeVector( optimizationVector=None ):
                latent = svdInfo["latentInitial"] \
                    + np.dot( svdInfo["rightMode"], optimizationVector )
                return(
                    _latentToVector(
                        latent=latent, lower=lower, upper=upper
                    )
                )
        else:
            optimizationInitial = initial.copy()
            optimizationBounds  = bounds

    def _decodeVector( optimizationVector=None ):
        return( np.asarray( optimizationVector, dtype=float ) )
        for x0, ( xmin, xmax ), variable in zip( initial, bounds, variables ):
            if not ( xmin <= x0 <= xmax ):
                raise ValueError( "{} initial={} is outside [{}, {}]"\
                                  .format( variable["name"], x0, xmin, xmax ) )

        # ------------------------------------------------- #
        # --- [2] objective function                    --- #
        # ------------------------------------------------- #
        def _objective( vector ):
            nonlocal evalCount, bestObjective, bestMaxResidual
            nonlocal bestVector, bestEvaluation, prevObjective

            evalCount += 1
            
            records       = _evaluateImpactX( vector=vector, keepStat=False )
            value, detail = _evaluateObjectives( records=records )
            value         = float( value )
            
            residualList = [
                metric["normResidual"] for metric in detail.values()
                if ( "normResidual" in metric )
            ]
            maxResidual = max( residualList ) if residualList else np.nan
            prevBest    = bestObjective
            
            if ( np.isfinite( value ) and value < bestObjective ):
                bestObjective   = value
                bestMaxResidual = maxResidual
                bestVector      = np.asarray( vector, dtype=float ).copy()
                bestEvaluation  = evalCount

            row = {
                "iteration"      : iterCount,
                "evaluation"     : evalCount,
                "objective"      : value,
                "bestObjective"  : bestObjective,
                "deltaObjective" : abs( value - prevObjective ) if np.isfinite( prevObjective ) else np.nan,
                "deltaBest"      : abs( bestObjective - prevBest ) if np.isfinite( prevBest ) else np.nan,
            }
            prevObjective = value
            
            for varIndex, variable in enumerate( variables ):
                row[variable["name"]] = float( vector[varIndex] )

            for objName, metric in detail.items():
                for metricName, metricValue in metric.items():
                    row["{}.{}".format( metricName, objName )] = metricValue

            history.append( row )
            pd.DataFrame( history ).to_csv( config["files"]["historyFile"], index=False )
            return( value )

        # ------------------------------------------------- #
        # --- [3] callback                              --- #
        # ------------------------------------------------- #
        def _callback( vector ):
            nonlocal iterCount

            iterCount += 1
            printEvery = int( optimizerCfg["printEvery"] )
            
            if ( printEvery > 0 and iterCount % printEvery == 0 ):
                print( " iteration={:5d}"
                       "  evaluation={:5d}"
                       "  best={:.8e}"
                       "  maxNormResidual={:.4e}".format(
                           iterCount, evalCount,
                           bestObjective, bestMaxResidual ) )

        # ------------------------------------------------- #
        # --- [4] optimization                          --- #
        # ------------------------------------------------- #
        method = optimizerCfg["method"]
        
        if ( method == "differential_evolution" ):
            result = opt.differential_evolution(
                _objective, bounds=bounds,
                maxiter=int( optimizerCfg["maxIter"] ),
                popsize=int( optimizerCfg["popSize"] ),
                tol=float( optimizerCfg["tol"] ),
                polish=bool( optimizerCfg["polish"] ),
            )

        elif ( method == "Nelder-Mead" ):
            options = {
                "maxiter" : int  ( optimizerCfg["maxIter"] ),
                "xatol"   : float( optimizerCfg["xtol"]    ),
                "fatol"   : float( optimizerCfg["ftol"]    ),
            }
            if ( "maxEval" in optimizerCfg ):
                options["maxfev"] = int( optimizerCfg["maxEval"] )

            result = opt.minimize( _objective, initial, method="Nelder-Mead",
                                   bounds=bounds, callback=_callback, options=options )

        elif ( method == "Powell" ):
            options = {
                "maxiter" : int  ( optimizerCfg["maxIter"] ),
                "xtol"    : float( optimizerCfg["xtol"]    ),
                "ftol"    : float( optimizerCfg["ftol"]    ),
            }
            if ( "maxEval" in optimizerCfg ):
                options["maxfev"] = int( optimizerCfg["maxEval"] )
                
            result = opt.minimize( _objective, initial, method="Powell",
                                   bounds=bounds, callback=_callback, options=options )
            
        elif ( method == "least_squares" ):
            lower = np.array( [ bound[0] for bound in bounds ], dtype=float )
            upper = np.array( [ bound[1] for bound in bounds ], dtype=float )

            def _residual( vector ):
                nonlocal evalCount, bestObjective, bestMaxResidual
                nonlocal bestVector, bestEvaluation, prevObjective

                evalCount += 1

                records     = _evaluateImpactX( vector=vector, keepStat=False )
                residual    = _evaluateResiduals( records=records, vector=vector, initial=initial )
                value       = float( 0.5 * np.dot( residual, residual ) )
                _, detail   = _evaluateObjectives( records=records )
                maxResidual = float( np.max( np.abs( residual ) ) )
                prevBest    = bestObjective

                if ( np.isfinite( value ) and value < bestObjective ):
                    bestObjective   = value
                    bestMaxResidual = maxResidual
                    bestVector      = np.asarray( vector, dtype=float ).copy()
                    bestEvaluation  = evalCount
                row = { "iteration"      : np.nan,
                        "evaluation"     : evalCount,
                        "objective"      : value,
                        "bestObjective"  : bestObjective,
                        "deltaObjective" : abs( value - prevObjective )
                        if np.isfinite( prevObjective ) else np.nan,
                        "deltaBest"      : abs( bestObjective - prevBest )
                        if np.isfinite( prevBest ) else np.nan, }
                prevObjective = value

                for varIndex, variable in enumerate( variables ):
                    row[variable["name"]] = float( vector[varIndex] )

                for objName, metric in detail.items():
                    for metricName, metricValue in metric.items():
                        row["{}.{}".format( metricName, objName )] = metricValue

                history.append( row )
                pd.DataFrame( history ).to_csv( config["files"]["historyFile"], index=False )

                printEvery = int( optimizerCfg["printEvery"] )
                if ( printEvery > 0 and evalCount % printEvery == 0 ):
                    print( " evaluation={:5d}  best={:.8e}  maxResidual={:.4e}"
                           .format( evalCount, bestObjective, bestMaxResidual ) )
                return( residual )

            result = opt.least_squares( _residual, initial,
                                        bounds=( lower, upper ),
                                        method="trf",
                                        jac="2-point",
                                        x_scale=optimizerCfg["xScale"],
                                        loss=optimizerCfg["loss"],
                                        max_nfev=int( optimizerCfg["maxEval"] ),
                                        xtol=float( optimizerCfg["xtol"] ),
                                        ftol=float( optimizerCfg["ftol"] ),
                                        gtol=float( optimizerCfg["gtol"] ),
                                        verbose=0, )
            bestVector    = np.asarray( result.x, dtype=float ).copy()
            bestObjective = float( result.cost )
            
        # ------------------------------------------------- #
        # --- [5] return                                --- #
        # ------------------------------------------------- #
        return( result, history, bestVector, bestEvaluation, evalCount )
    
    
    # ------------------------------------------------- #
    # --- [2] load settings                         --- #
    # ------------------------------------------------- #
    with open( inpFile, "r" ) as fk:
        config = json5.load( fk )

    for outFile in [ config["files"]["resultFile"], \
                     config["files"]["historyFile"], \
                     config["files"]["statFile"], \
                     config["files"]["matchedParamsFile"], \
                     config["files"]["matchedBeamlineFile"] ]:
        os.makedirs( os.path.dirname( outFile ), exist_ok=True )
    if ( os.path.exists( config["files"]["historyFile"] ) ):
        os.remove( config["files"]["historyFile"] )

    with open( paramsFile, "r" ) as fk:
        params = json5.load( fk )
    with open( config["files"]["beamlineFile"], "r" ) as fk:
        elements = json5.load( fk )

    if ( config["matching"]["model"].lower() != "impactx" ):
        raise NotImplementedError( "matching.model='linopt' is for future implementation." )
    if ( config["matching"]["runMode"].lower() not in [ "evaluate", "optimize" ] ):
        raise ValueError( "matching.runMode must be 'evaluate' or 'optimize'." )

    # ------------------------------------------------- #
    # --- [3] matching section and variables        --- #
    # ------------------------------------------------- #
    sectionCfg = config["matching"]["section"]
    simElements, activeKeys, startIndex = _selectSection(
        elements=elements, startElement=sectionCfg["startElement"],
        endElement=sectionCfg["endElement"], nUse=params["sim.nUse.elements"] )
    variables  = _buildVariables( params=params, elements=simElements, activeKeys=activeKeys )

    print( "\n === ImpactX matching ===" )
    print( " model   :: {}".format( config["matching"]["model"] ) )
    print( " mode    :: {}".format( config["matching"]["mode"] ) )
    print( " runMode :: {}".format( config["matching"]["runMode"] ) )
    print( " section :: {} -> {}".format( activeKeys[0], activeKeys[-1] ) )
    print( " nvar    :: {}\n".format( len( variables ) ) )

    # ------------------------------------------------- #
    # --- [4] optimization / evaluation             --- #
    # ------------------------------------------------- #
    startTime = time.perf_counter()
    runMode = config["matching"]["runMode"].lower()
    if ( runMode == "optimize" ):
        if ( len( variables ) == 0 ):
            raise ValueError( "No optimization variable is enabled." )
        result, history, bestVector, bestEvaluation, evalCount = _runOptimizer()
        if ( bestVector is None ):
            raise RuntimeError( "Optimizer did not produce a finite objective." )
        success = bool( result.success )
        message = str( result.message )
    else:
        bestVector = np.array( [ variable["initial"] for variable in variables ], dtype=float )
        records    = _evaluateImpactX( vector=bestVector, keepStat=False )
        bestValue,detail = _evaluateObjectives( records=records )
        history    = [ { "evaluation":1, "objective":float( bestValue ),
                         "bestObjective":float( bestValue ),
                         **{ "{}.{}".format( metricName, objName ):metricValue
                             for objName, metric in detail.items()
                             for metricName, metricValue in metric.items() } } ]
        pd.DataFrame( history ).to_csv( config["files"]["historyFile"], index=False )
        result, success, message = None, True, "single evaluation"

    bestRecords = _evaluateImpactX( vector=bestVector, keepStat=True )
    bestValue, bestDetail = _evaluateObjectives( records=bestRecords )

    # ------------------------------------------------- #
    # --- [5] save results                          --- #
    # ------------------------------------------------- #
    bestParams, bestElements = _applyVariables( vector=bestVector, params=params,
        elements=simElements, variables=variables, activeKeys=activeKeys )
    variableResult = { variable["name"]:float( bestVector[varIndex] )
                       for varIndex, variable in enumerate( variables ) }

    resultData = {
        "success"      : success,
        "message"      : message,
        "model"        : config["matching"]["model"],
        "mode"         : config["matching"]["mode"],
        "section"      : { "startElement":activeKeys[0], "endElement":activeKeys[-1] },
        "objective"    : float( bestValue ),
        "objectiveTerm": bestDetail,
        "variables"    : variableResult,
        "elapsedS"     : float( time.perf_counter() - startTime ),
    }
    if ( result is not None ):
        if ( config["optimizer"]["method"] == "least_squares" ):
            resultData["optimizer"] = { "method"     : config["optimizer"]["method"],
                                        "nEval"      : int( evalCount ),
                                        "nfev"       : int( result.nfev ),
                                        "njev"       : int( result.njev ),
                                        "cost"       : float( result.cost ),
                                        "optimality" : float( result.optimality ), }
        else:
            resultData["optimizer"] = { "method":config["optimizer"]["method"],
                                        "nIter" :int( result.nit ),
                                        "nEval" :int( evalCount ),
                                        "nfev"  :int( result.nfev ), }
            
    with open( config["files"]["resultFile"], "w" ) as fk:
        json5.dump( resultData, fk, indent=4 )
    with open( config["files"]["matchedParamsFile"], "w" ) as fk:
        json5.dump( bestParams, fk, indent=4 )

    fullElements = copy.deepcopy( elements )
    for elemKey in bestElements:
        fullElements[elemKey] = bestElements[elemKey]
    with open( config["files"]["matchedBeamlineFile"], "w" ) as fk:
        json5.dump( fullElements, fk, indent=4 )

    # ------------------------------------------------- #
    # --- [6] print optimization summary            --- #
    # ------------------------------------------------- #
    print( "\n" + "==================================" )
    print( "===      Matching summary      ===" )
    print( "==================================" + "\n" )
    print( " success   :: {}".format( success ) )
    print( " message   :: {}".format( message ) )
    if ( result is not None ):
        print( " method    :: {}".format( config["optimizer"]["method"] ) )
        if ( config["optimizer"]["method"] == "least_squares" ):
            print( " nEval     :: {}".format( int( evalCount ) ) )
            print( " nFunEval  :: {}".format( int( result.nfev ) ) )
            print( " nJacEval  :: {}".format( int( result.njev ) ) )
            print( " cost      :: {:.8e} -> {:.8e}"
                   .format( float( history[0]["objective"] ), float( result.cost ) ) )
        else:
            print( " nIter     :: {}".format( int( result.nit ) ) )
            print( " nEval     :: {}".format( int( result.nfev ) ) )
            print( " objective :: {:.8e} -> {:.8e}"
                   .format( float( history[0]["objective"] ), bestValue ) )
            
        if ( config["optimizer"]["method"] == "Nelder-Mead" ):
            print( " xatol     :: {:.4e}".format( config["optimizer"]["xtol"] ) )
            print( " fatol     :: {:.4e}".format( config["optimizer"]["ftol"] ) )
    else:
        print( " objective :: {:.8e}".format( bestValue ) )

    print( "\n objectives:" )
    for objName, metric in bestDetail.items():
        line = "  {:20s} value={:.8e}  penalty={:.8e}".format( objName,
               metric["value"], metric["penalty"] )
        if ( "target" in metric ):
            line += "  target={:.8e}  residual={:.4e}".format( metric["target"],
                    metric["residual"] )
        if ( "normResidual" in metric ):
            line += "  normResidual={:.4e}".format( metric["normResidual"] )
        print( line )

    print( "\n variables:" )
    for varName, value in variableResult.items():
        print( "  {:20s} {:.8e}".format( varName, value ) )
    print( "\n result file       :: {}".format( config["files"]["resultFile"] ) )
    print( " statistics        :: {}".format( config["files"]["statFile"] ) )
    print( " optimization log  :: {}".format( config["files"]["historyFile"] ) )
    print( "   columns 'bestObjective', 'deltaObjective', 'deltaBest'," )
    print( "   'value.*', 'residual.*', and 'normResidual.*' show convergence.\n" )
    print( "\n" + "==================================" + "\n" )
    return( resultData )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #
if ( __name__=="__main__" ):
    inpFile = "dat/matching.json"
    optimize__quadFromEnvelope( inpFile=inpFile )
