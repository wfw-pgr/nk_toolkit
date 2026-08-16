import copy, os, time
import json5
import numpy          as np
import pandas         as pd
import scipy.optimize as opt
import scipy.stats    as stt
import nk_toolkit.impactx.io_toolkit  as itk
import nk_toolkit.impactx.run_toolkit as rtk


# ========================================================= #
# ===  matching.elementType -> tunable element mapping  === #
# ========================================================= #
# -- selects which lattice elements the "quadAll"/"quadFD"/"quadEach"       -- #
# -- variables act on. "quadrupole" (default) matches dat/beamline_impactx -- #
# -- .json's quadrupole/quadrupole.linear elements ("k" [1/m2 or T/m]);    -- #
# -- "solenoid" matches a quad2solenoid-converted lattice (dat/beamline_   -- #
# -- impactx_solenoid.json)'s solenoid elements ("ks" [1/m]), keyed as     -- #
# -- sol1, sol2, ... one-to-one with the qm1, qm2, ... it replaced. Select -- #
# -- via matching.elementType in the matching config (e.g. matching_       -- #
# -- solenoid.json); the variable schema (quadAll/quadFD/quadEach/twiss)   -- #
# -- is unchanged between the two.                                        -- #
# ========================================================= #
ELEMENT_TYPE_CONFIG = {
    "quadrupole" : { "types" : [ "quadrupole", "quadrupole.linear" ],
                     "field" : "k",  "prefix" : "qm",  "label" : "QM", },
    "solenoid"   : { "types" : [ "solenoid" ],
                     "field" : "ks", "prefix" : "sol", "label" : "Solenoid", },
}


# ========================================================= #
# ===  match_toolkit.py                                 === #
# ========================================================= #
#
# Two independent optimizer families share this function, selected via
# matching.json's optimizer.method:
#  - least_squares / Powell / Nelder-Mead / differential_evolution / bayesian
#    -- driven by a residual-vector built from many independent objective
#    terms ([3] "objectives") + SVD-regularized quad space ([regularization]).
#  - cma-es -- an independent, from-scratch alternative built around a single
#    physically-scoped scalar cost ( envelope smoothness + over-expansion +
#    over-focusing + target beam size (+ transmission for tracking), see [3b]
#    "objective" ) and a (mu/mu_w,lambda)-CMA-ES optimizer with boundary-
#    penalty constraint handling ( optimizer.cmaEs ).
#
# Why CMA-ES as an alternative: quad matching under space charge is a rugged,
# mildly multimodal, ~O(10-40) dimensional black-box problem where the
# mode==tracking objective is inherently noisy (finite macro-particle
# sampling). Gradient-based methods (least_squares) need a clean finite-
# difference Jacobian and degrade on noisy objectives; CMA-ES estimates a
# full covariance of the search directions from rank information only (never
# from raw fitness differences or gradients), which makes it robust to that
# noise and self-adapting to the natural correlations between neighboring
# quads -- while still converging to least-squares-like precision once it is
# close to the optimum, which is why a short local Nelder-Mead polish is
# appended at the end.
#
# Optional CMA-ES second stage (matching.trackingRefine): at high beam
# current the space-charge force is strong enough that "envelope" mode's
# linear/ellipsoidal space-charge model can disagree substantially with full
# particle tracking (nonlinear fields, halo, real losses) -- squeezing the
# beam too tight locally is exactly where this gap opens up, since space
# charge defocusing grows sharply as the local beam size shrinks. Rather than
# run the whole global CMA-ES search directly in "tracking" mode (~1000x the
# per-evaluation cost of "envelope"), stage 1's envelope optimum seeds a
# second, small, local CMA-ES stage evaluated directly in "tracking" mode
# with bounds narrowed around that point, correcting for the gap at a
# fraction of the cost of a from-scratch tracking-mode search.
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
            prefix  = elementCfg["prefix"]
            for item in qmID:
                item = str( item ).strip()
                if ( "-" in item ):
                    startId, endId = item.split( "-", 1 )
                    startId       = int( startId )
                    endId         = int(   endId )
                    if ( startId > endId ):
                        raise ValueError( "Invalid qmID range: {}".format( item ) )
                    qmList += [ "{}{}".format( prefix, qmId ) for qmId in range( startId, endId + 1 ) ]
                else:
                    qmList.append( "{}{}".format( prefix, int( item ) ) )
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
                    raise ValueError( "{} is already assigned : {}"
                                      .format( elementCfg["label"], qmName ) )
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
                raise ValueError( "{} variable is outside matching section: {}"
                                  .format( elementCfg["label"], variable["target"] ) )
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

        elementTypes = elementCfg["types"]
        elementField = elementCfg["field"]
        for elemKey in activeKeys:
            elem = elements_[elemKey]
            if ( elem["type"] not in elementTypes ):
                continue

            elemName = elem["name"]
            k0       = float( elements[elemKey][elementField] )
            factor   = factors["quadAll"]
            if   ( k0 > 0.0 ):
                factor *= factors["quadFD"]["QF"]
            elif ( k0 < 0.0 ):
                factor *= factors["quadFD"]["QD"]

            if ( elemKey in factors["quadEach"] ):
                factor *= factors["quadEach"][elemKey]
            if ( elemName in factors["quadEach"] ):
                factor *= factors["quadEach"][elemName]
            elem[elementField] = k0 * factor

        return( params_, elements_ )

    # ------------------------------------------------- #
    # --- [1-6] make statistics                    --- #
    # ------------------------------------------------- #
    def _makeStats( rawStat=None, simElements=None, activeKeys=None, startIndex=0 ):
        elemKeys = list( simElements.keys() )
        stat = rawStat.sort_values( "step" )
        stat = stat.drop_duplicates( subset=["step"], keep="last" )
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
    def _evaluateImpactX( vector=None, mode=None, keepStat=False ):
        """ Evaluate ImpactX simulation """
        params_, elements_ = _applyVariables( vector=vector, params=params,
                                              elements=simElements, variables=variables,
                                              activeKeys=activeKeys )
        runMode   = mode if ( mode is not None ) else config["matching"]["mode"]
        runResult = rtk.execute__impactx( params=params_, elements=elements_,
                                          workDir=impactxDir, runMode=runMode,
                                          clearDiags=True, add_bpm=False, saveRecords=False,
                                          saveLattice=False, verbose=False )
        stat      = itk.get__beamStats  ( statFile=runResult["statFile"],
                                          refpFile=runResult["refpFile"] )
        records   = _makeStats          ( rawStat=stat, simElements=elements_,
                                          activeKeys=activeKeys, startIndex=startIndex )
        if ( keepStat ):
            records.to_csv( config["files"]["statFile"], index=False )
        return( records, params_, elements_ )

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

    def _regularizationResidual( vector=None, initial=None ):
        regularization = config["regularization"]
        if ( regularization["normalization"] != "bounds" ):
            raise ValueError( "regularization.normalization must be 'bounds'." )
        lower  = np.array( [ variable["min"] for variable in variables ] )
        upper  = np.array( [ variable["max"] for variable in variables ] )
        span   = np.maximum( upper - lower, 1.0e-12 )
        weight = float( regularization["weight"] )
        return( np.sqrt( weight ) * ( vector - initial ) / span )

    def _evaluateResiduals( records=None, vector=None, initial=None,
                            includeRegularization=True ):
        residualList = []

        for objective in config["objectives"]:
            if ( not( objective["enabled"] ) ):
                continue
            if ( objective["type"] == "periodicSigma" ):
                locationList = objective["locations"]
                residualPair = []
                for locIndex in range( 1, len( locationList ) ):
                    rowPrev = _selectRows( records=records,
                                           location=locationList[locIndex-1] ).iloc[0]
                    rowCurr = _selectRows( records=records,
                                           location=locationList[locIndex] ).iloc[0]
                    sigmaPrev = _sigmaMatrix( row=rowPrev )
                    sigmaCurr = _sigmaMatrix( row=rowCurr )
                    diagPrev  = np.abs( np.diag( sigmaPrev ) )
                    scaleMat  = np.sqrt( np.outer( diagPrev, diagPrev ) )
                    scaleMat  = np.maximum( scaleMat, 1.0e-30 )
                    residualPair.append( ( sigmaCurr - sigmaPrev ).ravel()
                                         / scaleMat.ravel() )
                residual = np.concatenate( residualPair )
                residualList.append( np.sqrt( float( objective["weight"] ) )
                                     * residual / np.sqrt( len( residual ) ) )
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
                if ( "target" not in objective ):
                    raise ValueError( "maximize requires target for least_squares/SVD: {}"
                                      .format( objective["name"] ) )
                normalized = ( values - float( objective["target"] ) ) / scale

            else:
                raise ValueError( "Unknown least_squares objective: {}".format( objType ) )

            residual = _compressResidual( values=normalized, method=aggregate )
            residualList.append( np.sqrt( weight ) * residual )

        if ( includeRegularization and config["regularization"]["enabled"] ):
            residualList.append( _regularizationResidual( vector=vector,
                                                          initial=initial ) )

        return( np.concatenate( residualList ) )

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
    
    def _buildSvdBasis( initial=None, lower=None, upper=None, variableIndex=None ):
        svdCfg        = config["regularization"]["svd"]
        latentInitial = _vectorToLatent( vector=initial, lower=lower, upper=upper )
        step          = float( svdCfg["step"] )
        jacobianList  = []

        for varIndex in variableIndex:
            latentPlus             = latentInitial.copy()
            latentMinus            = latentInitial.copy()
            latentPlus[varIndex]  += step
            latentMinus[varIndex] -= step

            vectorPlus  = _latentToVector( latent=latentPlus,
                                           lower=lower, upper=upper )
            vectorMinus = _latentToVector( latent=latentMinus,
                                           lower=lower, upper=upper )

            recordsPlus,  _, _ = _evaluateImpactX( vector=vectorPlus, keepStat=False )
            recordsMinus, _, _ = _evaluateImpactX( vector=vectorMinus, keepStat=False )

            residualPlus = _evaluateResiduals(
                records=recordsPlus, vector=vectorPlus, initial=initial,
                includeRegularization=False
            )
            residualMinus = _evaluateResiduals(
                records=recordsMinus, vector=vectorMinus, initial=initial,
                includeRegularization=False
            )
            jacobianList.append( ( residualPlus - residualMinus ) / ( 2.0 * step ) )

        jacobian      = np.column_stack( jacobianList )
        svdResult     = np.linalg.svd( jacobian, full_matrices=False )
        singularValue = svdResult[1]
        rightModeT    = svdResult[2]

        if ( not( np.all( np.isfinite( singularValue ) ) ) ):
            raise ValueError( "SVD contains non-finite singular values." )
        if ( len( singularValue ) == 0 or singularValue[0] <= 0.0 ):
            raise ValueError( "SVD Jacobian has no finite sensitivity." )

        relativeValue = singularValue / singularValue[0]
        modeIndex     = np.arange( len( singularValue ), dtype=int )
        if ( svdCfg["relativeCutoff"] is not None ):
            relativeCutoff = float( svdCfg["relativeCutoff"] )
            modeIndex = modeIndex[relativeValue[modeIndex] >= relativeCutoff]
        if ( svdCfg["nModes"] is not None ):
            nModes    = int( svdCfg["nModes"] )
            modeIndex = modeIndex[:nModes]
        if ( len( modeIndex ) == 0 ):
            raise ValueError( "No SVD mode satisfies the selection conditions." )
        if ( len( modeIndex ) == 0 ):
            modeIndex = np.array( [0], dtype=int )

        variableName = [ variables[varIndex]["name"] for varIndex in variableIndex ]

        jacobianData = pd.DataFrame( jacobian, columns=variableName )
        jacobianData.insert( 0, "residual", np.arange( len( jacobian ) ) )
        jacobianData.to_csv( config["files"]["jacobianFile"], index=False )

        svdData = {
            "mode"          : np.arange( 1, len( singularValue ) + 1 ),
            "singularValue" : singularValue,
            "relativeValue" : relativeValue,
            "retained"      : np.isin( np.arange( len( singularValue ) ), modeIndex ),
        }
        for modeVarIndex, varName in enumerate( variableName ):
            svdData[varName] = rightModeT[:,modeVarIndex]

        pd.DataFrame( svdData ).to_csv( config["files"]["svdFile"], index=False )

        return( { "latentInitial":latentInitial,
                  "rightMode"    :rightModeT[modeIndex,:].T,
                  "variableIndex":np.asarray( variableIndex, dtype=int ),
                  "singularValue":singularValue,
                  "relativeValue":relativeValue,
                  "modeIndex"    :modeIndex,
                  "nEval"        :2 * len( variableIndex ), } )

    # ------------------------------------------------- #
    # --- [1-12] print SVD summary                  --- #
    # ------------------------------------------------- #
    def _printSvdSummary( svdInfo=None, initial=None, lower=None, upper=None,
                          modeBoundVec=None, outerRound=0 ):
        svdCfg       = config["regularization"]["svd"]
        variableIndex = svdInfo["variableIndex"]
        modeIndex     = svdInfo["modeIndex"]
        retainedMode  = set( modeIndex.tolist() )
        if ( modeBoundVec is None ):
            modeBoundVec = np.full( len( modeIndex ), float( svdCfg["modeBound"] ) )

        elemLabel = elementCfg["label"]
        print( "\n" + "==================================" )
        print( "===  SVD summary (round {:2d})     ===".format( outerRound ) )
        print( "==================================" + "\n" )
        print( " {} variables :: {}".format( elemLabel, len( variableIndex ) ) )
        print( " SVD modes    :: {} / {}".format(
            len( modeIndex ), len( svdInfo["singularValue"] )
        ) )
        if ( np.allclose( modeBoundVec, modeBoundVec[0] ) ):
            print( " mode bound   :: +/- {:.4e}\n".format( float( modeBoundVec[0] ) ) )
        else:
            print( " mode bound   :: +/- [{:.4e} .. {:.4e}]  (per-mode scaled)\n".format(
                float( np.min( modeBoundVec ) ), float( np.max( modeBoundVec ) ) ) )

        print( " singular values:" )
        print( "  mode       singularValue      relativeValue   retained" )
        for modeNo in range( len( svdInfo["singularValue"] ) ):
            retained = "yes" if ( modeNo in retainedMode ) else "no"
            print( "  {:4d}   {:16.8e}   {:16.8e}   {:>8s}".format(
                modeNo + 1, svdInfo["singularValue"][modeNo],
                svdInfo["relativeValue"][modeNo], retained
            ) )

        latentInitial = svdInfo["latentInitial"][variableIndex]
        rightMode     = svdInfo["rightMode"]

        latentSpan = np.dot( np.abs( rightMode ), modeBoundVec )
        valueMin   = _latentToVector(
            latent=latentInitial - latentSpan,
            lower=lower[variableIndex], upper=upper[variableIndex]
        )
        valueMax = _latentToVector(
            latent=latentInitial + latentSpan,
            lower=lower[variableIndex], upper=upper[variableIndex]
        )

        initialQm = initial[variableIndex]
        deltaMin  = 100.0 * ( valueMin - initialQm ) / initialQm
        deltaMax  = 100.0 * ( valueMax - initialQm ) / initialQm

        dominantIndex  = np.argmax( np.abs( rightMode ), axis=1 )
        dominantMode   = modeIndex[dominantIndex] + 1
        dominantWeight = rightMode[np.arange( len( variableIndex ) ), dominantIndex]

        print( "\n {} reachable range:".format( elemLabel ) )
        print( "  variable                  initial        min        max"
               "     dMin[%]    dMax[%]   main mode" )

        for qmIndex, varIndex in enumerate( variableIndex ):
            print( "  {:24s} {:10.5f} {:10.5f} {:10.5f}"
                   " {:10.2f} {:10.2f}   {:3d} ({:+.3f})".format(
                       variables[varIndex]["name"], initialQm[qmIndex],
                       valueMin[qmIndex], valueMax[qmIndex],
                       deltaMin[qmIndex], deltaMax[qmIndex],
                       dominantMode[qmIndex], dominantWeight[qmIndex]
                   ) )

        print( "\n  min/max: each {} coordinate-wise reachable range".format( elemLabel ) )
        print( "  main mode: largest absolute SVD component for each {}\n".format( elemLabel ) )
    
    # ------------------------------------------------- #
    # --- [1-12] build sigma matrix                 --- #
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
    # --- [1-13] evaluate objectives               --- #
    # ------------------------------------------------- #
    def _evaluateObjectives( records=None, vector=None, initial=None ):
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
                    difference = ( sigmaCurr - sigmaPrev ) / scaleMat
                    pairPenalty.append( np.mean( difference**2 ) )

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
                    violation  = np.maximum( lower - values, 0.0 ) \
                               + np.maximum( values - upper, 0.0 )
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

        if ( config["regularization"]["enabled"] and vector is not None ):
            residual = _regularizationResidual( vector=vector, initial=initial )
            normValue = residual / np.sqrt( float( config["regularization"]["weight"] ) )
            penalty   = float( np.dot( residual, residual ) )
            detail["regularization"] = {
                "value"   :float( np.sqrt( np.mean( normValue**2 ) ) ), "target":0.0,
                "residual":float( np.linalg.norm( normValue ) ), "penalty":penalty,
                "normResidual":float( np.max( np.abs( normValue ) ) ), }
            total += penalty
        return( total, detail )

    # ------------------------------------------------- #
    # --- [1-14] run optimizer                      --- #
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
        outerRound      = 0

        optimizerCfg = config["optimizer"]
        initial = np.array( [ variable["initial"] for variable in variables ], dtype=float )
        lower   = np.array( [ variable["min"] for variable in variables ], dtype=float )
        upper   = np.array( [ variable["max"] for variable in variables ], dtype=float )
        bounds  = list( zip( lower, upper ) )

        for x0, ( xmin, xmax ), variable in zip( initial, bounds, variables ):
            if not ( xmin <= x0 <= xmax ):
                raise ValueError( "{} initial={} is outside [{}, {}]"\
                                  .format( variable["name"], x0, xmin, xmax ) )

        svdCfg  = config["regularization"]["svd"]
        svdInfo = None

        svdKind = [ "quadAll", "quadFD", "quadEach" ]
        svdVariableIndex = np.array( [ varIndex for varIndex, variable in enumerate( variables )
                                       if ( variable["kind"] in svdKind ) ], dtype=int )
        directVariableIndex = np.array( [ varIndex for varIndex, variable in enumerate( variables )
                                          if ( variable["kind"] not in svdKind ) ], dtype=int )
        if ( svdCfg["enabled"] and len( svdVariableIndex ) == 0 ):
            raise ValueError( "No QM variable is enabled for SVD." )

        # -- successive re-linearization: rebuild the SVD basis around the -- #
        # -- previous round's best point instead of relying on a single    -- #
        # -- local-linear model taken at the initial lattice.              -- #
        relinCfg      = svdCfg["relinearize"] if ( "relinearize" in svdCfg ) else {}
        relinEnabled  = bool( svdCfg["enabled"] and svdCfg["useModes"]
                              and relinCfg.get( "enabled", False ) )
        maxOuterRound = max( 1, int( relinCfg["maxRounds"] ) ) if relinEnabled else 1
        relinTol      = float( relinCfg["tol"] ) if relinEnabled else 0.0

        optimizationInitial = None
        optimizationBounds  = None
        _decodeVector        = None
        nModes                = 0

        # ------------------------------------------------- #
        # --- [2] (re-)linearize and build search space --- #
        # ------------------------------------------------- #
        def _prepareRound( linCenter=None ):
            nonlocal svdInfo, optimizationInitial, optimizationBounds, _decodeVector, nModes

            modeBoundVec = None
            if ( svdCfg["enabled"] ):
                svdInfo = _buildSvdBasis( initial=linCenter, lower=lower, upper=upper,
                                          variableIndex=svdVariableIndex )
                nModes        = len( svdInfo["modeIndex"] )
                modeBoundBase = float( svdCfg["modeBound"] )

                if ( svdCfg.get( "modeBoundScaling", False ) ):
                    # -- weakly-sensitive modes get proportionally more    -- #
                    # -- latent latitude so the search budget isn't spent  -- #
                    # -- equally on modes that barely move the objective.  -- #
                    maxScale = float( svdCfg.get( "maxModeBoundScale", 5.0 ) )
                    relativeRetained = np.maximum(
                        svdInfo["relativeValue"][svdInfo["modeIndex"]], 1.0e-6 )
                    boundScale = np.clip( 1.0 / np.sqrt( relativeRetained ), 1.0, maxScale )
                else:
                    boundScale = np.ones( nModes )
                modeBoundVec = modeBoundBase * boundScale

                _printSvdSummary( svdInfo=svdInfo, initial=linCenter, lower=lower, upper=upper,
                                  modeBoundVec=modeBoundVec, outerRound=outerRound )

            if ( svdCfg["enabled"] and svdCfg["useModes"] ):
                optimizationInitial = np.concatenate(
                    ( np.zeros( nModes ), linCenter[directVariableIndex] ) )
                optimizationBounds  = [ ( -b, b ) for b in modeBoundVec ] + \
                    [ bounds[varIndex] for varIndex in directVariableIndex ]

                def _decode( optimizationVector=None ):
                    vector       = linCenter.copy()
                    modeVector   = optimizationVector[:nModes]
                    directVector = optimizationVector[nModes:]

                    latent = svdInfo["latentInitial"][svdVariableIndex] \
                           + np.dot( svdInfo["rightMode"], modeVector )
                    vector[svdVariableIndex] = _latentToVector( latent=latent,
                                                                lower=lower[svdVariableIndex],
                                                                upper=upper[svdVariableIndex] )
                    vector[directVariableIndex] = directVector
                    return( vector )
                _decodeVector = _decode

            else:
                optimizationInitial = linCenter.copy()
                optimizationBounds  = bounds

                def _decode( optimizationVector=None ):
                    return( np.asarray( optimizationVector, dtype=float ) )
                _decodeVector = _decode

        # ------------------------------------------------- #
        # --- [3] objective function                    --- #
        # ------------------------------------------------- #
        def _objective( optimizationVector ):
            nonlocal evalCount, bestObjective, bestMaxResidual
            nonlocal bestVector, bestEvaluation, prevObjective

            evalCount += 1
            vector         = _decodeVector( optimizationVector=optimizationVector )
            records, _, _ = _evaluateImpactX( vector=vector, keepStat=False )
            value, detail = _evaluateObjectives( records=records, vector=vector,
                                                 initial=initial )
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
                "outerRound"     : outerRound,
                "iteration"      : iterCount,
                "evaluation"     : evalCount,
                "objective"      : value,
                "bestObjective"  : bestObjective,
                "deltaObjective" : abs( value - prevObjective ) \
                if np.isfinite( prevObjective ) else np.nan,
                "deltaBest"      : abs( bestObjective - prevBest ) \
                if np.isfinite( prevBest ) else np.nan,
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
        # --- [4] callback                              --- #
        # ------------------------------------------------- #
        def _callback( _optimizationVector ):
            nonlocal iterCount

            iterCount += 1
            printEvery = int( optimizerCfg["printEvery"] )

            if ( printEvery > 0 and iterCount % printEvery == 0 ):
                print( " round={:2d}  iteration={:5d}  evaluation={:5d}  best={:.8e}"
                       "  maxNormResidual={:.4e}"
                       .format( outerRound, iterCount, evalCount, bestObjective, bestMaxResidual ) )

        # ------------------------------------------------- #
        # --- [5] Bayesian optimization                 --- #
        # ------------------------------------------------- #
        def _runBayesian( evalBudget=None ):
            bayesCfg   = optimizerCfg["bayesian"]
            maxEval    = int( bayesCfg["maxEval"] )
            if ( evalBudget is not None ):
                maxEval = max( 1, min( maxEval, int( evalBudget ) ) )
            nInitial   = max( 1, min( int( bayesCfg["nInitial"] ), maxEval ) )
            nCandidate = int( bayesCfg["nCandidate"] )
            length     = float( bayesCfg["lengthScale"] )
            noise      = float( bayesCfg["noise"] )
            xi         = float( bayesCfg["xi"] )
            patience   = int( bayesCfg["patience"] )
            minImprove = float( bayesCfg["minImprovement"] )
            localRatio = float( bayesCfg["localRatio"] )
            localScale = float( bayesCfg["localScale"] )
            seed       = int( bayesCfg["seed"] )

            optLower = np.array( [ bound[0] for bound in optimizationBounds ], dtype=float )
            optUpper = np.array( [ bound[1] for bound in optimizationBounds ], dtype=float )
            optWidth = optUpper - optLower
            nVariable = len( optimizationInitial )
            rng       = np.random.default_rng( seed )

            def _toNormalized( vector=None ):
                return( ( np.asarray( vector ) - optLower ) / optWidth )

            def _fromNormalized( normalized=None ):
                return( optLower + np.asarray( normalized ) * optWidth )

            def _kernel( first=None, second=None ):
                delta    = ( first[:,None,:] - second[None,:,:] ) / length
                distance = np.sqrt( np.sum( delta**2, axis=2 ) )
                scaled   = np.sqrt( 5.0 ) * distance
                return( ( 1.0 + scaled + scaled**2 / 3.0 ) * np.exp( -scaled ) )

            initialNormalized = _toNormalized( vector=optimizationInitial )
            sampleList = [ initialNormalized ]
            if ( nInitial > 1 ):
                sampler = stt.qmc.LatinHypercube( d=nVariable, seed=seed )
                sampleList += list( sampler.random( n=nInitial - 1 ) )

            valueList = []
            for normalized in sampleList:
                valueList.append( _objective( _fromNormalized( normalized=normalized ) ) )
                _callback( normalized )

            stallCount = 0
            while ( len( valueList ) < maxEval and stallCount < patience ):
                sample = np.asarray( sampleList, dtype=float )
                value  = np.asarray( valueList, dtype=float )
                if ( not( np.all( np.isfinite( value ) ) ) ):
                    raise ValueError( "Bayesian optimization requires finite objectives." )

                valueMean = float( np.mean( value ) )
                valueStd  = float( np.std ( value ) )
                if ( valueStd <= 0.0 ):
                    valueStd = 1.0
                target = ( value - valueMean ) / valueStd

                covariance = _kernel( first=sample, second=sample )
                covariance += noise * np.eye( len( sample ) )
                cholesky = np.linalg.cholesky( covariance )
                alpha    = np.linalg.solve( cholesky.T,
                                           np.linalg.solve( cholesky, target ) )

                nLocal  = int( nCandidate * localRatio )
                nGlobal = nCandidate - nLocal
                sampler = stt.qmc.LatinHypercube( d=nVariable,
                                                  seed=seed + len( valueList ) )
                candidate = sampler.random( n=nGlobal )
                bestPoint = sample[int( np.argmin( value ) )]
                local = bestPoint + localScale * rng.normal( size=( nLocal, nVariable ) )
                candidate = np.vstack( ( candidate, np.clip( local, 0.0, 1.0 ) ) )

                crossKernel = _kernel( first=candidate, second=sample )
                predMean    = np.dot( crossKernel, alpha )
                solved      = np.linalg.solve( cholesky, crossKernel.T )
                predStd     = np.sqrt( np.maximum( 1.0 - np.sum( solved**2, axis=0 ),
                                                   1.0e-14 ) )
                improvement = np.min( target ) - predMean - xi
                ratio       = improvement / predStd
                acquisition = improvement * stt.norm.cdf( ratio ) \
                            + predStd * stt.norm.pdf( ratio )
                nextPoint = candidate[int( np.argmax( acquisition ) )]

                previousBest = float( np.min( value ) )
                nextValue    = _objective( _fromNormalized( normalized=nextPoint ) )
                sampleList.append( nextPoint )
                valueList.append( nextValue )
                _callback( nextPoint )

                threshold = minImprove * max( abs( previousBest ), 1.0 )
                if ( previousBest - nextValue > threshold ):
                    stallCount = 0
                else:
                    stallCount += 1

            bestIndex = int( np.argmin( valueList ) )
            message   = "maximum evaluations reached"
            if ( stallCount >= patience ):
                message = "no significant improvement for {} evaluations".format( patience )
            result = opt.OptimizeResult( x=_fromNormalized( sampleList[bestIndex] ),
                                         fun=float( valueList[bestIndex] ),
                                         nit=len( valueList ) - nInitial,
                                         nfev=len( valueList ), success=True,
                                         status=0, message=message )
            return( result )

        # ------------------------------------------------- #
        # --- [6] run the selected method once           --- #
        # ------------------------------------------------- #
        def _runMethodOnce( evalBudget=None ):
            method = optimizerCfg["method"]

            if ( method == "bayesian" ):
                return( _runBayesian( evalBudget=evalBudget ) )

            elif ( method == "differential_evolution" ):
                # -- differential_evolution has no direct nfev cap; maxiter/popSize --#
                # -- already bound its cost the same way in a single round, so the  --#
                # -- per-round budget is not enforced here (pre-existing behavior). --#
                return( opt.differential_evolution( _objective, bounds=optimizationBounds,
                                                     maxiter=int( optimizerCfg["maxIter"] ),
                                                     popsize=int( optimizerCfg["popSize"] ),
                                                     tol=float( optimizerCfg["tol"] ),
                                                     polish=bool( optimizerCfg["polish"] ) ) )

            elif ( method == "Nelder-Mead" ):
                options = {
                    "maxiter" : int  ( optimizerCfg["maxIter"] ),
                    "xatol"   : float( optimizerCfg["xtol"]    ),
                    "fatol"   : float( optimizerCfg["ftol"]    ),
                }
                if ( evalBudget is not None ):
                    options["maxfev"] = int( evalBudget )
                elif ( "maxEval" in optimizerCfg ):
                    options["maxfev"] = int( optimizerCfg["maxEval"] )

                return( opt.minimize( _objective, optimizationInitial, method="Nelder-Mead",
                                      bounds=optimizationBounds, callback=_callback,
                                      options=options ) )

            elif ( method == "Powell" ):
                options = {
                    "maxiter" : int  ( optimizerCfg["maxIter"] ),
                    "xtol"    : float( optimizerCfg["xtol"]    ),
                    "ftol"    : float( optimizerCfg["ftol"]    ),
                }
                if ( evalBudget is not None ):
                    options["maxfev"] = int( evalBudget )
                elif ( "maxEval" in optimizerCfg ):
                    options["maxfev"] = int( optimizerCfg["maxEval"] )

                return( opt.minimize( _objective, optimizationInitial, method="Powell",
                                      bounds=optimizationBounds, callback=_callback,
                                      options=options ) )

            elif ( method == "least_squares" ):
                optimizationLower = np.array( [ bound[0] for bound in optimizationBounds ],
                                              dtype=float )
                optimizationUpper = np.array( [ bound[1] for bound in optimizationBounds ],
                                              dtype=float )

                def _residual( optimizationVector ):
                    nonlocal evalCount, bestObjective, bestMaxResidual
                    nonlocal bestVector, bestEvaluation, prevObjective

                    evalCount += 1
                    vector         = _decodeVector( optimizationVector=optimizationVector )
                    records, _, _ = _evaluateImpactX( vector=vector, keepStat=False )
                    residual      = _evaluateResiduals( records=records, vector=vector,
                                                        initial=initial )
                    value         = float( 0.5 * np.dot( residual, residual ) )
                    _, detail     = _evaluateObjectives( records=records, vector=vector,
                                                         initial=initial )
                    maxResidual   = float( np.max( np.abs( residual ) ) )
                    prevBest      = bestObjective

                    if ( np.isfinite( value ) and value < bestObjective ):
                        bestObjective   = value
                        bestMaxResidual = maxResidual
                        bestVector      = np.asarray( vector, dtype=float ).copy()
                        bestEvaluation  = evalCount

                    row = { "outerRound"     : outerRound,
                            "iteration"      : np.nan,
                            "evaluation"     : evalCount,
                            "objective"      : value,
                            "bestObjective"  : bestObjective,
                            "deltaObjective" : abs( value - prevObjective ) \
                            if np.isfinite( prevObjective ) else np.nan,
                            "deltaBest"      : abs( bestObjective - prevBest ) \
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
                        print( " round={:2d}  evaluation={:5d}  best={:.8e}  maxResidual={:.4e}"
                               .format( outerRound, evalCount, bestObjective, bestMaxResidual ) )
                    return( residual )

                # -- scipy's own `nfev` (what `max_nfev` bounds) only counts    -- #
                # -- "major" evaluations: with jac="2-point" each major step   -- #
                # -- silently costs ~(ndim+1) additional real ImpactX calls    -- #
                # -- for the finite-difference Jacobian that nfev never sees.  -- #
                # -- Divide the real-call budget accordingly so `optimizer.    -- #
                # -- maxEval` bounds actual ImpactX evaluations, not nfev.     -- #
                targetRealCalls = int( evalBudget ) if ( evalBudget is not None ) \
                    else int( optimizerCfg["maxEval"] )
                nDim    = len( optimizationInitial )
                maxNfev = max( 1, targetRealCalls // ( nDim + 1 ) )

                return( opt.least_squares( _residual, optimizationInitial,
                                           bounds=( optimizationLower, optimizationUpper ),
                                           method="trf", jac="2-point",
                                           x_scale=optimizerCfg["xScale"],
                                           loss=optimizerCfg["loss"],
                                           max_nfev=maxNfev,
                                           xtol=float( optimizerCfg["xtol"] ),
                                           ftol=float( optimizerCfg["ftol"] ),
                                           gtol=float( optimizerCfg["gtol"] ), verbose=0 ) )
            else:
                raise ValueError( "Unknown optimizer method: {}".format( method ) )

        # ------------------------------------------------- #
        # --- [7] outer re-linearization loop            --- #
        # ------------------------------------------------- #
        # -- round 0 always linearizes at the configured initial vector;   -- #
        # -- when relinearize.enabled, later rounds re-center the SVD      -- #
        # -- basis on the previous round's best point and continue        -- #
        # -- optimizing, since a single linear model taken far from the   -- #
        # -- optimum can misrepresent which directions are worth moving.  -- #
        # -- optimizer.maxEval is a TOTAL budget across every round: each -- #
        # -- round only gets whatever is left after previous rounds, so   -- #
        # -- relinearize.maxRounds cannot multiply the configured budget. -- #
        globalMaxEval = int( optimizerCfg["maxEval"] )
        linCenter     = initial.copy()
        result        = None
        prevRoundBest = np.inf
        nRounds       = 0

        for roundIndex in range( maxOuterRound ):
            remainingEval = globalMaxEval - evalCount
            if ( remainingEval <= 0 ):
                break

            outerRound = roundIndex
            _prepareRound( linCenter=linCenter )
            result   = _runMethodOnce( evalBudget=remainingEval )
            nRounds += 1

            if ( bestVector is None ):
                break
            if ( roundIndex > 0 ):
                improvement = prevRoundBest - bestObjective
                threshold   = relinTol * max( abs( prevRoundBest ), 1.0 )
                if ( not( np.isfinite( improvement ) ) or improvement < threshold ):
                    break
            prevRoundBest = bestObjective
            linCenter     = bestVector.copy()

        # ------------------------------------------------- #
        # --- [8] return                                --- #
        # ------------------------------------------------- #
        return( result, history, bestVector, bestEvaluation, evalCount, svdInfo, nRounds )

    # ------------------------------------------------- #
    # --- [1-15] CMA-ES optimizer ( optimizer.method="cma-es" ) --- #
    # ------------------------------------------------- #
    # -- independent, from-scratch alternative to _runOptimizer() above: a  -- #
    # -- single physically-scoped scalar cost ( config["objective"], not    -- #
    # -- config["objectives"] ) evaluated by a (mu/mu_w,lambda)-CMA-ES       -- #
    # -- search + local polish, with an optional second stage that re-      -- #
    # -- refines locally under full tracking (matching.trackingRefine).     -- #
    # -- Every term is expressed directly in terms of sigma_x/sigma_y/      -- #
    # -- transmission (no generic eval() of user expressions).              -- #
    def _runCmaEsOptimizer():

        history    = []
        evalCount  = 0
        startTime  = time.perf_counter()
        runMode    = config["matching"]["runMode"].lower()
        stage1Mode = config["matching"]["mode"]

        lowerVector = np.array( [ variable["min"] for variable in variables ], dtype=float )
        upperVector = np.array( [ variable["max"] for variable in variables ], dtype=float )
        for x0, xmin, xmax, variable in zip( initialVector, lowerVector, upperVector, variables ):
            if not ( xmin <= x0 <= xmax ):
                raise ValueError( "{} initial={} is outside [{}, {}]"
                                  .format( variable["name"], x0, xmin, xmax ) )
        if ( runMode == "optimize" and len( variables ) == 0 ):
            raise ValueError( "No optimization variable is enabled." )

        # ------------------------------------------------- #
        # --- [a] physically-scoped scalar cost          --- #
        # ------------------------------------------------- #
        def _evaluateCost( records=None, vector=None, mode=None ):
            objCfg = config["objective"]
            total  = 0.0
            detail = {}

            allRows = _selectRows( records=records, location="all" )
            sigmaX  = allRows["sigma_x"].to_numpy()
            sigmaY  = allRows["sigma_y"].to_numpy()

            # -- [a] envelope smoothness : coefficient-of-variation of sigma  -- #
            # -- along the section -> 0 for a perfectly flat/non-oscillating  -- #
            # -- envelope, growing with the amplitude of the beat.            -- #
            cfg = objCfg["smoothness"]
            if ( cfg["enabled"] ):
                cvX = float( np.std( sigmaX ) / np.mean( sigmaX ) )
                cvY = float( np.std( sigmaY ) / np.mean( sigmaY ) )
                scale      = float( cfg["scale"] )
                normalized = np.array( [ cvX, cvY ] ) / scale
                penalty    = float( cfg["weight"] ) * float( np.mean( normalized**2 ) )
                total     += penalty
                detail["smoothness"] = { "value":0.5*(cvX+cvY), "target":0.0,
                                         "residual":0.5*(cvX+cvY),
                                         "normResidual":float( np.max( np.abs( normalized ) ) ),
                                         "penalty":penalty }

            # -- [b] don't over-expand : soft upper bound on sigma            -- #
            cfg = objCfg["aperture"]
            if ( cfg["enabled"] ):
                scale = float( cfg["scale"] )
                violX = np.maximum( sigmaX - float( cfg["sigmaXMax"] ), 0.0 )
                violY = np.maximum( sigmaY - float( cfg["sigmaYMax"] ), 0.0 )
                normalized = np.concatenate( [ violX, violY ] ) / scale
                penalty    = float( cfg["weight"] ) * float( np.mean( normalized**2 ) )
                total     += penalty
                detail["aperture"] = { "value":float( max( np.max( sigmaX ), np.max( sigmaY ) ) ),
                                       "target":float( max( cfg["sigmaXMax"], cfg["sigmaYMax"] ) ),
                                       "residual":float( np.max( np.concatenate( [ violX, violY ] ) ) ),
                                       "normResidual":float( np.max( normalized ) ),
                                       "penalty":penalty }

            # -- [c] don't over-focus : soft lower bound on sigma.  Space      -- #
            # -- charge defocusing / nonlinear emittance growth become        -- #
            # -- significant once the local beam size gets too small, so a    -- #
            # -- floor on sigma is used as a cheap proxy for "not squeezed     -- #
            # -- past the point where space charge starts dominating".        -- #
            cfg = objCfg["focusLimit"]
            if ( cfg["enabled"] ):
                scale = float( cfg["scale"] )
                violX = np.maximum( float( cfg["sigmaXMin"] ) - sigmaX, 0.0 )
                violY = np.maximum( float( cfg["sigmaYMin"] ) - sigmaY, 0.0 )
                normalized = np.concatenate( [ violX, violY ] ) / scale
                penalty    = float( cfg["weight"] ) * float( np.mean( normalized**2 ) )
                total     += penalty
                detail["focusLimit"] = { "value":float( min( np.min( sigmaX ), np.min( sigmaY ) ) ),
                                         "target":float( min( cfg["sigmaXMin"], cfg["sigmaYMin"] ) ),
                                         "residual":float( np.max( np.concatenate( [ violX, violY ] ) ) ),
                                         "normResidual":float( np.max( normalized ) ),
                                         "penalty":penalty }

            # -- [d] final beam size close to design target                   -- #
            cfg = objCfg["targetSize"]
            if ( cfg["enabled"] ):
                endRow = _selectRows( records=records, location=cfg["location"] )
                valX   = float( endRow["sigma_x"].iloc[0] )
                valY   = float( endRow["sigma_y"].iloc[0] )
                normX  = ( valX - float( cfg["sigmaXTarget"] ) ) / float( cfg["scale"] )
                normY  = ( valY - float( cfg["sigmaYTarget"] ) ) / float( cfg["scale"] )
                penalty = float( cfg["weight"] ) * 0.5 * ( normX**2 + normY**2 )
                total  += penalty
                detail["targetSize"] = { "value":0.5*(valX+valY),
                                         "target":0.5*(cfg["sigmaXTarget"]+cfg["sigmaYTarget"]),
                                         "residual":0.5*abs(valX-cfg["sigmaXTarget"])
                                         +0.5*abs(valY-cfg["sigmaYTarget"]),
                                         "normResidual":float( max( abs(normX), abs(normY) ) ),
                                         "penalty":penalty }

            # -- [e] x/y envelope balance                                      -- #
            cfg = objCfg["balance"]
            if ( cfg["enabled"] ):
                value      = ( sigmaX - sigmaY ) / ( sigmaX + sigmaY )
                normalized = value / float( cfg["scale"] )
                penalty    = float( cfg["weight"] ) * float( np.mean( normalized**2 ) )
                total     += penalty
                detail["balance"] = { "value":float( np.mean( value ) ), "target":0.0,
                                      "residual":float( np.mean( np.abs( value ) ) ),
                                      "normResidual":float( np.max( np.abs( normalized ) ) ),
                                      "penalty":penalty }

            # -- [f] transmission (tracking mode only; envelope mode carries   -- #
            # -- no particle-loss model so this term is meaningless there)     -- #
            cfg = objCfg["transmission"]
            if ( cfg["enabled"] and mode == "tracking" ):
                endRow = _selectRows( records=records, location="end" )
                value  = float( endRow["transmission"].iloc[0] )
                normalized = ( 1.0 - value ) / float( cfg["scale"] )
                penalty    = float( cfg["weight"] ) * normalized**2
                total     += penalty
                detail["transmission"] = { "value":value, "target":1.0,
                                           "residual":1.0-value,
                                           "normResidual":float( abs( normalized ) ),
                                           "penalty":penalty }

            # -- [g] regularization : stay close to the QM/Twiss vector that   -- #
            # -- calctwiss + track2impactx already produced, in bounds-        -- #
            # -- normalized units, so the search doesn't wander to a solution  -- #
            # -- that only "looks good" on this cost but abandons the design.  -- #
            # -- separate from top-level config["regularization"] (least_      -- #
            # -- squares-only, SVD-basis-normalized): this cost's scale differs.--#
            regCfg = objCfg.get( "regularization", { "enabled":False } )
            if ( regCfg["enabled"] ):
                lower  = np.array( [ variable["min"] for variable in variables ] )
                upper  = np.array( [ variable["max"] for variable in variables ] )
                span   = np.maximum( upper - lower, 1.0e-12 )
                normalized = ( vector - initialVector ) / span
                penalty    = float( regCfg["weight"] ) * float( np.mean( normalized**2 ) )
                total     += penalty
                detail["regularization"] = { "value":float( np.sqrt( np.mean( normalized**2 ) ) ),
                                             "target":0.0,
                                             "residual":float( np.max( np.abs( normalized ) ) ),
                                             "normResidual":float( np.max( np.abs( normalized ) ) ),
                                             "penalty":penalty }

            return( total, detail )

        # ------------------------------------------------- #
        # --- [b] CMA-ES ( mu/mu_w , lambda )            --- #
        # ------------------------------------------------- #
        def _runCmaEs( objectiveReal=None, initial=None, lower=None, upper=None, cmaCfg=None ):
            nDim   = len( initial )
            span   = upper - lower

            lambda_ = cmaCfg["popSize"]
            if ( lambda_ is None ):
                lambda_ = 4 + int( 3.0 * np.log( nDim ) )
            lambda_ = max( 4, int( lambda_ ) )
            mu      = lambda_ // 2

            weightsRaw = np.log( mu + 0.5 ) - np.log( np.arange( 1, mu + 1 ) )
            weights    = weightsRaw / np.sum( weightsRaw )
            muEff      = 1.0 / np.sum( weights**2 )

            cc     = ( 4.0 + muEff / nDim ) / ( nDim + 4.0 + 2.0 * muEff / nDim )
            cs     = ( muEff + 2.0 ) / ( nDim + muEff + 5.0 )
            c1     = 2.0 / ( ( nDim + 1.3 )**2 + muEff )
            cmu    = min( 1.0 - c1,
                         2.0 * ( muEff - 2.0 + 1.0 / muEff ) / ( ( nDim + 2.0 )**2 + muEff ) )
            damps  = 1.0 + 2.0 * max( 0.0, np.sqrt( ( muEff - 1.0 ) / ( nDim + 1.0 ) ) - 1.0 ) + cs
            chiN   = np.sqrt( nDim ) * ( 1.0 - 1.0 / ( 4.0 * nDim ) + 1.0 / ( 21.0 * nDim**2 ) )

            rng    = np.random.default_rng( int( cmaCfg["seed"] ) )
            xmean  = ( initial - lower ) / span
            sigma  = float( cmaCfg["sigma0"] )
            pc     = np.zeros( nDim )
            ps     = np.zeros( nDim )
            B      = np.eye( nDim )
            D      = np.ones( nDim )
            C      = np.eye( nDim )

            boundaryWeight = float( cmaCfg["boundaryWeight"] )
            maxEval        = int( cmaCfg["maxEval"] )
            maxGen         = int( cmaCfg["maxGen"] )
            tol            = float( cmaCfg["tol"] )
            printEvery     = int( cmaCfg["printEvery"] )

            evalCountLocal = 0
            generation     = 0
            recentBest     = []
            bestSoFarLocal = np.inf

            while ( evalCountLocal < maxEval and generation < maxGen ):
                arz = rng.standard_normal( ( lambda_, nDim ) )
                fitness  = np.empty( lambda_ )
                realCost = np.empty( lambda_ )

                for k in range( lambda_ ):
                    if ( evalCountLocal >= maxEval ):
                        lambda_used = k
                        arz         = arz[:k]
                        fitness     = fitness[:k]
                        realCost    = realCost[:k]
                        break
                    y         = B.dot( D * arz[k] )
                    xNorm     = xmean + sigma * y
                    xClipped  = np.clip( xNorm, 0.0, 1.0 )
                    boundaryPenalty = boundaryWeight * float( np.sum( ( xNorm - xClipped )**2 ) )
                    xReal     = lower + xClipped * span
                    cost      = objectiveReal( xReal, "cma", generation )
                    fitness[k]  = cost + boundaryPenalty
                    realCost[k] = cost
                    evalCountLocal += 1
                else:
                    lambda_used = lambda_

                if ( lambda_used < 2 ):
                    break

                order   = np.argsort( fitness[:lambda_used] )
                muUse   = min( mu, lambda_used // 2 ) if ( lambda_used < lambda_ ) else mu
                muUse   = max( 1, muUse )
                wUse    = weights[:muUse] / np.sum( weights[:muUse] )
                best    = order[:muUse]

                zBest = arz[best]
                yBest = np.array( [ B.dot( D * z ) for z in zBest ] )

                zmean = np.dot( wUse, zBest )
                ymean = np.dot( wUse, yBest )
                xmean = xmean + sigma * ymean

                ps = ( 1.0 - cs ) * ps \
                    + np.sqrt( cs * ( 2.0 - cs ) * muEff ) * B.dot( zmean )
                hsig = ( np.linalg.norm( ps )
                        / np.sqrt( 1.0 - ( 1.0 - cs )**( 2.0 * ( generation + 1 ) ) ) / chiN ) \
                    < ( 1.4 + 2.0 / ( nDim + 1.0 ) )
                pc = ( 1.0 - cc ) * pc \
                    + ( float( hsig ) * np.sqrt( cc * ( 2.0 - cc ) * muEff ) ) * ymean

                artmp = yBest.T                                    # nDim x muUse
                C = ( 1.0 - c1 - cmu ) * C \
                  + c1 * ( np.outer( pc, pc ) + ( 1.0 - float( hsig ) ) * cc * ( 2.0 - cc ) * C ) \
                  + cmu * artmp.dot( np.diag( wUse ) ).dot( artmp.T )
                sigma = sigma * np.exp( ( cs / damps ) * ( np.linalg.norm( ps ) / chiN - 1.0 ) )

                C = 0.5 * ( C + C.T )
                eigenValue, eigenVector = np.linalg.eigh( C )
                eigenValue = np.maximum( eigenValue, 1.0e-20 )
                D, B       = np.sqrt( eigenValue ), eigenVector

                generation += 1
                genBest        = float( np.min( realCost[:lambda_used] ) )
                bestSoFarLocal = min( bestSoFarLocal, genBest )
                recentBest.append( genBest )
                if ( printEvery > 0 and generation % printEvery == 0 ):
                    print( " generation={:4d}  evaluation={:5d}  sigma={:.4e}"
                           "  genBest={:.6e}  bestSoFar={:.6e}"
                           .format( generation, evalCountLocal, sigma, genBest, bestSoFarLocal ) )

                if ( len( recentBest ) >= 10 ):
                    window     = np.array( recentBest[-10:] )
                    improvement = window[0] - window[-1]
                    threshold   = tol * max( abs( window[0] ), 1.0 )
                    if ( improvement < threshold and sigma < 1.0e-6 ):
                        break
                if ( sigma * float( np.max( D ) ) < 1.0e-12 ):
                    break

            return( { "nGeneration":generation, "nEval":evalCountLocal } )

        # ------------------------------------------------- #
        # --- [c] evaluator factory ( shared logging )   --- #
        # ------------------------------------------------- #
        # -- returns an evaluator bound to one ImpactX run mode and one          -- #
        # -- stage-local "best so far" dict, so a stage 2 (tracking-mode local   -- #
        # -- refinement, see [e]) can track its own best independently of        -- #
        # -- stage 1 -- comparing "best objective under envelope mode" against   -- #
        # -- "best objective under tracking mode" is not meaningful since the    -- #
        # -- two modes don't even share the same objective terms (transmission   -- #
        # -- only exists under tracking).                                       -- #
        def _makeEvaluator( mode=None, stageBest=None, stageLabel="stage" ):
            def _evaluate( vector=None, stage=None, generation=0 ):
                nonlocal evalCount

                evalCount += 1
                # -- ImpactX's Poisson solver (MLMG) can throw instead of just    -- #
                # -- returning a bad answer when a candidate point pushes the     -- #
                # -- beam envelope somewhere the mesh can't represent (seen in    -- #
                # -- practice: "tracking" mode at high current, inside the local  -- #
                # -- refine stage). Treat that the same as any other bad point --  -- #
                # -- a large fixed penalty -- instead of letting it kill the      -- #
                # -- whole optimization run and lose every evaluation so far.     -- #
                try:
                    records, params_, elements_ = _evaluateImpactX( vector=vector, mode=mode )
                    total, detail = _evaluateCost( records=records, vector=vector, mode=mode )
                except Exception as excInfo:
                    total  = 1.0e8
                    detail = { "simulationFailure": { "value":1.0, "target":0.0,
                                                      "residual":1.0, "normResidual":1.0,
                                                      "penalty":total } }
                    print( "  [evaluation {:5d} failed: {}: {}] -> penalty={:.1e}".format(
                        evalCount, type( excInfo ).__name__, excInfo, total ) )

                if ( total < stageBest["value"] ):
                    stageBest["value"]  = total
                    stageBest["vector"] = np.array( vector, dtype=float ).copy()
                    stageBest["detail"] = detail
                    stageBest["eval"]   = evalCount

                row = { "stage":stage if ( stage is not None ) else stageLabel,
                       "generation":generation, "evaluation":evalCount,
                       "objective":total, "bestObjective":stageBest["value"] }
                for varIndex, variable in enumerate( variables ):
                    row[variable["name"]] = float( vector[varIndex] )
                for objName, metric in detail.items():
                    for metricName, metricValue in metric.items():
                        row["{}.{}".format( metricName, objName )] = metricValue
                history.append( row )
                if ( evalCount % max( 1, int( config["optimizer"]["cmaEs"]["historyFlushEvery"] ) ) == 0 ):
                    pd.DataFrame( history ).to_csv( config["files"]["historyFile"], index=False )

                return( total )
            return( _evaluate )

        # ------------------------------------------------- #
        # --- [d] optimize / evaluate ( stage 1 )        --- #
        # ------------------------------------------------- #
        stage1Best = { "value":np.inf, "vector":initialVector.copy(), "detail":{}, "eval":0 }
        evaluate1  = _makeEvaluator( mode=stage1Mode, stageBest=stage1Best, stageLabel="stage1" )
        cmaEsCfg   = config["optimizer"]["cmaEs"]

        if ( runMode == "optimize" ):
            # -- CMA-ES only ever evaluates sampled offspring, never its own    -- #
            # -- starting mean, so without this the calctwiss/track2impactx     -- #
            # -- baseline (often already a reasonable lattice) is never itself  -- #
            # -- a candidate and a few unlucky early generations could hand     -- #
            # -- back something worse than what the user started with. Seeding -- #
            # -- stage1Best with it up front makes "no worse than the input    -- #
            # -- lattice" a guarantee rather than a hope.                      -- #
            evaluate1( vector=initialVector, stage="stage1-seed", generation=0 )

            cmaInfo = _runCmaEs( objectiveReal=evaluate1, initial=initialVector,
                                 lower=lowerVector, upper=upperVector, cmaCfg=cmaEsCfg )

            polishCfg  = cmaEsCfg["polish"]
            polishInfo = None
            if ( polishCfg["enabled"] ):
                remaining = int( cmaEsCfg["maxEval"] ) - evalCount
                budget    = max( 0, min( int( polishCfg["maxEval"] ), remaining ) )
                if ( budget > 0 ):
                    bounds = list( zip( lowerVector, upperVector ) )

                    def _polishObjective( vector ):
                        return( evaluate1( vector=vector, stage="stage1-polish", generation=0 ) )

                    polishResult = opt.minimize(
                        _polishObjective, stage1Best["vector"], method=polishCfg["method"],
                        bounds=bounds,
                        options={ "maxfev":budget, "xatol":float( polishCfg["xtol"] ),
                                 "fatol":float( polishCfg["ftol"] ) }
                        if ( polishCfg["method"] == "Nelder-Mead" ) else
                        { "maxfev":budget, "xtol":float( polishCfg["xtol"] ),
                         "ftol":float( polishCfg["ftol"] ) } )
                    polishInfo = { "success":bool( polishResult.success ),
                                   "message":str( polishResult.message ),
                                   "nfev":int( polishResult.nfev ) }
            success, message = True, "CMA-ES + local polish"
        else:
            evaluate1( vector=initialVector, stage="evaluate", generation=0 )
            success, message = True, "single evaluation"
            cmaInfo, polishInfo = None, None

        bestVector, bestValue, bestDetail = ( stage1Best["vector"], stage1Best["value"],
                                              stage1Best["detail"] )
        stageUsed = "stage1"

        # ------------------------------------------------- #
        # --- [e] tracking-mode local refinement stage   --- #
        # ------------------------------------------------- #
        # -- when the beam is space-charge dominated (high current), the         -- #
        # -- linear/ellipsoidal space-charge model used by "envelope" mode can   -- #
        # -- diverge substantially from full particle tracking (nonlinear        -- #
        # -- fields, halo, real losses). Rather than pay tracking's ~1000x       -- #
        # -- per-eval cost for the *entire* global search, stage 1's envelope    -- #
        # -- optimum seeds a second, small, *local* CMA-ES run evaluated         -- #
        # -- directly in "tracking" mode, with bounds narrowed around that       -- #
        # -- point (matching.trackingRefine in the config).                     -- #
        refineCfg  = config["matching"].get( "trackingRefine", { "enabled":False } )
        refineInfo = None
        if ( runMode == "optimize" and refineCfg.get( "enabled", False )
            and stage1Mode != "tracking" ):
            stage2Best = { "value":np.inf, "vector":stage1Best["vector"].copy(),
                           "detail":{}, "eval":0 }
            evaluate2  = _makeEvaluator( mode="tracking", stageBest=stage2Best,
                                         stageLabel="refine" )
            # -- seeding guarantees the refine stage is never worse (under real  -- #
            # -- tracking physics) than simply accepting stage 1's point as-is.  -- #
            evaluate2( vector=stage1Best["vector"], stage="refine-seed", generation=0 )
            stage1TrackingObjective = float( stage2Best["value"] )

            radius   = float( refineCfg.get( "localRadius", 0.15 ) )
            fullSpan = upperVector - lowerVector
            refLower = np.clip( stage1Best["vector"] - radius * fullSpan, lowerVector, upperVector )
            refUpper = np.clip( stage1Best["vector"] + radius * fullSpan, lowerVector, upperVector )

            cmaInfo2 = _runCmaEs( objectiveReal=evaluate2, initial=stage1Best["vector"],
                                  lower=refLower, upper=refUpper, cmaCfg=refineCfg )

            refinePolishCfg  = refineCfg.get( "polish", { "enabled":False } )
            refinePolishInfo = None
            if ( refinePolishCfg.get( "enabled", False ) ):
                remaining = int( refineCfg["maxEval"] ) - cmaInfo2["nEval"]
                budget    = max( 0, min( int( refinePolishCfg["maxEval"] ), remaining ) )
                if ( budget > 0 ):
                    bounds2 = list( zip( refLower, refUpper ) )

                    def _refinePolishObjective( vector ):
                        return( evaluate2( vector=vector, stage="refine-polish", generation=0 ) )

                    refinePolishResult = opt.minimize(
                        _refinePolishObjective, stage2Best["vector"],
                        method=refinePolishCfg["method"], bounds=bounds2,
                        options={ "maxfev":budget, "xatol":float( refinePolishCfg["xtol"] ),
                                 "fatol":float( refinePolishCfg["ftol"] ) }
                        if ( refinePolishCfg["method"] == "Nelder-Mead" ) else
                        { "maxfev":budget, "xtol":float( refinePolishCfg["xtol"] ),
                         "ftol":float( refinePolishCfg["ftol"] ) } )
                    refinePolishInfo = { "success":bool( refinePolishResult.success ),
                                         "message":str( refinePolishResult.message ),
                                         "nfev":int( refinePolishResult.nfev ) }

            refineInfo = { "cmaEs":cmaInfo2, "polish":refinePolishInfo, "localRadius":radius,
                           "stage1Objective"        :float( stage1Best["value"] ),
                           "stage1TrackingObjective":stage1TrackingObjective,
                           "refineObjective"        :float( stage2Best["value"] ) }
            bestVector, bestValue, bestDetail = ( stage2Best["vector"], stage2Best["value"],
                                                  stage2Best["detail"] )
            stageUsed = "refine"

        finalMode = "tracking" if ( stageUsed == "refine" ) else stage1Mode
        bestRecords, bestParams, bestElements = _evaluateImpactX( vector=bestVector,
            mode=finalMode, keepStat=True )
        bestValue, bestDetail = _evaluateCost( records=bestRecords, vector=bestVector,
                                              mode=finalMode )
        pd.DataFrame( history ).to_csv( config["files"]["historyFile"], index=False )

        # ------------------------------------------------- #
        # --- [f] envelope-vs-tracking diagnostic        --- #
        # ------------------------------------------------- #
        # -- always measured at stage 1's (envelope-only) point, independent of -- #
        # -- whether the refine stage ran, so it stays a fixed "what if you'd   -- #
        # -- stopped after cheap envelope-mode optimization" reference.         -- #
        trackingVerification = None
        if ( config["matching"].get( "trackingVerify", False ) and stage1Mode == "envelope" ):
            stage1EnveRecords, _, _ = _evaluateImpactX( vector=stage1Best["vector"],
                                                         mode=stage1Mode )
            trackRecords, _, _ = _evaluateImpactX( vector=stage1Best["vector"], mode="tracking" )
            trackRows = _selectRows( records=trackRecords, location="end" )
            trackingVerification = {
                "transmission" : float( trackRows["transmission"].iloc[0] ),
                "end_sigma_x"  : float( trackRows["sigma_x"].iloc[0] ),
                "end_sigma_y"  : float( trackRows["sigma_y"].iloc[0] ),
                "envelope_end_sigma_x" : float(
                    _selectRows( records=stage1EnveRecords, location="end" )["sigma_x"].iloc[0] ),
                "envelope_end_sigma_y" : float(
                    _selectRows( records=stage1EnveRecords, location="end" )["sigma_y"].iloc[0] ),
            }

        # ------------------------------------------------- #
        # --- [g] save results                           --- #
        # ------------------------------------------------- #
        variableResult = { variable["name"]:float( bestVector[varIndex] )
                           for varIndex, variable in enumerate( variables ) }

        resultData = {
            "success"      : success,
            "message"      : message,
            "model"        : config["matching"]["model"],
            "mode"         : finalMode,
            "stageUsed"    : stageUsed,
            "optimizer"    : "cma-es + " + cmaEsCfg["polish"]["method"],
            "section"      : { "startElement":activeKeys[0], "endElement":activeKeys[-1] },
            "objective"    : float( bestValue ),
            "objectiveTerm": bestDetail,
            "variables"    : variableResult,
            "elapsedS"     : float( time.perf_counter() - startTime ),
            "nEval"        : int( evalCount ),
        }
        if ( cmaInfo is not None ):
            resultData["cmaEs"] = cmaInfo
        if ( polishInfo is not None ):
            resultData["polish"] = polishInfo
        if ( trackingVerification is not None ):
            resultData["trackingVerification"] = trackingVerification
        if ( refineInfo is not None ):
            resultData["trackingRefine"] = refineInfo

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
        # --- [h] print summary                          --- #
        # ------------------------------------------------- #
        print( "\n" + "==================================" )
        print( "===  Matching summary (CMA-ES)  ===" )
        print( "==================================" + "\n" )
        print( " success   :: {}".format( success ) )
        print( " message   :: {}".format( message ) )
        print( " stageUsed :: {}  (final mode={})".format( stageUsed, finalMode ) )
        print( " nEval     :: {}".format( evalCount ) )
        if ( cmaInfo is not None ):
            print( " stage1 CMA-ES :: {} generations, {} evaluations"
                   .format( cmaInfo["nGeneration"], cmaInfo["nEval"] ) )
        if ( polishInfo is not None ):
            print( " stage1 polish :: {} ({} nfev, success={})"
                   .format( cmaEsCfg["polish"]["method"],
                            polishInfo["nfev"], polishInfo["success"] ) )
        if ( refineInfo is not None ):
            print( " refine CMA-ES :: {} generations, {} evaluations, localRadius={}"
                   .format( refineInfo["cmaEs"]["nGeneration"], refineInfo["cmaEs"]["nEval"],
                            refineInfo["localRadius"] ) )
            print( " objective (stage1 / stage1-under-tracking / refine) :: "
                   "{:.6e} / {:.6e} / {:.6e}".format( refineInfo["stage1Objective"],
                            refineInfo["stage1TrackingObjective"], refineInfo["refineObjective"] ) )
        print( " objective :: {:.8e} -> {:.8e}"
               .format( history[0]["objective"] if history else float( "nan" ), bestValue ) )

        print( "\n objectives:" )
        for objName, metric in bestDetail.items():
            line = "  {:14s} value={:.6e}  target={:.6e}  penalty={:.6e}".format(
                objName, metric["value"], metric["target"], metric["penalty"] )
            print( line )

        if ( trackingVerification is not None ):
            print( "\n tracking verification (best point re-run with mode=tracking):" )
            for key, value in trackingVerification.items():
                print( "  {:24s} {:.6e}".format( key, value ) )

        print( "\n result file      :: {}".format( config["files"]["resultFile"] ) )
        print( " statistics       :: {}".format( config["files"]["statFile"] ) )
        print( " optimization log :: {}\n".format( config["files"]["historyFile"] ) )
        print( "==================================" + "\n" )
        return( resultData )


    # ------------------------------------------------- #
    # --- [2] load settings                         --- #
    # ------------------------------------------------- #
    with open( inpFile, "r" ) as fk:
        config = json5.load( fk )

    outFileList = [ config["files"]["resultFile"],
                    config["files"]["historyFile"],
                    config["files"]["statFile"],
                    config["files"]["matchedParamsFile"],
                    config["files"]["matchedBeamlineFile"] ]
    if ( config["regularization"]["svd"]["enabled"] ):
        outFileList += [ config["files"]["jacobianFile"],
                         config["files"]["svdFile"] ]
    for outFile in outFileList:
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
    if ( config["matching"]["mode"].lower() not in [ "envelope", "tracking" ] ):
        raise ValueError( "matching.mode must be 'envelope' or 'tracking'." )

    elementTypeKey = str( config["matching"].get( "elementType", "quadrupole" ) ).lower()
    if ( elementTypeKey not in ELEMENT_TYPE_CONFIG ):
        raise ValueError( "matching.elementType must be one of: {}"
                          .format( list( ELEMENT_TYPE_CONFIG.keys() ) ) )
    elementCfg = ELEMENT_TYPE_CONFIG[elementTypeKey]

    # ------------------------------------------------- #
    # --- [3] matching section and variables        --- #
    # ------------------------------------------------- #
    sectionCfg = config["matching"]["section"]
    simElements, activeKeys, startIndex = _selectSection( elements=elements,
        startElement=sectionCfg["startElement"], endElement=sectionCfg["endElement"],
        nUse=params["sim.nUse.elements"] )
    variables     = _buildVariables( params=params, elements=simElements,
                                     activeKeys=activeKeys )
    initialVector = np.array( [ variable["initial"] for variable in variables ],
                              dtype=float )

    print( "\n === ImpactX matching ===" )
    print( " model   :: {}".format( config["matching"]["model"] ) )
    print( " mode    :: {}".format( config["matching"]["mode"] ) )
    print( " runMode :: {}".format( config["matching"]["runMode"] ) )
    print( " method  :: {}".format( config["optimizer"]["method"] ) )
    print( " section :: {} -> {}".format( activeKeys[0], activeKeys[-1] ) )
    print( " nvar    :: {}\n".format( len( variables ) ) )

    # ------------------------------------------------- #
    # --- [4] optimization / evaluation             --- #
    # ------------------------------------------------- #
    if ( config["optimizer"]["method"] == "cma-es" ):
        return( _runCmaEsOptimizer() )

    startTime = time.perf_counter()
    runMode  = config["matching"]["runMode"].lower()
    svdInfo  = None
    nRounds  = 1
    if ( runMode == "optimize" ):
        if ( len( variables ) == 0 ):
            raise ValueError( "No optimization variable is enabled." )
        result, history, bestVector, bestEvaluation, evalCount, svdInfo, nRounds = _runOptimizer()
        if ( bestVector is None ):
            raise RuntimeError( "Optimizer did not produce a finite objective." )
        success = bool( result.success )
        message = str( result.message )
    else:
        bestVector = initialVector.copy()
        records, _, _ = _evaluateImpactX( vector=bestVector, keepStat=False )
        bestValue, detail = _evaluateObjectives( records=records, vector=bestVector,
                                                 initial=initialVector )
        history    = [ { "evaluation":1, "objective":float( bestValue ),
                         "bestObjective":float( bestValue ),
                         **{ "{}.{}".format( metricName, objName ):metricValue
                             for objName, metric in detail.items()
                             for metricName, metricValue in metric.items() } } ]
        pd.DataFrame( history ).to_csv( config["files"]["historyFile"], index=False )
        result, success, message = None, True, "single evaluation"

    bestRecords, _, _ = _evaluateImpactX( vector=bestVector, keepStat=True )
    bestValue, bestDetail = _evaluateObjectives( records=bestRecords, vector=bestVector,
                                                 initial=initialVector )

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
    if ( svdInfo is not None ):
        resultData["svd"] = {
            "useModes"      : bool( config["regularization"]["svd"]["useModes"] ),
            "nModes"        : int( len( svdInfo["modeIndex"] ) ),
            "nEval"         : int( svdInfo["nEval"] ),
            "nRounds"       : int( nRounds ),
            "singularValue" : svdInfo["singularValue"].tolist(),
            "relativeValue" : svdInfo["relativeValue"].tolist(),
        }

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
            print( " nEval     :: {}".format( int( evalCount ) ) )
            print( " objective :: {:.8e} -> {:.8e}"
                   .format( float( history[0]["objective"] ), bestValue ) )

        if ( config["optimizer"]["method"] == "Nelder-Mead" ):
            print( " xatol     :: {:.4e}".format( config["optimizer"]["xtol"] ) )
            print( " fatol     :: {:.4e}".format( config["optimizer"]["ftol"] ) )
        if ( svdInfo is not None ):
            print( " SVD modes :: {} / {}".format( len( svdInfo["modeIndex"] ), len( svdInfo["variableIndex"] ) ) )
            print( " SVD nEval :: {}".format( svdInfo["nEval"] ) )
            print( " SVD rounds:: {}".format( nRounds ) )
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
