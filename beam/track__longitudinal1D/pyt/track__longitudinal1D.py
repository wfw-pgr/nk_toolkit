import os, json5, re
import numpy             as np
import pandas            as pd
import scipy.special     as spsp


# ========================================================= #
# ===  track__longitudinal1D                            === #
# ========================================================= #
def track__longitudinal1D( parameterFile="inp/parameters.json" ):
    """
    # Track one-dimensional longitudinal motion in a low-beta linac.
    #  - T.Aoki コードの翻訳版ベース、多少変更 -
    #  - 元コードは青木さん。 - #
    #  ---  #
    """

    amu_MeV = 931.4941
    print( "\n" + "[track__longitudinal1D.py]" + "\n" )
    
    # ------------------------------------------------- #
    # --- [1] internal functions                    --- #
    # ------------------------------------------------- #
    def _transitionF( cellId=None, startCell=None, endCell=None ):
        if ( cellId <= startCell ):          # -- 電圧・位相の遷移関数：どうあげていくか -- #
            return( 0.0 )
        if ( cellId >  endCell   ):
            return( 1.0 )
        ratio = ( cellId - startCell ) / ( endCell - startCell ) - 0.5
        return( 0.5 + 0.5 * np.tanh( np.tan( np.pi * ratio ) ) )

    def _gammaAtK( energy_MeV_u=None ):      # -- エネルギー K での相対論的γ -- #
        return( 1.0 + massNumber * energy_MeV_u / massEnergy_MeV )

    def _betaAtK( energy_MeV_u=None ):       # -- エネルギー K での相対論的β -- #
        gamma = _gammaAtK( energy_MeV_u=energy_MeV_u )
        beta2 = 1.0 - 1.0 / gamma**2
        return( np.sqrt( np.maximum( beta2, 0.0 ) ) )

    def _phaseAdvance( energy_MeV_u=None ):  # -- 周期長による位相進みを計算 -- #
        beta = _betaAtK( energy_MeV_u=energy_MeV_u )
        return( 2.0*np.pi * periodLength_m / ( beta*wavelength_m ) )

    def _wrapPhase( phase_rad=None ):        # -- 2pi の余り -- #
        return( ( phase_rad+np.pi ) % (2.0*np.pi) - np.pi )

    # ------------------------------------------------- #
    # --- SC モデル: SPUNCH like 縦分割ディスク電荷 --- #
    # ------------------------------------------------- #
    def _build__spaceChargeModel( config=None ):
        """Return the longitudinal disc-disc field model of Baartman 1986 Eq.(2)."""
        # -- SC model : SPUNCH like = 縦方向分割ディスク近似 -- #
        enabled       = bool ( config.get( "enabled"           , False      ) )
        current_A     = float( config.get( "averageCurrent_A"  , 0.0        ) )
        beamRadius_m  = float( config.get( "beamRadius_m"      , 0.01       ) )
        pipeRadius_m  = float( config.get( "pipeRadius_m"      , aperture_m ) )
        numBessel     = int  ( config.get( "numBesselTerms"    , 80         ) )

        if ( not enabled or current_A == 0.0 ):
            def _zeroField( phase_rad=None, beta=None, reference_num_discs=None ):
                return( np.zeros_like( phase_rad, dtype=float ) )
            return( _zeroField )
        if ( beamRadius_m <= 0.0 or pipeRadius_m <= 0.0 ):
            raise ValueError( "Space-charge beam and pipe radii must be positive." )
        if ( beamRadius_m >= pipeRadius_m ):
            raise ValueError( "space_charge.beam_radius_m must be smaller than pipe_radius_m." )
        if ( numBessel < 1 ):
            raise ValueError( "space_charge.num_bessel_terms must be positive." )

        # -- SPUNCH like table :: make disc force table before tracking. -- # 
        eps0_F_m = 8.8541878128e-12
        kn       = spsp.jn_zeros( 0, numBessel )   # J0's first numBessel個の根 : ndarray [ J0=0 #1, J0=0 #2, ....  ]
        weight   = ( 2.0*spsp.j1( kn*beamRadius_m/pipeRadius_m ) / ( kn*spsp.j1( kn ) ) )**2     # [ ] of Eq.(2)
        zTable   = np.linspace( 0.0, 0.5 * wavelength_m * float(_betaAtK(targetEnergy) ), 5001 )
        kTable   = np.zeros_like( zTable )
        for kValue, wValue in zip( kn, weight ):
            kTable += wValue * np.exp( -kValue*zTable/pipeRadius_m )

        def _discField( phase_rad=None, beta=None, reference_num_discs=None ):
            """
            ディスク電荷による電場 [V/m]
            初期に位相(=z)方向に分割した粒子をディスク電荷としてクーロン反発を計算 (SPUNCH)
            """
            phase = np.asarray( phase_rad, dtype=float )
            nDisc = phase.size
            if ( nDisc < 2 ):
                return( np.zeros_like( phase ) )
            nReference      = nDisc if ( reference_num_discs is None ) else int( reference_num_discs )
            chargePerDisc_C = current_A / ( frequency_MHz*1.0e6*nReference )
            dPhase          = _wrapPhase( phase_rad=phase[:,None] - phase[None,:] )
            # +: RF arrival phase later particle ( dz = -beta*lambda*dphi/(2*pi). )
            dz_m      = -beta * wavelength_m * dPhase / ( 2.0*np.pi )
            absDz     = np.abs( dz_m )
            kernel    = np.interp( absDz, zTable, kTable ) * np.sign( dz_m )
            np.fill_diagonal( kernel, 0.0 )
            prefactor = chargePerDisc_C / ( 2.0*np.pi*eps0_F_m*beamRadius_m**2 )
            return( prefactor * np.sum( kernel, axis=1 ) )

        return( _discField )
    
    # ------------------------------------------------- #
    # --- TTF モデル: const / load-table            --- #
    # ------------------------------------------------- #
    def _build__ttfModel( config=None ):     # -- TTF(a,K) の関数を返却 -- #
        
        mode = config.get( "mode", "constant" )
        if   ( mode == "constant" ):
            constValue = float( config.get( "constant_value", 1.0 ) )
            def _ttfConstant( aperture_m=None, energy_MeV_u=None ):
                shape = np.broadcast( aperture_m, energy_MeV_u ).shape
                return( np.full( shape, constValue ) )
            return( _ttfConstant )

        elif ( mode == "csv" ):
            csvFile      = config.get( "csvFile", "inp/ttfModel.csv" )
            csvData      = pd.read_csv( csvFile )
            apertureAxis = np.sort( csvData["aperture_m"].unique() )
            energyAxis   = np.sort( csvData["energy_MeV_u"].unique() )
            absGrid   = csvData.pivot( index="aperture_m", columns="energy_MeV_u",
                                       values="abs_ttf" ).loc[apertureAxis, energyAxis].to_numpy()
            phaseGrid = csvData.pivot( index="aperture_m", columns="energy_MeV_u",
                                       values="phase_rad").loc[apertureAxis, energyAxis].to_numpy()

            def _interp2D( aperture_m=None, energy_MeV_u=None, grid=None ):
                apertureData,energyData = np.broadcast_arrays( aperture_m, energy_MeV_u )
                apertureFlat = apertureData.reshape( -1 )
                energyFlat   = energyData.reshape( -1 )

                # --- interpolation along energy for each aperture --- #
                valueAtAperture = np.asarray( [
                    np.interp( energyFlat, energyAxis, grid[apertureIndex,:] )
                    for apertureIndex in range( apertureAxis.size ) ] )

                # --- interpolation along aperture for each query --- #
                ret = np.asarray( [ np.interp( apertureValue, apertureAxis,
                                               valueAtAperture[:,queryIndex] )
                                    for queryIndex, apertureValue in enumerate( apertureFlat ) ] )

                return( ret.reshape( apertureData.shape ) )
            
            def _ttfCsv( aperture_m=None, energy_MeV_u=None ):
                absTtf = _interp2D( aperture_m=aperture_m, energy_MeV_u=energy_MeV_u,
                                    grid=absGrid   )
                phiTtf = _interp2D( aperture_m=aperture_m, energy_MeV_u=energy_MeV_u,
                                    grid=phaseGrid )
                return( np.clip( absTtf*np.cos( phiTtf ), 0.0, 1.0 ) )
            return( _ttfCsv )
        else:
            raise ValueError( " unknown ttf mode :: [ constant, csv ]" )


    # ------------------------------------------------- #
    # --- make RF buckect boundary coordinates      --- #
    # ------------------------------------------------- #
    def _calc__bucketBoundary( syncEnergy_MeV_u=None, syncPhase_rad=None,
                               voltage_MV=None, ttfValue=None ):
        """
        RFバケツ境界を返却.
        """
        # ------------------------------------------------- #
        # --- [1] internal functions                    --- #
        # ------------------------------------------------- #
        def _potential( phase_rad=None ):
            return( np.cos(phase_rad) + ( phase_rad-syncPhase_rad ) * np.sin(syncPhase_rad) )

        # ------------------------------------------------- #
        # --- [2] K -> beta, gamma / args check         --- #
        # ------------------------------------------------- #
        betaSync  = _betaAtK ( energy_MeV_u=syncEnergy_MeV_u )
        gammaSync = _gammaAtK( energy_MeV_u=syncEnergy_MeV_u )

        if ( voltage_MV <= 0.0 or ttfValue <= 0.0 ):
            return( None, None, None )
        if ( np.cos( syncPhase_rad ) <= 0.0 ):
            return( None, None, None )

        # ------------------------------------------------- #
        # --- [3] 不安定位相 / 位相滑り/エネルギー係数  --- #
        # ------------------------------------------------- #
        phaseUnst_rad = np.pi - syncPhase_rad         # - unstable phase pi-phi_s - #
        coeffPhase    = omega_rad_us * periodLength_m * massNumber \
            / ( massEnergy_MeV * betaSync**3 * gammaSync**3 * c_m_per_us )
        coeffK        = 2.0 * chargeNumber * voltage_MV * ttfValue \
            / ( massNumber * coeffPhase )

        # ------------------------------------------------- #
        # --- [4] セパラトリクスのポテンシャルを計算    --- #
        # ------------------------------------------------- #
        potSep       = _potential( phase_rad=phaseUnst_rad )
        phaseRaw     = np.linspace( syncPhase_rad-2.0*np.pi, phaseUnst_rad, 1001 )
        argRaw       = coeffK * ( _potential( phase_rad=phaseRaw ) - potSep )
        argTol       = 1.0e-12 * max( 1.0, np.nanmax( np.abs(argRaw) ) )
        valid        = ( argRaw >= -argTol )
        if ( not np.any( valid ) ):
            return( None, None, None )
        dKRaw        = np.full_like( argRaw, np.nan )
        dKRaw[valid] = np.sqrt( np.maximum( argRaw[valid], 0.0 ) )

        # ------------------------------------------------- #
        # --- [5] wrap phase / データ切れ目はNaN挿入    --- #
        # ------------------------------------------------- #
        phaseP = _wrapPhase( phase_rad=phaseRaw )
        jumpId = np.where( np.abs( np.diff(phaseP) ) > np.pi )[0] + 1
        phaseP = np.insert( phaseP, jumpId, np.nan )
        dKP    = np.insert( dKRaw,  jumpId, np.nan )
        upperK = syncEnergy_MeV_u + dKP
        lowerK = syncEnergy_MeV_u - dKP
        return( phaseP, upperK, lowerK )
            
    # ------------------------------------------------- #
    # --- separatrixの決定と捕獲効率                --- #
    # ------------------------------------------------- #
    def _calc__captureEfficiency( phase_rad=None, energy_MeV_u=None, syncEnergy_MeV_u=None,
                                  syncPhase_rad=None, voltage_MV=None, ttfValue=None,
                                  reference_num_particles=None ):
        
        betaSync  = _betaAtK ( energy_MeV_u=syncEnergy_MeV_u )
        gammaSync = _gammaAtK( energy_MeV_u=syncEnergy_MeV_u )
        if ( voltage_MV <= 0.0 or ttfValue <= 0.0 ):
            return( np.nan )
        if ( np.cos(syncPhase_rad) <= 0.0 ):
            return( np.nan )

        phaseUnst_rad = np.pi - syncPhase_rad
        coeffPhase    = omega_rad_us * periodLength_m * massNumber \
            / ( massEnergy_MeV * betaSync**3 * gammaSync**3 * c_m_per_us )
        coeffK        = 2.0 * chargeNumber * voltage_MV * ttfValue / ( massNumber * coeffPhase )

        phaseBranch   = syncPhase_rad + _wrapPhase( phase_rad=phase_rad-syncPhase_rad )
        potential     = np.cos(phaseBranch) + ( phaseBranch-syncPhase_rad ) * np.sin(syncPhase_rad)
        potentialSep  = np.cos(phaseUnst_rad) + ( phaseUnst_rad-syncPhase_rad ) * np.sin(syncPhase_rad)

        bucketHeight2 = coeffK * ( potential-potentialSep )
        finiteHeight  = np.abs( bucketHeight2[np.isfinite(bucketHeight2)] )
        tolerance     = 1.0e-12 * max( 1.0, np.max(finiteHeight, initial=0.0) )
        valid         = np.isfinite(phase_rad) & np.isfinite(energy_MeV_u)
        captured      = valid \
            & ( phaseBranch <= phaseUnst_rad+1.0e-12 ) \
            & ( bucketHeight2 >= -tolerance ) \
            & ( (energy_MeV_u-syncEnergy_MeV_u)**2
                <= np.maximum(bucketHeight2, 0.0)+tolerance )

        numReference = phase_rad.size if ( reference_num_particles is None ) \
            else int(reference_num_particles)
        return( 100.0 * np.count_nonzero(captured) / numReference )

    
    # ------------------------------------------------- #
    # --- [2] load parameters                       --- #
    # ------------------------------------------------- #
    with open( parameterFile, "r", encoding="utf-8" ) as inp:
        # Accept the trailing commas used by the original JSON5 parameter file.
        parameterText = re.sub( r",\s*([}\]])", r"\1", inp.read() )
        params = json5.loads( parameterText )

    particle        = params["particle"]
    rf              = params["rf"]
    lattice         = params["lattice"]
    tracking        = params["tracking"]
    output          = params["output"]
    spaceChargeCfg  = params["space_charge"]

    massEnergy_MeV  = float( particle["mass_amu"] ) * amu_MeV
    chargeNumber    = float( particle["chargeNumber"] )
    massNumber      = float( particle["massNumber"] )
    injectionEnergy = float( particle["injectionEnergy_MeV_u"] )
    targetEnergy    = float( particle["targetEnergy_MeV_u"] )

    frequency_MHz   = float( rf["frequency_MHz"] )
    periodLength_m  = float( rf["periodLength_m"] )
    voltageBunch    = float( rf["voltageBunch_MV"] )
    voltageAcc      = float( rf["voltageAccel_MV"] )
    voltagePower    = float( rf["voltagePow"] )
    phasePower      = float( rf["phasePow"] )
    accPhase_rad    = np.deg2rad( rf["phaseSync_deg"] )

    numBunchCells   = int( lattice["numBunchingCells"] )
    numTransCells   = int( lattice["numTransitionCells"] )
    maxAccCells     = int( lattice["numTotalAccelCells"] )
    aperture_m      = float( lattice["apertureRadius_m"] )
    minEnergy       = float( tracking.get( "minEnergy_MeV_u", 1.0e-6 ) )

    c_m_per_us      = 299.792458
    omega_rad_us    = 2.0 * np.pi * frequency_MHz
    wavelength_m    = c_m_per_us / frequency_MHz
    ttfModel        = _build__ttfModel( config=params["ttf"] )
    spaceChargeModel= _build__spaceChargeModel( config=spaceChargeCfg )

    if ( periodLength_m <= 0.0 or frequency_MHz <= 0.0 ):
        raise ValueError( "RF frequency and period length must be positive." )
    if ( injectionEnergy <= 0.0 or targetEnergy <= injectionEnergy ):
        raise ValueError( "Target energy must be larger than positive injection energy." )

    
    # ------------------------------------------------- #
    # --- [3] generate cell patterns                --- #
    # ------------------------------------------------- #
    maxCells  = numBunchCells + numTransCells + maxAccCells
    cellId    = np.arange( 1, maxCells + 1 )
    transEnd  = numBunchCells + numTransCells
    transF    = np.array( [ _transitionF( cellId=iCell, startCell=numBunchCells,
                                          endCell=transEnd ) for iCell in cellId ] )
    voltage   = voltageBunch + ( voltageAcc - voltageBunch ) * transF**voltagePower
    syncPhase = accPhase_rad * transF**phasePower
    aperture  = np.full( maxCells, aperture_m )
    

    # ------------------------------------------------- #
    # --- [4] synchronous particle design           --- #
    # ------------------------------------------------- #
    syncEnergy    = np.full( maxCells+1, np.nan )
    syncEnergy[0] = injectionEnergy
    numCells      = maxCells
    for iCell in range( maxCells ):
        ttf                 = float( ttfModel( aperture_m=aperture[iCell],
                                               energy_MeV_u=syncEnergy[iCell] ) )
        energyGain          = chargeNumber * voltage[iCell] * ttf \
            * np.sin( syncPhase[iCell] ) / massNumber
        syncEnergy[iCell+1] = syncEnergy[iCell] + energyGain
        if ( syncEnergy[iCell+1] >= targetEnergy ):
            numCells = iCell + 1
            break
    cellId      = cellId    [:numCells  ]
    voltage     = voltage   [:numCells  ]
    syncPhase   = syncPhase [:numCells  ]
    aperture    = aperture  [:numCells  ]
    syncEnergy  = syncEnergy[:numCells+1]
    numAccCells = max( 0, numCells-numBunchCells-numTransCells )

    if ( syncEnergy[-1] < targetEnergy ):
        print( "[WARNING] target energy was not reached within max cells." )

    syncBeta        = _betaAtK ( energy_MeV_u=syncEnergy[:-1] )
    syncGamma       = _gammaAtK( energy_MeV_u=syncEnergy[:-1] )
    ttfList         = ttfModel ( aperture_m=aperture,
                                 energy_MeV_u=syncEnergy[:-1] )
    syncAbsPhase    = np.full( numCells+1 , np.nan )
    syncArrival     = np.full( numCells   , np.nan )
    phaseCorr       = np.full( numCells   , np.nan )
    syncAbsPhase[0] = syncPhase[0]

    for iCell in range( numCells ):
        syncArrival[iCell]    = syncAbsPhase[iCell]+_phaseAdvance( energy_MeV_u=syncEnergy[iCell] )
        phaseCorr[iCell]      = syncPhase[iCell] - syncArrival[iCell]
        syncAbsPhase[iCell+1] = syncArrival[iCell]

    timeStep_us = periodLength_m / ( syncBeta * c_m_per_us )
    time_us     = np.cumsum( timeStep_us )
    zpos_m      = periodLength_m * cellId
    syncFreq    = np.sqrt( omega_rad_us * chargeNumber * c_m_per_us * voltage * ttfList * np.cos( syncPhase )
                           / ( massEnergy_MeV * syncBeta * syncGamma**3 * periodLength_m ) )
    syncTune    = syncFreq / ( 2.0 * np.pi * syncBeta * c_m_per_us / periodLength_m )

    # ------------------------------------------------- #
    # --- [5] phase scan tracking                   --- #
    # ------------------------------------------------- #
    # A full RF period is half-open so -180 and +180 are not the same disc twice.
    phaseMinMaxStep    = list( tracking["initialPhase_MinMaxStep"] )
    initialPhase_deg   = np.arange( *phaseMinMaxStep )
    numParticles       = initialPhase_deg.size
    phaseHistory       = np.full( ( numParticles, numCells+1 ), np.nan )
    energyHistory      = np.full( ( numParticles, numCells+1 ), np.nan )
    alive              = np.ones( numParticles, dtype=bool )
    phaseHistory[:,0]  = np.deg2rad( initialPhase_deg )
    energyHistory[:,0] = injectionEnergy

    for iCell in range( numCells ):
        valid = alive & np.isfinite( energyHistory[:, iCell] ) \
                      & ( energyHistory[:, iCell] > minEnergy )
        if ( not np.any( valid ) ):
            break

        phaseHistory[valid, iCell+1] = phaseHistory[valid, iCell] \
            + _phaseAdvance( energy_MeV_u=energyHistory[valid, iCell] )
        effectivePhase = phaseHistory[valid,iCell+1] + phaseCorr[iCell]
        partTtf        = ttfModel( aperture_m=aperture[iCell],
                                   energy_MeV_u=energyHistory[valid, iCell] )
        energyHistory[valid, iCell+1] = energyHistory[valid, iCell] \
            + chargeNumber * voltage[iCell] * partTtf \
            * np.sin( effectivePhase ) / massNumber
        # Baartman Eq. (2): Delta K_sc = q E_z L / A [MeV/u].  The field is
        # evaluated after the drift at the cell entrance and applied as a kick.
        betaMean  = float( _betaAtK( energy_MeV_u=np.nanmean( energyHistory[valid, iCell] ) ) )
        field_V_m = spaceChargeModel( phase_rad=effectivePhase, beta=betaMean,
                                      reference_num_discs=numParticles )
        energyHistory[valid,iCell+1] += chargeNumber * field_V_m * periodLength_m / ( massNumber * 1.0e6 )
        lost        = valid.copy()
        lost[valid] = ( energyHistory[valid,iCell+1] <= minEnergy )\
            | ~np.isfinite( energyHistory[valid, iCell+1] )
        alive[lost] = False

        
    # ------------------------------------------------- #
    # --- [6] write output data                     --- #
    # ------------------------------------------------- #
    cellDataFile   = output.get( "cellDataFile"  , "out/cellData.csv"   )
    phaseSpaceFile = output.get( "phaseSpaceFile", "out/phaseSpace.csv" )
    os.makedirs( os.path.dirname( cellDataFile   ) or ".", exist_ok=True )
    os.makedirs( os.path.dirname( phaseSpaceFile ) or ".", exist_ok=True )

    outData = np.column_stack( ( cellId, zpos_m, time_us, voltage, aperture, ttfList, syncPhase,
                                 _wrapPhase( phase_rad=phaseCorr ), syncEnergy[1:], syncTune ) )
    header  = ( "cell,position_m,time_us,voltage_MV,aperture_m,ttf,"
                "sync_phase_rad,phase_correction_rad,sync_energy_MeV_u,sync_tune" )
    np.savetxt( cellDataFile, outData, fmt="%.10e", delimiter=",", header=header, comments="" )

    
    # ------------------------------------------------- #
    # --- [7] calculate and save phase-space data   --- #
    # ------------------------------------------------- #
    elementId                    = np.arange( numCells + 1 )
    elementPosition_m            = elementId * periodLength_m
    particleEffectivePhase       = np.full_like( phaseHistory, np.nan )
    particleEffectivePhase[:,0]  = _wrapPhase( phase_rad=phaseHistory[:,0] )
    particleEffectivePhase[:,1:] = _wrapPhase( phase_rad=phaseHistory[:,1:] + phaseCorr[None,:] )
    localSyncPhase               = np.concatenate( ([syncPhase[0]], syncPhase) )
    localVoltage                 = np.concatenate( ([voltage[0]], voltage) )
    localTtf                     = np.concatenate( ([ttfList[0]], ttfList) )
    capturePercent               = np.full( numCells + 1, np.nan )
    separatrixDataList           = []
    for iElem in elementId:
        phaseSep, upperK, lowerK = _calc__bucketBoundary( syncEnergy_MeV_u=syncEnergy[iElem],
                                                          syncPhase_rad   =localSyncPhase[iElem],
                                                          voltage_MV      =localVoltage[iElem],
                                                          ttfValue        =localTtf[iElem] )
        capturePercent[iElem] = _calc__captureEfficiency(
            phase_rad              =particleEffectivePhase[:,iElem],
            energy_MeV_u           =energyHistory[:,iElem],
            syncEnergy_MeV_u       =syncEnergy[iElem],
            syncPhase_rad          =localSyncPhase[iElem],
            voltage_MV             =localVoltage[iElem],
            ttfValue               =localTtf[iElem],
            reference_num_particles=numParticles )

        if ( phaseSep is not None ):
            validSeparatrix    = np.isfinite(phaseSep) & np.isfinite(upperK) & np.isfinite(lowerK)
            segmentStart       = validSeparatrix & np.r_[ True, ~validSeparatrix[:-1] ]
            segmentId          = np.cumsum( segmentStart ) - 1
            numValidSeparatrix = np.count_nonzero( validSeparatrix )
            separatrixDataList.append( pd.DataFrame({
                "record_type"        : np.full( numValidSeparatrix, "separatrix" ),
                "element_id"         : np.full( numValidSeparatrix, iElem ),
                "segment_id"         : segmentId[validSeparatrix],
                "phase_rad"          : phaseSep[validSeparatrix],
                "upper_energy_MeV_u" : upperK[validSeparatrix],
                "lower_energy_MeV_u" : lowerK[validSeparatrix] }) )

    elementData = pd.DataFrame({
        "record_type"               : np.full( numCells + 1, "element" ),
        "element_id"                : elementId,
        "position_m"                : elementPosition_m,
        "sync_energy_MeV_u"         : syncEnergy,
        "sync_phase_rad"            : localSyncPhase,
        "capture_efficiency_percent": capturePercent })

    particleElement = np.repeat( elementId, numParticles )
    particleId      = np.tile( np.arange( numParticles ), numCells + 1 )
    particlePhase   = particleEffectivePhase.T.reshape( -1 )
    particleEnergy  = energyHistory.T.reshape( -1 )
    validParticle   = np.isfinite( particlePhase ) & np.isfinite( particleEnergy )

    particleData = pd.DataFrame({
        "record_type" : np.full( np.count_nonzero( validParticle ), "particle" ),
        "element_id"  : particleElement[validParticle],
        "item_id"     : particleId[validParticle],
        "phase_rad"   : particlePhase[validParticle],
        "energy_MeV_u": particleEnergy[validParticle] })

    if ( len( separatrixDataList ) > 0 ):
        separatrixData = pd.concat( separatrixDataList, ignore_index=True )
    else:
        columns        = [ "record_type", "element_id", "segment_id",
                           "phase_rad","upper_energy_MeV_u", "lower_energy_MeV_u" ]
        separatrixData = pd.DataFrame( columns=columns )

    phaseSpaceData = pd.concat( [ elementData, particleData, separatrixData ], ignore_index=True )
    phaseSpaceData.to_csv( phaseSpaceFile, index=False, float_format="%.10e" )

    
    # ------------------------------------------------- #
    # --- [8] report                                --- #
    # ------------------------------------------------- #
    mode     = params["ttf"].get( "mode", "constant" )
    print( f"[run] parameter file      :: {parameterFile}" )
    print( f"[run] TTF mode            :: {mode}" )
    print( f"[run] space charge        :: "
           f"{bool(spaceChargeCfg.get('enabled', False))}" )
    print( f"[run] cells               :: {numCells} "
           f"(bunch={numBunchCells}, transition={numTransCells}, acceleration={numAccCells})" )
    print( f"[run] final energy        :: {syncEnergy[-1]:.8f} MeV/u" )
    print( f"[run] final capture       :: {capturePercent[-1]:.2f} %" )
    print( f"[run] data output         :: {cellDataFile}" )
    print( f"[run] phase-space output  :: {phaseSpaceFile}" )

    result = { "parameters"        : params,
               "cell_id"           : cellId,
               "sync_energy"       : syncEnergy,
               "sync_phase"        : syncPhase,
               "phase_correction"  : phaseCorr,
               "particle_phase"    : phaseHistory,
               "particle_energy"   : energyHistory,
               "data_file"         : cellDataFile,
               "phase_space_file"  : phaseSpaceFile,
               "capture_efficiency": capturePercent, }
    return( result )
        


# ========================================================= #
# ===  Execution of Program                             === #
# ========================================================= #
if ( __name__=="__main__" ):
    parameterFile = "inp/parameters.json"
    track__longitudinal1D( parameterFile=parameterFile )
