import os, sys, json, re
import numpy  as np
import pandas as pd
import nk_toolkit.plot.load__config as lcf
import nk_toolkit.plot.gplot1D      as gp1
import nk_toolkit.plot.gplot2D      as gp2


# ========================================================= #
# ===  analyze__TTFbyEnergyAperture                     === #
# ========================================================= #
def analyze__TTFbyEnergyAperture( inpFile=None, outFile="out/ttf_summary.json",
                                  axisFile="out/ttf_axis_Ez.csv", csvFile="out/ttf_table.csv",
                                  frequency_MHz=36.5,
                                  massEnergy_MeV_u=931.49410242,
                                  energyMin_MeV_u=0.1, energyMax_MeV_u=5.0, energyNum=100,
                                  energyList_MeV_u=None, aperture_m=0.100, apertureList_m=None,
                                  minimumAperturePoints=2, apertureCopyStep_m=1.0e-5,
                                  coordUnit="cm", dataOrder="z-fastest", mirrorZ=True,
                                  phaseReference_m=0.0, plotFigure=True, pngDir="png/ttf/" ):
    """
    Calculate an on-axis transit-time-factor table from an r-z RF field map.

    The input field order is Er, Ez, Bphi. Coordinates are converted to metre.
    Kinetic energy and rest energy are per nucleon [MeV/u]. The returned table
    uses aperture [m], energy [MeV/u], TTF amplitude, phase [rad], and the
    signed in-phase component T*cos(phase).
    """

    # ------------------------------------------------- #
    # --- [1] internal functions                    --- #
    # ------------------------------------------------- #
    def _parseNumbers( line=None ):
        text = line.split( "#", 1 )[0].strip()
        if ( not text ):
            return( None )
        try:
            return( [ float( token.replace( "D", "E" ).replace( "d", "e" ) )
                      for token in text.split() ] )
        except ValueError:
            return( None )

    def _makeParentDir( fileName=None ):
        parentDir = os.path.dirname( fileName )
        if ( parentDir ):
            os.makedirs( parentDir, exist_ok=True )

    def _writeJson( fileName=None, data=None ):
        _makeParentDir( fileName=fileName )
        with open( fileName, "w", encoding="utf-8" ) as fileObj:
            json.dump( data, fileObj, ensure_ascii=False, indent=2,
                       allow_nan=False )
            fileObj.write( "\n" )

    def _readFieldMap( fileName=None ):
        with open( fileName, "r", encoding="utf-8" ) as fileObj:
            lineList = fileObj.readlines()

        cellIndex = None
        for il, line in enumerate( lineList ):
            if ( re.match( r"^\s*Cell\s*:\s*\d+", line, re.IGNORECASE ) ):
                cellIndex = il
                break
        if ( cellIndex is None ):
            raise ValueError( "Cell : nn was not found in the input file." )

        headerList = []
        lineIndex  = cellIndex + 1
        while ( ( len( headerList ) < 4 ) and
                ( lineIndex < len( lineList ) ) ):
            values = _parseNumbers( line=lineList[lineIndex] )
            if ( values is not None ):
                headerList.append( values )
            lineIndex += 1

        if ( len( headerList ) < 4 ):
            raise ValueError( "The two flags and r/z grid headers are incomplete." )
        if ( ( len( headerList[0] ) != 1 ) or
             ( len( headerList[1] ) != 1 ) or
             ( len( headerList[2] ) != 3 ) or
             ( len( headerList[3] ) != 3 ) ):
            raise ValueError( "Unexpected field-map header format." )

        rMinRaw, rMaxRaw, numRRaw = headerList[2]
        zMinRaw, zMaxRaw, numZRaw = headerList[3]
        numR = int( round( numRRaw ) )
        numZ = int( round( numZRaw ) )
        if ( ( numR < 1 ) or ( numZ < 2 ) ):
            raise ValueError( "numR >= 1 and numZ >= 2 are required." )

        fieldList = []
        while ( ( len( fieldList ) < numR * numZ ) and
                ( lineIndex < len( lineList ) ) ):
            values = _parseNumbers( line=lineList[lineIndex] )
            if ( ( values is not None ) and ( len( values ) >= 3 ) ):
                fieldList.append( values[:3] )
            lineIndex += 1
        if ( len( fieldList ) != numR * numZ ):
            message = ( f"RF field data are incomplete: {len(fieldList)} / "
                        f"{numR*numZ} rows." )
            raise ValueError( message )

        unitScale = { "m":1.0, "cm":1.0e-2, "mm":1.0e-3 }
        if ( coordUnit not in unitScale ):
            raise ValueError( "coordUnit must be 'm', 'cm', or 'mm'." )
        rAxis_m = np.linspace( rMinRaw, rMaxRaw, numR ) * unitScale[coordUnit]
        zAxis_m = np.linspace( zMinRaw, zMaxRaw, numZ ) * unitScale[coordUnit]
        field   = np.asarray( fieldList, dtype=float )

        if ( dataOrder == "z-fastest" ):
            field = field.reshape( numR, numZ, 3 )
        elif ( dataOrder == "r-fastest" ):
            field = field.reshape( numZ, numR, 3 ).transpose( 1, 0, 2 )
        else:
            raise ValueError(
                "dataOrder must be 'z-fastest' or 'r-fastest'." )

        axisIndex = int( np.argmin( np.abs( rAxis_m ) ) )
        if ( abs( rAxis_m[axisIndex] ) > 1.0e-12 ):
            raise ValueError( "The field map does not contain the r=0 axis." )

        ret = {
            "header_flags" : [ int( round( headerList[0][0] ) ),
                               int( round( headerList[1][0] ) ) ],
            "r_axis_m"     : rAxis_m,
            "z_axis_m"     : zAxis_m,
            "Er_axis_V_m"  : field[axisIndex,:,0],
            "Ez_axis_V_m"  : field[axisIndex,:,1],
            "Bphi_axis_T"  : field[axisIndex,:,2],
            "num_r"        : numR,
            "num_z"        : numZ,
        }
        return( ret )

    
    def _plot1DByAperture( xAxis=None, yData=None, apertureAxis=None,
                           pngFile=None, xLabel=None, yLabel=None ):
        config = {
            **lcf.load__config(),
            "figure.size"        : [ 5.0, 4.0 ],
            "figure.pngFile"     : pngFile,
            "figure.position"    : [ 0.16, 0.16, 0.94, 0.94 ],
            "ax1.x.range"        : { "auto":True, "min":0.0, "max":1.0, "num":11 },
            "ax1.y.range"        : { "auto":True, "min":0.0, "max":1.0, "num":11 },
            "ax1.x.label"        : xLabel,
            "ax1.y.label"        : yLabel,
            "ax1.x.minor.nticks" : 1,
            "plot.marker"        : "none",
            "legend.fontsize"    : 9.0,
        }
        _makeParentDir( fileName=pngFile )
        fig = gp1.gplot1D( config=config, pngFile=pngFile )
        for apertureIndex, apertureValue in enumerate( apertureAxis ):
            label = f"a = {apertureValue:.6g} m"
            fig.add__plot( xAxis=xAxis, yAxis=yData[apertureIndex,:],
                           color=f"C{apertureIndex%10}", label=label )
        fig.set__axis()
        fig.set__legend()
        fig.save__figure()
        

    def _plot2DMap( xGrid=None, yGrid=None, zData=None, pngFile=None,
                    xLabel=None, yLabel=None, zLabel=None ):
        config = {
            **lcf.load__config(),
            "figure.size"     : [ 5.0, 5.0 ],
            "figure.pngFile"  : pngFile,
            "figure.position" : [ 0.16, 0.16, 0.84, 0.84 ],
            "ax1.x.range"     : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
            "ax1.y.range"     : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
            "ax1.x.label"     : xLabel,
            "ax1.y.label"     : yLabel,
            "cmp.level"       : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
            "cmp.colortable"  : "jet",
        }
        _makeParentDir( fileName=pngFile )
        fig    = gp2.gplot2D( xAxis=xGrid.reshape( (-1,) ), yAxis=yGrid.reshape( (-1,) ),
                              cMap=zData.reshape( (-1,) ), config=config )

    # ------------------------------------------------- #
    # --- [2] input and output settings             --- #
    # ------------------------------------------------- #
    try:
        if ( inpFile is None ):
            raise ValueError( "inpFile must be specified." )
        if ( frequency_MHz <= 0.0 ):
            raise ValueError( "frequency_MHz must be positive." )
        if ( massEnergy_MeV_u <= 0.0 ):
            raise ValueError( "massEnergy_MeV_u must be positive." )
        if ( minimumAperturePoints < 1 ):
            raise ValueError( "minimumAperturePoints must be at least one." )

        # ------------------------------------------------- #
        # --- [3] read axial field                      --- #
        # ------------------------------------------------- #
        fieldMap = _readFieldMap( fileName=inpFile )
        zAxis_m  = fieldMap["z_axis_m"]
        EzAxis   = fieldMap["Ez_axis_V_m"]

        mirroredZ = False
        if ( mirrorZ and ( zAxis_m[0] >= -1.0e-14 ) ):
            if ( abs( zAxis_m[0] ) > 1.0e-12 ):
                raise ValueError( "mirrorZ=True requires zmin=0 for a half map." )
            zAxis_m = np.concatenate( ( -zAxis_m[:0:-1], zAxis_m ) )
            EzAxis  = np.concatenate( (  EzAxis[:0:-1], EzAxis  ) )
            mirroredZ = True

        if ( np.any( np.diff( zAxis_m ) <= 0.0 ) ):
            raise ValueError( "z coordinates must be strictly increasing." )

        axisData = pd.DataFrame( { "z_m":zAxis_m, "Ez_V_m":EzAxis } )
        _makeParentDir( fileName=axisFile )
        axisData.to_csv( axisFile, index=False )

        # ------------------------------------------------- #
        # --- [4] energy and aperture grids             --- #
        # ------------------------------------------------- #
        if ( energyList_MeV_u is None ):
            energyList_MeV_u = np.linspace( energyMin_MeV_u,
                                            energyMax_MeV_u, energyNum )
        else:
            energyList_MeV_u = np.asarray( energyList_MeV_u, dtype=float )
        if ( np.any( energyList_MeV_u <= 0.0 ) ):
            raise ValueError( "All kinetic energies must be positive." )

        if ( apertureList_m is None ):
            apertureList_m = np.asarray( [ aperture_m ], dtype=float )
        else:
            apertureList_m = np.asarray( apertureList_m, dtype=float )
        if ( np.any( apertureList_m <= 0.0 ) ):
            raise ValueError( "All aperture radii must be positive." )

        copiedAperture = False
        if ( apertureList_m.size < minimumAperturePoints ):
            copiedAperture = True
            apertureCenter_m = float( apertureList_m[0] )
            offset = ( np.arange( minimumAperturePoints )
                       - 0.5 * ( minimumAperturePoints - 1 ) )
            apertureList_m = apertureCenter_m + offset * apertureCopyStep_m
            if ( np.any( apertureList_m <= 0.0 ) ):
                raise ValueError( "Copied aperture radii became non-positive." )

        # ------------------------------------------------- #
        # --- [5] TTF calculation                       --- #
        # ------------------------------------------------- #
        lightSpeed_m_s = 299792458.0
        omega_rad_s    = 2.0 * np.pi * frequency_MHz * 1.0e6
        gamma          = 1.0 + energyList_MeV_u / massEnergy_MeV_u
        beta           = np.sqrt( 1.0 - 1.0 / gamma**2 )
        voltageNorm_V  = np.trapezoid( np.abs( EzAxis ), zAxis_m )
        voltageDC_V    = np.trapezoid( EzAxis, zAxis_m )
        if ( voltageNorm_V <= 0.0 ):
            raise ValueError( "Integral of |Ez| is zero." )

        absTtfList   = []
        phaseTtfList = []
        realTtfList  = []
        for betaValue in beta:
            phase = omega_rad_s * ( zAxis_m - phaseReference_m ) \
                    / ( betaValue * lightSpeed_m_s )
            voltageComplex_V = np.trapezoid( EzAxis * np.exp( 1.0j * phase ),
                                             zAxis_m )
            absTtfList.append( float( abs( voltageComplex_V ) / voltageNorm_V ) )
            phaseTtfList.append( float( np.angle( voltageComplex_V ) ) )
            realTtfList.append( float( voltageComplex_V.real / voltageNorm_V ) )

        absTtfList   = np.asarray( absTtfList )
        phaseTtfList = np.asarray( phaseTtfList )
        realTtfList  = np.asarray( realTtfList )

        # ------------------------------------------------- #
        # --- [6] make plot                             --- #
        # ------------------------------------------------- #
        pngFileDict = {
            "ttf_abs_1d"  : os.path.join( pngDir, "ttf_abs_vs_energy.png" ),
            "ttf_phase_1d": os.path.join( pngDir, "ttf_phase_vs_energy.png" ),
            "ttf_real_1d" : os.path.join( pngDir, "ttf_real_vs_energy.png" ),
            "ttf_abs_2d"  : os.path.join( pngDir, "ttf_abs_map.png" ),
            "ttf_phase_2d": os.path.join( pngDir, "ttf_phase_map.png" ),
            "ttf_real_2d" : os.path.join( pngDir, "ttf_real_map.png" ),
            "Ez_axis_1d"  : os.path.join( pngDir, "Ez_axis_vs_z.png" ),
        }
        absTtfGrid   = np.tile( absTtfList   , ( apertureList_m.size, 1 ) )
        phaseTtfGrid = np.tile( phaseTtfList , ( apertureList_m.size, 1 ) )
        realTtfGrid  = np.tile( realTtfList  , ( apertureList_m.size, 1 ) )

        if ( plotFigure ):
            _plot1DByAperture( xAxis=energyList_MeV_u, yData=absTtfGrid,
                               apertureAxis=apertureList_m, pngFile=pngFileDict["ttf_abs_1d"],
                               xLabel="K [MeV/u]", yLabel="|TTF|" )
            _plot1DByAperture( xAxis=energyList_MeV_u, yData=phaseTtfGrid,
                               apertureAxis=apertureList_m,
                               pngFile=pngFileDict["ttf_phase_1d"],
                              xLabel="K [MeV/u]", yLabel="TTF phase [rad]" )
            _plot1DByAperture( xAxis=energyList_MeV_u, yData=realTtfGrid,
                               apertureAxis=apertureList_m, pngFile=pngFileDict["ttf_real_1d"],
                               xLabel="K [MeV/u]", yLabel="Re(TTF)" )
            
            energyGrid, apertureGrid = np.meshgrid( energyList_MeV_u, apertureList_m )
            _plot2DMap( xGrid=energyGrid, yGrid=apertureGrid, zData=absTtfGrid,
                        pngFile=pngFileDict["ttf_abs_2d"], xLabel="K [MeV/u]",
                        yLabel="a [m]", zLabel="|TTF|" )
            _plot2DMap( xGrid=energyGrid, yGrid=apertureGrid, zData=phaseTtfGrid,
                        pngFile=pngFileDict["ttf_phase_2d"], xLabel="K [MeV/u]",
                        yLabel="a [m]", zLabel="TTF phase [rad]" )
            _plot2DMap( xGrid=energyGrid, yGrid=apertureGrid, zData=realTtfGrid,
                        pngFile=pngFileDict["ttf_real_2d"], xLabel="K [MeV/u]",
                        yLabel="a [m]", zLabel="Re(TTF)" )

            EzGrid = np.tile( EzAxis, ( apertureList_m.size, 1 ) )
            _plot1DByAperture( xAxis=zAxis_m, yData=EzGrid, apertureAxis=apertureList_m,
                               pngFile=pngFileDict["Ez_axis_1d"],
                               xLabel="z [m]", yLabel="Ez [V/m]" )
        
        # ------------------------------------------------- #
        # --- [7] table and summary output              --- #
        # ------------------------------------------------- #
        tableList = []
        for apertureValue in apertureList_m:
            for ik, energyValue in enumerate( energyList_MeV_u ):
                tableList.append( {
                    "aperture_m"  : float( apertureValue ),
                    "energy_MeV_u": float( energyValue ),
                    "abs_ttf"     : float( absTtfList[ik] ),
                    "phase_rad"   : float( phaseTtfList[ik] ),
                    "real_ttf"    : float( realTtfList[ik] ),
                } )
        tableData = pd.DataFrame( tableList )
        _makeParentDir( fileName=csvFile )
        tableData.to_csv( csvFile, index=False )

        ret = {
            "status"        : "success",
            "failure_reason": "",
            "settings": {
                "source_file"          : inpFile,
                "data_order"           : dataOrder,
                "input_coordinate_unit": coordUnit,
                "field_components"     : [ "Er_V_m", "Ez_V_m", "Bphi_T" ],
                "frequency_MHz"        : float( frequency_MHz ),
                "mass_energy_MeV_u"    : float( massEnergy_MeV_u ),
                "mirror_z_even"        : bool( mirroredZ ),
                "phase_reference_m"    : float( phaseReference_m ),
                "phase_convention"     : "+omega*(z-zref)/(beta*c)",
                "normalization"        : "integral(abs(Ez))*dz",
                "aperture_copy_step_m" : float( apertureCopyStep_m ),
                "plot_figure"          : bool( plotFigure ),
            },
            "output_files": {
                "axis_field_csv_file" : axisFile,
                "ttf_table_csv_file"  : csvFile,
                "png_files"           : pngFileDict if plotFigure else {},
            },
            "field_map_summary": {
                "header_flags": fieldMap["header_flags"],
                "r_min_m"     : float( fieldMap["r_axis_m"][0] ),
                "r_max_m"     : float( fieldMap["r_axis_m"][-1] ),
                "num_r"       : int( fieldMap["num_r"] ),
                "z_min_m"     : float( zAxis_m[0] ),
                "z_max_m"     : float( zAxis_m[-1] ),
                "num_z"       : int( zAxis_m.size ),
                "voltage_norm_V"     : float( voltageNorm_V ),
                "voltage_dc_signed_V": float( voltageDC_V ),
            },
            "table_summary": {
                "num_energy"          : int( energyList_MeV_u.size ),
                "energy_min_MeV_u"    : float( np.min( energyList_MeV_u ) ),
                "energy_max_MeV_u"    : float( np.max( energyList_MeV_u ) ),
                "num_aperture"        : int( apertureList_m.size ),
                "aperture_m"          : apertureList_m.tolist(),
                "aperture_data_copied": bool( copiedAperture ),
                "num_table_rows"      : int( tableData.shape[0] ),
                "abs_ttf_min"         : float( np.min( absTtfList ) ),
                "abs_ttf_max"         : float( np.max( absTtfList ) ),
                "phase_rad_min"       : float( np.min( phaseTtfList ) ),
                "phase_rad_max"       : float( np.max( phaseTtfList ) ),
                "real_ttf_min"        : float( np.min( realTtfList ) ),
                "real_ttf_max"        : float( np.max( realTtfList ) ),
            },
        }
        _writeJson( fileName=outFile, data=ret )

        # ------------------------------------------------- #
        # --- [8] standard output                       --- #
        # ------------------------------------------------- #
        print( "[analyze__TTFbyEnergyAperture] summary JSON ::", outFile )
        print( "[analyze__TTFbyEnergyAperture] axis Ez      ::", axisFile )
        print( "[analyze__TTFbyEnergyAperture] table CSV    ::", csvFile )
        print( "[analyze__TTFbyEnergyAperture] data order   ::", dataOrder )
        print( "[analyze__TTFbyEnergyAperture] mirror z     ::", mirroredZ )
        print( "[analyze__TTFbyEnergyAperture] max|Er| axis ::",
               np.max( np.abs( fieldMap["Er_axis_V_m"] ) ) )
        print( "[analyze__TTFbyEnergyAperture] max|Bp| axis ::",
               np.max( np.abs( fieldMap["Bphi_axis_T"] ) ) )
        print( tableData.to_string( index=False ) )
        if ( plotFigure ):
            for plotName, pngFile in pngFileDict.items():
                print( f"[analyze__TTFbyEnergyAperture] {plotName:13s} ::", pngFile )
                print( "[analyze__TTFbyEnergyAperture] status       :: success" )
        return( ret )

    except Exception as error:
        ret = { "status":"failed", "failure_reason":str( error ) }
        print( "[analyze__TTFbyEnergyAperture] status       :: failed" )
        print( "[analyze__TTFbyEnergyAperture] reason       ::", str( error ) )
        if ( outFile is not None ):
            try:
                _writeJson( fileName=outFile, data=ret )
                print( "[analyze__TTFbyEnergyAperture] summary JSON ::", outFile )
            except Exception:
                pass
        return( ret )


# ========================================================= #
# ===  Execution of Program                             === #
# ========================================================= #
if ( __name__=="__main__" ):
    inpFile = "inp/eh_DTL.#01"
    outFile = "out/ttf_summary.json"
    csvFile = "out/ttf_table.csv"
    ret = analyze__TTFbyEnergyAperture( inpFile=inpFile, outFile=outFile, csvFile=csvFile,
                                        frequency_MHz=36.5, coordUnit="cm", dataOrder="z-fastest",
                                        aperture_m=0.100, minimumAperturePoints=2, mirrorZ=True )

