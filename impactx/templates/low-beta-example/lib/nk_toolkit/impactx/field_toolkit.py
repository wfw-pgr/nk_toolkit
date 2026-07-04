import os, sys
import numpy                        as np
import pandas                       as pd
import nk_toolkit.vtk.write_toolkit as wtk


# ========================================================= #
# ===  save__DTLBfield.py                               === #
# ========================================================= #
def save__DTLBfield( data=None, icell=1, ksymx=0, ksymy=0, ksymz=0,
                     coordScale=1.0, fieldScale=1.0,  
                     outFile="out/eh_PMQ.#01" ):
    """
    Save a 3D magnetic-field map in the ASCII eh_PMQ.#nn format.

    Parameters
    ----------
    data : dict
        3D arrays with common shape ( Nx, Ny, Nz ).

        Coordinate keys are selected in this order:
            xg -> x
            yg -> y
            zg -> z

        Magnetic-field keys are selected in this order:
            Bx -> bx
            By -> by
            Bz -> bz

        Default assumption:
            coordinate : [m]
            B-field    : [T]

    icell : int
        First header value of eh_PMQ.#nn.

    ksymx, ksymy, ksymz : int
        Symmetry flags in the second header line.

    coordScale : float
        Output coordinate = coordScale * input coordinate.
        (Default) 1.0.
        1.0e2 converts [m] to [cm].

    fieldScale : float
        (Default) 1.0
        impactx: T/m, Track: kG/cm, 1.0e4 converts [T] to [G].

    outFile : str
        Output file name.

    Returns
    -------
    outFile : str or None
        Output file name on success.
        None on failure.

    Notes
    -----
    Field rows are written in this order:

        iz fastest, then iy, then ix

    Equivalent loop:

        for ix in range( Nx ):
            for iy in range( Ny ):
                for iz in range( Nz ):
                    write( Bx[ix,iy,iz], By[ix,iy,iz], Bz[ix,iy,iz] )
    """

    # ========================================================= #
    # ===  [0] internal functions                           === #
    # ========================================================= #
    # ------------------------------------------------- #
    # --- [0-1] select data key                     --- #
    # ------------------------------------------------- #
    def _select__DataKey( candidates=None, dataNames=None,
                          quantityName="data" ):

        for candidateName in candidates:

            if ( candidateName in dataNames ):
                return( candidateName )

        raise KeyError(
            f"{quantityName} is required. "
            f"Use one of {candidates}."
        )

    # ------------------------------------------------- #
    # --- [0-2] extract axis coordinate             --- #
    # ------------------------------------------------- #
    def _extract__AxisCoordinate( coordData=None, axisIndex=None,
                                  gridShape=None, coordName="xg" ):

        dimension = len( gridShape )

        indexer = [ 0 ] * dimension
        indexer[axisIndex] = slice( None )

        axisValues = np.asarray(
            coordData[tuple( indexer )], dtype=float
        )

        refShape = [ 1 ] * dimension
        refShape[axisIndex] = gridShape[axisIndex]

        refData = axisValues.reshape( refShape )

        if ( not np.allclose(
                coordData, refData,
                rtol=1.0e-10, atol=1.0e-12
        ) ):
            raise ValueError(
                f"'{coordName}' must depend only on its own axis. "
                "Curvilinear grids cannot be saved as eh_PMQ."
            )

        if ( not np.all( np.isfinite( axisValues ) ) ):
            raise ValueError(
                f"'{coordName}' must contain finite values."
            )

        if ( axisValues.size > 1 ):

            deltaAxis = np.diff( axisValues )

            if ( np.any( deltaAxis <= 0.0 ) ):
                raise ValueError(
                    f"'{coordName}' must be monotonically increasing."
                )

            if ( not np.allclose(
                    deltaAxis, deltaAxis[0],
                    rtol=1.0e-10, atol=1.0e-12
            ) ):
                raise ValueError(
                    f"'{coordName}' must be uniformly spaced."
                )

        return( axisValues )

    # ------------------------------------------------- #
    # --- [1] input check                           --- #
    # ------------------------------------------------- #
    try:

        if ( not isinstance( data, dict ) ):
            raise ValueError( "data must be a dictionary." )

        if ( coordScale == 0.0 ):
            raise ValueError( "coordScale must not be zero." )

        if ( fieldScale == 0.0 ):
            raise ValueError( "fieldScale must not be zero." )

        coordKeys = {
            "x" : [ "xg", "x" ],
            "y" : [ "yg", "y" ],
            "z" : [ "zg", "z" ],
        }
        fieldKeys = {
            "Bx" : [ "Bx", "bx" ],
            "By" : [ "By", "by" ],
            "Bz" : [ "Bz", "bz" ],
        }

        xName = _select__DataKey(
            candidates=coordKeys["x"],
            dataNames=data.keys(),
            quantityName="x coordinate",
        )
        yName = _select__DataKey(
            candidates=coordKeys["y"],
            dataNames=data.keys(),
            quantityName="y coordinate",
        )
        zName = _select__DataKey(
            candidates=coordKeys["z"],
            dataNames=data.keys(),
            quantityName="z coordinate",
        )

        BxName = _select__DataKey(
            candidates=fieldKeys["Bx"],
            dataNames=data.keys(),
            quantityName="Bx field",
        )
        ByName = _select__DataKey(
            candidates=fieldKeys["By"],
            dataNames=data.keys(),
            quantityName="By field",
        )
        BzName = _select__DataKey(
            candidates=fieldKeys["Bz"],
            dataNames=data.keys(),
            quantityName="Bz field",
        )

        xg = np.asarray( data[xName] )
        yg = np.asarray( data[yName] )
        zg = np.asarray( data[zName] )

        Bx = np.asarray( data[BxName], dtype=float )
        By = np.asarray( data[ByName], dtype=float )
        Bz = np.asarray( data[BzName], dtype=float )

        if ( Bx.ndim != 3 ):
            raise ValueError(
                f"'{BxName}' must be a 3D array, but shape={Bx.shape}."
            )

        gridShape = Bx.shape
        Nx, Ny, Nz = gridShape

        dataArrays = {
            xName : xg,
            yName : yg,
            zName : zg,
            BxName: Bx,
            ByName: By,
            BzName: Bz,
        }

        for dataName, values in dataArrays.items():

            if ( values.shape != gridShape ):
                raise ValueError(
                    f"'{dataName}' shape={values.shape} does not match "
                    f"'{BxName}' shape={gridShape}."
                )

            if ( not np.issubdtype( values.dtype, np.number ) ):
                raise ValueError(
                    f"'{dataName}' must have a numeric dtype."
                )

            if ( np.issubdtype( values.dtype, np.complexfloating ) ):
                raise ValueError(
                    f"'{dataName}' must be real."
                )

            if ( not np.all( np.isfinite( values ) ) ):
                raise ValueError(
                    f"'{dataName}' must contain finite values."
                )

        # ------------------------------------------------- #
        # --- [2] grid header                           --- #
        # ------------------------------------------------- #
        xa = _extract__AxisCoordinate(
            coordData=xg,
            axisIndex=0,
            gridShape=gridShape,
            coordName=xName,
        )
        ya = _extract__AxisCoordinate(
            coordData=yg,
            axisIndex=1,
            gridShape=gridShape,
            coordName=yName,
        )
        za = _extract__AxisCoordinate(
            coordData=zg,
            axisIndex=2,
            gridShape=gridShape,
            coordName=zName,
        )

        xMin, xMax = coordScale * xa[0], coordScale * xa[-1]
        yMin, yMax = coordScale * ya[0], coordScale * ya[-1]
        zMin, zMax = coordScale * za[0], coordScale * za[-1]

        # ------------------------------------------------- #
        # --- [3] flatten field map                     --- #
        # ------------------------------------------------- #
        fieldData = np.column_stack( [
            fieldScale * Bx.ravel( order="C" ),
            fieldScale * By.ravel( order="C" ),
            fieldScale * Bz.ravel( order="C" ),
        ] )
        # fieldData = np.column_stack( [
        #     fieldScale * Bx.ravel( order="F" ),
        #     fieldScale * By.ravel( order="F" ),
        #     fieldScale * Bz.ravel( order="F" ),
        # ] )

        # ------------------------------------------------- #
        # --- [4] write field map                       --- #
        # ------------------------------------------------- #
        outDir = os.path.dirname( outFile )

        if ( outDir != "" ):
            os.makedirs( outDir, exist_ok=True )

        with open( outFile, "w" ) as fileUnit:

            fileUnit.write( f"{int(icell):d}\n" )
            fileUnit.write(
                f"{int(ksymx):d} {int(ksymy):d} {int(ksymz):d}\n"
            )

            fileUnit.write(
                f"{xMin:15.8E} {xMax:15.8E} {Nx:d}\n"
            )
            fileUnit.write(
                f"{yMin:15.8E} {yMax:15.8E} {Ny:d}\n"
            )
            fileUnit.write(
                f"{zMin:15.8E} {zMax:15.8E} {Nz:d}\n"
            )

            np.savetxt( fileUnit, fieldData, fmt="%15.8E" )

        # ------------------------------------------------- #
        # --- [5] report                                --- #
        # ------------------------------------------------- #
        print( "[save__DTLBfield]" )
        print( f"  outFile     = {outFile}" )
        print( f"  shape       = ( {Nx}, {Ny}, {Nz} )" )
        print( f"  header      = icell={int(icell)}" )
        print(
            f"  symmetry    = "
            f"( {int(ksymx)}, {int(ksymy)}, {int(ksymz)} )"
        )
        print(
            f"  x-range[cm] = ( {xMin:.8E}, {xMax:.8E} )"
        )
        print(
            f"  y-range[cm] = ( {yMin:.8E}, {yMax:.8E} )"
        )
        print(
            f"  z-range[cm] = ( {zMin:.8E}, {zMax:.8E} )"
        )
        print( f"  coordScale  = {coordScale:.8E}" )
        print( f"  fieldScale  = {fieldScale:.8E}" )
        print( "  data order  = ix fastest, then iy, then iz" )
        print( "  status      = success" )

        return( outFile )

    except Exception as exc:

        print( "[save__DTLBfield]" )
        print( f"  outFile     = {outFile}" )
        print( "  status      = failed" )
        print( f"  reason      = {exc}" )

        return( None )




# ========================================================= #
# ===  generate__QMField.py                             === #
# ========================================================= #
def generate__QMField( L_qm =1.0, G_qm=1.0, L_fringe=None, zc=0.0, 
                       xMin=-1.0, xMax=1.0, xNum=11, yMin=-1.0, yMax=1.0, yNum=11,
                       zMin=-1.0, zMax=1.0, zNum=11, 
                       csvFile="out/ideal_QMField.dat", vtiFile="out/ideal_QMField.vti", ):
    """
    Ideal normal quadrupole field map.

    Parameters
    ----------
    L_qm     : magnetic length [m]
    G_qm     : quadrupole gradient [T/m]
    L_fringe : None -> hard-edge model
               float [m] -> smooth tanh fringe model
    """

    # ------------------------------------------------- #
    # --- [1] grid making                           --- #
    # ------------------------------------------------- #
    xa       = np.linspace( xMin, xMax, xNum )
    ya       = np.linspace( yMin, yMax, yNum )
    za       = np.linspace( zMin, zMax, zNum )
    xg,yg,zg = np.meshgrid( xa, ya, za, indexing="ij" )
    
    if ( L_fringe is None ):
        # ------------------------------------------------- #
        # --- [1-1]  Hard-edge                          --- #
        # ------------------------------------------------- #
        G_reg  = G_qm  * ( np.abs( zg - zc ) <= L_qm / 2.0)
        Bx     = G_reg * yg
        By     = G_reg * xg
        Bz     = np.zeros_like(Bx)

    else:
        # ------------------------------------------------- #
        # --- [1-2]  Soft-edge                          --- #
        # ------------------------------------------------- #
        L_fringe = float(L_fringe)
        if ( L_fringe <= 0.0 ):
            raise ValueError(" L_fringe must be positive.")

        z1 = zc - L_qm / 2.0
        z2 = zc + L_qm / 2.0

        u1 = ( zg - z1 ) / L_fringe
        u2 = ( zg - z2 ) / L_fringe

        # ------------------------------------------------- #
        # --- [1-3] d/dx tanh (x)                       --- #
        # ------------------------------------------------- #
        def tanh_derivatives(u):
            """
            tanh(u), d/dz tanh(u), d2/dz2 tanh(u), d3/dz3 tanh(u)
            where u=(z-z0)/a.
            """
            t  = np.tanh(u)
            s2 = 1.0 - t**2
            f0 = t
            f1 = s2 / L_fringe
            f2 = -2.0 * t * s2 / L_fringe**2
            f3 = -2.0 * s2 * (1.0 - 3.0 * t**2) / L_fringe**3
            return( f0, f1, f2, f3 )

        # ------------------------------------------------- #
        # --- [1-4] calculate field                     --- #
        # ------------------------------------------------- #
        t1, dt1, d2t1, d3t1 = tanh_derivatives(u1)
        t2, dt2, d2t2, d3t2 = tanh_derivatives(u2)

        g   = 0.5 * G_qm * ( t1   - t2   )
        gp  = 0.5 * G_qm * ( dt1  - dt2  )
        gpp = 0.5 * G_qm * ( d2t1 - d2t2 )
        g3  = 0.5 * G_qm * ( d3t1 - d3t2 )

        r2  = xg**2 + yg**2

        Bx  = yg * ( g - gpp * (3.0 * xg**2 + yg**2) / 12.0 )
        By  = xg * ( g - gpp * (xg**2 + 3.0 * yg**2) / 12.0 )
        Bz  = xg * yg * ( gp - r2 * g3 / 12.0 )

    # ------------------------------------------------- #
    # --- [2] stack data / output data              --- #
    # ------------------------------------------------- #
    data_ = { "xg": xg.ravel(), "yg": yg.ravel(), "zg": zg.ravel(), \
              "Bx": Bx.ravel(), "By": By.ravel(), "Bz": Bz.ravel(), }
    df    = pd.DataFrame.from_dict( data_ )
    df.to_csv( csvFile )
    
    data  = { "xg": xg, "yg": yg, "zg": zg, \
              "Bx": Bx, "By": By, "Bz": Bz, }
    ret = wtk.write__vtkImageData( data=data.copy(), vtiFile=vtiFile )

    # ------------------------------------------------- #
    # --- [4] save and return                       --- #
    # ------------------------------------------------- #
    return( data )




# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    mode = "generate__QMField"
    
    if ( mode == "generate__QMField" ):
        
        params = {
            "L_qm"     : 1.0 , "G_qm": 1.0,
            "L_fringe" : 1.0 , "zc"  : 0.0, 
            "xMin"     :-1.0 , "xMax": 1.0, "xNum":11,
            "yMin"     :-1.0 , "yMax": 1.0, "yNum":11,
            "zMin"     :-1.0 , "zMax": 1.0, "zNum":11, 
            "csvFile"  :"out/ideal_QMField.dat" ,
            "vtiFile"  :"out/ideal_QMField.vti" ,        
        }
        data = generate__QMField( **params )
        print( data.keys() )
        ret = save__DTLBfield( data=data, outFile="out/eh_PMQ.#01", )
