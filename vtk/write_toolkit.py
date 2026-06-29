import os
import numpy as np
import vtk
import vtk.util.numpy_support as vtknp


# ========================================================= #
# ===  write__vtkImageData.py                            === #
# ========================================================= #
def write__vtkImageData( data=None, vtiFile="out/out.vti", ):
    """
    Save 2D / 3D scalar grid arrays in a dictionary as VTK ImageData.

    Parameters
    ----------
    data : dict
        Dictionary containing same-shape scalar arrays.

        Reserved metadata keys:
            "origin"  : [ x0, y0 ] or [ x0, y0, z0 ]
            "spacing" : [ dx, dy ] or [ dx, dy, dz ]

        Coordinate arrays named "x", "y", "z" are written as PointData.
        When origin / spacing are omitted, they are inferred from these
        arrays. Missing coordinate axes use index coordinates.

    vtiFile : str
        Output VTK ImageData file name.

    Returns
    -------
    imageData : vtk.vtkImageData or None
        Created VTK ImageData object. None on failure.

    Notes
    -----
    Each physical quantity must be a scalar ndarray with shape:
        2D : ( Nx, Ny )
        3D : ( Nx, Ny, Nz )

    VTI supports only uniformly spaced, axis-aligned grids.
    """

    # ========================================================= #
    # ===  [0] functions                                    === #
    # ========================================================= #
    # ------------------------------------------------- #
    # --- [0-1] expand grid parameters              --- #
    # ------------------------------------------------- #
    def _expand__GridParameter( values=None, dimension=None, defaultValue=0.0,
                                valueName="parameter", ):
        """
        Expand 2D / 3D grid parameter into a 3-component VTK tuple.
        """
        if ( values is None ):
            return( None )
        
        values = np.asarray( values, dtype=float ).reshape( -1 )
        
        if ( values.size not in [ dimension, 3 ] ):
            raise ValueError(
                f"{valueName} must have {dimension} or 3 components."
            )
        gridValues = np.full( 3, defaultValue, dtype=float )
        gridValues[:values.size] = values
        
        if ( not np.all( np.isfinite( gridValues[:dimension] ) ) ):
            raise ValueError( f"{valueName} must contain finite values." )
        
        return( gridValues )

    # ------------------------------------------------- #
    # --- [0-2] infer axis geometry                 --- #
    # ------------------------------------------------- #
    def _infer__AxisGeometry( coordData=None, axisIndex=None,
                              gridShape=None, axisName="x" ):
        """
        Infer origin and spacing from one coordinate array.
        """
        
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
                f"'{axisName}' is not an axis-aligned coordinate array. "
                "Use .vts for curvilinear grids."
            )
        
        if ( not np.all( np.isfinite( axisValues ) ) ):
            raise ValueError(
                f"'{axisName}' must contain finite coordinate values."
            )
        
        axisOrigin = axisValues[0]
        
        if ( axisValues.size == 1 ):
            return( axisOrigin, None )
        
        deltaAxis = np.diff( axisValues )
        axisSpacing = deltaAxis[0]

        if ( np.isclose( axisSpacing, 0.0 ) ):
            raise ValueError(
                f"'{axisName}' has zero spacing."
            )
        
        if ( not np.allclose(
                deltaAxis, axisSpacing,
                rtol=1.0e-10, atol=1.0e-12
        ) ):
            raise ValueError(
                f"'{axisName}' is not uniformly spaced. "
                "Use .vtr for nonuniform rectilinear grids."
            )
    
        return( axisOrigin, axisSpacing )


    # ------------------------------------------------- #
    # --- [1] input data                            --- #
    # ------------------------------------------------- #
    try:

        if ( not isinstance( data, dict ) ):
            raise ValueError( "data must be a dictionary." )

        metaNames  = [ "origin", "spacing" ]
        arrayNames = [
            dataName for dataName in data.keys()
            if dataName not in metaNames
        ]

        if ( len( arrayNames ) == 0 ):
            raise ValueError(
                "data must contain at least one grid-data array."
            )

        arrayData = {}

        for dataName in arrayNames:

            if ( not isinstance( dataName, str ) ):
                raise ValueError(
                    "Dictionary keys for grid data must be strings."
                )

            values = np.asarray( data[dataName] )

            if ( values.ndim not in [ 2, 3 ] ):
                raise ValueError(
                    f"'{dataName}' must be a 2D or 3D scalar array, "
                    f"but shape={values.shape}."
                )

            if ( values.dtype == np.bool_ ):
                values = values.astype( np.uint8 )

            if ( not np.issubdtype( values.dtype, np.number ) ):
                raise ValueError(
                    f"'{dataName}' must have a numeric dtype."
                )

            if ( np.issubdtype( values.dtype, np.complexfloating ) ):
                raise ValueError(
                    f"'{dataName}' is complex. "
                    "Store real and imaginary parts separately."
                )

            arrayData[dataName] = values

        gridShape = arrayData[arrayNames[0]].shape
        dimension = len( gridShape )

        for dataName in arrayNames:

            if ( arrayData[dataName].shape != gridShape ):
                raise ValueError(
                    f"'{dataName}' shape={arrayData[dataName].shape} "
                    f"does not match grid shape={gridShape}."
                )

        # ------------------------------------------------- #
        # --- [2] origin and spacing                    --- #
        # ------------------------------------------------- #
        coordCandidates = {
            "x" : [ "x", "xg" ],
            "y" : [ "y", "yg" ],
            "z" : [ "z", "zg" ],
        }

        axisNames  = [ "x", "y", "z" ][:dimension]
        coordNames = []

        for axisName in axisNames:
            
            coordName = None

            for candidateName in coordCandidates[axisName]:

                if ( candidateName in arrayData ):
                    coordName = candidateName
                    break

            coordNames.append( coordName )

        explicitOrigin = _expand__GridParameter(
            values=data.get( "origin" ),
            dimension=dimension,
            defaultValue=0.0,
            valueName="origin",
        )
        explicitSpacing = _expand__GridParameter(
            values=data.get( "spacing" ),
            dimension=dimension,
            defaultValue=1.0,
            valueName="spacing",
        )
        
        vtkOrigin  = np.zeros( 3, dtype=float )
        vtkSpacing = np.ones( 3, dtype=float )
        
        for axisIndex, axisName in enumerate( axisNames ):
            
            coordName = coordNames[axisIndex]
            
            if ( coordName is None ):
                continue

            axisOrigin, axisSpacing = _infer__AxisGeometry(
                coordData=arrayData[coordName],
                axisIndex=axisIndex,
                gridShape=gridShape,
                axisName=coordName,
            )

            vtkOrigin[axisIndex] = axisOrigin
            
            if ( axisSpacing is not None ):
                vtkSpacing[axisIndex] = axisSpacing
                
        if ( explicitOrigin is not None ):

            for axisIndex, axisName in enumerate( axisNames ):

                coordName = coordNames[axisIndex]

                if (
                        coordName is not None
                        and not np.isclose(
                            explicitOrigin[axisIndex],
                            vtkOrigin[axisIndex],
                            rtol=1.0e-10,
                            atol=1.0e-12,
                        )
                ):
                    raise ValueError(
                        f"origin[{axisIndex}] is inconsistent with "
                        f"coordinate array '{coordName}'."
                    )

            vtkOrigin[:dimension] = explicitOrigin[:dimension]

        if ( explicitSpacing is not None ):

            for axisIndex, axisName in enumerate( axisNames ):
                
                coordName = coordNames[axisIndex]
                
                if (
                        coordName is not None
                        and gridShape[axisIndex] > 1
                        and not np.isclose(
                            explicitSpacing[axisIndex],
                            vtkSpacing[axisIndex],
                            rtol=1.0e-10,
                            atol=1.0e-12,
                        )
                ):
                    raise ValueError(
                        f"spacing[{axisIndex}] is inconsistent with "
                        f"coordinate array '{coordName}'."
                    )
                
            vtkSpacing[:dimension] = explicitSpacing[:dimension]

        if ( np.any( np.isclose( vtkSpacing[:dimension], 0.0 ) ) ):
            raise ValueError( "spacing must not contain zero." )

        # ------------------------------------------------- #
        # --- [3] vtkImageData                          --- #
        # ------------------------------------------------- #
        vtkShape = [ 1, 1, 1 ]
        vtkShape[:dimension] = gridShape

        imageData = vtk.vtkImageData()
        imageData.SetDimensions( *vtkShape )
        imageData.SetOrigin( *vtkOrigin )
        imageData.SetSpacing( *vtkSpacing )

        pointData = imageData.GetPointData()
        axisOrder = tuple( range( dimension - 1, -1, -1 ) )

        for dataName in arrayNames:

            vtkValues = np.ascontiguousarray(
                arrayData[dataName].transpose( axisOrder )
            ).reshape( -1 )

            vtkArray = vtknp.numpy_to_vtk(
                num_array=vtkValues,
                deep=True,
            )
            vtkArray.SetName( dataName )

            pointData.AddArray( vtkArray )

        coordDataNames = [
            coordName for coordName in coordNames
            if coordName is not None
        ]
        
        scalarNames = [
            dataName for dataName in arrayNames
            if dataName not in coordDataNames
        ]
        
        activeName = scalarNames[0] if ( len( scalarNames ) > 0 ) \
            else arrayNames[0]
        pointData.SetActiveScalars( activeName )
        
        # ------------------------------------------------- #
        # --- [4] write VTI file                        --- #
        # ------------------------------------------------- #
        fileRoot, fileExt = os.path.splitext( vtiFile )

        if ( fileExt.lower() != ".vti" ):
            vtiFile = f"{fileRoot}.vti"

        outDir = os.path.dirname( vtiFile )

        if ( outDir != "" ):
            os.makedirs( outDir, exist_ok=True )

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName( vtiFile )
        writer.SetInputData( imageData )
        writer.SetDataModeToAppended()

        status = writer.Write()

        if ( status != 1 ):
            raise IOError( "vtkXMLImageDataWriter.Write() failed." )

        print( "[write__vtkImageData]" )
        print( f"  vtiFile    = {vtiFile}" )
        print( f"  dimension  = {dimension}" )
        print( f"  shape      = {gridShape}" )
        print( f"  origin     = {vtkOrigin.tolist()}" )
        print( f"  spacing    = {vtkSpacing.tolist()}" )
        print( f"  pointData  = {arrayNames}" )
        print( "  status     = success" )

        return( imageData )

    except Exception as exc:

        print( "[write__vtkImageData]" )
        print( f"  vtiFile    = {vtiFile}" )
        print( "  status     = failed" )
        print( f"  reason     = {exc}" )

        return( None )


# ========================================================= #
# ===  Execution of Program                             === #
# ========================================================= #
if ( __name__=="__main__" ):

    mode = "vti"
    
    if ( mode == "vti" ):
    
        Nx, Ny, Nz = 21, 21, 101
        
        xa = np.linspace( -10.0, 10.0, Nx )
        ya = np.linspace( -10.0, 10.0, Ny )
        za = np.linspace( -50.0, 50.0, Nz )
        
        xg, yg, zg = np.meshgrid( xa, ya, za, indexing="ij" )
        
        Bx =  1.0e-3 * yg
        By =  1.0e-3 * xg
        Bz = np.zeros_like( xg )
        
        Data = { "x" : xg, "y" : yg, "z" : zg,
                 "Bx": Bx, "By": By, "Bz": Bz }

        imageData = write__vtkImageData( data=Data, vtiFile="out/QMField.vti" )
