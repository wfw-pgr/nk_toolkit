import os, sys, io
import h5py
import scipy                        as sp
import numpy                        as np
import pandas                       as pd
import nk_toolkit.plot.load__config as lcf
import nk_toolkit.plot.gplot1D      as gp1
import nk_toolkit.plot.gplot2D      as gp2

# ========================================================= #
# ===  load__opal_t7                                    === #
# ========================================================= #

def load__opal_t7( inpFile=None, structured=True ):

    typelist = [ "2delectrostatic", "2dmagnetostatic" ]
    
    # ------------------------------------------------- #
    # --- [1]  arguments                            --- #
    # ------------------------------------------------- #
    if ( inpFile is None ): sys.exit( "[load__opal_t7.py] inpFile == ???" )

    # ------------------------------------------------- #
    # --- [2] load header                           --- #
    # ------------------------------------------------- #
    with open(inpFile, "r") as f:
        for line in f:
            stripped = line.strip()
            if ( stripped and not( stripped.startswith("#") ) ):
                stripped =  stripped.split()
                ftype    = (stripped[0]).lower()
                break
        if ( not( ftype in typelist ) ):
            sys.exit( "[load__opal_t7] unknown type :: {}".format( ftype ) )
        if ( ftype in [ "2delectrostatic", "2dmagnetostatic" ] ):
            order        = stripped[1].lower()
            nHeaderLines = 2
        header = []
        while( len( header ) < nHeaderLines ):
            line     = next(f)  # 読み進める
            stripped = line.strip()
            if ( stripped and not( stripped.startswith("#") ) ):
                header += [ stripped ]
        if ( ftype in [ "2delectrostatic", "2dmagnetostatic" ] ):
            x1Grid = [ float(val) for val in ( header[0] ).split() ]
            x2Grid = [ float(val) for val in ( header[1] ).split() ]
            if ( order.lower() == "xz" ):
                zGrid, xGrid = x1Grid, x2Grid
            else:
                sys.exit( "not implemented" )
        Data = np.loadtxt( f, comments="#" )

    # ------------------------------------------------- #
    # --- [3] return data                           --- #
    # ------------------------------------------------- #
    if ( structured ):
        shape = ( int(zGrid[2]+1), int(xGrid[2]+1), 2 )
        Data  = np.reshape( Data, shape )
    return( Data )



# ========================================================= #
# ===  save__opal_t7                                    === #
# ========================================================= #

def save__opal_t7( outFile=None, Data=None, type=None, fmt="%15.8e", \
                   xGrid=None, yGrid=None, zGrid=None, \
                   freq=None, order=None, Nfourier=None ):

    #  -- [HowTo] -- #
    #  -- e.g. )  save = save__opal_t7( outFile=outFile, Data=Data, **params )
    #  ------------- #
    
    # ------------------------------------------------- #
    # --- [1] Arguments                             --- #
    # ------------------------------------------------- #
    if ( outFile is None ): sys.exit( "[load__opal_t7] outFile == ???" )
    if ( Data    is None ): sys.exit( "[load__opal_t7] Data    == ???" )
    if ( type    is None ): sys.exit( "[load__opal_t7] type    == ???" )
    
    # ------------------------------------------------- #
    # --- [2] output depending on type              --- #
    # ------------------------------------------------- #
    header = []
    if ( type.lower() in [ "2delectrostatic", "2dmagnetostatic" ] ):   # 2D - ElectroStatic / MagnetoStatic 
        if ( order is None ): order = "XZ"
        Nx, Nz  = xGrid[2]-1, zGrid[2]-1
        header += [ f"{type} {order}"     ]
        header += [ f"{zGrid[0]} {zGrid[1]} {Nz}" ]
        header += [ f"{xGrid[0]} {xGrid[1]} {Nx}" ]
        header  = "\n".join( header )
        np.savetxt( outFile, Data, header=header, comments="",fmt=fmt )
        print( "[opal_toolkit.py] output :: {} ".format( outFile ) )
        
    elif ( type.lower() in [ "2ddynamic"] ):                           # 2D Dynamic
        Ez_,Er_ = 0, 1
        Ea_,Hp_ = 2, 3
        if ( order is None ): order = "XZ"
        Nx, Nz  = xGrid[2]-1, zGrid[2]-1
        header += [ f"{type} {order}" ]
        header += [ f"{zGrid[0]} {zGrid[1]} {Nz}" ]
        header += [ f"{freq}" ]
        header += [ f"{xGrid[0]} {xGrid[1]} {Nx}" ]
        header  = "\n".join( header )
        if ( Data.shape[1] == 3 ):
            Ea   = np.sqrt( Data[:,Ez_]**2 + Data[:,Er_]**2 )
            Data = np.insert( Data, 2, Ea, axis=1 )
        np.savetxt( outFile, Data, header=header, comments="",fmt=fmt )
        print( "[opal_toolkit.py] output :: {} ".format( outFile ) )
    
    else:
        print( "[opal_toolkit.py] unknown type :: {} ".format( type ) )
        sys.exit()
        
    return()



# ========================================================= #
# ===  load__opalStatistics                             === #
# ========================================================= #
def load__statistics( inpFile=None ):

    # ------------------------------------------------- #
    # --- [1] read lines                            --- #
    # ------------------------------------------------- #
    if ( inpFile is None ): sys.exit( "[opal_toolkit.py] inpFile == ???" )
    with open( inpFile, 'r') as f:
        lines = f.readlines()

    # ------------------------------------------------- #
    # --- [2] column name                           --- #
    # ------------------------------------------------- #
    columns = []
    for i, line in enumerate(lines):
        if line.strip().startswith("&column"):
            for j in range(i, len(lines)):
                if "name=" in lines[j]:
                    name_line = lines[j].strip()
                    name = name_line.split("=", 1)[1].strip().strip('",')
                    columns.append(name)
                if lines[j].strip() == "&end":
                    break

    # ------------------------------------------------- #
    # --- [3] number of parameters                  --- #
    # ------------------------------------------------- #
    num_parameters = sum(1 for line in lines if line.strip().startswith("&parameter"))

    # ------------------------------------------------- #
    # --- [4] search &data section                  --- #
    # ------------------------------------------------- #
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("&data"):
            for j in range(i, len(lines)):
                if lines[j].strip() == "&end":
                    data_start = j + 1
                    break
            break

    # ------------------------------------------------- #
    # --- [5] skip parameters                       --- #
    # ------------------------------------------------- #
    data_lines = lines[data_start + num_parameters:]

    # ------------------------------------------------- #
    # --- [6] read data                             --- #
    # ------------------------------------------------- #
    data = pd.read_csv( io.StringIO("".join( data_lines) ), sep=r"\s+", names=columns )
    return( data )




# ========================================================= #
# ===  TM010 mode electric field ( Ez, Er, Hp )         === #
# ========================================================= #
def ef__TM010( Lcav=None, Rcav=None, Nz=11, Nr=11, E0=1.0e6, freq=None, \
               outFile="ef__TM010.T7", pngFile=None ):

    z_,r_   = 0, 1
    ez_,er_ = 0, 1
    ea_,hp_ = 2, 3
    x01     = 2.405            # J0 の1つ目のゼロ
    m2cm    = 1.0e2
    MHz     = 1.0e6            # unit : [MHz]
    mu0     = 4.0*np.pi*1.0e-7 # permeability
    MV      = 1.0e6            # unit : [MV/m]
    
    # ------------------------------------------------- #
    # --- [1] grid make    ( calc in MKS )          --- #
    # ------------------------------------------------- #
    omega   = 2.0*np.pi * freq * MHz
    kc01    = x01 / Rcav
    import nkUtilities.equiSpaceGrid as esg
    zGrid   = [ 0.0, Lcav, Nz ]
    xGrid   = [ 0.0, Rcav, Nr ]
    coord   = esg.equiSpaceGrid( x1MinMaxNum=zGrid, x2MinMaxNum=xGrid, returnType="point" )
    Er      = 0.0 * coord[:,r_]
    Ez      =                  E0 * sp.special.j0( kc01*coord[:,r_] )  # 時刻 t = 0 のEz
    Bp      = ( kc01/omega ) * E0 * sp.special.j1( kc01*coord[:,r_] )
    Hp      = Bp / mu0
    Ea      = np.sqrt( Er**2 + Ez**2 )
    Data    = np.concatenate( [ Ez[:,np.newaxis]/MV, Er[:,np.newaxis]/MV,\
                                Ea[:,np.newaxis]/MV, Hp[:,np.newaxis] ], axis=1 )
    # -- re-define grid in cm for output -- #
    zGrid   = [ 0.0, Lcav*m2cm, Nz ]
    xGrid   = [ 0.0, Rcav*m2cm, Nr ]
    coord   = esg.equiSpaceGrid( x1MinMaxNum=zGrid, x2MinMaxNum=xGrid, returnType="point" )
    
    # ------------------------------------------------- #
    # --- [2] return                                --- #
    # ------------------------------------------------- #
    if ( outFile is not None ):
        type = "2DDynamic"
        ret  = save__opal_t7( outFile=outFile, Data=Data, freq=freq, \
                              xGrid=xGrid, zGrid=zGrid, type=type )
    if ( pngFile is not None ):
        config   = lcf.load__config()
        ext      = os.path.splitext( pngFile )[1]
        config_  = {
            "figure.size"        : ( 6, 6 ), 
            "figure.position"    : [ 0.16, 0.16, 0.86, 0.86 ], 
            "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":6 },
            "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":6 },
            "ax1.x.label"        : "Z (cm)",
            "ax1.y.label"        : "R (cm)",
            "ax1.x.minor.nticks" : 1,
            "cmp.level"          : { "auto":True, "min": 0.0, "max":1.0, "num":100 },
            "cmp.colortable"     : "jet",
        }
        config   = { **config, **config_ }
        pngFile_ = pngFile.replace( ext, "_Ez"+ext )
        fig      = gp2.gplot2D( xAxis=coord[:,z_], yAxis=coord[:,r_], cMap=Data[:,ez_], \
        	 	        config=config, pngFile=pngFile_ )
        pngFile_ = pngFile.replace( ext, "_Er"+ext )
        fig      = gp2.gplot2D( xAxis=coord[:,z_], yAxis=coord[:,r_], cMap=Data[:,er_], \
        	 	        config=config, pngFile=pngFile_ )
        pngFile_ = pngFile.replace( ext, "_Hp"+ext )
        fig      = gp2.gplot2D( xAxis=coord[:,z_], yAxis=coord[:,r_], cMap=Data[:,hp_], \
        	 	        config=config, pngFile=pngFile_ )
    return( Data )



# ========================================================= #
# === load__opalHDF5.py                                 === #
# ========================================================= #

def load__opalHDF5( inpFile=None, steps=None, series=None ):
    
    # ------------------------------------------------- #
    # --- [1] load HDF5 file                        --- #
    # ------------------------------------------------- #
    stack = {}
    with h5py.File( inpFile, "r" ) as f:
        if ( steps is not None ):
            steps = sorted( steps )
            keys  = [ "Step#{}".format( step ) for step in steps ]
        else:
            steps = sorted( [ int( key.replace( "Step#", "" ) ) for key in f.keys() ] )
            keys  = [ "Step#{}".format( step ) for step in steps ]
            
        for key in keys:
            df         = pd.DataFrame( { ikey:f[key][ikey] for ikey in f[key].keys() } )
            stack[key] = df.sort_values( by="id" ).reset_index(drop=True)
            
        if ( series is not None ):
            stack = [ stack[key] for key in keys ]
            stack = pd.concat( stack, axis=0, ignore_index=True )
            stack = stack[ stack["id"].isin( series ) ]

    # ------------------------------------------------- #
    # --- [2] return step                           --- #
    # ------------------------------------------------- #
    return( stack )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1] calculate ef__TM010                   --- #
    # ------------------------------------------------- #
    print( " -- calculate ef__TM010 --" )
    ef = ef__TM010( Lcav=0.5, Rcav=0.1, Nz=11, Nr=11, outFile="test/ef__TM010.T7", \
                    pngFile="test/ef__TM010.png", freq=36.5 )

    # ------------------------------------------------- #
    # --- [2] save opal t7                          --- #
    # ------------------------------------------------- #
    print( " -- save opal t7 -- " )
    type    = "2DElectroStatic"
    outFile = "test/ef__sample.T7"
    xGrid   = [ 0.0, 1.0, 6 ]
    zGrid   = [ -1.0, 1.0, 11 ]
    import nkUtilities.equiSpaceGrid as esg
    coord   = esg.equiSpaceGrid( x1MinMaxNum=zGrid, x2MinMaxNum=xGrid, \
                                 returnType = "point" )
    Ez      = coord[:,0]
    Ex      = coord[:,1]
    Data    = np.concatenate( [ Ez[:,np.newaxis], Ex[:,np.newaxis] ], axis=1 )
    params  = { "zGrid":zGrid, "xGrid":xGrid }
    save__opal_t7( outFile=outFile, Data=Data, type="2DElectroStatic", **params )

    # ------------------------------------------------- #
    # --- [3] load opal t7                          --- #
    # ------------------------------------------------- #
    print( " -- load opal t7 -- " )
    ret     = load__opal_t7( inpFile="test/ef__sample.T7" )
    print( ret )
    
    # ------------------------------------------------- #
    # --- [4] load sddds ( sample.stat )            --- #
    # ------------------------------------------------- #
    print( " -- load statistics -- " )
    inpFile = "test/main.stat"
    Data    = load__statistics( inpFile=inpFile )
    print( Data )
    
    # ------------------------------------------------- #
    # --- [5] load opal particle info               --- #
    # ------------------------------------------------- #
    inpFile = "test/main.h5"
    series  = [ ik for ik in range(300) ]
    ret     = load__opalHDF5( inpFile=inpFile, series=series )

    
