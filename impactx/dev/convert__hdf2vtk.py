


# ========================================================= #
# ===  convert__hdf2vtk.py                              === #
# ========================================================= #

def convert__hdf2vtk( hdf5File=None, outFile=None, \
                      pids=None, steps=None, random_choice=None ):

    # ------------------------------------------------- #
    # --- [1] arguments check                       --- #
    # ------------------------------------------------- #
    if ( os.path.exists( hdf5File ) ):
        Data = load__impactHDF5( inpFile=hdf5File, random_choice=random_choice, \
                                 pids=pids, steps=steps )
    else:
        raise FileNotFoundError( f"[ERROR] HDF5 File not Found :: {hdf5File}" )
    if ( outFile is None ):
        raise TypeError( f"[ERROR] outFile == {outFile} ???" )
    else:
        ext = os.path.splitext( outFile )[1]

    # ------------------------------------------------- #
    # --- [2] save as vtk poly data                 --- #
    # ------------------------------------------------- #
    steps = sorted( Data["step"].unique() )

    for ik,step in enumerate(steps):
        # -- points coordinate make -- #
        df     = Data[ Data["step"] == step ]
        df     = df[ np.isfinite(df["xp"]) & \
                     np.isfinite(df["yp"]) & \
                     np.isfinite(df["tp"]) ]
        if ( df.shape[0] == 0 ):
            print( "[impactx_toolkit.py] [WARNING] no appropriate point data :: ik={0}, step={1}".format( ik, step ) )
            continue
        df["dt"] = df["tp"] - df["tp"].mean()
        coords   = df[ ["xp", "yp", "dt"] ].to_numpy()
        cloud    = pv.PolyData( coords )
        # -- momentum & pid -- #
        cloud.point_data["pid"]      = df["pid"].to_numpy()
        cloud.point_data["x"]        = df["xp" ].to_numpy()
        cloud.point_data["y"]        = df["yp" ].to_numpy()
        cloud.point_data["t"]        = df["tp" ].to_numpy()
        cloud.point_data["dt"]       = df["dt" ].to_numpy()
        cloud.point_data["momentum"] = df[ ["px", "py", "pz"] ].to_numpy()
    
        # -- save file -- #
        houtFile = outFile.replace( ext, "-{0:06}".format(ik+1) + ext )
        cloud.save( houtFile )
        print( "[convert__hdf2vtk.py] outFile :: {} ".format( houtFile ) )
    return()
        
