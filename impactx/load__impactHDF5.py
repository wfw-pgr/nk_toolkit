import pandas as pd
import numpy  as np
import h5py


# ========================================================= #
# === load__impactHDF5.py                               === #
# ========================================================= #

def load__impactHDF5( inpFile=None, ):

    # ------------------------------------------------- #
    # --- [1] load HDF5 file                        --- #
    # ------------------------------------------------- #
    stack = {}
    with h5py.File( inpFile, "r" ) as f:
        steps = sorted( [ int( key ) for key in f["data"].keys() ] )
        for step in steps:
            key, df    = str(step), {}
            df["pid"]  = f["data"][key]["particles"]["beam"]["id"][:]
            df["xp"]   = f["data"][key]["particles"]["beam"]["position"]["x"][:]
            df["yp"]   = f["data"][key]["particles"]["beam"]["position"]["y"][:]
            df["tp"]   = f["data"][key]["particles"]["beam"]["position"]["t"][:]
            df["px"]   = f["data"][key]["particles"]["beam"]["momentum"]["x"][:]
            df["py"]   = f["data"][key]["particles"]["beam"]["momentum"]["y"][:]
            df["pz"]   = f["data"][key]["particles"]["beam"]["momentum"]["t"][:]
            df["step"] = np.ones( ( df["pid"].shape[0], ) ) * step
            stack[key] = pd.DataFrame( df )
    ret = pd.concat( stack, ignore_index=True )

    # ------------------------------------------------- #
    # --- [2] return step                           --- #
    # ------------------------------------------------- #
    return( ret )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    inpFile = "impactx/diags/openPMD/bpm.h5"
    Data    = load__impactHDF5( inpFile=inpFile )
    print( Data )
