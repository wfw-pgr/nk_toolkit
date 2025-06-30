import os, sys
import pandas as pd

# ========================================================= #
# ===  load__tfs                                        === #
# ========================================================= #
def load__tfs( tfsFile=None, silent=True ):
    # ------------------------------------------------- #
    # --- [1] load file                             --- #
    # ------------------------------------------------- #
    try:
        with open( tfsFile, 'r' ) as f:
            lines = f.readlines()
    except Exception as e:
        print( f"[load__tfs.py]  Error !! :: {e}" )
        sys.exit()
        
    # ------------------------------------------------- #
    # --- [2] parse header contents                 --- #
    # ------------------------------------------------- #
    headers    = []
    datatypes  = []
    data_start = 0
    metadata   = {}
    for i, line in enumerate(lines):
        if line.startswith('@'):
            parts         = line.split()
            key           = parts[1]
            value         = parts[3].strip('"')
            metadata[key] = value
        elif line.startswith('*'):
            headers       = line.split()[1:]
        elif line.startswith('$'):
            datatypes     = line.split()[1:]
        else:
            data_start    = i
            break
    keys = list( metadata.keys() ) + ["data"]
        
    # ------------------------------------------------- #
    # --- [3] load data contents                    --- #
    # ------------------------------------------------- #
    df  = pd.read_csv( tfsFile, sep=r"\s+", skiprows=data_start, names=headers )
    ret = { **metadata, **{ "keys":keys, "df":df } }
    return( ret )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    tfsFile = "out/survey.tfs"
    ret     = load__tfs( tfsFile=tfsFile )
    print( ret )
    
