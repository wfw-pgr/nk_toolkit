import numpy as np
import pandas as pd
import math


# ========================================================= #
# ===  generate__beamDrivenNeutronSource.py             === #
# ========================================================= #
def generate__beamDrivenNeutronSource( inpFile="dat/angle_energy_vs_neutrons.dat", surface=50 ):

    s_section = "[source]"
    s_format  = """
  <source>={weight}
    s-type=26
    suf={surface}
    proj=neutron
    dir=data
    a-type=11
    na=2
    {al} 1.0
    {ah} 1.0
    e-type=22
{values}
"""

    # .dat sample

    """
    al ah    el           eh      n[count/sr/MeV] r.err.
    0 10    1.0000E-10   1.0000E-09   0.0000E+00  0.0000
    0 10    1.0000E-09   1.0000E-08   0.0000E+00  0.0000
    0 10    1.0000E-08   1.0000E-07   2.6135E+14  0.7071
    ...... 
    """

    # ========================================================= #
    # ===  round__digit                                     === #
    # ========================================================= #
    def round__digit( x, digit=3 ):
        if ( ( x == 0 ) or ( x is None ) ):
            ret = 0
        else:
            ret = round( x, digit - int( math.floor( math.log10( abs( x ) ) ) ) - 1 )
        return( ret )

    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    data = pd.read_csv( inpFile, sep=r"\s+" )
    for col in [ "al", "ah", "el", "eh", "n[count/sr/MeV]" ]:
        data[col] = pd.to_numeric( data[col], errors="coerce" )

    # ------------------------------------------------- #
    # --- [2] bin yield                             --- #
    # ------------------------------------------------- #
    data["dE"]       = data["eh"] - data["el"]
    data["sr"]       = 2.0*np.pi*( np.cos(np.deg2rad(data["al"]) ) - \
                                   np.cos(np.deg2rad(data["ah"])) )
    data["n[count]"] = data["n[count/sr/MeV]"] * data["dE"] * data["sr"]

    # ------------------------------------------------- #
    # --- [3] group by angle                        --- #
    # ------------------------------------------------- #
    angleGroups = []
    for ( al, ah ), grp in data.groupby( [ "al", "ah" ], sort=True ):
        grp          = grp.sort_values( by=[ "el", "eh" ] ).copy()
        integ        = grp["n[count]"].sum()
        grp["eProb"] = 0.0 if ( integ <= 0.0 ) else grp["n[count]"] / integ

        angleGroups.append(
            {
                "al"    : float( al )   ,
                "ah"    : float( ah )   ,
                "yield" : float( integ ),
                "data"  : grp.copy(),
            }
        )
    totalYield = sum( [ ag["yield"] for ag in angleGroups ] )

    # ------------------------------------------------- #
    # --- [4] make each <source>                    --- #
    # ------------------------------------------------- #
    sourceBlocks = []
    for ag in angleGroups:
        weight = 0.0 if totalYield <= 0.0 else ag["yield"] / totalYield
        
        eLines = []
        for _, row in ag["data"].iterrows():
            if ( row["eProb"] <= 0.0 ):
                continue
            eLines.append( "    {0} {1} {2}".format( round__digit( row["el"]    ),
                                                     round__digit( row["eh"]    ),
                                                     round__digit( row['eProb'], digit=8 ) ) )
        values = "\n".join( eLines )

        sourceBlocks.append(
            s_format.format(
                weight  = round__digit( weight, digit=8 ), 
                surface = surface, 
                al      = round__digit( ag["al"] ), 
                ah      = round__digit( ag["ah"] ),
                values  = values,
            )
        )

    # ------------------------------------------------- #
    # --- [5] return                                 -- #
    # ------------------------------------------------- #
    text = s_section + "\n" + "".join( sourceBlocks )
    return( text )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    ret     = generate__beamDrivenNeutronSource()
    outFile = "dat/source_phits.inp"
    
    with open( outFile, "w" ) as f:
        f.write( ret )
