import os, sys, json5, re
import numpy          as np
import pandas         as pd
import nk_toolkit.opal.opal_toolkit as opk

# ========================================================= #
# ===  translate__track2opal.py                         === #
# ========================================================= #

def translate__track2opal( paramsFile="dat/parameters.json" ):
    
    cm       = 1.e-2
    MeV      = 1.e+6
    gauss    = 1.e-4           # [T]
    
    # ------------------------------------------------- #
    # --- [1] read track file                       --- #
    # ------------------------------------------------- #
    #  -- [1-1] load parameters                     --  #
    with open( paramsFile, "r" ) as f:
        params = json5.load( f )
    #  -- [1-2] load track-v38's results            --  #
    with open( params["file.sclinac"], "r" ) as f:
        lines  = f.readlines()
    
    # ------------------------------------------------- #
    # --- [2] analyze each line                     --- #
    # ------------------------------------------------- #
    seq    = []
    counts = { "at":0.0, "N":0, "Nqm":0, "Nrf":0, "Ndr":0, "Nps":0, "fmapfnList":[] }
    for ik,line in enumerate(lines):
        words = ( line.split() )
        if ( not( len( words ) == 0 ) ):
            counts["N"]   += 1
            
            if   ( words[1].lower()  == "quad" ):
                seq += convert_to_quad ( words, counts, params )

            elif ( words[1].lower()  == "drift" ):
                seq += convert_to_drift( words, counts, params )
                
            elif ( words[1].lower()  == "rfgap" ):
                seq += convert_to_rfcavity( words, counts, params )
            
            else:
                sys.exit( "[ERROR] undefined keyword :: {} ".format( words[1] ) )

    L_tot     = counts["at"]
    Nelements = counts["N"]
    print( f" -- all of the {Nelements} elements were loaded... " )
    print( f" --    total length of the beam line == {L_tot:.8} " )
    print()

    # ------------------------------------------------- #
    # --- [3] modify rfcavity thickness             --- #
    # ------------------------------------------------- #
    if ( params["translate.rf.thickness"] ):
        at   = 0.0
        Lcav = params["translate.rf.thickness"]
        
        # ------------------------------------------------- #
        # --- [3-1] modify length   - DR - RF - DR -    --- #
        # ------------------------------------------------- #
        for ik in range( len(seq) ):
            
            if ( seq[ik]["type"].lower() == "rfcavity" ):
                seq[ik]["L"] = Lcav
                
                # -- check previous drift &  L_dr = L_dr - L_cav/2
                if ( ik != 0 ):
                    if ( seq[ik-1]["type"].lower()=="drift" ):
                        seq[ik-1]["L"] = seq[ik-1]["L"] - 0.5*Lcav
                    else:
                        print( "[CAUTION] {0} is rfcavity, but {1} is not drift... [CAUTION]"\
                               .format( seq[ik]["tag"], seq[ik-1]["tag"] ) )
                        sys.exit()
                        
                # -- check next drift     &  L_dr = L_dr - L_cav/2
                if ( ik != len(seq) ):
                    if ( seq[ik+1]["type"].lower()=="drift" ):
                        seq[ik+1]["L"] = seq[ik+1]["L"] - 0.5*Lcav
                    else:
                        print( "[CAUTION] {0} is rfcavity, but {1} is not drift... [CAUTION]"\
                               .format( seq[ik]["tag"], seq[ik+1]["tag"] ) )
                        sys.exit()

        # ------------------------------------------------- #
        # --- [3-2] sum up "at" position again          --- #
        # ------------------------------------------------- #
        for ik,elem in enumerate( seq ):
            elem["at"]  = at
            at         += elem["L"]
            
    # ------------------------------------------------- #
    # --- [4] convert into mad-x sequence           --- #
    # ------------------------------------------------- #
    contents  = "// -- [2] beam line components   -- //;\n"
    drift_f   = "{0}: drift, L={1:.8}, elemedge={2:.8};\n"
    quadr_f   = "{0}: quadrupole, L={1:.8}, K1={2:.8}, elemedge={3:.8};\n"
    rfcav_f   = '{0}: rfcavity, L={1:.8}, volt={2:.8}, lag={3:.8}, fmapfn="{4}", ' \
        + "elemedge={5:.8};\n"
    
    for ik,elem in enumerate(seq):
        
        if   ( elem["type"].lower() == "drift"      ):
            keys      = [ "tag", "L", "at" ]
            contents += drift_f.format( *( [ elem[key] for key in keys ] ) )
            
        elif ( elem["type"].lower() == "quadrupole" ):
            keys      = [ "tag", "L", "K1", "at" ]
            contents += quadr_f.format( *( [ elem[key] for key in keys ] ) )
            
        elif ( elem["type"].lower() == "rfcavity"   ):
            keys      = [ "tag", "L", "volt", "lag", "fmapfn", "at" ]
            contents += rfcav_f.format( *( [ elem[key] for key in keys ] ) )
            
        else:
            print( "[translate__track2opal.py] unknown keywords :: {} ".format( elem["type"] ) )
            sys.exit()

    elements  = [ elem["tag"] for elem in seq ]
    contents += "l1: line=(" + ",".join( elements ) + ");\n"
            
    with open( params["file.sequence"], "w" ) as f:
        f.write( contents )


    # ------------------------------------------------- #
    # --- [4] generate fmap                         --- #
    # ------------------------------------------------- #
    fmapfnList = list( set( counts["fmapfnList"] ) )
    pattern    = r"Rcav([\d\.]+)_Lcav([\d\.]+)"
    for fmapfn in fmapfnList:
        print( fmapfn )
        text   = os.path.splitext( os.path.basename( fmapfn ) )[0]
        search = re.search( pattern, text )
        if ( search ):
            Rcav = float( search.group(1) )
            Lcav = float( search.group(2) )
        else:
            print( "[translate__track2opal.py] cannnot recognize (Rcav, Lcav) ... [ERROR] " )
            print( "   text   :: {}".format( text    ) )
            print( "   saerch :: {}".format( pattern ) )
            sys.exit()
        opk.ef__TM010( outFile=fmapfn, Lcav=Lcav, Rcav=Rcav, \
                       Nz=params["translate.rf.Nz"], Nr=params["translate.rf.Nr"] )
            
    return()
        
# ========================================================= #
# ===  convert_to_quad                                  === #
# ========================================================= #
def convert_to_quad( words, counts, params ):
    
    MeV            = 1.0e+6
    cm             = 1.0e-2
    gauss          = 1.0e-4
    
    tag            = "qm{}".format( (counts["Nqm"]+1) )
    Bq             = float(words[2]) * gauss
    Ra             = float(words[5]) * cm
    L              = float(words[4]) * cm
    gradB          = Bq / Ra
    K1             = gradB 
    ret            = [ { "type":"quadrupole", "tag":tag,
                         "K1":K1, "L":L, "at":counts["at"] } ]
    counts["Nqm"] += 1
    counts["at"]  += L
    return( ret )


# ========================================================= #
# ===  convert_to_drift                                 === #
# ========================================================= #
def convert_to_drift( words, counts, params ):

    cm             = 1.0e-2

    tag            = "dr{}".format( (counts["Ndr"]+1) )
    L              = float(words[2]) * cm    
    ret            = [ { "type":"drift", "tag":tag, "L":L, "at":counts["at"] } ]
    counts["Ndr"] += 1
    counts["at"]  += L
    return( ret )


# ========================================================= #
# ===  convert_to_rfcavity                              === #
# ========================================================= #
def convert_to_rfcavity( words, counts, params ):

    MHz       = 1.0e+6
    cm        = 1.0e-2
    tag       = "rf{}".format( (counts["Nrf"]+1) )
    
    L         = 0.0
    Lcav      = params["translate.rf.thickness"]
    volt      = float(words[2])                    # volt :    [MV]
    lag       = float(words[3])/180.0 *np.pi       #           [deg]
    harmonics =   int(words[4])                    # harmonics
    Rcav      =   int(words[5]) * cm               # R-cavity  [cm]
    freq      = params["freq_0"] * harmonics * MHz # -- for LINAC, freq = f0 * harmon,  harmon=1
    fmapfn    = os.path.join( params["file.fmap.dir"], \
                              "TM010__Rcav{0}_Lcav{1}.T7".format( Rcav, Lcav ) )
    fmapfn_   = re.sub( r"^opal/", "", fmapfn )
    ret       = [ { "type":"rfcavity", "tag":tag, "L":L, "Rcav":Rcav, "volt":volt, "freq":freq, \
                    "lag":lag, "hormonics":harmonics, "fmapfn":fmapfn_, "at":counts["at"], } ]
    counts["Nrf"] += 1
    counts["at"]  += L
    counts["fmapfnList"] += [ fmapfn ]
    return( ret )

    
            
# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    translate__track2opal()


    
