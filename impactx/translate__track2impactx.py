import os, sys, json5, re
import numpy          as np
import pandas         as pd

# ========================================================= #
# ===  translate__track2impactx.py                      === #
# ========================================================= #

def translate__track2impactx( paramsFile="dat/parameters.json" ):
    
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
    #  -- [1-3] preparation                         --  #
    params["rf.freq"]   = params["beam.freq"] * params["rf.harmonics"]
        
    # ------------------------------------------------- #
    # --- [2] analyze each line                     --- #
    # ------------------------------------------------- #
    seq    = []
    counts = { "at":0.0, "N":0, "Nqm":0, "Nrf":0, "Ndr":0, }
    for ik,line in enumerate(lines):
        line_ = re.sub( r"#.*", "", line ).strip()
        words = ( line_.split() )
        if ( not( len( words ) == 0 ) ):
            counts["N"]   += 1
            
            if   ( words[1].lower()  == "quad" ):
                if not( params["translate.skip.qm"] ):
                    seq += convert_to_quad ( words, counts, params )

            elif ( words[1].lower()  == "drift" ):
                if not( params["translate.skip.dr"] ):
                    seq += convert_to_drift( words, counts, params )
                
            elif ( words[1].lower()  == "rfgap" ):
                if not( params["translate.skip.rf"] ):
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
    contents  = "import impactx" + "\n\n"
    contents += "ns = {}\n".format( params["translate.nslice"] )
    base_f    = "{0:<8} = impactx.elements."
    if ( params["translate.trackmode"].lower() in ["nonlinear"] ):
        drift_f   = base_f + 'ExactDrift( name="{0}", ds={1:.8}, aperture_x={2:.8}, aperture_y={2:.8}, nslice=ns )\n'
        quadr_f   = base_f + 'ExactQuad ( name="{0}", ds={1:.8}, unit=1, k={2:.8}, aperture_x={3:.8}, aperture_y={3:.8}, nslice=ns )\n'
        rfcav_f   = base_f + 'RFCavity  ( name="{0}", ds={1:.8}, escale={2:.8}, freq={3:.8}, phase={4:.8}, cos_coefficients={5}, sin_coefficients={6}, aperture_x={7:.8}, aperture_y={7:.8}, nslice=ns )\n'
    else:
        drift_f   = base_f + 'Drift   ( name="{0}", ds={1:.8}, aperture_x={2:.8}, aperture_y={2:.8}, nslice=ns )\n'
        quadr_f   = base_f + 'Quad    ( name="{0}", ds={1:.8}, k={2:.8}, aperture_x={3:.8}, aperture_y={3:.8}, nslice=ns )\n'
        rfcav_f   = base_f + 'RFCavity( name="{0}", ds={1:.8}, escale={2:.8}, freq={3:.8}, phase={4:.8}, cos_coefficients={5}, sin_coefficients={6}, aperture_x={7:.8}, aperture_y={7:.8}, nslice=ns )\n'

    if ( params["translate.add_monitor"] ):
        contents  += ( base_f + 'BeamMonitor( "{0}", backend="h5")\n' ).format( "bpm" )
    
    for ik,elem in enumerate(seq):
        
        if   ( elem["type"].lower() == "drift"      ):
            keys      = [ "tag", "L", "aperture" ]
            contents += drift_f.format( *( [ elem[key] for key in keys ] ) )
            
        elif ( elem["type"].lower() == "quadrupole" ):
            keys      = [ "tag", "L", "K1", "aperture" ]
            contents += quadr_f.format( *( [ elem[key] for key in keys ] ) )
            
        elif ( elem["type"].lower() == "rfcavity"   ):
            keys      = [ "tag", "L", "escale", "freq", "phase", \
                          "cos_coeff", "sin_coeff", "aperture"]
            contents += rfcav_f.format( *( [ elem[key] for key in keys ] ) )
            
        else:
            print( "[translate__track2impactx.py] unknown keywords :: {} ".format( elem["type"] ) )
            sys.exit()

    # ------------------------------------------------- #
    # --- [5] beam line definition                  --- #
    # ------------------------------------------------- #
    if ( params["translate.add_monitor"] ):
        elements = ["bpm"]
        for elem in seq:
            if ( elem["type"].lower() in [ "quadrupole", "rfcavity" ] ):
                elements += [ elem["tag"], "bpm" ]
            else:
                elements += [ elem["tag"] ]
    else:
        elements = [ elem["tag"] for elem in seq ]
        
    contents += "beamline = [ " + ",".join( elements ) + " ]\n"

    # ------------------------------------------------- #
    # --- [6] write in a file                       --- #
    # ------------------------------------------------- #
    with open( params["file.sequence"], "w" ) as f:
        f.write( contents )
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
    if ( params["translate.nonlinear"] ):
        K1         = gradB
    else:
        Brho       = 1.3
        K1         = gradB / Brho
    if ( params["translate.aperture"] is not None ):
        aperture   = params["translate.aperture"]
    else:
        aperture   = Ra
    ret            = [ { "type":"quadrupole", "tag":tag,
                         "K1":K1, "L":L, "at":counts["at"], "aperture":aperture } ]
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
    Ra             = float(words[3]) * cm
    if ( params["translate.aperture"] is not None ):
        aperture   = params["translate.aperture"]
    else:
        aperture   = Ra
    
    ret            = [ { "type":"drift", "tag":tag, "L":L, \
                         "aperture":aperture, "at":counts["at"] } ]
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
    volt      = float(words[2])                         # volt :    [MV]
    phase     = float(words[3])                         #           [deg]
    harmonics =   int(words[4])                         # harmonics
    Rcav      =   int(words[5]) * cm                    # R-cavity  [cm]
    freq      = params["beam.freq"] * harmonics * MHz   # -- freq = fbeam * harmon
    if ( params["translate.aperture"] is not None ):
        aperture   = params["translate.aperture"]
    else:
        aperture   = Rcav
    escale    = volt / Lcav / params["beam.mass"]
    cos_coeff = [ 1.0, ]
    sin_coeff = [ 0.0, ]
    cos_coeff = "[{}]".format( ",".join( [ str(val) for val in cos_coeff ] ) )
    sin_coeff = "[{}]".format( ",".join( [ str(val) for val in sin_coeff ] ) )
    ret       = [ { "type":"rfcavity", "tag":tag, "L":L, "Rcav":Rcav, "escale":escale, \
                    "freq":freq, "phase":phase, "hormonics":harmonics, "cos_coeff":cos_coeff, "sin_coeff":sin_coeff, \
                    "at":counts["at"], "aperture":aperture } ]
    counts["Nrf"] += 1
    counts["at"]  += L
    return( ret )

    
            
# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    translate__track2impactx()


    
