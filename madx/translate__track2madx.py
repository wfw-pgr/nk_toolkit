import os, sys, json5
import numpy          as np
import pandas         as pd

# ========================================================= #
# ===  translate__track2madx.py                         === #
# ========================================================= #

def translate__track2madx( paramsFile="dat/parameters.json" ):
    
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
    with open( params["file.beam"], "r" ) as f:
        energy = pd.read_csv( f, header=0, sep=r"\s+" )
    #  -- [1-3] parameters                          --  #
    E0 = params["mass/u"] * params["umass"]
    
    # ------------------------------------------------- #
    # --- [2] analyze each line                     --- #
    # ------------------------------------------------- #
    seq    = []
    counts = { "at":0.0, "N":0, "Nqm":0, "Nrf":0, "Ndr":0 }
    for ik,line in enumerate(lines):
        words = ( line.split() )
        if ( not( len( words ) == 0 ) ):
            counts["N"] += 1
            Ek = energy["Energy[MeV/u]"][ counts["N"] ] * params["Nu"]      #  -- [MeV]
            
            if   ( words[1].lower()  == "quad" ):
                seq += [ convert_to_quad ( words, counts, params, Ek, E0 ) ]

            elif ( words[1].lower()  == "drift" ):
                seq += [ convert_to_drift( words, counts, params ) ]
                
            elif ( words[1].lower()  == "rfgap" ):
                seq += [ convert_to_rfgap( words, counts, params, Ek, E0 ) ]
            
            else:
                sys.exit( "[ERROR] undefined keyword :: {} ".format( words[1] ) )

    L_tot     = counts["at"]
    Nelements = counts["N"]
    print( f" -- all of the {Nelements} elements were loaded... " )
    print( f" --    total length of the beam line == {L_tot:.8} " )
    print()
                
    # ------------------------------------------------- #
    # --- [3] convert into mad-x sequence           --- #
    # ------------------------------------------------- #
    contents  = ""
    contents += "{0}: sequence, L={1:.8}, refer=entry;\n"\
        .format( params["translate.seqLabel"], L_tot )
    quadr_f   = "  {0}: quadrupole, L={1:.8}, K1={2:.8}, at={3:.8};\n"
    rfcav_f   = "  {0}: RFCavity, L={1:.8}, volt={2:.8}, lag={3:.8}, "\
        + "freq={4:.4}, harmon={5}, n_bessel={6}, at={7:.8};\n"
    rfgap_f   = "  {0}: matrix, L={1:.8}, rm11={2:.8}, rm21={3:.8}, rm22={4:.8}, rm33={5:.8}, "\
        + "rm43={6:.8}, rm44={7:.8}, rm55={8:.8}, rm65={9:.8}, rm66={10:.8}, kick6={11:.8}, at={12:.8};\n"
    drift_f   = "  {0}: drift, L={1:.8}, at={2:.8};\n"
    
    for ik,elem in enumerate(seq):
        
        if   ( elem["type"].lower() == "quadrupole" ):
            keys      = [ "tag", "L", "K1", "at" ]
            contents += quadr_f.format( *( [ elem[key] for key in keys ] ) )
            
        elif ( elem["type"].lower() == "rfcavity"   ):
            keys      = [ "tag", "L", "volt", "lag", "freq", "harmon", "n_bessel", "at" ]
            contents += rfcav_f.format( *( [ elem[key] for key in keys ] ) )

        elif ( elem["type"].lower() == "rfgap"      ):
            keys      = [ "tag", "L", "rm11", "rm21", "rm22", "rm33", \
                          "rm43", "rm44", "rm55", "rm65", "rm66", "kick6", "at" ]
            contents += rfgap_f.format( *( [ elem[key] for key in keys ] ) )
            
        elif ( elem["type"].lower() == "drift"   ):
            keys      = [ "tag", "L", "at" ]
            contents += drift_f.format( *( [ elem[key] for key in keys ] ) )
            
        else:
            print( "[translate__track2madx.py] unknown keywords :: {} ".format( elem["type"] ) )
            sys.exit()
            
    contents += "endsequence;"
    with open( params["file.sequence"], "w" ) as f:
        f.write( contents )

    # ------------------------------------------------- #
    # --- [4] set observe points                    --- #
    # ------------------------------------------------- #
    contents = ""
    if ( params["ptc.sw"] ):
        cmd = "ptc_observe"
    else:
        cmd = "observe"
    for ik,elem in enumerate(seq):
        contents += "{0}, place={1};\n".format( cmd, elem["tag"] )
    with open( params["file.observe"], "w" ) as f:
        f.write( contents )




        
# ========================================================= #
# ===  convert_to_quad                                  === #
# ========================================================= #
def convert_to_quad( words, counts, params, Ek, E0 ):
    
    MeV            = 1.0e+6
    cm             = 1.0e-2
    gauss          = 1.0e-4
    
    tag            = "qm{}".format( (counts["Nqm"]+1) )
    pc             = np.sqrt( Ek**2 + 2.0*Ek*E0 ) #  -- [MeV] = c*p
    BRho           = pc * MeV / params["cv"]      #  -- [Tm]  = p/q = pc*qe/(cv*qe)
    Bq             = float(words[2]) * gauss
    Ra             = float(words[5]) * cm
    L              = float(words[4]) * cm
    gradB          = Bq / Ra                      #  -- [T/m]
    K1             = gradB / BRho                 #  -- [1/m2]
    ret            = { "type":"quadrupole", "tag":tag,
                       "K1":K1, "L":L, "at":counts["at"] }
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
    ret            = { "type":"drift", "tag":tag, "L":L, "at":counts["at"] }
    counts["Ndr"] += 1
    counts["at"]  += L
    return( ret )



# ========================================================= #
# ===  convert_to_rfgap                                 === #
# ========================================================= #
def convert_to_rfgap( words, counts, params, Ek, E0 ):

    MeV            = 1.0e+6
    MHz            = 1.0e+6
    cm             = 1.0e-2
    gauss          = 1.0e-4
    
    tag            = "rf{}".format( (counts["Nrf"]+1) )
    
    if   ( params["translate.RFmode"] == "rfcavity" ):
        L        = 0.0 * cm
        volt     = float(words[2])                 # volt : (MV)
        lag      = float(words[3]) / 360.0 * params["translate.lagSign"]
        harmon   =   int(words[4])
        freq     = params["freq_0"] * harmon       # -- for LINAC, freq = f0 * harmon,  harmon=1
        harmon   = 1                               # -- harmon is for ring. see MAD-X's manual.
        n_bessel = params["translate.n_bessel"]
        ret      = { "type":"RFcavity", "tag":tag, "L":L, "volt":volt, "lag":lag, \
                     "freq":freq, "harmon":harmon, "n_bessel":n_bessel, "at":counts["at"] }
        
    elif ( params["translate.RFmode"] == "rfgap" ):
        L        = 0.0 * cm
        Vg       = float(words[2])                 # volt : (MV)
        phi      = float(words[3]) * params["translate.lagSign"]                # [deg]
        harmon   =   int(words[4])
        freq     = params["freq_0"] * harmon * MHz # -- for LINAC, freq = f0 * harmon,  harmon=1
        rm       = rfgap( Vg=Vg, phi=phi, Ek=Ek, mass=E0, charge=1.0, freq=freq )
        ret      = { "type":"rfgap", "tag":tag, "L":L, "at":counts["at"] }
        ret      = { **ret, **rm }

    else:
        print( "[translate__trac2madx.py] unknown translate.RFmode :: {} "\
               .format( params["translate.RFmode"] ) )
        sys.exit()
        
    counts["Nrf"] += 1
    counts["at"]  += L
    return( ret )


    
# ========================================================= #
# ===  rfgap model of the trace3D code                  === #
# ========================================================= #

def rfgap( Vg     = 1.0,       # [MV]
           phi    = -45.0,     # [deg]
           Ek     = 0.0,       # [MeV]
           mass   = 931.494,   # [MeV]
           charge = 1.0,       # [C/e]
           freq   = 146.0e6 ): # [Hz]
    
    cv      = 2.9979e8         # (m/s)
    qa      = abs( charge )    # (C)
    
    # ------------------------------------------------- #
    # --- [1] calculation                           --- #
    # ------------------------------------------------- #
    phis    =  phi/180.0*np.pi
    lamb    =  cv / freq
    Wi      =  Ek
    dW      =  qa * Vg * np.cos(phis)
    Wm      =  Wi + 0.5*dW
    Wf      =  Wi +     dW
    Weff    =  qa * Vg * np.sin(phis)
    gamma_i = 1.0 + Wi/mass
    gamma_m = 1.0 + Wm/mass
    gamma_f = 1.0 + Wf/mass
    bg_i    = np.sqrt( gamma_i**2 - 1.0 )
    bg_m    = np.sqrt( gamma_m**2 - 1.0 )
    bg_f    = np.sqrt( gamma_f**2 - 1.0 )
    beta_m  = bg_m / gamma_m
    kx      = (-1.0)*( np.pi*Weff ) / ( mass *   bg_m**2 * lamb ) 
    ky      = (-1.0)*( np.pi*Weff ) / ( mass *   bg_m**2 * lamb )
    kz      = (+2.0)*( np.pi*Weff ) / ( mass * beta_m**2 * lamb )
    p0c     = np.sqrt( Ek**2 + 2.0*Ek*mass )  # (MeV)
    
    # ------------------------------------------------- #
    # --- [2] r-matrix                              --- #
    # ------------------------------------------------- #
    #  -- trace3D notation --  #
    # rmdiag  = bg_i / bg_f
    # rm11    =   1.0
    # rm22    = rmdiag
    # rm33    =   1.0
    # rm44    = rmdiag
    # rm55    =   1.0
    # rm66    = rmdiag
    # rm21    = kx / bg_f
    # rm43    = ky / bg_f
    # rm65    = kz / bg_f
    # kick6   = 0.0

    #  -- my notation ( for [T, PT] ) --  #
    rm11    =   1.0
    rm22    = bg_i / bg_f
    rm33    =   1.0
    rm44    = bg_i / bg_f
    rm55    =   1.0
    rm66    = gamma_i / gamma_f    #  PT = dE/(p0 c)  !=  dp/p0
    rm21    = kx / bg_f
    rm43    = ky / bg_f
    rm65    = kz / ( gamma_f )
    kick6   = dW / p0c

    # ------------------------------------------------- #
    # --- [3] return                                --- #
    # ------------------------------------------------- #
    ret     = { "rm11":rm11, "rm21":rm21, "rm22":rm22, \
                "rm33":rm33, "rm43":rm43, "rm44":rm44, \
                "rm55":rm55, "rm65":rm65, "rm66":rm66, \
                "kick6":kick6, \
               }
    return( ret )

    
            
# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    translate__track2madx()


    
