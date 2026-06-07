import os, sys, json5, re
import numpy          as np
import pandas         as pd


# ========================================================= #
# ===  translate__track2impactx.py                      === #
# ========================================================= #

def translate__track2impactx( paramsFile   ="dat/parameters.json", \
                              trackFile    ="track/sclinac.dat", \
                              impactxBLFile="dat/beamline_impactx.json", \
                              trackBLFile  ="dat/beamline_track.json", \
                              phaseFile    ="dat/rfphase.csv" ):

    ret = extract__trackv38_beamline( inpFile=trackFile, outFile=trackBLFile )
    ret = translate__impactxElements( paramsFile=paramsFile, inpFile=trackBLFile, \
                                      phaseFile =phaseFile , outFile=impactxBLFile )
    
    with open( paramsFile, "r" ) as f:
        params    = json5.load( f )
        Lcav      = params["translate.cavity.length"]
        qm_factor = params["translate.quad.factor"]
    
    if ( Lcav > 0.0 ):
        ret = adjust__driftlength    ( inpFile=impactxBLFile, outFile=impactxBLFile, \
                                       Lcav=Lcav )
    if ( qm_factor != 1.0 ):
        ret = adjust__QmagnetStrength( inpFile=impactxBLFile, outFile=impactxBLFile, \
                                       factor=qm_factor )
    return( ret )
    


# ========================================================= #
# ===  extract__trackv38_beamline.py                    === #
# ========================================================= #

def extract__trackv38_beamline( inpFile="track/sclinac.dat", \
                                outFile="dat/beamline_track.json" ):

    MeV   = 1.0e+6
    cm    = 1.0e-2
    gauss = 1.0e-4
    MHz   = 1.0e+6
    amu   = 931.494    # [MeV]

    # ------------------------------------------------- #
    # --- [1] read track file                       --- #
    # ------------------------------------------------- #
    with open( inpFile, "r" ) as f:
        lines  = f.readlines()

    # ------------------------------------------------- #
    # --- [2] convert_quad                          --- #
    # ------------------------------------------------- #
    def convert_to_quad( words, counts ):
        name  = "qm{}".format( (counts["Nqm"]+1) )
        Bq    = float(words[2]) * gauss
        Ra    = float(words[5]) * cm
        ds    = float(words[4]) * cm
        gradB = Bq / Ra
        K1    = gradB
        ret   = [ { "type":"quadrupole", "name":name, "ds":ds, "k":K1, \
                    "aperture_x":Ra, "aperture_y":Ra } ]
        counts["Nqm"] += 1
        counts["at"]  += ds
        return( ret )

    # ------------------------------------------------- #
    # --- [3] convert_to_drift                      --- #
    # ------------------------------------------------- #
    def convert_to_drift( words, counts ):
        name           = "dr{}".format( (counts["Ndr"]+1) )
        ds             = float(words[2]) * cm
        Ra             = float(words[3]) * cm
        ret            = [ { "type":"drift", "name":name, "ds":ds, \
                             "aperture_x":Ra, "aperture_y":Ra } ]
        counts["Ndr"] += 1
        counts["at"]  += ds
        return( ret )

    # ------------------------------------------------- #
    # --- [4] convert_to_rfcavity                   --- #
    # ------------------------------------------------- #
    def convert_to_rfcavity( words, counts ):
        name      = "rf{}".format( (counts["Nrf"]+1) )
        ds        = 0.0   # tempolarily set as 0 to identify other element's position.
        volt      = float(words[2])                         # volt :    [MV]
        phase     = float(words[3])                         #           [deg]
        harmonics =   int(words[4])                         # harmonics
        Ra        = float(words[5]) * cm                    # R-cavity  [cm]
        ret       = [ { "type":"rfcavity", "name":name, "ds":ds, \
                        "volt":volt, "phase":phase, "harmonics":harmonics, \
                        "aperture_x":Ra, "aperture_y":Ra,  } ]
        counts["Nrf"] += 1
        counts["at"]  += ds
        return( ret )

    # ------------------------------------------------- #
    # --- [5] analyze each line                     --- #
    # ------------------------------------------------- #
    seq    = []
    counts = { "at":0.0, "N":0, "Nqm":0, "Nrf":0, "Ndr":0, }
    for ik,line in enumerate(lines):
        line_ = re.sub( r"#.*", "", line ).strip()
        words = ( line_.split() )
        if ( not( len( words ) == 0 ) ):
            counts["N"]   += 1
            
            if   ( words[1].lower()  == "quad" ):
                seq += convert_to_quad ( words, counts )

            elif ( words[1].lower()  == "drift" ):
                seq += convert_to_drift( words, counts )
                
            elif ( words[1].lower()  == "rfgap" ):
                seq += convert_to_rfcavity( words, counts )
            
            else:
                raise ValueError( "[ERROR] undefined keyword :: {} ".format( words[1] ) )

    L_tot     = counts["at"]
    Nelements = counts["N"]
    print( f" -- all of the {Nelements} elements were loaded... " )
    print( f" --    total length of the beam line == {L_tot:.8} " )
    print()

    # ------------------------------------------------- #
    # --- [6] save as a json file                   --- #
    # ------------------------------------------------- #
    elements = { el["name"]:el for el in seq }
    if ( outFile is not None ):
        with open( outFile, "w" ) as f:
            json5.dump( elements, f, indent=4 )
            print( "[extract__trackv38_beamline] output :: {}".format( outFile ) )
    return( elements )
        


# ========================================================= #
# ===  translate__impactxElements.py                    === #
# ========================================================= #

def translate__impactxElements( paramsFile="dat/parameters.json", \
                                inpFile   ="dat/beamline_track.json", \
                                outFile   ="dat/beamline_impactx.json", \
                                phaseFile ="dat/rfphase.csv" ):

    # ========================================================= #
    # ===  functions                                        === #
    # ========================================================= #
    
    # ------------------------------------------------- #
    # --- [0-1] RFcavity's phase                    --- #
    # ------------------------------------------------- #
    def guess__RFcavityPhase( elements=None, params=None, phaseFile=None ):

        amu = 931.494       # [MeV]
        cv  = 299792458.0   # light of speed [m/s]
        
        # ------------------------------------------------- #
        # --- [0-1-1] use functions                     --- #
        # ------------------------------------------------- #
        def norm_angle( deg ):
            deg  = np.asarray( deg, dtype=float )
            ret  = ( ( deg + 180.0 ) % 360.0 ) - 180.0
            if ( np.ndim(ret) == 0 ): ret = float( ret )
            return( ret )
    
        # ------------------------------------------------- #
        # --- [0-1-2] guess initial rfcavity phase      --- #
        # ------------------------------------------------- #
        Ek0         = params["beam.Ek.MeV/u"] * params["beam.u.nucleon"]
        Em0         = params["beam.mass.amu"] * amu
        omega       = params["beam.freq.Hz"]  * params["beam.harmonics"] * 2.0*np.pi
        df          = pd.DataFrame.from_dict( elements, orient="index" )
        df          = df.reindex( columns=df.columns.union(["volt", "phase", "ds"]), fill_value=0.0 )
        df          = df.drop( columns=[ "k","aperture_x", "aperture_y", "harmonics" ], \
                               errors="ignore" )
        mask        = df["type"] == "rfcavity"
        df[["volt","phase","ds"]] = df[["volt","phase","ds"]].fillna(0.0)
        df.loc[mask,"ds"] = params["translate.cavity.length"]
        egain       = ( df["volt"] * np.cos( df["phase"] /180.0*np.pi ) ).fillna(0)
        df["Ek_i"]  = np.concatenate( ([0.0], np.cumsum(egain)[:-1]) ) + Ek0
        df["Ek_o"]  = np.cumsum( egain ) + Ek0
        df["Ek"]    = 0.5*( df["Ek_i"] + df["Ek_o"] )
        df["gamma"] = 1.0 + df["Ek"]/Em0
        df["beta"]  = np.sqrt( 1.0 - 1.0/df["gamma"]**2 )
        Em0_GeV     = Em0 * 1e6 / 1e9   # MeV -> GeV 
        df["Brho"]  = 3.3356 * df["beta"] * df["gamma"] * Em0_GeV / params["beam.charge.qe"]
        vp          = df["beta"] * cv
        dt          = df["ds"].to_numpy() / vp.to_numpy()
        t_in        = np.concatenate( ([0.0], np.cumsum(dt)[:-1] ) )
        t_out       = t_in + dt
        t_mid       = 0.5*( t_in + t_out )
        df["tpass"] = t_mid
        df["phi_o"] = norm_angle( t_mid * omega / np.pi * 180.0 )
        df["phi_t"] = norm_angle( params["translate.cavity.phase"] )
        df["phi_c"] = norm_angle( df["phi_t"] - df["phi_o"] )
        df["phi_b"] = 0.0
        
        # ------------------------------------------------- #
        # --- [0-1-3] return                            --- #
        # ------------------------------------------------- #
        if ( phaseFile is not None ):
            df.to_csv( phaseFile )
        return( df )

    
    # ------------------------------------------------- #
    # --- [0-2] convert MaryLie unit => MADX unit   --- #
    # ------------------------------------------------- #
    def convert__MaryLie2MADX( elements=None, phase_df=None, retype=True ):
        for key in elements.keys():
            if ( elements[key]["type"] == "quadrupole" ):
                elements[key]["k"] = elements[key]["k"] / phase_df.loc[ key, "Brho" ]
                if ( retype ):
                    elements[key]["type"] = "quadrupole.linear"
            if ( elements[key]["type"] == "drift" ):
                if ( retype ):
                    elements[key]["type"] = "drift.linear"
        return( elements )

    
    # ------------------------------------------------- #
    # --- [0-3] convert routine                     --- #
    # ------------------------------------------------- #
    def convert__rfcavity( element, params, phase_df ):
        amu    = 931.494  # [MeV]
        Em0    = params["beam.mass.amu"] * amu
        ds     = params["translate.cavity.length"]
        freq   = params["beam.freq.Hz"] * params["beam.harmonics"]
        escale = element["volt"] / ds / Em0
        phase  = phase_df["phi_c"].loc[ element["name"] ]  # <= use "delay from global phase"
        ret    = { "type":element["type"], "name":element["name"], "ds":ds, "escale":escale, \
                   "freq":freq, "phase":phase, \
                   "aperture_x":element["aperture_x"], "aperture_y":element["aperture_y"] } 
        return( ret )

    # ------------------------------------------------- #
    # --- [0-4] convert shortrf                     --- #
    # ------------------------------------------------- #
    def convert__shortrf( element, params, phase_df ):
        amu    = 931.494  # [MeV]
        Em0    = params["beam.mass.amu"] * amu
        V      = element["volt"] / Em0
        freq   = params["beam.freq.Hz"] * params["beam.harmonics"]
        phase  = params["translate.cavity.phase"]           # <= correct!! use "tobe" phase.
        # phase  = phase_df["phi_c"].loc[ element["name"] ] # <= [Incorrect!!]  for shortRF
        ret    = { "type":"shortrf", "name":element["name"], "V":V, "freq":freq, "phase":phase } 
        return( ret )


    # ------------------------------------------------- #
    # --- [0-5] convert rfgap ( TRACE3D's element ) --- #
    # ------------------------------------------------- #
    def convert__rfgap( element, params, phase_df ):
        
        amu    = 931.494
        
        # ------------------------------------------------- #
        # --- [0-5-1] rfgap model of the trace3D code   --- #
        # ------------------------------------------------- #
        def rfgap( Vg     = 1.0,       # [MV]
                   phi    = -45.0,     # [deg]   # for particle ??
                   Ek     = 0.0,       # [MeV]
                   mass   = 931.494,   # [MeV]
                   charge = 1.0,       # [C/e]
                   freq   = 146.0e6 ): # [Hz]
        
            cv      = 2.99792458e8     # (m/s)
            qa      = abs( charge )    # (C)
        
            # ------------------------------------------------- #
            # --- [0-5-2] calculation                       --- #
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
            # --- [0-5-3] r-matrix                          --- #
            # ------------------------------------------------- #            
            #  -- validated ver. -- #
            #   - reduced component to avoid duplication with shortRF - #
            rmat_     = np.zeros( (7,7) )
            rmat_[1,1] =   1.0
            rmat_[2,2] =   1.0
            rmat_[3,3] =   1.0
            rmat_[4,4] =   1.0
            rmat_[5,5] =   1.0
            rmat_[6,6] =   1.0    #  PT = dE/(p0 c)  !=  dp/p0
            rmat_[2,1] = kx / bg_f
            rmat_[4,3] = ky / bg_f
            rmat_[6,5] =   0.0
            rmat       = ( np.copy( rmat_[1:,1:] ) ).tolist()  # list for json5 dump
            
            #  -- original ver. ( all-component ) ( for [T, PT] ) --  #
            # rmat_     = np.zeros( (7,7) )
            # rmat_[1,1] =   1.0
            # rmat_[2,2] = bg_i / bg_f
            # rmat_[3,3] =   1.0
            # rmat_[4,4] = bg_i / bg_f
            # rmat_[5,5] =   1.0
            # rmat_[6,6] = gamma_i / gamma_f    #  PT = dE/(p0 c)  !=  dp/p0
            # rmat_[2,1] = kx / bg_f
            # rmat_[4,3] = ky / bg_f
            # rmat_[6,5] = (-1.0) * kz / ( gamma_f ) # -1 for impactx
            # # rmat_[6,5] = kz / ( gamma_f ) # -1 for impactx
            # # kick6     = dW / p0c
            # rmat       = ( np.copy( rmat_[1:,1:] ) ).tolist()    # list for json5 dump
            
            # ------------------------------------------------- #
            # --- [0-5-4] return                            --- #
            # ------------------------------------------------- #
            return( rmat )

        # ------------------------------------------------- #
        # --- [0-5-5] set values for LinearMap          --- #
        # ------------------------------------------------- #
        name   = element["name"].replace( "rf", "gap" )
        Vg     = element["volt"]
        phi    = phase_df["phi_t"].loc[ element["name"] ]
        Ek     = phase_df["Ek_i"].loc[ element["name"] ]
        mass   = params["beam.mass.amu"] * amu
        charge = params["beam.charge.qe"]
        freq   = params["beam.freq.Hz"]  * params["beam.harmonics"]
        Rmat   = rfgap( Vg=Vg, phi=phi, Ek=Ek, mass=mass, charge=charge, freq=freq )
        ret    = { "type":"rfgap", "name":name, "ds":0.0, "R":Rmat  }
        return( ret )


    # ========================================================= #
    # ===  Main Routine                                     === #
    # ========================================================= #
    
    # ------------------------------------------------- #
    # --- [1] load files                            --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        params   = json5.load( f )
    with open( inpFile   , "r" ) as f:
        elements = json5.load( f )

    # ------------------------------------------------- #
    # --- [2] call converter                        --- #
    # ------------------------------------------------- #
    phase_df  = guess__RFcavityPhase( elements=elements, params=params, phaseFile=phaseFile )
    if   ( params["mode.linear"] ):
        elements = convert__MaryLie2MADX( elements=elements, phase_df=phase_df, retype=True  )
    elif ( params["translate.quad.options"]["unit"] == 0 ):
        elements = convert__MaryLie2MADX( elements=elements, phase_df=phase_df, retype=False )
    if   ( params["translate.cavity.phase.fromfile"] is not None ):
        phase_df_ = pd.read_csv( params["translate.cavity.phase.fromfile"] )
        phase_df_ = phase_df_.set_index( "name" )
        phase_df  = phase_df .set_index( "name" )
        phase_df.update( phase_df_, overwrite=True )
        phase_df.to_csv( "dat/temp.csv" )
    elements_ = {}
    for key in elements.keys():
        if   ( elements[key]["type"].lower() in [ "rfcavity" ] ):
            if   ( params["translate.cavity.type"].lower() in ["rfcavity"] ):
                ret            = convert__rfcavity( elements[key], params, phase_df )
            elif ( params["translate.cavity.type"].lower() in ["shortrf" ] ):
                ret            = convert__shortrf ( elements[key], params, phase_df )
            elements_[key] = { **ret, **params["translate.cavity.options" ] }
            if   ( params["translate.cavity.rfgap"] ):
                ret                      = convert__rfgap( elements[key], params, phase_df )
                elements_[ ret["name"] ] = ret      # ret's name is gapXX -> define another element
                
        elif ( elements[key]["type"].lower() in [ "quadrupole", "quadrupole.linear" ] ):
            elements_[key] = { **elements[key], **params["translate.quad.options" ] }
        elif ( elements[key]["type"].lower() in [ "drift", "drift.linear" ] ):
            elements_[key] = { **elements[key], **params["translate.drift.options"] }

    # ------------------------------------------------- #
    # --- [3] skip components                       --- #
    # ------------------------------------------------- #
    if ( params["translate.cavity.skip"] ):
        elements_ = { k:v for k,v in elements_.items()
                      if not( v["type"].lower() in ["rfcavity","rfgap","shortrf"] ) }
    if ( params["translate.quad.skip"] ):
        elements_ = { k:v for k,v in elements_.items()
                      if not( v["type"].lower() in ["quadrupole","quadrupole.linear"] ) }
    if ( params["translate.drift.skip"] ):
        elements_ = { k:v for k,v in elements_.items()
                      if not( v["type"].lower() in ["drift","drift.linear"] ) }
    
    # ------------------------------------------------- #
    # --- [4] save and return                       --- #
    # ------------------------------------------------- #
    if ( outFile is not None ):
        with open( outFile, "w" ) as f:
            json5.dump( elements_ , f, indent=2 )
            print( "[translate__track2impactx.py] output :: {}".format( outFile ) )
    return( elements_ )



# ========================================================= #
# ===  adjust__driftlength.py                           === #
# ========================================================= #
def adjust__driftlength( Lcav=None, elements=None, \
                         inpFile=None, outFile=None ):

    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    if ( Lcav     is None ): raise ValueError( "[adjust__driftlength] Lcav == ???" )
    if ( elements is None ):
        if ( inpFile is not None ):
            with open( inpFile, "r" ) as f:
                elements = json5.load( f )
    sequence = list( elements.keys() )
    nSeq     = len( sequence )
    Lcav_h   = 0.5 * Lcav
    
    # ------------------------------------------------- #
    # --- [2] re-length                             --- #
    # ------------------------------------------------- #
    for ik,seq in enumerate(sequence):
        prev, next = None, None
        if ( ik-1 >= 0    ): prev = sequence[ik-1]
        if ( ik+1 <  nSeq ): next = sequence[ik+1]
        if ( elements[seq]["type"].lower() in [ "rfcavity" ] ):
            if ( prev is not None ):
                if ( elements[prev]["type"] in ["drift", "drift.linear"] ):
                    if ( elements[prev]["ds"] > Lcav_h ):
                        elements[prev]["ds"] -= Lcav_h
            if ( next is not None ):
                if ( elements[next]["type"] in ["drift", "drift.linear"] ):
                    if ( elements[next]["ds"] > Lcav_h ):
                        elements[next]["ds"] -= Lcav_h

    # ------------------------------------------------- #
    # --- [3] save and return                       --- #
    # ------------------------------------------------- #
    if ( outFile is not None ):
        with open( outFile, "w" ) as f:
            json5.dump( elements, f, indent=4 )
            print( "[adjust__driftlength] output :: {}".format( outFile ) )
    return( elements )



# ========================================================= #
# ===  adjust__QmagnetStrength                          === #
# ========================================================= #
def adjust__QmagnetStrength( factor =1.0 , elements=None, \
                             inpFile=None, outFile =None ):
    
    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    if ( elements is None ):
        if ( inpFile is None ):
            raise TypeError( "[adjust__QmagnetStrength] no elements... [ERROR]" )
        else:
            with open( inpFile, "r" ) as f:
                elements = json5.load( f )
                
    # ------------------------------------------------- #
    # --- [2] beam line                             --- #
    # ------------------------------------------------- #
    for key,item in elements.items():
        if ( elements[key]["type"].lower() in ["quadrupole", "quadrupole.linear"] ):
            elements[key]["k"] = elements[key]["k"] * factor

    # ------------------------------------------------- #
    # --- [3] dump again                            --- #
    # ------------------------------------------------- #
    if ( outFile is not None ):
        with open( outFile, "w" ) as f:
            json5.dump( elements, f, indent=4 )
            print( "[adjust__driftlength] output :: {}".format( outFile ) )
    return( elements )




# ========================================================= #
# ===  Trackv38's track.dat => parameters.json          === #
# ========================================================= #
def calculate__twissFromTrackv38( trackFile=None, full2rms=1.0/8.0,
                                  namelists=None, ):

    # # ========================================================= #
    # # ===  calc twiss parameter from track.dat              === #
    # # ========================================================= #
    # @invoke.task()
    # def calctwiss( ctx, trackFile="track/track.dat" ):
    #     """calculate twiss parameter for impactx from track.dat"""
    #     ret = ctk.calculate__twissFromTrackv38( trackFile=trackFile, )
    #     print( ret )
    #     return( ret )

    
    import f90nml
    if ( trackFile is not None ):
        namelists = f90nml.read( trackFile )
        namelists = namelists.get( "tran", {} )
    if ( namelists is None ):
        namelists = {
            "Win"      :5.0e6    ,
            "freqb"    :36.5e6   ,
            "amass"    :2.0      ,
            "epsnx"    :20.0     ,
            "alfax"    :0.173982 ,
            "betax"    :535.875  ,
            "epsny"    :20.0     ,
            "alfay"    :0.115839 ,
            "betay"    :113.997  ,
            "epsnz"    :5.5      ,
            "alfaz"    :0.0      ,
            "betaz"    :10.0     ,
        }
    cv    = 299792458.0
    amu   = 931.4941024e6
    cm    = 1.0e-2
    cm2mm = 10.0

    # ------------------------------------------------- #
    # --- [1] calculation                           --- #
    # ------------------------------------------------- #
    # -- relativistic gamma, beta                   --  #
    gamma = 1.0 + namelists["Win"] / ( amu )
    beta  = np.sqrt( 1.0 - 1.0 / gamma**2 )

    # -- TRACKv38: Normalized full [pi cm mrad] -> geometric rms [mm mrad]  -- #
    emitx = namelists["epsnx"] / ( beta * gamma ) * full2rms * cm2mm
    emity = namelists["epsny"] / ( beta * gamma ) * full2rms * cm2mm

    # -- TRACK beta_x, beta_y: [cm/rad] -> [m/rad]  --  #
    betax = namelists["betax"] * cm
    betay = namelists["betay"] * cm

    # -- z: phi[deg] -> t=-c dt [m], dW/W -> pt=dE/(p0 c)
    kt    = cv / ( 360.0 * namelists["freqb"] )
    kpt   = ( gamma - 1.0 ) / ( beta*gamma ) * 1.0e-2

    betat = abs( kt / kpt ) *   namelists["betaz"]
    emitt = abs( kt * kpt ) * ( namelists["epsnz"]*full2rms ) * 1.0e6
    
    alfax = namelists["alfax"]
    alfay = namelists["alfay"]
    alfaz = namelists["alfaz"]
    print( f'"beam.twiss.alpha"    : [{alfax:.8g}, {alfay:.8g}, {alfaz:.8g}],')
    print( f'"beam.twiss.beta"     : [{betax:.8g}, {betay:.8g}, {betat:.8g}],')
    print( f'"beam.emittance.geom" : [{emitx:.8g}, {emity:.8g}, {emitt:.8g}],')
    ret   = { "alphax": alfax, "alphay": alfay, "alphaz": alfaz,
              "betax" : betax, "betay" : betay, "betat" : betat,
              "emitx" : emitx, "emity" : emity, "emitt" : emitt, }
    return( ret )




# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    translate__track2impactx()
