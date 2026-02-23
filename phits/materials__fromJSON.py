import os, sys, json5

# ========================================================= #
# ===  materials__fromJSON.py                           === #
# ========================================================= #

def materials__fromJSON( matFile="dat/materials.json", \
                         outFile="inp/materials.phits.j2", keys=[], \
                         tetra_auto_mat=False, \
                         ibegin_normal=1, ibegin_tetra=5001 ):
    
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if not( os.path.exists( matFile ) ):
        raise FileNotFoundError( "[materials__fromJSON.py] matFile == {}".format( matFile ) )
    
    # ------------------------------------------------- #
    # --- [2] load json file                        --- #
    # ------------------------------------------------- #
    with open( matFile, "r" ) as f:
        matDB = json5.load( f )
    try:
        settings = matDB.pop( "settings" )
    except KeyError:
        settings = None

    if ( settings is not None ):
        if ( "materialList" in settings ):
            adds  = list( set( settings["materialList"] ) - set( keys ) )
            keys += adds

    if ( tetra_auto_mat ):
        ibegin = ibegin_tetra
    else:
        ibegin = ibegin_normal
            
    # ------------------------------------------------- #
    # --- [3] format as a material_phits.inp        --- #
    # ------------------------------------------------- #
    ret = generate__materialFile( matDB=matDB, outFile=outFile, settings=settings, \
                                  keys=keys, ibegin=ibegin )

    # ------------------------------------------------- #
    # --- [4] pack density and number info          --- #
    # ------------------------------------------------- #
    dnDB = {}
    for key in keys:
        dkey       = matDB[key]["Name"] + ".density"
        nkey       = matDB[key]["Name"] + ".matNum"
        dnDB[dkey] = ( matDB[key] )["Density"]
        dnDB[nkey] = ( matDB[key] )["matNum"]
    return( dnDB )


# ========================================================= #
# ===  generate__materialFile                           === #
# ========================================================= #

def generate__materialFile( outFile="inp/materials_phits.inp", matDB=None, \
                            keys=[], settings=None, ibegin=1 ):

    default_settings = { "characterSize":2.0,  }
    
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( matDB    is None ): sys.exit( "[save__materialFile] matDB   == ???" )
    if ( settings is None ): settings = default_settings
    if ( len(keys) == 0   ): keys     = matDB.keys()

    # ------------------------------------------------- #
    # --- [2] make contents                         --- #
    # ------------------------------------------------- #
    pageTitle  = show__section( section="materials_phits.inp (PHITS)", \
                                bar_mark="=", comment_mark="$$", newLine=False )
    matTitle   = show__section( section="Material Section", \
                                bar_mark="-", comment_mark="$$", newLine=False )
    matSection = "[Material]\n"
    block1     = "\n"+pageTitle+"\n"+matTitle+"\n"+matSection+"\n"
    for ik,key in enumerate(keys):
        item    = matDB[key]
        title   = "matNum[{0}] :: {1}".format( ik+ibegin, item["Name"] )
        section = show__section( section=title, bar_mark="-", comment_mark="$$" )
        if ( len( item["Comment"] ) > 0 ):
            comment = "$$ comment :: {}\n".format( item["Comment"] )
        else:
            comment = ""
        matNumSection = "mat[{}]\n".format( ik+ibegin )
        composition   = item["Composition"]
        composit_note = [ " "*4 + "{0:<10} {1:12.5e}\n".format(key,rate) \
                          for key,rate in composition.items() ]
        block1       += section + comment + matNumSection + "".join( composit_note ) + "\n"
        # -- not needed now -- #
        # matNumDefine  = "$ <define> @{0:<25} = {1:10}\n"\
        #     .format( item["Name"]+".matNum" , ik+ibegin )
        # densityDefine = "$ <define> @{0:<25} = {1:10}\n"\
        #     .format( item["Name"]+".density", item["Density"] )
        # block1       += matNumDefine + densityDefine
        # -- not needed now -- #
        item["matNum"] = ik+ibegin

    # ------------------------------------------------- #
    # --- [3] matNameColor section                  --- #
    # ------------------------------------------------- #
    colTitle = show__section( section="matNameColor section (PHITS)", \
                              bar_mark="-", comment_mark="$$" )
    block2   = colTitle + "[MatNameColor]\n"
    block2  += "    {0:<4} {1:<18} {2:<10} {3:<20}\n".format("mat","name","size","color")
    for ik,key in enumerate(keys):
        item = matDB[key]
        line = "    {0:<4} {1:<18} {2:<10} {3:<20}\n"\
            .format( ik+ibegin, item["Name"], settings["characterSize"], item["Color"] )
        block2 += line
    block    = block1 + "\n" + block2

    # ------------------------------------------------- #
    # --- [4] save in a file                        --- #
    # ------------------------------------------------- #
    with open( outFile, "w" ) as f:
        f.write( block )
    print( "[materials__fromJSON.py] outFile :: {} ".format( outFile ) )
    return( block )



# ========================================================= #
# ===  show__section                                    === #
# ========================================================= #

def show__section( section=None, length=71, bar_mark="-", comment_mark="#", \
                   sidebarLen=3, sideSpaceLen=1, newLine=True ):

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( section is None ): sys.exit( "[show__section.py] section == ???" )

    # ------------------------------------------------- #
    # --- [2] Length determination                  --- #
    # ------------------------------------------------- #
    sectLen        = len(section)
    uprlwrbar_Len  = length - ( len( comment_mark ) + sideSpaceLen )*2
    space_t_Len    = ( length - len(section) - 2*( len( comment_mark ) \
                                                   + sideSpaceLen*2 + sidebarLen ) )
    space_f_Len    = space_t_Len // 2
    space_r_Len    = space_t_Len - space_f_Len

    # ------------------------------------------------- #
    # --- [3] preparation                           --- #
    # ------------------------------------------------- #
    space_f        = " "*space_f_Len
    space_r        = " "*space_r_Len
    side1          = comment_mark + " "*sideSpaceLen
    side2          = comment_mark + " "*sideSpaceLen + bar_mark*sidebarLen + " "*sideSpaceLen

    # ------------------------------------------------- #
    # --- [4] section contents                      --- #
    # ------------------------------------------------- #
    line1          = side1 + bar_mark*uprlwrbar_Len + side1[::-1] + "\n"
    line2          = side2 + space_f + section + space_r + side2[::-1] + "\n"
    lines          = line1 + line2 + line1
    if ( newLine ):
        lines = "\n" + lines + "\n"
    return( lines )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    matFile = "test/materials__fromJSON/materials.json"
    materials__fromJSON( matFile=matFile )


