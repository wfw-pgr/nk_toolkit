# ========================================================= #
# ===  make__sectionTitle                               === #
# ========================================================= #

def make__sectionTitle( section=None, length=71,
                        bar_mark="-", comment_mark="#",
                        sidebar_len=3, side_space_len=1,
                        new_line=True ):
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( section is None ):
        raise ValueError("section must not be None")
    section = str(section)

    # ------------------------------------------------- #
    # --- [2] length determination                  --- #
    # ------------------------------------------------- #
    left_margin    = comment_mark + " " * side_space_len
    right_margin   = " " * side_space_len + comment_mark
    side_left      = ( comment_mark + " " * side_space_len \
                       + bar_mark * sidebar_len + " " * side_space_len )
    side_right     = ( " " * side_space_len + bar_mark * sidebar_len \
                       + " " * side_space_len + comment_mark )
    bar_len        = length - len(left_margin) - len(right_margin)
    text_space_len = length - len(section) - len(side_left) - len(side_right)

    if bar_len < 0 or text_space_len < 0:
        raise ValueError("length is too short for the given section title")

    left_space_len  = text_space_len // 2
    right_space_len = text_space_len - left_space_len
    
    line1 = left_margin + bar_mark * bar_len + right_margin
    line2 = ( side_left
              + " " * left_space_len
              + section
              + " " * right_space_len
              + side_right )
    lines = "\n".join([line1, line2, line1])

    if new_line:
        lines = "\n" + lines + "\n"

    return lines

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    print( make__sectionTitle( section="sample execution" ) )
    
