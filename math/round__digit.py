import math

# ========================================================= #
# ===  round__digit                                     === #
# ========================================================= #
def round__digit( x, digit=3 ):
    if ( ( x == 0 ) or ( x is None ) ):
        ret = 0
    else:
        ret = round( x, digit - int( math.floor( math.log10( abs( x ) ) ) ) - 1 )
    return( ret )
