import json5, re, math, sys, decimal, io
import asteval
import numpy as np

# ========================================================= #
# ===  json__formulaParser.py                           === #
# ========================================================= #

def json__formulaParser( inpFile=None, verbose=False, table=None, \
                         expr_fml=None, formula_mark="", variable_mark="$", iterMax=100 ):
    
    if ( expr_fml is None ):
        if   ( formula_mark == "`" ):
            expr_fml = r"\s*`([\s\S]+)`\s*"
        elif ( formula_mark == "" ):
            expr_fml = r"([\s\S]+)"
        else:
            print( "[json__formulaParser.py] formula_mark == {} ??? ".format( variable_mark ) )
            sys.exit()

    if ( variable_mark in [ "$" ] ):
        variable_mark = "\\{}".format( variable_mark )

    # ------------------------------------------------- #
    # --- [1] load json file as json5               --- #
    # ------------------------------------------------- #
    with open( inpFile, "r" ) as f:
        varDict = json5.load( f )
    if ( table is not None ):
        varDict = { **table, **varDict }

    # ------------------------------------------------- #
    # --- [2] convert formula string into value     --- #
    # ------------------------------------------------- #
    for iloop in range( iterMax ):
        updated = False
        old     = repr(varDict)
        
        for key,val in varDict.items():
            varDict[key] = evaluate_value( val, varDict, expr_fml, \
                                           variable_mark, verbose=verbose )
        new     = repr( varDict )
        
        if ( old != new ): updated = True
        if not updated: break

    # ------------------------------------------------- #
    # --- [3] align__digits                         --- #
    # ------------------------------------------------- #
    for key,val in varDict.items():
        if ( type(val) in [ int, np.int32 ] ):
            varDict[key] = int( return__digitsAligned( val ) )
        if ( type(val) in [ float, np.float64 ] ):
            varDict[key] = float( return__digitsAligned( val ) )
        if ( type(val) in [ list ] ):
            stack = []
            for vh in val:
                if   ( type(vh) in [   int, np.int32   ] ):
                    stack += [   int( return__digitsAligned( vh ) ) ]
                elif ( type(vh) in [ float, np.float64 ] ):
                    stack += [ float( return__digitsAligned( vh ) ) ]
                else:
                    stack += [ vh ]
            varDict[key] = stack
    
    # ------------------------------------------------- #
    # --- [4] print parsed items                    --- #
    # ------------------------------------------------- #
    return( varDict )


# ========================================================= #
# ===  evaluate value at once                           === #
# ========================================================= #
def evaluate_value( obj, varDict, expr_fml, variable_mark, aeval=None, verbose=False ):

    # ------------------------------------------------- #
    # --- [0] preparation                           --- #
    # ------------------------------------------------- #
    if ( aeval is None ):
        aeval = asteval.Interpreter( usersyms={ "math":math, "np":np, "decimal":decimal },
                                     err_writer=io.StringIO(), )
        
    # ------------------------------------------------- #
    # --- [1] for str                               --- #
    # ------------------------------------------------- #
    if isinstance( obj, str ):
        match_fml = re.match(expr_fml, obj)
        if match_fml:
            formula = match_fml.group(1).strip()
            vmark   = variable_mark.replace("\\", "")

            for var_, val_ in sorted( varDict.items(), key=lambda kv: len(kv[0]), reverse=True ):
                expr_from = variable_mark + r"\{*" + re.escape(var_) + r"\}*\s*"
                expr_into = "{}".format(val_)
                formula = re.sub(expr_from, expr_into, formula)

            try:
                ret = aeval(formula)

                # asteval は例外を投げずに error に溜める場合がある
                if len(aeval.error) > 0:
                    err_msgs = [str(err.get_error()) for err in aeval.error]
                    aeval.error = []

                    # @を含む式は「評価したかった可能性が高い」ので verbose 時にだけ出す
                    if ( ( verbose ) and ( vmark in obj ) ):
                        print()
                        print("[json__formulaParser.py] Cannot evaluate [ERROR]")
                        print(f"[json__formulaParser.py] original :: {obj}")
                        print(f"[json__formulaParser.py] formula  :: {formula}")
                        print(f"[json__formulaParser.py] error    :: {err_msgs}")
                        print()
                    return( obj )
                
                return( ret )

            except Exception as e:
                if verbose and ( vmark in obj ):
                    print()
                    print("[json__formulaParser.py] Cannot evaluate [ERROR]")
                    print(f"[json__formulaParser.py] original :: {obj}")
                    print(f"[json__formulaParser.py] formula  :: {formula}")
                    print(f"[json__formulaParser.py] error    :: {e}")
                    print()
                return ( obj )
            
        return( obj )

    # ------------------------------------------------- #
    # --- [2] for list                              --- #
    # ------------------------------------------------- #
    elif isinstance(obj, list):
        return [
            evaluate_value(v, varDict, expr_fml, variable_mark, aeval=aeval, verbose=verbose)
            for v in obj
        ]

    # ------------------------------------------------- #
    # --- [3] for dict                              --- #
    # ------------------------------------------------- #
    elif isinstance(obj, dict):
        return {
            k: evaluate_value(v, varDict, expr_fml, variable_mark, aeval=aeval, verbose=verbose)
            for k, v in obj.items()
        }

    else:
        return obj


# ========================================================= #
# ===  for clean aligned digits                         === #
# ========================================================= #
def return__digitsAligned( value, maxDigit=8, precision=12 ):

    if ( type( value ) in [ int, float, np.float64, np.int32 ] ):
        decimal.getcontext().prec     = precision
        decimal.getcontext().rounding = decimal.ROUND_HALF_UP
        exp = math.floor( math.log10( abs( value ) ) ) if ( value != 0 ) else 0.0
        if ( abs( exp ) >= maxDigit ):
            ret = ( "{:15." + str(maxDigit) + "e}" ).format( value )
        else:
            lowest = int( exp - maxDigit )
            quant  = decimal.Decimal( "1e{0}".format( lowest ) )
            dval   = decimal.Decimal.from_float( value ).quantize( quant )
            ret   = format(dval, "f").rstrip("0").rstrip(".")
        return( ret )
    else:
        return( value )



# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
        
    # ------------------------------------------------- #
    # --- [1] declare regexp and file name          --- #
    # ------------------------------------------------- #
    inpFile  = "test/json__formulaParser/sample1.json"
    ret      = json__formulaParser( inpFile=inpFile, variable_mark="$" )
    print()
    print( ret )
    print()
    
    # ------------------------------------------------- #
    # --- [2] variable mark is "@" case             --- #
    # ------------------------------------------------- #
    inpFile  = "test/json__formulaParser/sample2.json"
    ret      = json__formulaParser( inpFile=inpFile, variable_mark="@" )
    print()
    print( ret )
    print()


    # ------------------------------------------------- #
    # --- [2] variable mark is "@" case             --- #
    # ------------------------------------------------- #
    inpFile  = "test/json__formulaParser/sample3.json"
    ret      = json__formulaParser( inpFile=inpFile, variable_mark="@" )
    print()
    print( ret )
    print()

