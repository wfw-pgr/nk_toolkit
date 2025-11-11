import os, sys, json5
import pandas as pd
import numpy  as np
import nk_toolkit.math.fitting_toolkit as fit


# ========================================================= #
# ===  extract__paretoFrontier                          === #
# ========================================================= #
def get__paretoFrontier( df=None, xkey=None, ykey=None, \
                         opt_method="platypus", fit_method="spline_under_ristriction", \
                         directions=["max","max"], monotone="decreasing", \
                         concave_convex="concave", nFront=101, smoothness=0.1, \
                         elbow=True, elbow_method="distance", outFile=None ):

    # ------------------------------------------------- #
    # --- [1-1] extract pareto primitive way        --- #
    # ------------------------------------------------- #
    def extract__paretoFrontierPoints_primitive( df=None, xkey=None, ykey=None, \
                                                 directions=None ):
        # ------------------------------------------------- #
        # --- [1] prepare and sort                      --- #
        # ------------------------------------------------- #
        df_  = df[ [xkey,ykey] ].copy()
        ascending   = [ (directions[0].lower() == "min"), not( directions[1].lower() == "min" ) ]
        df_  = df_.sort_values( [ xkey, ykey ], ascending=ascending ).reset_index()    
        df_  = df_.drop_duplicates( subset=[xkey], keep='first' )
        # ------------------------------------------------- #
        # --- [2] compare                               --- #
        # ------------------------------------------------- #
        idx  = []
        if ( directions[1].lower() == "max" ):
            best = - np.inf
            cmp  = lambda a, b: a > b
        else:
            best = np.inf
            cmp  = lambda a, b: a < b

        for _, row in df_.iterrows():
            if ( cmp( row[ykey], best ) ):
                 idx += [ row["index"] ]
                 best = row[ ykey ]
        ret = df.loc[idx].copy()
        ret = ret.sort_values( xkey ).reset_index( drop=True )
        return( ret )


    # ------------------------------------------------- #
    # --- [1-2] extract pareto using platypus       --- #
    # ------------------------------------------------- #
    def extract__paretoFrontierPoints_platypus( df=None, xkey=None, ykey=None, directions=None ):

        import platypus
        # ------------------------------------------------- #
        # --- [1] evaluation function using index       --- #
        # ------------------------------------------------- #
        def eval_func( df_in, solution ):
            idx                    = int( solution.variables[0] )
            solution.objectives[:] = [ df_in.iloc[idx][xkey], df_in.iloc[idx][ykey] ]

        # ------------------------------------------------- #
        # --- [2] definition of problems                --- #
        # ------------------------------------------------- #
        table       = { "min":platypus.Problem.MINIMIZE, "max":platypus.Problem.MAXIMIZE }
        directions_ = [ table[direction] for direction in directions ]
        #   -- variable   : 1 ( index )
        #   -- objectives : 2 ( EY, Pmiss ) etc.
        problem               = platypus.Problem( 1, 2 )
        problem.types[0]      = platypus.Real( 0, len(df)-1 )
        problem.directions[:] = directions_
        problem.function      = eval_func
        
        # ------------------------------------------------- #
        # --- [3] pareto optimization                   --- #
        # ------------------------------------------------- #
        solutions = []
        for i in range(len(df)):
            sol              = platypus.Solution( problem )
            sol.variables[:] = [i]
            problem.function( df, sol )
            solutions.append( sol )
            
        # ------------------------------------------------- #
        # --- [4] return non-dominated solutions        --- #
        # ------------------------------------------------- #
        pareto         = platypus.nondominated( solutions )
        pareto_indices = [ int( round( sol.variables[0] ) )  for sol in pareto ]
        pareto_df      = df.iloc[ pareto_indices ].drop_duplicates().reset_index( drop=True )
        return( pareto_df )
    
    
    # ------------------------------------------------- #
    # --- [2] prepare fittings of paretoFrontier    --- #
    # ------------------------------------------------- #
    if   ( opt_method == "primitive" ):
        pareto = extract__paretoFrontierPoints_primitive( df=df, xkey=xkey, ykey=ykey, \
                                                          directions=directions )
    elif ( opt_method == "platypus"  ):
        pareto = extract__paretoFrontierPoints_platypus ( df=df, xkey=xkey, ykey=ykey, \
                                                          directions=directions )
    else:
        sys.exit( "unknown opt_method != ( primitive, platypus ) [ERROR]" )
    xPareto  = pareto[xkey].to_numpy()
    yPareto  = pareto[ykey].to_numpy()
    xFront = np.linspace( np.min( xPareto ), np.max( xPareto ), nFront )
    
    # ------------------------------------------------- #
    # --- [4] call interpolator                     --- #
    # ------------------------------------------------- #
    if ( fit_method == "polynomial" ):
        yFront = fit.fit__polynomial_5th( xFront, xPareto, yPareto )
    if ( fit_method == "quantile_regression" ):
        yFront = fit.fit__quantile_regression( xFront, xPareto, yPareto )
    if ( fit_method == "spline_under_ristriction" ):
        yFront = fit.fit__spline_under_ristriction( xFront, xPareto, yPareto, \
                                                    smoothness=smoothness, \
                                                    monotone=monotone, \
                                                    concave_convex=concave_convex )

    # ------------------------------------------------- #
    # --- [5] search elbow point                    --- #
    # ------------------------------------------------- #
    if ( elbow ):
        xElbow, yElbow = search__elbowPoint( xp=xFront, yp=yFront, method=elbow_method, \
                                             concave_convex=concave_convex )

    # ------------------------------------------------- #
    # --- [6] save and return                       --- #
    # ------------------------------------------------- #
    ret = { "xPareto":xPareto.tolist(), "yPareto":yPareto.tolist(), \
            "xFront" : xFront.tolist(), "yFront" : yFront.tolist(), \
            "xElbow" :float(xElbow)   , "yElbow" :float(yElbow)   , \
           }
    if ( outFile is not None ):
        with open( outFile, "w" ) as f:
            json5.dump( ret, f, indent=2 )
    return( ret )



# ========================================================= #
# ===  search__elbowPoint                               === #
# ========================================================= #

def search__elbowPoint( xp=None, yp=None, method="distance-curvature",
                        concave_convex=None, spline_s=None ):

    # method :: [ distance, curvature, second_derivative ]
    
    # ------------------------------------------------- #
    # --- [1] preparation                           --- #
    # ------------------------------------------------- #
    xv = np.asarray( xp )
    yv = np.asarray( yp )
    def normalize( arg ):
        mn, mx = np.min(arg), np.max(arg)
        if ( mx - mn == 0 ):
            return np.zeros_like( arg )
        return( (arg - mn) / (mx - mn) )
    xn     = normalize( xv )
    yn     = normalize( yv )

    # ------------------------------------------------- #
    # --- [2] search                                --- #
    # ------------------------------------------------- #
    def __distance_method( xn, yn ):
        dx, dy  = xn[-1]-xn[0], yn[-1]-yn[0]
        denom   = np.hypot( dx, dy ) + 1.e-18
        dists   = ( dx*(yn[0]-yn) - dy*(xn[0]-xn) ) / denom
        if   ( concave_convex == "concave" ):
            idx = int( np.argmax( dists ) )
        elif ( concave_convex == "convex"  ):
            idx = int( np.argmin( dists ) )
        else:
            idx = int( np.argmax( np.abs( dists ) ) )
        ret     = np.array( [ xv[idx], yv[idx] ] )
        return( ret )

    def __curvature_method( xn, yn, spline_s=None ):
        import scipy.interpolate 
        if ( spline_s is None ):
            spline_s = 1.0e-3 * len( xn ) * np.var( yn )
        sp     = scipy.interpolate.UnivariateSpline( xn, yn, s=spline_s, k=3 )
        y1,y2  = sp.derivative(1)(xn), sp.derivative(2)(xn)
        denom  = ( 1.0 + y1**2 )**(3.0/2.0)
        kappa  = y2 / denom
        if   ( concave_convex == "concave" ):
            idx    = int( np.argmax(kappa) )
        elif ( concave_convex == "convex"  ):
            idx    = int( np.argmin(kappa) )
        else:
            idx    = int( np.argmax( np.abs( kappa ) ) )
        ret    = np.array( [ xv[idx], yv[idx] ] )
        return( ret )

    def __2nd_derivative( xn, yn ):
        skips  = 5
        dydx   = np.gradient(   yn, xn, edge_order=2 )
        d2y    = np.gradient( dydx, xn, edge_order=2 )
        idx    = int( np.argmax( np.abs( d2y[skips:(-1*skips)] )) ) + skips
        ret    = np.array( [ xv[idx], yv[idx] ] )
        return( ret )

    if   ( method.lower() == "distance"  ):
        ret = __distance_method ( xn, yn )
    elif ( method.lower() == "curvature" ):
        ret = __curvature_method( xn, yn )
    elif ( method.lower() == "2nd_derivative" ):
        ret = __2nd_derivative  ( xn, yn )
    elif ( method.lower() == "distance-curvature" ):
        ret = 0.5 * ( __distance_method ( xn, yn ) + __curvature_method( xn, yn ) )

    # ------------------------------------------------- #
    # --- [3] return                                --- #
    # ------------------------------------------------- #
    return( ret )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1] sample generation                     --- #
    # ------------------------------------------------- #
    #  -- [1-1] example 1 -- #
    rng      = np.random.default_rng(100)
    mu       = np.log( (1.0, 1.0) )
    theta    = np.deg2rad( 30.0 )
    rmat     = np.array( [ [ np.cos(theta), -np.sin(theta)],
                           [ np.sin(theta),  np.cos(theta)] ] )
    sxlog    = 0.30
    sylog    = 0.15
    Det      = np.diag( [ sxlog**2, sylog**2])
    sigma    = rmat @ Det @ rmat.T
    zv       = rng.multivariate_normal( mu, sigma, size=500, )
    xy       = np.exp( zv )
    df       = pd.DataFrame( {"X": xy[:,0], "Y": xy[:,1]} )

    #  -- [1-2] example 2 -- #
    # xg     = np.linspace( 0.5, 10.0, 101 )
    # y_true = 10.0 * ( 1.0 - (xg/10.0)**2 )
    # noise  = np.random.normal( loc=0.0, scale=2.0, size=len(xg) )
    # yg     = y_true + noise + np.random.rand(len(xg))*5.0
    # yg    += np.random.choice( [0, 10, -5], size=len(xg), p=[0.7, 0.2, 0.1] )
    # df     = pd.DataFrame( { "X":xg, "Y":yg } )

    # ------------------------------------------------- #
    # --- [2] optimization test                     --- #
    # ------------------------------------------------- #
    pareto = get__paretoFrontier( df=df, xkey="X", ykey="Y", \
                                  opt_method="platypus", fit_method="spline_under_ristriction", \
                                  directions=[ "max", "max" ], \
                                  monotone="decreasing", concave_convex="convex", \
                                  elbow_method="distance-curvature",  \
                                  outFile = "save.json"
                                 )
    xP,yP  = np.array( pareto["xPareto"] ), np.array( pareto["yPareto"] )
    xF,yF  = np.array( pareto["xFront"]  ), np.array( pareto["yFront"]  ) 
    xE,yE  = np.array([pareto["xElbow"]] ), np.array([pareto["yElbow"]] )
    
    # ------------------------------------------------- #
    # --- [3] plot results                          --- #
    # ------------------------------------------------- #
    import nk_toolkit.plot.load__config   as lcf
    import nk_toolkit.plot.gplot1D        as gp1
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [4.5,4.5],
        "figure.pngFile"     : "pareto_test.png", 
        "figure.position"    : [ 0.16, 0.16, 0.94, 0.94 ],
        "ax1.x.range"        : { "auto":False, "min": 0.0, "max":3.0, "num":4 },
        "ax1.y.range"        : { "auto":False, "min": 0.0, "max":3.0, "num":4 },
        "ax1.x.label"        : "x",
        "ax1.y.label"        : "y",
        "plot.marker"        : "o",
        "plot.linestyle"     : "none", 
        "plot.markersize"    : 3.0,
        "legend.fontsize"    : 9.0, 
    }
    config = { **config, **config_ }
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=df["X"] , yAxis=df["Y"], label="Data"   )
    fig.add__plot( xAxis=xP      , yAxis=yP     , label="Pareto" )
    fig.add__plot( xAxis=xF      , yAxis=yF     , label="Front", marker="none", linestyle="--" )
    fig.add__plot( xAxis=xE      , yAxis=yE     , label="Elbow", marker="*", markersize=6.0 )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()
