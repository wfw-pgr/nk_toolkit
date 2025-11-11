import os, sys
import pandas as pd
import numpy  as np


# ========================================================= #
# ===  fit__polynomial_5th                              === #
# ========================================================= #
def fit__polynomial_5th( xnews, xvals, yvals ):
    import scipy.optimize as opt
    def __func_polynomial_5th( x, a5, a4, a3, a2, a1, a0 ):
        ret = a5*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0
        return( ret )
    pop, _ = opt.curve_fit( __func_polynomial_5th, xvals, yvals )
    func   = lambda x: __func_polynomial_5th( x, *pop )
    ynews  = func( xnews )
    return( ynews )


# ========================================================= #
# ===  fit__quantile_regression                         === #
# ========================================================= #
def fit__quantile_regression( xnews, xvals, yvals, \
                              tau=0.98, df=4, degree=4, smoother="spline",
                              make_envelope=True, grid_n=400, spline_s=None ):
    # - tau :: quantile point      :: ( def. 0.98 )
    # - df  :: degree of freedom of B-spline  ::
    # - 
    import patsy
    import statsmodels.api   as sm
    import scipy.interpolate as itp
    xvals = np.asarray(xvals).ravel()
    yvals = np.asarray(yvals).ravel()
    xnews = np.asarray(xnews).ravel()
    order = np.argsort( xvals )
    x, y  = xvals[order], yvals[order]
    
    # --- [1] Bスプライン分位回帰 --- #
    Xspl = patsy.dmatrix( f"bs(x, df={df}, degree={degree}, include_intercept=False)",
                          {"x": x}, return_type="dataframe")
    X    = sm.add_constant( Xspl )
    mod  = sm.QuantReg( y, X )
    res  = mod.fit( q=tau, max_iter=2000, p_tol=1e-7 )
        
    # --- [2] 高密度グリッドで予測 → 滑らか化 --- #
    xg   = np.linspace( x.min(), x.max(), grid_n )
    Xg   = sm.add_constant( patsy.dmatrix(
        f"bs(xg, df={df}, degree={degree}, include_intercept=False)",
        {"xg": xg}, return_type="dataframe") )
    yg   = res.predict( Xg )
        
    # --- [3] 上包絡（単調増加に寄せたい有効） --- #
    if ( make_envelope ):
        yg = np.maximum.accumulate( yg )

    # --- [4] 平滑化 --- #
    if   ( smoother == "pchip" ):
        f = itp.PchipInterpolator( xg, yg, extrapolate=True )
        return f(xnews)
    elif ( smoother == "spline" ):
        # 連続C2。s が大きいほど“なだらか”
        if ( spline_s is None ):
            # 自動スケール（経験則）：データスパンと点数から適当に
            rng = x.max() - x.min() + 1e-12
            spline_s = 0.002 * rng * grid_n
            f = itp.UnivariateSpline(xg, yg, s=spline_s, k=3)
        return f(xnews)
    else:
        raise ValueError("smoother must be 'pchip' or 'spline'")



    
# ========================================================= #
# ===  fit__spline_under_ristriction                    === #
# ========================================================= #
def fit__spline_under_ristriction( xnews, xvals, yvals, \
                                   dof=2, n_basis=None, smoothness=0.1, \
                                   monotone=None, concave_convex=None ):
    
    # monotone       :: [ "increasing" , "decreasing" ]
    # concave_convex :: [ "concave", "convex" ]
    import cvxpy             as cp
    import scipy.interpolate as itp
    # ------------------------------------------------- #
    # --- [1] 内部関数                              --- #
    # ------------------------------------------------- #
    def __uniform_knots( xmin, xmax, dof, n_basis ):
        k        = dof
        n_int    = n_basis - k - 1
        if ( n_int > 0 ):
            internal = np.linspace( xmin, xmax, n_int+2 )[1:-1]
        else:
            internal = np.array( [] )
        return( np.r_[[xmin]*(k+1), internal, [xmax]*(k+1)] )
        
    def __bspline_design( x, t, dof ):
        x     = np.asarray(x)
        ncoef = len(t) - dof - 1
        B     = np.empty( (len(x), ncoef) )
        for j in range(ncoef):
            c       = np.zeros(ncoef)
            c[j]    = 1.0
            B[:, j] = itp.BSpline(t, c, dof)(x)
        return( B )
    
    def __diff_mat( n, order=1 ):
        D = np.zeros((n-order, n))
        if   ( order == 1 ):
            for i in range(n-1):
                D[i,i], D[i,i+1] = -1, 1
        elif ( order == 2 ):
            for i in range(n-2):
                D[i,i], D[i,i+1], D[i,i+2] = 1, -2, 1
        else:
            raise ValueError
        return( D )

    # ------------------------------------------------- #
    # --- [2] 変数                                  --- #
    # ------------------------------------------------- #
    xvals_     = np.asarray( xvals, float )
    yvals_     = np.asarray( yvals, float )
    xnews_     = np.asarray( xnews, float )
    xmin, xmax = np.min(xvals_), np.max(xvals_)
    if ( n_basis is None ):
        n_basis = min( max( 12, len( xvals_ )//3 + dof ), 50 )
        
    # ------------------------------------------------- #
    # --- [2] 基底行列                              --- #
    # ------------------------------------------------- #
    t    = __uniform_knots( xmin, xmax, dof, n_basis )
    Bx   = __bspline_design( xvals_, t, dof )
    Bz   = __bspline_design( xnews_, t, dof )
    m    = Bx.shape[1]
    
    # ------------------------------------------------- #
    # --- [3] 差分行列                              --- #
    # ------------------------------------------------- #
    D1   = __diff_mat(m, 1)
    D2   = __diff_mat(m, 2)
    
    # ------------------------------------------------- #
    # --- [4] 制約  ( 単調性・凹凸 )                --- #
    # ------------------------------------------------- #
    c    = cp.Variable(m)
    cons = []
    if ( monotone is None ):
        if ( xvals_[0] > xvals_[-1] ):
            monotone = "decreasing"
        else:
            monotone = "increasing"
        
    #  -- [4-1] 単調増加・減少    -- #
    if   ( monotone == "increasing" ):
        cons += [ D1 @ c >= 0 ]     # f' > 0
    elif ( monotone == "decreasing" ):
        cons += [ D1 @ c <= 0 ]

    #  -- [4-2] 凸・凹 制約       -- #
    if   ( concave_convex == "convex" ):
        cons += [ D2 @ c <= 0 ]   # f'' <= 0 （上に凸）
    elif ( concave_convex == "concave" ):
        cons += [ D2 @ c >= 0 ]   # f'' >= 0 （下に凸）
    elif ( concave_convex is None ):
        pass
    else:
        print( "[fit__spline_under_ristriction] unknown concave_convex: {}".format(concave_convex))
        print( "[fit__spline_under_ristriction] NO concave or convex ristriction [WARNING] " )
        
    # ------------------------------------------------- #
    # --- [5] 目的関数：最小二乗+sm*(曲率の二乗)    --- #
    # ------------------------------------------------- #
    obj  = cp.Minimize( cp.sum_squares( Bx @ c - yvals_ ) + smoothness * cp.sum_squares( D2 @ c ) )
    cp.Problem( obj, cons ).solve( solver="OSQP", verbose=False )
    ynew = ( Bz @ c.value ).ravel()
    return( ynew )
    


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    
    # ------------------------------------------------- #
    # --- [1] make example                          --- #
    # ------------------------------------------------- #
    Np = 21
    xv = np.linspace( 0.0, 1.0, Np )
    nv = np.random.uniform( 0.0, 1.0, (Np,) ) * 0.2
    yv = xv**2 + nv
    yv = - np.sin( xv**2 ) + nv
    
    # ------------------------------------------------- #
    # --- [2] interpolation                         --- #
    # ------------------------------------------------- #
    xn = np.linspace( np.min(xv), np.max(xv), 101 )
    # yf = fit__polynomial_5th( xn, xv, yv )
    yf = fit__spline_under_ristriction( xn, xv, yv, \
                                      monotone="decreasing", concave_convex="concave" )

    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    import nk_toolkit.plot.load__config   as lcf
    import nk_toolkit.plot.gplot1D        as gp1
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [4.5,4.5],
        "figure.pngFile"     : "fitting.png", 
        "figure.position"    : [ 0.16, 0.16, 0.94, 0.94 ],
        "ax1.y.normalize"    : 1.0e0, 
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "x",
        "ax1.y.label"        : "y",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 3.0,
        "legend.fontsize"    : 9.0, 
    }
    config = { **config, **config_ }
        
    fig    = gp1.gplot1D( config=config )
    fig.add__plot( xAxis=xv, yAxis=yv, label="data", linestyle="none", marker="o"    )
    fig.add__plot( xAxis=xn, yAxis=yf, label="fit" , linestyle="-"   , marker="none" )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()

    
