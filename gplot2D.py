import os, sys, math
import numpy                   as np
import matplotlib.pyplot       as plt
import matplotlib.tri          as mtr
import matplotlib.ticker       as tic
import scipy.interpolate       as itp
import nk_toolkit.load__config as lcf

# ========================================================= #
# ===  2次元 カラーマップ描画用クラス                   === #
# ========================================================= #

class gplot2D:
    
    # ------------------------------------------------- #
    # --- クラス初期化用ルーチン                    --- #
    # ------------------------------------------------- #
    def __init__( self, \
                  xAxis   = None, yAxis  = None, \
                  cMap    = None, Cntr   = None, vect    = None, \
                  pngFile = None, config = None, cmpmode = None ):
        
        # ------------------------------------------------- #
        # --- 引数の引き渡し                            --- #
        # ------------------------------------------------- #
        self.xAxis, self.yAxis  = xAxis, yAxis
        self.cMap , self.Cntr   =  cMap, Cntr
        self.vect , self.config =  vect, config
        
        # ------------------------------------------------- #
        # --- コンフィグの設定                          --- #
        # ------------------------------------------------- #
        if ( self.config  is     None ): self.config = lcf.load__config()
        if ( pngFile      is not None ): self.config["figure.pngFile"] = pngFile
        if ( cmpmode      is not None ): self.config["cmp.cmpmode"]    = cmpmode
        self.configure__rcParams()

        # ------------------------------------------------- #
        # --- x軸 - y軸の作成                           --- #
        # ------------------------------------------------- #
        #  -- もし，引数に x,y 変数が渡されてない場合，インデックスで代用不可 -- #
        if ( ( self.xAxis is None ) and ( self.cMap is not None ) ):
            self.xAxis = np.arange( ( self.cMap.shape[0] ) )
        if ( ( self.yAxis is None ) and ( self.cMap is not None ) ):
            self.yAxis = np.arange( ( self.cMap.shape[1] ) )

        # ------------------------------------------------- #
        # --- レベルの設定  ( カラー / コンター )       --- #
        # ------------------------------------------------- #
        self.cmpLevels = np.linspace( self.config["cmp.level"]["min"], \
                                      self.config["cmp.level"]["max"], \
                                      self.config["cmp.level"]["num"]  )
        self.cntLevels = np.linspace( self.config["cnt.level"]["min"], \
                                      self.config["cnt.level"]["max"], \
                                      self.config["cnt.level"]["num"]  )
        
        # ------------------------------------------------- #
        # --- 描画領域の作成                            --- #
        # ------------------------------------------------- #
        #  -- 描画領域                                  --  #
        cmppos     = self.config["figure.position"]
        self.fig   = plt.figure( figsize=self.config["figure.size"] )
        self.ax1   = self.fig.add_axes( [ cmppos[0], cmppos[1], \
                                          cmppos[2]-cmppos[0], cmppos[3]-cmppos[1] ] )
        self.set__axis()
        self.set__grid()
        
        # ------------------------------------------------- #
        # --- 速攻描画                                  --- #
        # ------------------------------------------------- #
        instantOut = False
        #  -- もし cMap が渡されていたら，即，描く      --  #
        if ( self.cMap is not None ):
            self.add__cMap( xAxis     = self.xAxis, yAxis   = self.yAxis,     \
                            cMap      = self.cMap , levels  = self.cmpLevels  )
            if ( self.config["clb.sw"]      ):
                self.set__colorBar()
            if ( self.config["cmp.point.sw"] ):
                self.add__point( xAxis=self.xAxis, yAxis=self.yAxis )
            instantOut = True
        #  -- もし Cntrが渡されていたら，即，描く       --  #
        if ( self.Cntr is not None ):
            self.add__contour( xAxis   = self.xAxis, yAxis   = self.yAxis,     \
                               Cntr    = self.Cntr,  levels  = self.cntLevels  )
            if ( self.config["cnt.separatrix.sw"] ): self.add__separatrix()
            instantOut = True
        # -- もし xvec, yvec が渡されていたら，即，描く --  #
        #  -- revision need -- #
        if ( self.vect is not None ):
            self.add__vector ( vect=self.vect )
            instantOut = True
        # -- もし 何かを描いてたら，出力する．          --  #
        if ( instantOut ):
            self.save__figure( pngFile=self.config["figure.pngFile"] )

            
    # ========================================================= #
    # === カラーマップ 追加 ルーチン  ( add__cMap )         === #
    # ========================================================= #
    def add__cMap( self, xAxis=None, yAxis=None, cMap=None, \
                   levels=None, alpha=None, cmpmode=None ):
        
        # ------------------------------------------------- #
        # --- 引数情報 更新                             --- #
        # ------------------------------------------------- #
        self.xAxis, self.yAxis, self.cMap = xAxis, yAxis, cMap
        if ( levels  is not None ): self.cmpLevels             = levels
        if ( alpha   is not None ): self.config["cmp.alpha"]   = alpha
        if ( cmpmode is not None ): self.config["cmp.cmpmode"] = cmpmode
        
        # ------------------------------------------------- #
        # --- コンター情報を設定する                    --- #
        # ------------------------------------------------- #
        if ( self.config["cmp.level"]["auto"] ):
            self.set__cmpLevels()
        else:
            self.set__cmpLevels( levels=self.cmpLevels )
            
        # ------------------------------------------------- #
        # --- 軸情報整形 : 1次元軸 を 各点情報へ変換    --- #
        # ------------------------------------------------- #
        xAxis_, yAxis_ = np.copy( xAxis ), np.copy( yAxis )
        
        # ------------------------------------------------- #
        # --- カラーマップを作図                        --- #
        # ------------------------------------------------- #
        eps = 1.e-10 * abs( self.cmpLevels[-1] - self.cmpLevels[0] )
        self.cMap[ np.where( self.cMap < float( self.cmpLevels[ 0] ) ) ] = self.cmpLevels[ 0] + eps
        self.cMap[ np.where( self.cMap > float( self.cmpLevels[-1] ) ) ] = self.cmpLevels[-1] - eps
        # -- transparent for [ lowerbound , upperbound ] -- #
        if ( self.config["cmp.transparent"] is not None ):
            if ( self.config["cmp.transparent"][0] is not None ):
                index = np.where( self.cMap < float( self.config["cmp.transparent"][0] ) )
                self.cMap[ index ] = np.nan
            if ( self.config["cmp.transparent"][1] is not None ):
                index = np.where( self.cMap > float( self.config["cmp.transparent"][1] ) )
                self.cMap[ index ] = np.nan
        if   ( self.config["cmp.cmpmode"].lower() in [ "pcolor", "pcolormesh" ] ):
            self.cImage = self.ax1.pcolormesh( xAxis_, yAxis_, self.cMap, \
                                               alpha =self.config["cmp.alpha"], \
                                               cmap  =self.config["cmp.colortable"], \
                                               zorder=0 )
        elif ( self.config["cmp.cmpmode"].lower() in [ "tricontourf" ] ):
            triangulated = mtr.Triangulation( xAxis_, yAxis_ )
            self.cImage = self.ax1.tricontourf( triangulated, self.cMap, self.cmpLevels, \
                                                alpha =self.config["cmp.alpha"], \
                                                cmap  =self.config["cmp.colortable"], \
                                                zorder=0, extend="both" )
        elif ( self.config["cmp.cmpmode"].lower() in [ "contourf" ] ):
            xAxis_, yAxis_ = np.reshape( xAxis_, cMap.shape ), np.reshape( yAxis_, cMap.shape )
            self.cImage = self.ax1.contourf( xAxis_, yAxis_, self.cMap, self.cmpLevels, \
                                             alpha =self.config["cmp.alpha"], \
                                             cmap  =self.config["cmp.colortable"], \
                                             zorder=0, extend="both" )
        else:
            sys.exit( "[add__cmap] cmp.cmpmode == ??? [tricontourf, pcolormesh, contourf]" )
            
        # ------------------------------------------------- #
        # --- 軸調整 / 最大 / 最小 表示                 --- #
        # ------------------------------------------------- #
        print( "[gplot2D] :: size :: x, y, z    =  "\
               .format( xAxis_.shape, yAxis_.shape, self.cMap.shape ) )
        print( "[gplot2D] :: ( min(x), max(x) ) = ( {0}, {1} ) "\
               .format( np.min( self.xAxis ), np.max( self.xAxis ) ) )
        print( "[gplot2D] :: ( min(y), max(y) ) = ( {0}, {1} ) "\
               .format( np.min( self.yAxis ), np.max( self.yAxis ) ) )
        print( "[gplot2D] :: ( min(z), max(z) ) = ( {0}, {1} ) "\
               .format( np.min( self.cMap  ), np.max( self.cMap  ) ) )
        self.set__axis()

        
    # ========================================================= #
    # === 等高線 追加 ルーチン  ( add__contour )            === #
    # ========================================================= #
    def add__contour( self, xAxis=None, yAxis=None, Cntr=None, levels=None ):
        # ------------------------------------------------- #
        # --- 引数情報 更新                             --- #
        # ------------------------------------------------- #
        if ( xAxis  is not None ): self.xAxis = xAxis
        if ( yAxis  is not None ): self.yAxis = yAxis
        if ( levels is not None ): self.cntLevels = levels
        if ( Cntr   is None     ):
            sys.exit( "[add__contour] Cntr == ???" )
        else:
            self.Cntr = Cntr
        # ------------------------------------------------- #
        # --- コンター情報を設定する                    --- #
        # ------------------------------------------------- #
        if ( self.config["cnt.level"]["auto"] ):
            self.set__cntLevels()
        else:
            self.set__cntLevels( levels=self.cntLevels )
        # ------------------------------------------------- #
        # --- 軸情報整形 : 1次元軸 を 各点情報へ変換    --- #
        # ------------------------------------------------- #
        xAxis_, yAxis_ = np.copy( xAxis ), np.copy( yAxis )
        # ------------------------------------------------- #
        # --- カラーマップを作成                        --- #
        # ------------------------------------------------- #
        eps = 1.e-10 * abs( self.cntLevels[-1] - self.cntLevels[0] )
        self.Cntr[ np.where( self.Cntr < float( self.cntLevels[ 0] ) ) ] = self.cntLevels[ 0] + eps
        self.Cntr[ np.where( self.Cntr > float( self.cntLevels[-1] ) ) ] = self.cntLevels[-1] - eps
        triangulated = mtr.Triangulation( xAxis_, yAxis_ )
        # ------------------------------------------------- #
        # --- 等高線をプロット                          --- #
        # ------------------------------------------------- #
        self.cImage = self.ax1.tricontour( triangulated, self.Cntr, self.cntLevels, \
                                           colors     = self.config["cnt.color"], \
                                           linewidths = self.config["cnt.linewidth"], \
                                           zorder=1 )
        if ( self.config["cnt.clabel.sw"] ):
            self.ax1.clabel( self.cImage, fontsize=self.config["cnt.clabel.fontsize"] )
        self.set__axis()

        
    # ========================================================= #
    # ===   ベクトル 追加  ルーチン                         === #
    # ========================================================= #
    def add__vector( self, vect=None, ):

        # -- vect :: [ nData, 4 ]   -- #
        #         :: [ x, y, u, v ]
        x_, y_, u_, v_ = 0, 1, 2, 3
        # ------------------------------------------------- #
        # --- [1] 引数チェック                          --- #
        # ------------------------------------------------- #
        if ( vect  is None ): sys.exit("[add__vector] vect  == ???")
        print( vect.shape )
        
        # ------------------------------------------------- #
        # --- [2] set datarange / down sampling         --- #
        # ------------------------------------------------- #
        if ( self.config["vec.x.range"]["auto"] ):
            config["vec.x.range"]["min"] = np.min( self.vect[:,x_] )
            config["vec.x.range"]["max"] = np.max( self.vect[:,x_] )
        if ( self.config["vec.y.range"]["auto"] ):
            config["vec.y.range"]["min"] = np.min( self.vect[:,y_] )
            config["vec.y.range"]["max"] = np.max( self.vect[:,y_] )
        xa     = np.linspace( self.config["vec.x.range"]["min"], \
                              self.config["vec.x.range"]["max"], \
                              self.config["vec.x.range"]["num"]  )
        ya     = np.linspace( self.config["vec.y.range"]["min"], \
                              self.config["vec.y.range"]["max"], \
                              self.config["vec.x.range"]["num"]  )
        xg, yg = np.meshgrid( xa, ya )
        pAxis  = np.copy( vect[:, x_:y_+1] )
        uxIntp = itp.griddata( pAxis, vect[:,u_], (xg,yg), \
                               method=self.config["vec.interpolation"] )
        vyIntp = itp.griddata( pAxis, vect[:,v_], (xg,yg), \
                               method=self.config["vec.interpolation"] )
        
        # ------------------------------------------------- #
        # --- プロット 設定                             --- #
        # ------------------------------------------------- #
        if ( self.config["vec.scale.auto"] ):
            maxLength = np.max( np.sqrt( uxIntp**2 + vyIntp**2 ) )
            self.config["vec.scale"] = self.config["vec.scale.ref"] / maxLength
            print( self.config["vec.scale"] )
            
        # ------------------------------------------------- #
        # -- ベクトルプロット                            -- #
        # ------------------------------------------------- #
        self.ax1.quiver( xg, yg, uxIntp, vyIntp, angles='uv', scale_units='xy', \
                         color=self.config["vec.color"], scale=self.config["vec.scale"], \
                         width=self.config["vec.width"], pivot=self.config["vec.pivot"], \
                         headwidth =self.config["vec.head.width"], \
                         headlength=self.config["vec.head.length"] )


    # ========================================================= #
    # ===  点 追加                                          === #
    # ========================================================= #
    def add__point( self, xAxis=None, yAxis=None, color=None, marker=None ):
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( xAxis      is None ): xAxis      = 0
        if ( yAxis      is None ): yAxis      = 0
        if ( color      is None ): color      = self.config["cmp.point.color"]
        if ( marker     is None ): marker     = self.config["cmp.point.marker"]
        # ------------------------------------------------- #
        # --- 点 描画                                   --- #
        # ------------------------------------------------- #
        self.ax1.plot( xAxis, yAxis, marker=marker, color=color, \
                       markersize     =self.config["cmp.point.size"], \
                       markeredgewidth=self.config["cmp.point.width"], \
                       linewidth      =0.0 )


    # ========================================================= #
    # ===  プロット 追加                                    === #
    # ========================================================= #
    def add__plot( self, xAxis=None, yAxis=None, label=None, color=None, \
                   linestyle=None, linewidth=None, marker=None, alpha=0.95 ):
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( yAxis     is None ): yAxis     = self.yAxis
        if ( xAxis     is None ): xAxis     = self.xAxis
        if ( yAxis     is None ): sys.exit( " [USAGE] add__plot( xAxis=xAxis, yAxis=yAxis )")
        if ( xAxis     is None ): xAxis     = np.arange( yAxis.size ) # - xはyサイズで代用 - #
        if ( label     is None ): label     = ' '
        if ( linewidth is None ): linewidth = self.config["plot.linewidth"]
        if ( marker    is None ): marker    = self.config["plot.marker"]
        if ( color     is None ): color     = self.config["plot.color"]
        if ( alpha     is None ): alpha     = self.config["plot.alpha"]
        # ------------------------------------------------- #
        # --- プロット                                  --- #
        # ------------------------------------------------- #
        self.ax1.plot( xAxis, yAxis, \
                       alpha =alpha,  marker=marker, \
                       label =label, linewidth=linewidth, \
                       color =color, linestyle=linestyle, )
    

    # ========================================================= #
    # ===  セパラトリクス 描画                              === #
    # ========================================================= #
    def add__separatrix( self, mask=None, xAxis=None, yAxis=None, separatrix=None ):
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( xAxis      is None ): xAxis      = self.xAxis
        if ( yAxis      is None ): yAxis      = self.yAxis
        if ( mask       is None ): mask       = self.Cntr / np.max( self.Cntr )
        if ( separatrix is None ): separatrix = self.config["cnt.separatrix.value"]
        # ------------------------------------------------- #
        #  -- レベル 作成                               --  #
        # ------------------------------------------------- #
        if ( type( separatrix ) is not list or type( separatrix ) is not np.ndarray ):
            sepLevels = [ separatrix ]
        # ------------------------------------------------- #
        # --- 軸情報整形 : 1次元軸 を 各点情報へ変換    --- #
        # ------------------------------------------------- #
        if ( ( xAxis.ndim == 1 ) and ( yAxis.ndim == 1 ) ):
            xAxis_, yAxis_ = np.meshgrid( xAxis, yAxis, indexing='ij' )
        else:
            xAxis_, yAxis_ = xAxis, yAxis
        # ------------------------------------------------- #
        # --- セパラトリクス 描画                       --- #
        # ------------------------------------------------- #
        self.ax1.contour( xAxis_, yAxis_, mask, sepLevels, \
                          color     = self.config["cnt.separatrix.color"], \
                          linewidth = self.config["cnt.separatrix.linewidth"]  )

        
    # ========================================================= #
    # === 軸設定用ルーチン                                  === #
    # ========================================================= #
    def set__axis( self, xRange=None, yRange=None ):

        # ------------------------------------------------- #
        # --- 自動レンジ調整   ( 優先順位 2 )           --- #
        # ------------------------------------------------- #
        #  -- オートレンジ (x)                          --  #
        if ( ( self.config["ax1.x.range"]["auto"] ) and ( self.xAxis is not None ) ):
            vMin, vMax  = np.min( self.xAxis ), np.max( self.xAxis )
            ret         = self.auto__griding( vMin =vMin, vMax=vMax, \
                                              nGrid=self.config["ax1.x.range"]["num"] )
            self.config["ax1.x.range"]["min"] = ret[0]
            self.config["ax1.x.range"]["max"] = ret[1]
        #  -- オートレンジ (y)                          --  #
        if ( ( self.config["ax1.y.range"]["auto"] ) and ( self.yAxis is not None ) ):
            vMin, vMax  = np.min( self.yAxis ), np.max( self.yAxis )
            ret         = self.auto__griding( vMin =vMin, vMax=vMax, \
                                              nGrid=self.config["ax1.y.range"]["num"] )
            self.config["ax1.y.range"]["min"] = ret[0]
            self.config["ax1.y.range"]["max"] = ret[1]

        # ------------------------------------------------- #
        # --- 軸範囲 直接設定  ( 優先順位 1 )           --- #
        # ------------------------------------------------- #
        if ( xRange is not None ):
            self.config["ax1.x.range"]["min"] = xRange[0]
            self.config["ax1.x.range"]["max"] = xRange[1]
        if ( yRange is not None ):
            self.config["ax1.y.range"]["min"] = yRange[0]
            self.config["ax1.y.range"]["max"] = yRange[1]
        self.ax1.set_xlim( self.config["ax1.x.range"]["min"],
                           self.config["ax1.x.range"]["max"] )
        self.ax1.set_ylim( self.config["ax1.y.range"]["min"],
                           self.config["ax1.y.range"]["max"] )

        # ------------------------------------------------- #
        # --- 軸タイトル 設定                           --- #
        # ------------------------------------------------- #
        self.ax1.set_xlabel( self.config["ax1.x.label"], \
                             fontsize=self.config["ax1.x.label.fontsize"] )
        self.ax1.set_ylabel( self.config["ax1.y.label"], \
                             fontsize=self.config["ax1.y.label.fontsize"] )

        # ------------------------------------------------- #
        # --- 目盛を調整する                            --- #
        # ------------------------------------------------- #
        self.set__ticks()


    # ========================================================= #
    # ===  軸目盛 設定 ルーチン                             === #
    # ========================================================= #
    def set__ticks( self ):

        # ------------------------------------------------- #
        # --- 軸目盛 自動調整                           --- #
        # ------------------------------------------------- #
        #  -- 軸目盛 整数設定                           --  #
        xtick_dtype     = np.int32 if ( self.config["ax1.x.major.integer"] ) else np.float64
        ytick_dtype     = np.int32 if ( self.config["ax1.y.major.integer"] ) else np.float64
        #  -- 軸目盛 自動調整 (x)                       --  #
        if ( self.config["ax1.x.major.auto"] ):
            xMin, xMax  = self.ax1.get_xlim()
            self.xticks = np.linspace( xMin, xMax, self.config["ax1.x.range"]["num"], \
                                       dtype=xtick_dtype  )
        else:
            self.xticks = np.array( self.config["ax1.x.major.ticks"], dtype=xtick_dtype )
        #  -- 軸目盛 自動調整 (y)                       --  #
        if ( self.config["ax1.y.major.auto"] ):
            yMin, yMax  = self.ax1.get_ylim()
            self.yticks = np.linspace( yMin, yMax, self.config["ax1.y.range"]["num"], \
                                       dtype=ytick_dtype  )
        else:
            self.yticks = np.array( self.config["ax1.y.major.ticks"], dtype=ytick_dtype )
        #  -- Minor 軸目盛                              --  #
        if ( self.config["ax1.x.minor.sw"] is False ): self.config["ax1.x.minor.nticks"] = 1
        if ( self.config["ax1.y.minor.sw"] is False ): self.config["ax1.y.minor.nticks"] = 1
        self.ax1.xaxis.set_minor_locator( tic.AutoMinorLocator( self.config["ax1.x.minor.nticks"] ) )
        self.ax1.yaxis.set_minor_locator( tic.AutoMinorLocator( self.config["ax1.y.minor.nticks"] ) )
        #  -- 軸目盛 調整結果 反映                      --  #
        self.ax1.set_xticks( self.xticks )
        self.ax1.set_yticks( self.yticks )
        # ------------------------------------------------- #
        # --- 軸目盛 スタイル                           --- #
        # ------------------------------------------------- #
        #  -- 対数表示 ( x,y )                          --  #
        if ( self.config["ax1.x.log"] ):
            self.ax1.set_xscale("log")
            if ( self.config["ax1.x.major.auto"] ):
                pass
            else:
                self.ax1.set_xticks( self.config["ax1.x.major.ticks"] )
        if ( self.config["ax1.y.log"] ):
            self.ax1.set_yscale("log")
            if ( self.config["ax1.y.major.auto"] ):
                pass
            else:
                self.ax1.set_yticks( self.config["ax1.y.major.ticks"] )
        #  -- 軸スタイル (x)                            --  #
        self.ax1.tick_params( axis  ="x", labelsize=self.config["ax1.x.major.fontsize"], \
                              length=self.config["ax1.x.major.length"], \
                              width =self.config["ax1.x.major.width"])
        #  -- 軸スタイル (y)                            --  #
        self.ax1.tick_params( axis  ="y", labelsize=self.config["ax1.y.major.fontsize"], \
                              length=self.config["ax1.y.major.length"], \
                              width =self.config["ax1.y.major.width"])
        # ------------------------------------------------- #
        # --- 軸目盛  オフ                              --- #
        # ------------------------------------------------- #
        if ( self.config["ax1.x.major.noLabel"] ):
            self.ax1.set_xticklabels( ['' for i in self.ax1.get_xaxis().get_ticklocs()])
        if ( self.config["ax1.y.major.noLabel"] ):
            self.ax1.set_yticklabels( ['' for i in self.ax1.get_yaxis().get_ticklocs()])



        # # ------------------------------------------------- #
        # # --- 軸目盛 設定                               --- #
        # # ------------------------------------------------- #
        # #  -- 整数  軸目盛り                            --  #
        # xtick_dtype = np.int32 if ( self.config["ax1.x.major.integer"] ) else np.float64
        # ytick_dtype = np.int32 if ( self.config["ax1.y.major.integer"] ) else np.float64
        # #  -- 自動 / 手動 軸目盛り (x)                  --  #
        # if ( self.config["cmp_xAutoTicks"] ):
        #     xMin, xMax = self.ax1.get_xlim()
        #     self.ax1.set_xticks( np.linspace( xMin, xMax, self.config["ax1.x.major.nticks"], \
        #                                       dtype=xtick_dtype ) )
        # else:
        #     self.ax1.set_xticks( np.array(self.config["ax1.x.major.ticks"],dtype=xtick_dtype) )
        # #  -- 自動 / 手動 軸目盛り (y)                  --  #
        # if ( self.config["cmp_yAutoTicks"] ):
        #     yMin, yMax            = self.ax1.get_ylim()
        #     self.ax1.set_yticks( np.linspace( yMin, yMax, self.config["ax1.y.major.nticks"], \
        #                                       dtype=ytick_dtype ) )
        # else:
        #     self.ax1.set_yticks( np.array(self.config["ax1.y.major.nticks"],dtype=ytick_dtype) )
            
        # # ------------------------------------------------- #
        # # --- 目盛 スタイル 設定                        --- #
        # # ------------------------------------------------- #
        # self.ax1.tick_params( axis  ="x", labelsize=self.config["ax1.x.major.fontsize"], \
        #                       length=self.config["ax1.x.major.length"], \
        #                       width =self.config["ax1.x.major.width" ]  )
        # self.ax1.tick_params( axis  ="y", labelsize=self.config["ax1.y.major.fontsize"], \
        #                       length=self.config["ax1.y.major.length"], \
        #                       width =self.config["ax1.y.major.width" ]  )
        
        # #  -- Minor 軸目盛                              --  #
        # if ( self.config["ax1.x.minor.sw"] is False ): self.config["ax1.x.minor.nticks"] = 1
        # if ( self.config["ax1.y.minor.sw"] is False ): self.config["ax1.y.minor.nticks"] = 1
        # self.ax1.xaxis.set_minor_locator( tic.AutoMinorLocator( self.config["ax1.x.minor.nticks"] ) )
        # self.ax1.yaxis.set_minor_locator( tic.AutoMinorLocator( self.config["ax1.y.minor.nticks"] ) )
        # # ------------------------------------------------- #
        # # --- 軸目盛 無し                               --- #
        # # ------------------------------------------------- #
        # if ( self.config["ax1.x.major.off"] ): self.ax1.get_xaxis().set_ticks([])
        # if ( self.config["ax1.y.major.off"] ): self.ax1.get_yaxis().set_ticks([])
        # # ------------------------------------------------- #
        # # --- 軸目盛 ラベル 無し                        --- #
        # # ------------------------------------------------- #
        # if ( self.config["ax1.x.major.noLabel"] ):
        #     self.ax1.set_xticklabels( ['' for i in self.ax1.get_xaxis().get_ticklocs()])
        # if ( self.config["ax1.y.major.noLabel"] ):
        #     self.ax1.set_yticklabels( ['' for i in self.ax1.get_yaxis().get_ticklocs()])

        
    # ========================================================= #
    # ===  カラー レベル設定                                === #
    # ========================================================= #
    def set__cmpLevels( self, levels=None, nLevels=None ):
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( nLevels is None ): nLevels = self.config["cmp.level"]["num"]
        if (  levels is None ):
            minVal, maxVal  = np.min( self.cMap ), np.max( self.cMap )
            eps             = 1.0e-12
            # -- レベルが一定値である例外処理 -- #
            if ( abs( maxVal - minVal ) < eps ):
                if ( abs( minVal ) > eps ):
                    minVal, maxVal =  0.0, 2.0*maxVal
                else:
                    minVal, maxVal = -1.0, 1.0
            levels = np.linspace( minVal, maxVal, nLevels )
        # ------------------------------------------------- #
        # --- cmpLevels を 設定                         --- #
        # ------------------------------------------------- #
        self.cmpLevels  = levels

        
    # ========================================================= #
    # ===  コンター レベル設定                              === #
    # ========================================================= #
    def set__cntLevels( self, levels=None, nLevels=None ):
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( nLevels is None ): nLevels = self.config["cnt.level"]["num"]
        if (  levels is None ):
            minVal, maxVal  = np.min( self.Cntr ), np.max( self.Cntr )
            levels          = np.linspace( minVal, maxVal, nLevels )
        # ------------------------------------------------- #
        # --- cntLevels を 設定                         --- #
        # ------------------------------------------------- #
        self.cntLevels  = levels

        
    # ========================================================= #
    # ===  カラーバー 描画 ルーチン                         === #
    # ========================================================= #
    def set__colorBar( self ):
        # ------------------------------------------------- #
        # --- 準備                                      --- #
        # ------------------------------------------------- #
        #  -- color bar の 作成                         --  #
        clbdata         = np.array( [ np.copy( self.cmpLevels ), np.copy( self.cmpLevels ) ] )
        lbrt            = self.config["clb.position"]
        clbax           = self.fig.add_axes( [lbrt[0],lbrt[1],lbrt[2]-lbrt[0],lbrt[3]-lbrt[1]] )
        
        # ------------------------------------------------- #
        # --- 横向き カラーバーの描画                   --- #
        # ------------------------------------------------- #        
        if ( self.config["clb.orientation"].lower() in [ "h", "horizontal" ] ):
            clbax.set_xlim( self.cmpLevels[0] , self.cmpLevels[-1] )
            clbax.set_ylim( [0.0, 1.0] )
            clb_tickLabel = np.linspace( self.cmpLevels[0], self.cmpLevels[-1], \
                                         self.config["clb.x.major.nticks"] )
            #  -- color bar の 軸目盛  設定                 --  #
            clbax.tick_params( labelsize=self.config["clb.fontsize"], \
                               length=self.config["clb.x.major.length"], \
                               width=self.config["clb.x.major.width" ]  )
            clbax.xaxis.set_minor_locator( tic.AutoMinorLocator( self.config["clb.x.minor.nticks"] ) )
            clbax.get_xaxis().set_ticks( clb_tickLabel )
            clbax.get_yaxis().set_ticks([])
            self.myCbl  = clbax.contourf( self.cmpLevels, [0.0,1.0], clbdata, \
                                          self.cmpLevels, zorder=0, \
                                          cmap = self.config["cmp.colortable"] )
            
        # ------------------------------------------------- #
        # --- 縦向き カラーバーの描画                   --- #
        # ------------------------------------------------- #        
        if ( self.config["clb.orientation"].lower() in [ "v", "vertical" ] ):
            clbax.set_xlim( [0.0, 1.0] )
            clbax.set_ylim( self.cmpLevels[0], self.cmpLevels[-1] )
            clbax.get_xaxis().set_ticks([])
            clb_tickLabel = np.linspace( self.cmpLevels[0], self.cmpLevels[-1], \
                                         self.config["clb.y.major.nticks"] )
            clbax.tick_params( labelsize=self.config["clb.fontsize"], \
                               length=self.config["clb.y.major.length"], \
                               width=self.config["clb.y.major.width" ]  )
            clbax.yaxis.set_minor_locator( tic.AutoMinorLocator( self.config["clb.y.minor.nticks"] ) )
            clbax.get_yaxis().set_ticks( clb_tickLabel )
            clbax.yaxis.tick_right()
            self.myCbl  = clbax.contourf( [0.0,1.0], self.cmpLevels, np.transpose( clbdata ), \
                                          self.cmpLevels, zorder=0, \
                                          cmap = self.config["cmp.colortable"] )
            
        # ------------------------------------------------- #
        # --- カラーバー タイトル 追加                  --- #
        # ------------------------------------------------- #        
        if ( self.config["clb.title"] is not None ):
            textax = self.fig.add_axes( [0,0,1,1] )
            ctitle = r"{}".format( self.config["clb.title"] )
            textax.text( *self.config["clb.title.position"], ctitle, \
                         fontsize=self.config["clb.title.fontsize"] )
            textax.set_axis_off()

            
    # ========================================================= #
    # ===  グリッド / y=0 軸線 追加                         === #
    # ========================================================= #
    def set__grid( self ):
        
        # ------------------------------------------------- #
        # --- グリッド ( 主グリッド :: Major )          --- #
        # ------------------------------------------------- #
        if ( self.config["grid.major.sw"]      ):
            self.ax1.grid( visible  =self.config["grid.major.sw"]       , \
                           which    ='major'                            , \
                           color    =self.config["grid.major.color"]    , \
                           alpha    =self.config["grid.major.alpha"]    , \
                           linestyle=self.config["grid.major.linestyle"], \
                           linewidth=self.config["grid.major.linewidth"]  )
        # ------------------------------------------------- #
        # --- グリッド ( 副グリッド :: Minor )          --- #
        # ------------------------------------------------- #
        if ( self.config["grid.minor.sw"] ):
            self.ax1.grid( visible  =self.config["grid.minor.sw"]   , \
                           which    ='minor'                        , \
                           color    =self.config["grid.minor.color"], \
                           alpha    =self.config["grid.minor.alpha"], \
                           linestyle=self.config["grid.minor.linestyle"], \
                           linewidth=self.config["grid.minor.linewidth"]  )

    # ========================================================= #
    # ===  軸の値 自動算出ルーチン                          === #
    # ========================================================= #
    def auto__griding( self, vMin=None, vMax=None, nGrid=5 ):

        eps = 1.e-8

        # ------------------------------------------------- #
        # --- check Arguments                           --- #
        # ------------------------------------------------- #
        if ( vMax  <  vMin ):
            sys.exit( "[auto__griding] ( vMin,vMax ) == ( {0},{1} ) ??? ".format( vMin, vMax ) )
        if ( nGrid <= 0 ):
            sys.exit( "[auto__griding] nGrid == {0} ??? ".format( nGrid ) )
        if ( vMin == vMax  ):
            return( [ vMin-eps, vMax+eps] )
            
        # ------------------------------------------------- #
        # --- auto grid making                          --- #
        # ------------------------------------------------- #
        minimum_tick = ( vMax - vMin ) / float( nGrid )
        magnitude    = 10**( math.floor( math.log( minimum_tick, 10 ) ) )
        significand  = minimum_tick / magnitude
        if   ( significand > 5    ):
            grid_size = 10 * magnitude
        elif ( significand > 2    ):
            grid_size =  5 * magnitude
        elif ( significand > 1    ):
            grid_size =  2 * magnitude
        else:
            grid_size =  1 * magnitude
        tick_below   = grid_size * math.floor( vMin / grid_size ) 
        tick_above   = grid_size * math.ceil ( vMax / grid_size )
        return( [ tick_below, tick_above, nGrid ] )
        

            
    # ========================================================= #
    # ===  ファイル 保存                                    === #
    # ========================================================= #
    def save__figure( self, pngFile=None, dpi=None, transparent=None, minimal=None ):

        # ------------------------------------------------- #
        # --- 引数設定                                  --- #
        # ------------------------------------------------- #
        if ( pngFile     is None ): pngFile     = self.config["figure.pngFile"]
        if ( dpi         is None ): dpi         = self.config["figure.dpi"]
        if ( transparent is None ): transparent = self.config["figure.transparent"]
        if ( minimal     is None ): minimal     = self.config["figure.minimal"]
        
        # ------------------------------------------------- #
        # --- ファイル ( png ) 出力                     --- #
        # ------------------------------------------------- #
        if ( minimal ):
            # -- 最小プロット -- #
            self.fig.savefig( pngFile, dpi=dpi, bbox_inches='tight', \
                              pad_inches=0, transparent=transparent )
        else:
            # -- 通常プロット -- #
            self.fig.savefig( pngFile, dpi=dpi, pad_inches=0, transparent=transparent )
        print( "[ save__figure() @gplot1D ] output :: {0}".format( pngFile ) )
        # plt.close()
        return()

    # ========================================================= #
    # ===  configure__rcParams                              === #
    # ========================================================= #
    def configure__rcParams( self ):
        
        # ------------------------------------------------- #
        # --- 全体設定                                  --- #
        # ------------------------------------------------- #
        # plt.style.use('seaborn-white')
        plt.rcParams['figure.dpi']             = self.config["figure.dpi"]

        # ------------------------------------------------- #
        # --- 画像 サイズ / 余白 設定                   --- #
        # ------------------------------------------------- #
        # -- 相対座標  --  #
        plt.rcParams['figure.subplot.left']    = 0.0
        plt.rcParams['figure.subplot.bottom']  = 0.0
        plt.rcParams['figure.subplot.right']   = 1.0
        plt.rcParams['figure.subplot.top']     = 1.0
        plt.rcParams['figure.subplot.wspace']  = 0.0
        plt.rcParams['figure.subplot.hspace']  = 0.0
        # -- 余白設定  --  #
        plt.rcParams['axes.xmargin']           = 0
        plt.rcParams['axes.ymargin']           = 0
        
        # ------------------------------------------------- #
        # --- フォント 設定                             --- #
        # ------------------------------------------------- #
        # -- フォント 種類                              --  #
        plt.rcParams['font.family']            = self.config["figure.fontname"]
        plt.rcParams['font.serif']             = self.config["figure.fontname"]
        plt.rcParams['mathtext.fontset']       = self.config["figure.mathfont"]
        # -- other settings                         --  #
        #     :: 'dejavusans', 'cm', 'custom'       ::  #
        #     :: 'stix', 'stixsans', 'dejavuserif'  ::  #
        # --                                        --  #
        # -- 通常 フォント                          --  #
        plt.rcParams['font.size']              = self.config["figure.font.size"]
        # -- 軸タイトル                             --  #
        plt.rcParams['axes.labelsize']         = self.config["figure.font.size"]
        plt.rcParams['axes.labelweight']       = 'regular'
        
        # ------------------------------------------------- #
        # --- 目盛 設定 ( xticks, yticks )              --- #
        # ------------------------------------------------- #
        # -- 目盛線向き :: 内向き('in'), 外向き('out')   -- #
        # --            :: 双方向か('inout')             -- #
        # -- xTicks -- #
        plt.rcParams['xtick.direction']        = 'in'
        plt.rcParams['xtick.bottom']           = True
        plt.rcParams['xtick.top']              = True
        plt.rcParams['xtick.major.size']       = self.config["ax1.x.major.size"]
        plt.rcParams['xtick.major.width']      = self.config["ax1.x.major.width"]
        plt.rcParams['xtick.minor.size']       = self.config["ax1.x.minor.size"]
        plt.rcParams['xtick.minor.width']      = self.config["ax1.x.minor.width"]
        plt.rcParams['xtick.minor.visible']    = self.config["ax1.x.minor.sw"]
        # -- yTicks -- #
        plt.rcParams['ytick.direction']        = 'in'
        plt.rcParams['ytick.left']             = True
        plt.rcParams['ytick.right']            = True
        plt.rcParams['ytick.major.size']       = self.config["ax1.y.major.size"]
        plt.rcParams['ytick.major.width']      = self.config["ax1.y.major.width"]
        plt.rcParams['ytick.minor.visible']    = self.config["ax1.y.minor.sw"]
        plt.rcParams['ytick.minor.size']       = self.config["ax1.y.minor.size"]
        plt.rcParams['ytick.minor.width']      = self.config["ax1.y.minor.width"]
        
        # ------------------------------------------------- #
        # --- プロット線 / 軸 の線の太さ                --- #
        # ------------------------------------------------- #
        plt.rcParams['lines.linewidth']        = self.config["plot.linewidth"]
        plt.rcParams['axes.linewidth']         = self.config["figure.axes.linewidth"]
        return()

    
# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    x_, y_, z_ = 0, 1, 2

    # ------------------------------------------------- #
    # --- [1] preparation                           --- #
    # ------------------------------------------------- #
    os.makedirs( "test/gplot2D/", exist_ok=True )

    import nkUtilities.equiSpaceGrid as esg
    x1MinMaxNum = [ -1.0, 1.0, 21 ]
    x2MinMaxNum = [ -1.0, 1.0, 21 ]
    x3MinMaxNum = [  0.0, 0.0,  1 ]
    coord       = esg.equiSpaceGrid( x1MinMaxNum=x1MinMaxNum, x2MinMaxNum=x2MinMaxNum, \
                                     x3MinMaxNum=x3MinMaxNum, returnType = "point" )
    coord[:,z_] = np.sqrt( coord[:,x_]**2 + coord[:,y_]**2 )
    
    xvec = coord[:,x_]
    yvec = coord[:,y_]
    uvec = coord[:,x_] / coord[:,z_]
    vvec = coord[:,y_] / coord[:,z_]
    vect = np.concatenate( [ xvec[:,np.newaxis], yvec[:,np.newaxis], \
                             uvec[:,np.newaxis], vvec[:,np.newaxis] ], axis=1 )

    # ------------------------------------------------- #
    # --- [2] config settings                       --- #
    # ------------------------------------------------- #
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [4.5,4.5],
        "figure.position"    : [ 0.16, 0.16, 0.84, 0.84 ],
        "ax1.y.normalize"    : 1.0e0, 
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "x",
        "ax1.y.label"        : "y",
        "ax1.x.minor.nticks" : 1,
        "cmp.level"          : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "cmp.colortable"     : "jet",
    }
    config = { **config, **config_ }

    # ------------------------------------------------- #
    # --- [3] generate profile 1                    --- #
    # ------------------------------------------------- #
    gplot2D( xAxis=coord[:,x_], yAxis=coord[:,y_], cMap=coord[:,z_], vect=vect, \
             config=config, pngFile="test/gplot2D/tricontourf.png", cmpmode="tricontourf" )

    # ------------------------------------------------- #
    # --- [4] generate profile 2                    --- #
    # ------------------------------------------------- #
    shape = (21,21,3)
    coord = np.reshape( coord, shape )
    gplot2D( xAxis=coord[:,:,x_], yAxis=coord[:,:,y_], cMap=coord[:,:,z_], \
             config=config, pngFile="test/gplot2D/pcolormesh.png", cmpmode="pcolormesh" )

    
    
