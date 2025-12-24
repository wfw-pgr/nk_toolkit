import os, sys, math
import scipy                        as sp
import numpy                        as np 
import matplotlib.pyplot            as plt
import matplotlib.ticker            as tic
import nk_toolkit.plot.load__config as lcf

# ========================================================= #
# === 1次元プロット描画用クラス                         === #
# ========================================================= #

class gplot1D:
    
    # ------------------------------------------------- #
    # --- クラス初期化用ルーチン                    --- #
    # ------------------------------------------------- #
    def __init__( self, xAxis=None, yAxis=None, label=None, \
                  pngFile=None, config=None, window=False ):

        # ------------------------------------------------- #
        # --- 引数の引き渡し                            --- #
        # ------------------------------------------------- #
        self.config                 = config
        self.xAxis , self.yAxis     = xAxis, yAxis
        self.xticks, self.yticks    = None, None
        self.DataRange              = None
        self.DataRange_ax2          = None
        
        # ------------------------------------------------- #
        # --- コンフィグの設定                          --- #
        # ------------------------------------------------- #
        if ( self.config is     None ): self.config                   = lcf.load__config()
        if ( pngFile     is not None ): self.config["figure.pngFile"] = pngFile
        self.configure__rcParams()
        
        # ------------------------------------------------- #
        # --- 描画領域の作成                            --- #
        # ------------------------------------------------- #
        #  -- 描画領域                                  --  #
        pos      = self.config["figure.position"]
        self.fig = plt.figure( figsize=self.config["figure.size"]  )
        self.ax1 = self.fig.add_axes( [ pos[0], pos[1], pos[2]-pos[0], pos[3]-pos[1] ] )
        self.ax2 = None
        self.set__axis()
        self.set__grid()
        
        # ------------------------------------------------- #
        # --- class 定義時に画像ファイル出力            --- #
        # ------------------------------------------------- #
        instantOut = False
        #  -- yAxis あり -> 即，描く  --  #
        if ( self.yAxis is not None ):
            instantOut = True
            if ( self.xAxis is None ): # -- xAxis なし、インデックスで代用 -- #
                self.xAxis = np.arange( float( self.yAxis.size ) )
            # -- ploting -- #
            self.add__plot( self.xAxis, self.yAxis, label=label )
            self.set__axis()
            if ( self.config["legend.sw"] ): self.set__legend()

        # ------------------------------------------------- #
        # --- カーソルの追加                            --- #
        # ------------------------------------------------- #
        if ( self.config["ax1.cursor.x"] is not None ):
            self.add__cursor( xAxis=self.config["ax1.cursor.x"], axis="ax1" )
        if ( self.config["ax1.cursor.y"] is not None ):
            self.add__cursor( yAxis=self.config["ax1.cursor.y"], axis="ax1" )
        if ( self.config["ax2.cursor.y"] is not None ):
            self.add__cursor( yAxis=self.config["ax2.cursor.y"], axis="ax2" )

        # ------------------------------------------------- #
        # --- 出力 / ディスプレイ                       --- #
        # ------------------------------------------------- #
        #  -- もし 何かを描いてたら，出力する．         --  #
        if ( instantOut ): self.save__figure( pngFile=self.config["figure.pngFile"] )
        #  -- window に表示する．                       --  #
        if ( window     ): self.display__window()

        
    # ========================================================= #
    # ===  プロット 追加                                    === #
    # ========================================================= #
    def add__plot( self, xAxis=None, yAxis=None, label=None, color=None, alpha=None, \
                   linestyle=None, linewidth=None,
                   xRange=None, yRange=None, xlabel=None, ylabel=None, \
                   marker=None, markersize=None, markerwidth=None, kw={} ):
        
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( yAxis       is None ): yAxis       = self.yAxis
        if ( xAxis       is None ): xAxis       = self.xAxis
        if ( yAxis       is None ): sys.exit( " [gplot1D.py] yAxis == ?? " )
        if ( xAxis       is None ): xAxis       = np.arange( yAxis.size ) # - インデックス代用-#
        if ( color       is None ): color       = self.config["plot.color"]
        if ( alpha       is None ): alpha       = self.config["plot.alpha"]
        if ( linestyle   is None ): linestyle   = self.config["plot.linestyle"]
        if ( linewidth   is None ): linewidth   = self.config["plot.linewidth"]
        if ( marker      is None ): marker      = self.config["plot.marker"]
        if ( markersize  is None ): markersize  = self.config["plot.markersize"]
        if ( markerwidth is None ): markerwidth = self.config["plot.markerwidth"]

        # ------------------------------------------------- #
        # --- 軸設定                                    --- #
        # ------------------------------------------------- #
        if ( xlabel is not None ): self.config["ax1.x.label"] = xlabel
        if ( ylabel is not None ): self.config["ax1.y.label"] = ylabel
        self.xAxis, self.yAxis = xAxis, yAxis
        self.update__DataRange( xAxis=xAxis, yAxis=yAxis )
        self.set__axis( xRange=xRange, yRange=yRange )
        xAxis_, yAxis_ = np.copy(xAxis), np.copy( yAxis )
        if ( self.config["ax1.x.normalize"] is not None ):
            xAxis_   = xAxis / self.config["ax1.x.normalize"]
        if ( self.config["ax1.y.normalize"] is not None ):
            yAxis_   = yAxis / self.config["ax1.y.normalize"]
        if ( self.config["plot.colorStack"] is not None ):
            color    = ( self.config["plot.colorStack"] ).pop(0)
            
        # ------------------------------------------------- #
        # --- プロット 追加                             --- #
        # ------------------------------------------------- #
        self.ax1.plot( xAxis_, yAxis_, \
                       color =color , linestyle =linestyle , \
                       label =label , linewidth =linewidth , \
                       marker=marker, markersize=markersize, \
                       markeredgewidth=markerwidth, alpha =alpha, **kw )
        

    # ========================================================= #
    # ===  プロット 追加                                    === #
    # ========================================================= #
    def add__errorbar( self, xAxis=None, yAxis=None, xerr=None, yerr=None, \
                       capsize=None, capthick=None, fmt="none", \
                       color=None, alpha=None, linestyle=None, linewidth=None ):
        
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( yAxis     is None ): yAxis     = self.yAxis
        if ( xAxis     is None ): xAxis     = self.xAxis
        if ( yAxis     is None ): sys.exit( " [gplot1D.py] yAxis == ?? " )
        if ( xAxis     is None ): xAxis     = np.arange( yAxis.size ) # インデックス代用
        if ( color     is None ): color     = self.config["plot.error.color"]
        if ( alpha     is None ): alpha     = self.config["plot.error.alpha"]
        if ( linewidth is None ): linewidth = self.config["plot.error.linewidth"]
        if ( capsize   is None ): capsize   = self.config["plot.error.cap.size"]
        if ( capthick  is None ): capthick  = self.config["plot.error.cap.thick"]
        if ( ( xerr is None ) and ( yerr is None ) ):
            sys.exit("[add__errorbar] xerr=None & yerr=None ")
            
        # ------------------------------------------------- #
        # --- プロット 追加                             --- #
        # ------------------------------------------------- #
        self.ax1.errorbar( xAxis, yAxis, xerr=xerr, yerr=yerr, \
                           capsize=capsize, capthick=capthick, fmt=fmt, alpha=alpha, \
                           ecolor=color, elinewidth=linewidth )


    # ========================================================= #
    # ===  軸 レンジ 自動調整用 ルーチン                    === #
    # ========================================================= #
    def set__axis( self, xRange=None, yRange=None ):

        if ( len( self.config["ax1.x.range"] ) == 2 ):
            self.config["ax1.x.range"] += [ self.config["ax1.x.major.nticks"] ]
        if ( len( self.config["ax1.y.range"] ) == 2 ):
            self.config["ax1.y.range"] += [ self.config["ax1.y.major.nticks"] ]
        
        # ------------------------------------------------- #
        # --- 自動レンジ調整   ( 優先順位 2 )           --- #
        # ------------------------------------------------- #
        #  -- オートレンジ (x)                          --  #
        if ( ( self.config["ax1.x.range"]["auto"] ) and ( self.DataRange is not None ) ):
            ret = self.auto__griding( vMin =self.DataRange[0], vMax=self.DataRange[1], \
                                      nGrid=self.config["ax1.x.range"]["num"] )
            self.config["ax1.x.range"]["min"] = ret[0]
            self.config["ax1.x.range"]["max"] = ret[1]
        #  -- オートレンジ (y)                          --  #
        if ( ( self.config["ax1.y.range"]["auto"] ) and ( self.DataRange is not None ) ):
            ret = self.auto__griding( vMin=self.DataRange[2], vMax=self.DataRange[3], \
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
        self.ax1.set_xlim( float( self.config["ax1.x.range"]["min"] ),
                           float( self.config["ax1.x.range"]["max"] ) )
        self.ax1.set_ylim( float( self.config["ax1.y.range"]["min"] ),
                           float( self.config["ax1.y.range"]["max"] ) )

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
    # ===  軸 レンジ 自動調整用 ルーチン for axis2          === #
    # ========================================================= #
    def set__axis2( self, xRange=None, yRange=None ):

        # ------------------------------------------------- #
        # --- 自動レンジ調整   ( 優先順位 2 )           --- #
        # ------------------------------------------------- #
        #  -- オートレンジ (y)                          --  #
        if ( ( self.config["ax2.y.range"]["auto"] ) and ( self.DataRange_ax2 is not None ) ):
            ret = self.auto__griding( vMin=self.DataRange_ax2[2], vMax=self.DataRange_ax2[3], \
                                      nGrid=self.config["ax2.y.range"]["num"] )
            self.config["ax2.y.range"]["min"] = ret[0]
            self.config["ax2.y.range"]["max"] = ret[1]
            
        # ------------------------------------------------- #
        # --- 軸範囲 直接設定  ( 優先順位 1 )           --- #
        # ------------------------------------------------- #
        if ( yRange is not None ): self.config["ax2.y.range"] = yRange
        self.ax2.set_ylim( float( self.config["ax2.y.range"]["min"] ), \
                           float( self.config["ax2.y.range"]["max"] ), )
        
        # ------------------------------------------------- #
        # --- 軸タイトル 設定                           --- #
        # ------------------------------------------------- #
        self.ax2.set_ylabel( self.config["ax2.y.label"], \
                             fontsize=self.config["ax2.y.label.fontsize"] )
        # ------------------------------------------------- #
        # --- 目盛を調整する                            --- #
        # ------------------------------------------------- #
        self.set__ticks2()

        
    # ========================================================= #
    # ===  軸目盛 設定 ルーチン for axis2                   === #
    # ========================================================= #
    def set__ticks2( self ):

        # ------------------------------------------------- #
        # --- 軸目盛 自動調整                           --- #
        # ------------------------------------------------- #
        #  -- 軸目盛 整数設定                           --  #
        ytick_dtype      = np.int32 if ( self.config["ax2.y.major.integer"] ) else np.float64
        #  -- 軸目盛 自動調整 (y)                       --  #
        if ( self.config["ax2.y.major.auto"] ):
            yMin, yMax   = self.ax2.get_ylim()
            self.yticks2 = np.linspace( yMin, yMax, self.config["ax2.y.range"]["num"], dtype=ytick_dtype  )
        else:
            self.yticks2 = np.array( self.config["ax2.y.major.ticks"], dtype=ytick_dtype )
        #  -- Minor 軸目盛                              --  #
        if ( self.config["ax2.y.minor.sw"] is False ):
            self.config["ax2.y.minor.nticks"] = 1
        self.ax2.yaxis.set_minor_locator( tic.AutoMinorLocator( self.config["ax2.y.minor.nticks"] ) )
        #  -- 軸目盛 調整結果 反映                      --  #
        self.ax2.set_yticks( self.yticks2 )
        # ------------------------------------------------- #
        # --- 軸目盛 スタイル                           --- #
        # ------------------------------------------------- #
        #  -- 対数表示 ( x,y )                          --  #
        if ( self.config["ax2.y.log"] ):
            self.ax2.set_yscale("log")
            if ( self.config["ax2.y.major.auto"] ):
                pass
            else:
                self.ax2.set_yticks( self.config["ax2.y.major.ticks"] )

        #  -- 軸スタイル (y)                            --  #
        self.ax2.tick_params( axis     ="y", \
                              labelsize=self.config["ax2.y.major.fontsize"], \
                              length   =self.config["ax2.y.major.length"  ], \
                              width    =self.config["ax2.y.major.width"   ]  )

        # ------------------------------------------------- #
        # --- 軸目盛  オフ                              --- #
        # ------------------------------------------------- #
        if ( self.config["ax2.y.major.noLabel"] ):
            self.ax2.set_yticklabels( [ "" for i in self.ax2.get_yaxis().get_ticklocs() ] )

        

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
            self.ax1.xaxis.set_major_locator( tic.LogLocator( base=10.0, numticks=10 ) )
            # self.ax1.xaxis.set_minor_locator( tic.LogLocator( base=10.0, numticks=10 ) )
            self.ax1.xaxis.set_minor_locator(
                tic.LogLocator( base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10 ) )
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
        self.ax1.tick_params( axis  ="x", which="minor", \
                              labelsize=self.config["ax1.x.minor.fontsize"], \
                              length=self.config["ax1.x.minor.length"], \
                              width =self.config["ax1.x.minor.width"])
        #  -- 軸スタイル (y)                            --  #
        self.ax1.tick_params( axis  ="y", labelsize=self.config["ax1.y.major.fontsize"], \
                              length=self.config["ax1.y.major.length"], \
                              width =self.config["ax1.y.major.width"])
        self.ax1.tick_params( axis  ="y", which="minor", \
                              labelsize=self.config["ax1.y.minor.fontsize"], \
                              length=self.config["ax1.y.minor.length"], \
                              width =self.config["ax1.y.minor.width"])
        # ------------------------------------------------- #
        # --- 10^X notation                             --- #
        # ------------------------------------------------- #
        if ( self.config["ax1.y.power.sw"] ):
            formatter = tic.ScalarFormatter( useMathText=True )
            formatter.set_powerlimits( tuple(self.config["ax1.y.power.range"]) )
            self.ax1.yaxis.set_major_formatter( formatter )
            
        
        # ------------------------------------------------- #
        # --- 軸目盛  オフ                              --- #
        # ------------------------------------------------- #
        if ( self.config["ax1.x.major.noLabel"] ):
            self.ax1.set_xticklabels( ['' for i in self.ax1.get_xaxis().get_ticklocs()])
        if ( self.config["ax1.y.major.noLabel"] ):
            self.ax1.set_yticklabels( ['' for i in self.ax1.get_yaxis().get_ticklocs()])


    # =================================================== #
    # === データレンジ更新 for 複数プロット自動範囲用 === #
    # =================================================== #
    def update__DataRange( self, xAxis=None, yAxis=None, ax2=False ):
        
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( ( xAxis is None ) or ( yAxis is None ) ):
            sys.exit(  "[ERROR] [@update__DataRange] xAxis or yAxis is None [ERROR]" )

        # ------------------------------------------------- #
        # --- NaN 除去                                  --- #
        # ------------------------------------------------- #
        mask   = np.isfinite( xAxis ) & np.isfinite( yAxis )
        xAxis_ = xAxis[mask].astype( np.float64 )
        yAxis_ = yAxis[mask].astype( np.float64 )

        if ( xAxis_.size == 0 ) or ( yAxis_.size == 0 ):
            print("[WARNING] [@update__DataRange] All data is NaN. Skipping range update.")
            return

        # ------------------------------------------------- #
        # --- ax2 case                                  --- #
        # ------------------------------------------------- #
        if ( ax2 ):
            if ( self.DataRange_ax2 is None ):
                # -- DataRange_ax2 未定義のとき -- #
                self.DataRange_ax2    = np.array( [ np.min( xAxis_ ), np.max( xAxis_ ),
                                                    np.min( yAxis_ ), np.max( yAxis_ ) ] )
            else:
                # -- DataRange_ax2 を更新する -- #
                self.DataRange_ax2 = np.array( [ min( self.DataRange_ax2[0], np.min( xAxis_ ) ), \
                                                 max( self.DataRange_ax2[1], np.max( xAxis_ ) ), \
                                                 min( self.DataRange_ax2[2], np.min( yAxis_ ) ), \
                                                 max( self.DataRange_ax2[3], np.max( yAxis_ ) ), ] )
            return()
        # ------------------------------------------------- #
        # --- update DataRange for ax1                  --- #
        # ------------------------------------------------- #
        if ( self.DataRange is None ):
            # -- DataRange 未定義のとき -- #
            self.DataRange    = np.zeros( (4,) )
            self.DataRange[0] = np.min( xAxis_ )
            self.DataRange[1] = np.max( xAxis_ )
            self.DataRange[2] = np.min( yAxis_ )
            self.DataRange[3] = np.max( yAxis_ )
        else:
            # -- DataRange を更新する -- #
            if( self.DataRange[0] > np.min( xAxis_ ) ): self.DataRange[0] = np.min( xAxis_ )
            if( self.DataRange[1] < np.max( xAxis_ ) ): self.DataRange[1] = np.max( xAxis_ )
            if( self.DataRange[2] > np.min( yAxis_ ) ): self.DataRange[2] = np.min( yAxis_ )
            if( self.DataRange[3] < np.max( yAxis_ ) ): self.DataRange[3] = np.max( yAxis_ )

        
    # ========================================================= #
    # ===  グリッド / y=0 軸線 追加                         === #
    # ========================================================= #
    def set__grid( self ):
        
        # ------------------------------------------------- #
        # --- y=0 軸線 描画                             --- #
        # ------------------------------------------------- #
        if ( self.config["plot.y=0_sw"] ):
            self.ax1.axhline( y        = 0.0, \
                              linestyle=self.config["plot.y=0_linestyle"], \
                              color    =self.config["plot.y=0_color"]    , \
                              linewidth=self.config["plot.y=0_linewidth"] )
            
        # ------------------------------------------------- #
        # --- グリッド ( 主グリッド :: Major )          --- #
        # ------------------------------------------------- #
        if ( self.config["grid.major.sw"]      ):
            self.ax1.grid( visible  =self.config["grid.major.sw"]       , \
                           which    ='major'                      , \
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
    # ===  凡例を表示                                       === #
    # ========================================================= #
    def set__legend( self, loc=None, fontsize=None ):

        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( loc      is not None ): self.config["legend.location"] = loc
        if ( fontsize is not None ): self.config["legend.fontsize"] = fontsize
        
        # ------------------------------------------------- #
        # --- 凡例 ( 第２軸 )                           --- #
        # ------------------------------------------------- #
        h1, l1 = self.ax1.get_legend_handles_labels()
        if ( self.ax2 is not None ):
            h2, l2 = self.ax2.get_legend_handles_labels()
            h1, l1 = h1+h2, l1+l2

        bbox_to_anchor = None
        if ( self.config["ax1.legend.position"] is not None ):
            bbox_to_anchor = tuple( self.config["ax1.legend.position"] )
        
        # ------------------------------------------------- #
        # --- 凡例 描画                                 --- #
        # ------------------------------------------------- #
        self.ax1.legend( h1, l1, loc   =self.config["legend.location"]    , \
                         fontsize      =self.config["legend.fontsize"]    , \
                         ncol          =self.config["legend.nColumn" ]    , \
                         frameon       =self.config["legend.frameOn" ]    , \
                         labelspacing  =self.config["legend.labelGap"]    , \
                         columnspacing =self.config["legend.columnGap"]   , \
                         handlelength  =self.config["legend.handleLength"], \
                         bbox_to_anchor=bbox_to_anchor )
        
    # ========================================================= #
    # ===  alias ( set__legend )                            === #
    # ========================================================= #
    def add__legend( self, loc=None, fontsize=None ):
        self.set__legend( loc=loc, fontsize=fontsize )

    # ========================================================= #
    # ===  カーソル 描画                                    === #
    # ========================================================= #
    def add__cursor( self, xAxis=None, yAxis=None, axis="ax1", label=None, \
                     color=None, linestyle=None, linewidth=None ):

        confname = axis + ".cursor." + "{}"
        
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( color     is not None ): self.config[confname.format("color")]     = color
        if ( linestyle is not None ): self.config[confname.format("linewidth")] = linestyle
        if ( linewidth is not None ): self.config[confname.format("linewidth")] = linewidth
        if   ( axis == "ax1" ):
            theAxis = self.ax1
        elif ( axis == "ax2" ):
            if ( self.ax2  is None ):
                self.ax2 = self.ax1.twinx()
            theAxis = self.ax2
        else:
            print( "[gplot1D.py] illegal axis name ??  -> [ ax1 or ax2 ] " )
            sys.exit()
            
        # ------------------------------------------------- #
        # --- カーソル ( x ) 追加                       --- #
        # ------------------------------------------------- #
        plot_settings = { "colors"   : self.config[confname.format( "color"     )], \
                          "linestyle": self.config[confname.format( "linestyle" )], \
                          "linewidth": self.config[confname.format( "linewidth" )]  }
        if ( xAxis is not None ):
            MinMax = theAxis.get_ylim()
            theAxis.vlines( xAxis, MinMax[0], MinMax[1], \
                            **plot_settings )
             
        # ------------------------------------------------- #
        # --- カーソル ( y ) 追加                       --- #
        # ------------------------------------------------- #
        if ( yAxis is not None ):
            MinMax = theAxis.get_xlim()
            theAxis.hlines( yAxis, MinMax[0], MinMax[1], label=label, \
                            **plot_settings )

            
    # =================================================== #
    # === 2軸目 プロット用 ルーチン                   === #
    # =================================================== #
    def add__plot2( self, xAxis=None, yAxis=None, label=None, color=None, alpha=None, \
                    linestyle=None, linewidth=None, \
                    marker=None, markersize=None, markerwidth=None ):

        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( yAxis       is None ): sys.exit( " [add__plot2] yAxis for axis 2 == ?? " )
        if ( xAxis       is None ): xAxis       = np.arange( yAxis.size ) # -インデックス代用- #
        if ( label       is None ): label       = self.config["legend.labelLength"]*' '
        if ( color       is None ): color       = self.config["plot.color"]
        if ( alpha       is None ): alpha       = self.config["plot.alpha"]
        if ( linestyle   is None ): linestyle   = self.config["plot.linestyle"]
        if ( linewidth   is None ): linewidth   = self.config["plot.linewidth"]
        if ( marker      is None ): marker      = self.config["plot.marker"]
        if ( markersize  is None ): markersize  = self.config["plot.markersize"]
        if ( markerwidth is None ): markerwidth = self.config["plot.markerwidth"]
        
        # ------------------------------------------------- #
        # --- 軸設定                                    --- #
        # ------------------------------------------------- #
        if ( self.ax2    is None ): self.ax2 = self.ax1.twinx()
        self.update__DataRange( xAxis=xAxis, yAxis=yAxis, ax2=True )
        self.set__axis2()
        xAxis_, yAxis_ = np.copy(xAxis), np.copy( yAxis )
        if ( self.config["ax1.x.normalize"] is not None ):
            xAxis_   = xAxis / self.config["ax1.x.normalize"]
        if ( self.config["ax2.y.normalize"] is not None ):
            yAxis_   = yAxis / self.config["ax2.y.normalize"]
        
        # ------------------------------------------------- #
        # --- プロット 追加                             --- #
        # ------------------------------------------------- #
        self.ax2.plot( xAxis_, yAxis_, \
                       color =color , linestyle =linestyle , \
                       label =label , linewidth =linewidth , \
                       marker=marker, markersize=markersize, \
                       markeredgewidth=markerwidth, alpha =alpha )
        

    # ========================================================= #
    # ===  bar 追加                                         === #
    # ========================================================= #
    def add__bar( self, xAxis=None, yAxis=None, xMin=None, xMax=None, yerr=None, \
                  color=None, alpha=None, width=None, \
                  label=None, align="center", bottom=None ):
        
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( yAxis is None ): yAxis      = self.yAxis
        if ( xAxis is None ): xAxis      = self.xAxis
        if ( yAxis is None ): sys.exit( " [add__bar] yAxis == ?? " )
        if ( xAxis is None ): xAxis      = np.arange( yAxis.size ) # - インデックス代用 - #
        if ( label is None ): label      = ' '*self.config["legend.labelLength"]
        if ( width is None ): width      = self.config["bar.width"]
        if ( color is None ): color      = self.config["bar.color"]
        if ( alpha is None ): alpha      = self.config["bar.alpha"]
        if ( ( xMin  is not None ) and ( xMax is not None ) ):
            xAxis = 0.5*( xMin + xMax )
            width = xMax - xMin
            
        # ------------------------------------------------- #
        # --- 軸設定                                    --- #
        # ------------------------------------------------- #
        if   ( type(width) is float ): # --  relative value ( 0 - 1.0 )  -- #
            bar_width  = ( xAxis[1]-xAxis[0] ) * width
        elif ( type(width) in [ list, np.ndarray ] ):
            bar_width  = width
        else:
            print( "[add__bar @ gplot1D.py] width == ??? ( float, list, np.ndarray ) " )
            sys.exit()
        self.xAxis = xAxis
        self.yAxis = yAxis
        self.update__DataRange( xAxis=xAxis, yAxis=yAxis )
        self.set__axis()
        
        # ------------------------------------------------- #
        # --- プロット 追加                             --- #
        # ------------------------------------------------- #
        self.ax1.bar( xAxis, yAxis, label =label, width=bar_width, yerr =yerr, \
                      color =color, alpha =alpha, \
                      align =align, bottom=bottom, \
                      edgecolor=self.config["bar.line.color"], \
                      linewidth=self.config["bar.line.width"], \
        )


    # ========================================================= #
    # ===  vector 追加                                      === #
    # ========================================================= #
    def add__arrow( self, xAxis=None, yAxis=None, uvec=None, vvec=None, color=None, width=None, \
                    scale=1.0, nvec=10 ):
    
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( yAxis      is None ): yAxis      = self.yAxis
        if ( xAxis      is None ): xAxis      = self.xAxis
        if ( yAxis      is None ): sys.exit( " [add__plot] yAxis == ?? " )
        if ( xAxis      is None ): xAxis      = np.arange( yAxis.size ) # - インデックス代用 - #
        if ( width      is None ): width      = 1.0
        if ( color      is None ): color      = "blue"
        
        # ------------------------------------------------- #
        # --- 軸設定                                    --- #
        # ------------------------------------------------- #
        self.xAxis   = xAxis
        self.yAxis   = yAxis
        self.update__DataRange( xAxis=xAxis, yAxis=yAxis )
        self.set__axis()
        nData = self.yAxis.shape[0]
        uvec_ = scale * uvec
        vvec_ = scale * vvec
        index = np.linspace( 0.0, nData-1, nvec, dtype=np.int64 )
        index = np.array( index, dtype=np.int64 )
        
        # ------------------------------------------------- #
        # --- プロット 追加                             --- #
        # ------------------------------------------------- #
        for ik in index:
            self.ax1.arrow( xAxis[ik], yAxis[ik] , uvec_[ik], vvec_[ik], \
                            color =color , width=width  )
    
        
    # ========================================================= #
    # ===  色付きライン                                     === #
    # ========================================================= #
    def add__colorline( self, xAxis=None, yAxis    =None , label =None, alpha     =None, \
                        linestyle  =None, linewidth=None , marker=None, markersize=None, \
                        color      =None, cmap     ="jet", norm  =plt.Normalize(0.0, 1.0) ):

        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( yAxis      is None ): yAxis      = self.yAxis
        if ( xAxis      is None ): xAxis      = self.xAxis
        if ( yAxis      is None ): sys.exit( " [add__colorline] yAxis == ?? " )
        if ( xAxis      is None ): xAxis      = np.arange( yAxis.size ) # - インデックス代用 - #
        if ( label      is None ): label      = " "*self.config["legend.labelLength"]
        if ( alpha      is None ): alpha      = self.config["plot.alpha"]
        if ( linestyle  is None ): linestyle  = self.config["plot.linestyle"]
        if ( linewidth  is None ): linewidth  = self.config["plot.linewidth"]
        if ( marker     is None ): marker     = self.config["plot.marker"]
        if ( markersize is None ): markersize = self.config["plot.markersize"]
        
        # ------------------------------------------------- #
        # --- フィルタリング                            --- #
        # ------------------------------------------------- #
        # xAxis, yAxis = gfl.generalFilter( xAxis=xAxis, yAxis=yAxis, config=self.config )
        
        # ------------------------------------------------- #
        # --- make & check color_array                  --- #
        # ------------------------------------------------- #
        if ( color is None ):
            color  = np.linspace(0.0, 1.0, len(xAxis) )
        if ( not( hasattr( color, "__iter__" ) ) ):
            color  = np.array( [ color ] )
        if ( ( np.min( color ) < 0.0 ) or ( np.max( color ) > 1.0 ) ):
            print( "[add__colorline @ gplot1D.py] color range exceeds [0,1] :: Normalize.... " )
            color  = ( color - np.min( color ) ) / ( np.max( color ) - np.min( color ) )
            
        # ------------------------------------------------- #
        # --- 軸設定                                    --- #
        # ------------------------------------------------- #
        self.xAxis   = xAxis
        self.yAxis   = yAxis
        color        = np.asarray( color )
        points       = np.array( [xAxis, yAxis] ).T.reshape( -1,1,2 )
        segments     = np.concatenate( [points[:-1], points[1:]], axis=1)
        self.update__DataRange( xAxis=xAxis, yAxis=yAxis )
        self.set__axis()
        # ------------------------------------------------- #
        # --- プロット 追加                             --- #
        # ------------------------------------------------- #
        import matplotlib.collections as mcoll
        lc           = mcoll.LineCollection( segments, array=color, cmap=cmap, norm=norm,
                                             linewidth=linewidth, alpha=alpha )
        self.ax1.add_collection( lc )
        
    
    # ========================================================= #
    # ===  scatter プロット 追加                            === #
    # ========================================================= #
    def add__scatter( self, xAxis=None, yAxis=None, cAxis=None, color=None, cmap=None, \
                      density=False, bins=100, \
                      label=None, alpha=None, marker=None, markersize=None, markerwidth=None ):
        
        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #
        if ( yAxis       is None ): yAxis       = self.yAxis
        if ( xAxis       is None ): xAxis       = self.xAxis
        if ( yAxis       is None ): sys.exit( " [add__plot] yAxis == ?? " )
        if ( xAxis       is None ): xAxis       = np.arange( yAxis.size ) #-インデックス代用-#
        if ( color       is None ): color       = self.config["plot.color"]
        if ( label       is None ): label       = ' '*self.config["legend.labelLength"]
        if ( alpha       is None ): alpha       = self.config["plot.alpha"]
        if ( marker      is None ): marker      = self.config["plot.marker"]
        if ( markersize  is None ): markersize  = self.config["plot.markersize"]
        
        # ------------------------------------------------- #
        # --- 軸設定                                    --- #
        # ------------------------------------------------- #
        self.xAxis   = xAxis
        self.yAxis   = yAxis
        self.update__DataRange( xAxis=xAxis, yAxis=yAxis )
        self.set__axis()
        if ( cmap is None ):
            cmap = "jet"
        if ( type( cmap ) is list ):
            if ( type( cmap[0] ) == str ):
                import matplotlib.colors as mcl
                cmap = mcl.ListedColormap( cmap )
        if ( cAxis is None ) and ( density is True ):  #  -- density color -- 
            stat, xedge, yedge, binNum = sp.stats.binned_statistic_2d(
                xAxis, yAxis, None, statistic='count', bins=bins, expand_binnumbers=True )
            flat_index = ( binNum[0]-1, binNum[1]-1 )
            cAxis      = stat[ flat_index ]
            
        # ------------------------------------------------- #
        # --- プロット 追加                             --- #
        # ------------------------------------------------- #
        self.ax1.scatter( xAxis, yAxis , c=cAxis, cmap=cmap, label=label, \
                          marker=marker, s=markersize, alpha =alpha   )

    # ========================================================= #
    # ===  テキスト 追加                                    === #
    # ========================================================= #
    def add__text( self, xpos=0.5, ypos=0.5, text="", color="black", fontsize=None ):

        # ------------------------------------------------- #
        # --- 引数チェック                              --- #
        # ------------------------------------------------- #        
        xMin  , xMax   = self.ax1.get_xlim()
        yMin  , yMax   = self.ax1.get_ylim()
        xcoord, ycoord = xpos*(xMax-xMin)+xMin, ypos*(yMax-yMin)+yMin
        
        # ------------------------------------------------- #
        # --- プロット 追加                             --- #
        # ------------------------------------------------- #
        self.ax1.text( xcoord, ycoord, text, color=color, fontsize=fontsize )


        
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
            if ( self.config["legend.sw"] ): self.set__legend()
            self.fig.savefig( pngFile, dpi=dpi, pad_inches=0, transparent=transparent )
        print( "[ save__figure() @gplot1D ] output :: {0}".format( pngFile ) )
        plt.close()
        return()
   

    # ========================================================= #
    # ===  Display window                                   === #
    # ========================================================= #
    def display__window( self ):
        
        # ------------------------------------------------- #
        # --- Window へ出力                             --- #
        # ------------------------------------------------- #
        print( "\n" + "[ display__window @gplot1D ]" + "\n" )
        self.fig.show()
               

        
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

        
# ======================================== #
# ===  実行部                          === #
# ======================================== #
if ( __name__=="__main__" ):

    os.makedirs( "test/gplot1D/", exist_ok=True )
    
    xAxis    = np.linspace( 0.0, 2.0*np.pi, 101 )
    yAxis1   = np.sin( xAxis ) * 0.06
    yAxis2   = np.cos( xAxis ) * 100.0
    
    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [5.5,5.5],
        "figure.pngFile"     : "test/gplot1D/gplot1D.png", 
        "figure.position"    : [ 0.20, 0.20, 0.92, 0.92 ],
        "ax1.y.normalize"    : 1.0e0, 
        "ax1.x.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax2.y.range"        : { "auto":True, "min": 0.0, "max":1.0, "num":11 },
        "ax1.x.label"        : "x",
        "ax1.y.label"        : "y",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 3.0,
        "legend.fontsize"    : 9.0, 
    }
    config = { **config, **config_ }
        
    fig    = gplot1D( config=config )
    fig.add__plot ( xAxis=xAxis, yAxis=yAxis1, label="y1", color="C0" )
    fig.add__plot2( xAxis=xAxis, yAxis=yAxis2, label="y2", color="C1" )
    fig.set__axis()
    fig.set__axis2()
    fig.set__legend()
    fig.save__figure()

    
