import os, sys
import pandas as pd
import plotly.subplots
import plotly.graph_objects      as go
import nk_toolkit.madx.load__tfs as ltf

# ========================================================= #
# ===  plotly__beamline.py                              === #
# ========================================================= #
def plotly__beamline( survey=None, twiss=None, plotKeys=None, html=None ):

    # ------------------------------------------------- #
    # --- [0] constants                             --- #
    # ------------------------------------------------- #
    twissSets = [
        { "title":"position (x,y)"                , "keys":[ "x"    , "y"     ] },
        { "title":"momentum (p_x,p_y)"            , "keys":[ "px"   , "py"    ] },
        { "title":"beta_x, beta_y"                , "keys":[ "betx" , "bety"  ] },
        { "title":"epsilon_x, epsilon_y"          , "keys":[ "emitx", "emity" ] },
        { "title":"alpha_x, alpha_y"              , "keys":[ "alfx" , "alfy"  ] },
        { "title":"D_x, D_y"                      , "keys":[ "dx"   , "dy"    ] },
        { "title":"mu_x, mu_y"                    , "keys":[ "mux"  , "muy"   ] },
        { "title":"gamma_x, gamma_y"              , "keys":[ "gamx" , "gamy"  ] },
        { "title":"beam size (sigma_x, sigma_y)"  , "keys":[ "sigx" , "sigy"  ] },
    ]
    color_map = {
        "sbend"      : "red",
        "quadrupole" : "blue",
        "drift"      : "lightgray",
        "marker"     : "black",
        "solenoid"   : "green",
        "monitor"    : "orange",
        "rfcavity"   : "purple",
        "multipole"  : "cyan",
    }
    linesets = [ { "color":"blue","width":1.2 },
                 { "color":"red" ,"width":1.2 } ]


    # ------------------------------------------------- #
    # --- [1] preparation                           --- #
    # ------------------------------------------------- #
    #  -- [1-1] filter plot variables               --  #
    nGraph, titles, targets = 0, [], []
    if ( survey is not None ):
        nGraph += 1
        titles += [ "Beamline layout" ]
    if ( twiss  is not None ):
        twiss_keys          = { s.lower() for s in twiss["df"].columns }
        twiss["df"].columns = twiss["df"].columns.str.lower()
        for htwiss in twissSets:
            for hkey in htwiss["keys"]:
                if ( hkey in twiss_keys ):
                    targets += [ htwiss ]
                    break
    #  -- [1-2] prepare subplots                    --  #
    nGraph      += len( targets )
    row_heights  = [ 1.0/float(nGraph) for ik in range( nGraph ) ]
    titles      += [ tgt["title"] for tgt in targets ]
    fig          = plotly.subplots.make_subplots( rows=nGraph, cols=1,
                                                  shared_xaxes     = True,
                                                  vertical_spacing = 0.05,
                                                  row_heights      = row_heights, 
                                                  subplot_titles   = titles )
    
    # ------------------------------------------------- #
    # --- [2] plot beam line layout                 --- #
    # ------------------------------------------------- #
    if ( survey is not None ):
        # -- [2-1] display dataframes               --  #
        print( "\n -- beam line (.tfs) -- \n" )
        print( survey["df"] )
        print()
        # -- [2-2] plot beam line components        --  #
        width     = 1.0
        y0, y1    = -0.5*width, +0.5*width
        line      = dict( color="white", width=2.0, )
        for _, row in survey["df"].iterrows():
            s,l       = row["S"]-row["L"], row["L"]
            name,typ  = row["NAME"], row["KEYWORD"]
            xv, yv    = [ s, s+l, s+l, s, s ], [ y0, y0, y1, y1, y0 ]
            color     = color_map.get( typ.lower(), "lightgray" )
            hovertext = f"{name}<br>type={typ}<br>s={s:.3f} m<br>l={l:.3f} m",
            if ( l == 0 ):
                fig.add_trace( go.Scatter( x=[s], y=[0], mode="markers",\
                                           marker    = { "color":color, "size":8, }, \
                                           hovertext = hovertext, \
                                           hoverinfo = "text",\
                                           showlegend= False ), 
                               row=1, col=1 )
            else:
                fig.add_trace( go.Bar    ( x=[l], y=[0], base=[s], width=0.6,
                                           orientation="h",
                                           marker     = { "color":color },
                                           hovertext  = hovertext, 
                                           hoverinfo  = "text",
                                           showlegend = False ),
                               row=1, col=1 )
        # -- [2-3] draw center line                 --  #
        lineset1 = { "color":"chartreuse", "width":1.5, "dash":"solid" }
        fig.add_shape( type="line", 
                       x0=0.0, x1=1.0, y0=0.0, y1=0.0, xref="paper", yref="y", \
                       line=lineset1 )
    
    # ------------------------------------------------- #
    # --- [3] plot twiss variables                  --- #
    # ------------------------------------------------- #
    if ( twiss is not None ):
        # -- [3-1] display dataframes               --  #
        print( "\n --   twiss  (.tfs)  -- \n" )
        print( twiss["df"] )
        print()
        # -- [3-2] subplot に注釈を追加 -- #
        for it,target in enumerate(targets):
            row_idx = it + 2
            for ik,pkey in enumerate( target["keys"] ):
                if pkey in twiss_keys:
                    fig.add_annotation(
                        text=pkey,
                        xref="x domain", yref="y domain",
                        x=0.98, y=0.85 - ik * 0.15,
                        row=row_idx, col=1,
                        showarrow=False,
                        font=dict( size=12, color=linesets[ik]["color"] ),
                        align="right"
                    )
        # --  [3-3] plot twiss data  -- #
        s = twiss["df"]["s"]
        for it,target in enumerate( targets ):
            for ik,pkey in enumerate( target["keys"] ):
                if ( pkey in twiss_keys ):
                    fig.add_trace( go.Scatter( mode="lines", line=linesets[ik], name=pkey, \
                                               x=s, y=( twiss["df"] )[pkey], showlegend=False ),
                                   row=it+2, col=1 )
        
    # ------------------------------------------------- #
    # --- [3] update_layout and show                --- #
    # ------------------------------------------------- #
    fig.update_layout(
        title       ="Beamline Layout from MAD-X",
        xaxis_title ="s [m}",
        yaxis_title ="BeamLine",
        yaxis       = {"showticklabels":False, "type":"linear" },
        barmode     ="overlay",
        height      =400*nGraph,
    )
    for ik in range(nGraph):
        fig.update_xaxes( title_text="s [m]", row=ik+1, col=1, \
                          showticklabels=True)
    if ( html ):
        if ( not( os.path.exists( os.path.dirname(html) ) ) ):
            print( f" [warning] Cannot Find html = {html}" )
            html = "plot.html"
        print( f" HTML File == {html}" )
        fig.write_html( html, auto_open=False)
    else:
        fig.show()



# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    surveyFile = "out/survey.tfs"
    twissFile  = "out/twiss.tfs"
    survey     = ltf.load__tfs( tfsFile=surveyFile )
    twiss      = ltf.load__tfs( tfsFile= twissFile )
    plotly__beamline( survey=survey, twiss=twiss )
