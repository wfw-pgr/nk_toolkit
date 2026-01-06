import os, sys, json5, glob
import numpy  as np
import pandas as pd
import nk_toolkit.impactx.io_toolkit as itk


# ========================================================= #
# ===  analyze_toolkit.py                               === #
# ========================================================= #
def analyze_toolkit():
    postProcess__beam()


# ========================================================= #
# ===  get__postprocessed                               === #
# ========================================================= #
def get__postprocessed( recoFile=None, statFile=None, refpFile=None, postFile=None, \
                        stat_from_bpms=True, bpmsFile=None, correlation=True ):
    
    cv     = 2.99792458e8   # [m/s]
    amu    = 931.494        # [MeV]
    
    # ------------------------------------------------- #
    # --- [1] arguments check                       --- #
    # ------------------------------------------------- #
    if ( recoFile is None ): recoFile="impactx/diags/records.json"
    if ( statFile is None ): statFile="impactx/diags/reduced_beam_characteristics.0"
    if ( refpFile is None ): refpFile="impactx/diags/ref_particle.0"
    if ( postFile is None ): postFile="impactx/diags/posts.csv"
    if ( bpmsFile is None ): bpmsFile="impactx/diags/openPMD/bpm.h5"

    # ------------------------------------------------- #
    # --- [2] load data                             --- #
    # ------------------------------------------------- #
    records, beam, bpms = None, None, None
    
    with open( recoFile, "r" ) as f:
        records = json5.load( f )
        
    beam = itk.get__beamStats( statFile=statFile, refpFile=refpFile )
    
    if ( stat_from_bpms ):
        # bpms  = itk.get__particles( refpFile=refpFile, bpmsFile=bpmsFile, recoFile=recoFile )
        # stats = calc__statsFromBPMs( bpms=bpms )
        stats = calc__statsFromBPMs( bpmsFile=bpmsFile, refpFile=refpFile )
        # -- overwrite on beam -- #
        cols        = beam.columns.difference( stats.columns )
        stats[cols] = np.nan
        stats, beam = stats.set_index("step"), beam.set_index("step")
        common_idx  = stats.index.intersection( beam.index )
        stats.loc[ common_idx, cols ] = beam.loc[ common_idx, cols ]
        beam        = stats
        
    if ( correlation ):
        # if ( bpms is None ):
        #     bpms = itk.get__particles( refpFile=refpFile, bpmsFile=bpmsFile, recoFile=recoFile )
        # corr = calc__correlations( bpms=bpms )
        corr = calc__correlations( bpmsFile=bpmsFile )
        
    # ------------------------------------------------- #
    # --- [3] calculations                          --- #
    # ------------------------------------------------- #
    Em0    = records["beam.mass.amu"] * amu
    Ek_ref = Em0 * ( beam["gamma"] - 1.0 )
    Et_ref = Ek_ref + Em0
    p0c    = np.sqrt( Et_ref**2 - Em0**2 )
    posts                   = {}
    posts["s"]              = beam["s"]
    posts["Ek_ref"]         = Ek_ref
    posts["Ek_min"]         = Ek_ref + beam["min_pt"]  * p0c
    posts["Ek_avg"]         = Ek_ref + beam["mean_pt"] * p0c
    posts["Ek_max"]         = Ek_ref + beam["max_pt"]  * p0c
    posts["dphi_min"]       = beam["min_t"]    / cv * records["beam.freq.rf.Hz"] * 360.0
    posts["dphi_avg"]       = beam["mean_t"]   / cv * records["beam.freq.rf.Hz"] * 360.0
    posts["dphi_max"]       = beam["max_t"]    / cv * records["beam.freq.rf.Hz"] * 360.0
    posts["dphi_rms"]       = beam["sigma_t"]  / cv * records["beam.freq.rf.Hz"] * 360.0
    posts["dp/p_min"]       = beam["min_pt"]   / beam["beta"] * 100.0 # (%)
    posts["dp/p_avg"]       = beam["mean_pt"]  / beam["beta"] * 100.0
    posts["dp/p_max"]       = beam["max_pt"]   / beam["beta"] * 100.0
    posts["dp/p_rms"]       = beam["sigma_pt"] / beam["beta"] * 100.0
    posts["dE/E_min"]       = beam["min_pt"]   * beam["beta"] * 100.0
    posts["dE/E_avg"]       = beam["mean_pt"]  * beam["beta"] * 100.0
    posts["dE/E_max"]       = beam["max_pt"]   * beam["beta"] * 100.0
    posts["dE/E_rms"]       = beam["sigma_pt"] * beam["beta"] * 100.0
    posts["transmission"]   = beam["charge_C"] / beam["charge_C"].iloc[0] * 100.0
    posts["max/sigma_x"]    = np.maximum( np.abs(beam["min_x"]),
                                          np.abs(beam["max_x"] ) ) / beam["sigma_x"]
    posts["max/sigma_y"]    = np.maximum( np.abs(beam["min_y"]),
                                          np.abs(beam["max_y"] ) ) / beam["sigma_y"]
    posts["max/sigma_t"]    = np.maximum( np.abs(beam["min_t"]),
                                          np.abs(beam["max_t"] ) ) / beam["sigma_t"]
    posts["max/sigma_dphi"] = np.maximum( np.abs(posts["dphi_min"]),
                                          np.abs(posts["dphi_max"] ) ) / posts["dphi_rms"]
    df_posts                = pd.DataFrame( posts )
    if ( correlation ):
        df_posts = df_posts.merge( corr.drop( columns=["s"], errors="ignore" ),
                                   on="step", how="left" )
    
    # ------------------------------------------------- #
    # --- [4] save and return                       --- #
    # ------------------------------------------------- #
    df_posts.to_csv( postFile, index=False )
    return()


# ========================================================= #
# ===  calculate statistic values from BPMs data        === #
# ========================================================= #

def calc__statsFromBPMs( bpmsFile=None, refpFile=None, steps=None ) -> pd.DataFrame:
    
    qe = 1.60217663e-19

    if ( refpFile is None ): refpFile="impactx/diags/ref_particle.0"
    if ( bpmsFile is None ): bpmsFile="impactx/diags/openPMD/bpm.h5"
    
    # ------------------------------------------------- #
    # --- [1] functions                             --- #
    # ------------------------------------------------- #
    def _weighted_mean( val, weights ):
        weight_sum = np.sum(weights)
        if weight_sum == 0.0:
            return np.nan
        return( float( np.sum( val*weights ) / weight_sum ) )

    
    def _weighted_variance( val, weights, mean ):
        weight_sum = np.sum(weights)
        if weight_sum == 0.0:
            return np.nan
        return( float( np.sum( weights*( val-mean )**2 ) / weight_sum ) )

    
    def _weighted_covariance( valX, valY, weights, meanX, meanY ):
        weight_sum = np.sum(weights)
        if weight_sum == 0.0:
            return np.nan
        return( float( np.sum( weights * ( valX-meanX )*( valY-meanY ) ) / weight_sum ) )


    # ------------------------------------------------- #
    # --- [2] main loop                             --- #
    # ------------------------------------------------- #
    columns = [ "step", "s",
                "mean_x","min_x","max_x", "mean_y","min_y","max_y", "mean_t","min_t","max_t",
                "sigma_x", "sigma_y", "sigma_t", "mean_px", "min_px", "max_px",
                "mean_py", "min_py", "max_py", "mean_pt", "min_pt", "max_pt",
                "sigma_px", "sigma_py", "sigma_pt", "emittance_x", "emittance_y", "emittance_t",
                "alpha_x", "alpha_y", "alpha_t", "beta_x",  "beta_y",  "beta_t", 
                "dispersion_x",  "dispersion_px", "dispersion_y",  "dispersion_py",
                "emittance_xn", "emittance_yn", "emittance_tn", "charge_C" ]
    stack   = []

    with open( refpFile, "r" ) as f:
        refp         = pd.read_csv( f, sep=r"\s+" )
        refp["step"] = refp["step"].astype(int)
        refp         = refp.set_index( "step", drop=False )
    
    with h5py.File( bpmsFile, "r" ) as f:
        isteps = sorted( int(ik) for ik in f["data"].keys() )
        if ( steps is not None ):
            isteps = [ s for s in isteps if s in set( steps ) ]
            
        for step in isteps:
            key = str( step )
            if ( not( key in f["data"] ) ):
                stack.append( dict.fromkeys( columns, np.nan ) | { "step": step } )
                continue
        
            # ------------------------------------------------- #
            # --- [2-1] prepare variables                   --- #
            # ------------------------------------------------- #
            xp        = f["data"][key]["particles"]["beam"]["position"]["x"][:]
            yp        = f["data"][key]["particles"]["beam"]["position"]["y"][:]
            tp        = f["data"][key]["particles"]["beam"]["position"]["t"][:]
            px        = f["data"][key]["particles"]["beam"]["momentum"]["x"][:]
            py        = f["data"][key]["particles"]["beam"]["momentum"]["y"][:]
            pt        = f["data"][key]["particles"]["beam"]["momentum"]["t"][:]
            weights   = f["data"][key]["particles"]["beam"]["weighting"][:]
            # reference particle ( assumed constant per BPM )
            ref_s     = refp.at[ step, "s"     ]
            ref_beta  = refp.at[ step, "beta"  ]
            ref_gamma = refp.at[ step, "gamma" ]

            # ------------------------------------------------- #
            # --- [2-2] mean, min, max                      --- #
            # ------------------------------------------------- #
            mean_xp        = _weighted_mean( xp, weights )
            mean_yp        = _weighted_mean( yp, weights )
            mean_tp        = _weighted_mean( tp, weights )
            mean_px        = _weighted_mean( px, weights )
            mean_py        = _weighted_mean( py, weights )
            mean_pt        = _weighted_mean( pt, weights )
            
            min_xp, max_xp = xp.min(), xp.max()
            min_yp, max_yp = yp.min(), yp.max()
            min_tp, max_tp = tp.min(), tp.max()
            min_px, max_px = px.min(), px.max()
            min_py, max_py = py.min(), py.max()
            min_pt, max_pt = pt.min(), pt.max()

            # ------------------------------------------------- #
            # --- [2-3] 2nd moment / covariance             --- #
            # ------------------------------------------------- #
            xp_ms = _weighted_variance( xp, weights, mean_xp )
            yp_ms = _weighted_variance( yp, weights, mean_yp )
            tp_ms = _weighted_variance( tp, weights, mean_tp )
            px_ms = _weighted_variance( px, weights, mean_px )
            py_ms = _weighted_variance( py, weights, mean_py )
            pt_ms = _weighted_variance( pt, weights, mean_pt )

            xp_px = _weighted_covariance( xp, px, weights, mean_xp, mean_px )
            yp_py = _weighted_covariance( yp, py, weights, mean_yp, mean_py )
            tp_pt = _weighted_covariance( tp, pt, weights, mean_tp, mean_pt )
            xp_pt = _weighted_covariance( xp, pt, weights, mean_xp, mean_pt )
            px_pt = _weighted_covariance( px, pt, weights, mean_px, mean_pt )
            yp_pt = _weighted_covariance( yp, pt, weights, mean_yp, mean_pt )
            py_pt = _weighted_covariance( py, pt, weights, mean_py, mean_pt )
            
            # ------------------------------------------------- #
            # --- [2-4] sigma ( RMS )                       --- #
            # ------------------------------------------------- #
            sigma_xp = np.sqrt( xp_ms )
            sigma_yp = np.sqrt( yp_ms )
            sigma_tp = np.sqrt( tp_ms )
            sigma_px = np.sqrt( px_ms )
            sigma_py = np.sqrt( py_ms )
            sigma_pt = np.sqrt( pt_ms )
            
            # ------------------------------------------------- #
            # --- [2-5] emittance                           --- #
            # ------------------------------------------------- #
            emittance_x = np.sqrt( xp_ms * px_ms - xp_px**2 )
            emittance_y = np.sqrt( yp_ms * py_ms - yp_py**2 )
            emittance_t = np.sqrt( tp_ms * pt_ms - tp_pt**2 )
            
            # ------------------------------------------------- #
            # --- [2-6] dispersion                          --- #
            # ------------------------------------------------- #
            if ( pt_ms > 0.0 ):
                dispersion_x  = - xp_pt / pt_ms
                dispersion_px = - px_pt / pt_ms
                dispersion_y  = - yp_pt / pt_ms
                dispersion_py = - py_pt / pt_ms
            else:
                dispersion_x  = dispersion_px = np.nan
                dispersion_y  = dispersion_py = np.nan
                
            # ------------------------------------------------- #
            # --- [2-7] dispersion corrected                --- #
            # ------------------------------------------------- #
            xp_msd       = xp_ms - pt_ms * dispersion_x**2
            px_msd       = px_ms - pt_ms * dispersion_px**2
            xp_px_d      = xp_px - pt_ms * dispersion_x * dispersion_px
            yp_msd       = yp_ms - pt_ms * dispersion_y**2
            py_msd       = py_ms - pt_ms * dispersion_py**2
            yp_py_d      = yp_py - pt_ms * dispersion_y * dispersion_py
            emittance_xd = np.sqrt( xp_msd * px_msd - xp_px_d**2 )
            emittance_yd = np.sqrt( yp_msd * py_msd - yp_py_d**2 )
            
            # ------------------------------------------------- #
            # --- [2-8] beta alpha                          --- #
            # ------------------------------------------------- #
            beta_x  =   xp_msd  / emittance_xd if emittance_xd > 0 else np.nan
            alpha_x = - xp_px_d / emittance_xd if emittance_xd > 0 else np.nan
            beta_y  =    yp_msd / emittance_yd if emittance_yd > 0 else np.nan
            alpha_y = - yp_py_d / emittance_yd if emittance_yd > 0 else np.nan
            beta_t  =     tp_ms / emittance_t  if emittance_t  > 0 else np.nan
            alpha_t =   - tp_pt / emittance_t  if emittance_t  > 0 else np.nan
            
            # ------------------------------------------------- #
            # --- [2-9] normalized emittance                --- #
            # ------------------------------------------------- #
            emittance_xn = emittance_x * ref_beta * ref_gamma
            emittance_yn = emittance_y * ref_beta * ref_gamma
            emittance_tn = emittance_t * ref_beta * ref_gamma
            
            # ------------------------------------------------- #
            # --- [2-10] charge_C                           --- #
            # ------------------------------------------------- #
            charge_C     = qe * np.sum( weights )
            
            # ------------------------------------------------- #
            # --- [2-10] store and return                   --- #
            # ------------------------------------------------- #
            stack.append( {
                "step"     : step   ,  "s"       : ref_s,
                "mean_x"   : mean_xp,  "min_x"   : min_xp,   "max_x"   : max_xp,
                "mean_y"   : mean_yp,  "min_y"   : min_yp,   "max_y"   : max_yp,
                "mean_t"   : mean_tp,  "min_t"   : min_tp,   "max_t"   : max_tp,
                "sigma_x"  : sigma_xp, "sigma_y" : sigma_yp, "sigma_t" : sigma_tp,
                "mean_px"  : mean_px,  "min_px"  : min_px,   "max_px"  : max_px,
                "mean_py"  : mean_py,  "min_py"  : min_py,   "max_py"  : max_py,
                "mean_pt"  : mean_pt,  "min_pt"  : min_pt,   "max_pt"  : max_pt,
                "sigma_px" : sigma_px, "sigma_py": sigma_py, "sigma_pt": sigma_pt,
                
                "emittance_x": emittance_x, "emittance_y": emittance_y, "emittance_t": emittance_t,
                "alpha_x"    : alpha_x,     "alpha_y"    : alpha_y,     "alpha_t"    : alpha_t,
                "beta_x"     : beta_x,      "beta_y"     : beta_y,      "beta_t"     : beta_t,
                
                "dispersion_x" : dispersion_x, "dispersion_px" : dispersion_px,
                "dispersion_y" : dispersion_y, "dispersion_py" : dispersion_py,
                "emittance_xn" : emittance_xn, "emittance_yn"  : emittance_yn ,
                "emittance_tn" : emittance_tn, "charge_C"      : charge_C, 
            } )
    ret = pd.DataFrame( stack )
    return( ret )
    

# # ========================================================= #
# # ===  calculate statistic values from BPMs data        === #
# # ========================================================= #

# def calc__statsFromBPMs( bpms: pd.DataFrame,  bpmsFile:  ) -> pd.DataFrame:

#     qe = 1.60217663e-19
    
#     # ------------------------------------------------- #
#     # --- [1] functions                             --- #
#     # ------------------------------------------------- #
#     def _weighted_mean( val, weights ):
#         weight_sum = np.sum(weights)
#         if weight_sum == 0.0:
#             return np.nan
#         return( float( np.sum( val*weights ) / weight_sum ) )

    
#     def _weighted_variance( val, weights, mean ):
#         weight_sum = np.sum(weights)
#         if weight_sum == 0.0:
#             return np.nan
#         return( float( np.sum( weights*( val-mean )**2 ) / weight_sum ) )

    
#     def _weighted_covariance( valX, valY, weights, meanX, meanY ):
#         weight_sum = np.sum(weights)
#         if weight_sum == 0.0:
#             return np.nan
#         return( float( np.sum( weights * ( valX-meanX )*( valY-meanY ) ) / weight_sum ) )


#     # ------------------------------------------------- #
#     # --- [2] main loop                             --- #
#     # ------------------------------------------------- #
#     columns = [ "step", "s",
#                 "mean_x","min_x","max_x", "mean_y","min_y","max_y", "mean_t","min_t","max_t",
#                 "sigma_x", "sigma_y", "sigma_t", "mean_px", "min_px", "max_px",
#                 "mean_py", "min_py", "max_py", "mean_pt", "min_pt", "max_pt",
#                 "sigma_px", "sigma_py", "sigma_pt", "emittance_x", "emittance_y", "emittance_t",
#                 "alpha_x", "alpha_y", "alpha_t", "beta_x",  "beta_y",  "beta_t", 
#                 "dispersion_x",  "dispersion_px", "dispersion_y",  "dispersion_py",
#                 "emittance_xn", "emittance_yn", "emittance_tn", "charge_C" ]
#     stack   = []
#     for step_index,group in bpms.groupby( "step", sort=True ):

#         if ( group.shape[0] == 0 ):
#             stack.append( dict.fromkeys( columns, np.nan ) | { "step": step_index } )
#             continue
        
#         # ------------------------------------------------- #
#         # --- [2-1] prepare variables                   --- #
#         # ------------------------------------------------- #
#         xp, yp, tp = group["xp"].to_numpy(), group["yp"].to_numpy(), group["tp"].to_numpy()
#         px, py, pt = group["px"].to_numpy(), group["py"].to_numpy(), group["pt"].to_numpy()
#         weights    = group["wt"].to_numpy()
#         # reference particle ( assumed constant per BPM )
#         a_ref_info = group.iloc[0]
#         ref_s      = a_ref_info["ref_s"]
#         ref_beta   = a_ref_info["ref_beta"]
#         ref_gamma  = a_ref_info["ref_gamma"]

#         # ------------------------------------------------- #
#         # --- [2-2] mean, min, max                      --- #
#         # ------------------------------------------------- #
#         mean_xp        = _weighted_mean( xp, weights )
#         mean_yp        = _weighted_mean( yp, weights )
#         mean_tp        = _weighted_mean( tp, weights )
#         mean_px        = _weighted_mean( px, weights )
#         mean_py        = _weighted_mean( py, weights )
#         mean_pt        = _weighted_mean( pt, weights )

#         min_xp, max_xp = xp.min(), xp.max()
#         min_yp, max_yp = yp.min(), yp.max()
#         min_tp, max_tp = tp.min(), tp.max()
#         min_px, max_px = px.min(), px.max()
#         min_py, max_py = py.min(), py.max()
#         min_pt, max_pt = pt.min(), pt.max()

#         # ------------------------------------------------- #
#         # --- [2-3] 2nd moment / covariance             --- #
#         # ------------------------------------------------- #
#         xp_ms = _weighted_variance( xp, weights, mean_xp )
#         yp_ms = _weighted_variance( yp, weights, mean_yp )
#         tp_ms = _weighted_variance( tp, weights, mean_tp )
#         px_ms = _weighted_variance( px, weights, mean_px )
#         py_ms = _weighted_variance( py, weights, mean_py )
#         pt_ms = _weighted_variance( pt, weights, mean_pt )

#         xp_px = _weighted_covariance( xp, px, weights, mean_xp, mean_px )
#         yp_py = _weighted_covariance( yp, py, weights, mean_yp, mean_py )
#         tp_pt = _weighted_covariance( tp, pt, weights, mean_tp, mean_pt )
#         xp_pt = _weighted_covariance( xp, pt, weights, mean_xp, mean_pt )
#         px_pt = _weighted_covariance( px, pt, weights, mean_px, mean_pt )
#         yp_pt = _weighted_covariance( yp, pt, weights, mean_yp, mean_pt )
#         py_pt = _weighted_covariance( py, pt, weights, mean_py, mean_pt )

#         # ------------------------------------------------- #
#         # --- [2-4] sigma ( RMS )                       --- #
#         # ------------------------------------------------- #
#         sigma_xp = np.sqrt( xp_ms )
#         sigma_yp = np.sqrt( yp_ms )
#         sigma_tp = np.sqrt( tp_ms )
#         sigma_px = np.sqrt( px_ms )
#         sigma_py = np.sqrt( py_ms )
#         sigma_pt = np.sqrt( pt_ms )

#         # ------------------------------------------------- #
#         # --- [2-5] emittance                           --- #
#         # ------------------------------------------------- #
#         emittance_x = np.sqrt( xp_ms * px_ms - xp_px**2 )
#         emittance_y = np.sqrt( yp_ms * py_ms - yp_py**2 )
#         emittance_t = np.sqrt( tp_ms * pt_ms - tp_pt**2 )

#         # ------------------------------------------------- #
#         # --- [2-6] dispersion                          --- #
#         # ------------------------------------------------- #
#         if ( pt_ms > 0.0 ):
#             dispersion_x  = - xp_pt / pt_ms
#             dispersion_px = - px_pt / pt_ms
#             dispersion_y  = - yp_pt / pt_ms
#             dispersion_py = - py_pt / pt_ms
#         else:
#             dispersion_x  = dispersion_px = np.nan
#             dispersion_y  = dispersion_py = np.nan

#         # ------------------------------------------------- #
#         # --- [2-7] dispersion corrected                --- #
#         # ------------------------------------------------- #
#         xp_msd       = xp_ms - pt_ms * dispersion_x**2
#         px_msd       = px_ms - pt_ms * dispersion_px**2
#         xp_px_d      = xp_px - pt_ms * dispersion_x * dispersion_px
#         yp_msd       = yp_ms - pt_ms * dispersion_y**2
#         py_msd       = py_ms - pt_ms * dispersion_py**2
#         yp_py_d      = yp_py - pt_ms * dispersion_y * dispersion_py
#         emittance_xd = np.sqrt( xp_msd * px_msd - xp_px_d**2 )
#         emittance_yd = np.sqrt( yp_msd * py_msd - yp_py_d**2 )

#         # ------------------------------------------------- #
#         # --- [2-8] beta alpha                          --- #
#         # ------------------------------------------------- #
#         beta_x  =   xp_msd  / emittance_xd if emittance_xd > 0 else np.nan
#         alpha_x = - xp_px_d / emittance_xd if emittance_xd > 0 else np.nan
#         beta_y  =    yp_msd / emittance_yd if emittance_yd > 0 else np.nan
#         alpha_y = - yp_py_d / emittance_yd if emittance_yd > 0 else np.nan
#         beta_t  =     tp_ms / emittance_t  if emittance_t  > 0 else np.nan
#         alpha_t =   - tp_pt / emittance_t  if emittance_t  > 0 else np.nan

#         # ------------------------------------------------- #
#         # --- [2-9] normalized emittance                --- #
#         # ------------------------------------------------- #
#         emittance_xn = emittance_x * ref_beta * ref_gamma
#         emittance_yn = emittance_y * ref_beta * ref_gamma
#         emittance_tn = emittance_t * ref_beta * ref_gamma

#         # ------------------------------------------------- #
#         # --- [2-10] charge_C                           --- #
#         # ------------------------------------------------- #
#         charge_C     = qe * np.sum( weights )

#         # ------------------------------------------------- #
#         # --- [2-10] store and return                   --- #
#         # ------------------------------------------------- #
#         stack.append( {
#             "step"     : step_index, "s"     : ref_s,
#             "mean_x"   : mean_xp,  "min_x"   : min_xp,   "max_x"   : max_xp,
#             "mean_y"   : mean_yp,  "min_y"   : min_yp,   "max_y"   : max_yp,
#             "mean_t"   : mean_tp,  "min_t"   : min_tp,   "max_t"   : max_tp,
#             "sigma_x"  : sigma_xp, "sigma_y" : sigma_yp, "sigma_t" : sigma_tp,
#             "mean_px"  : mean_px,  "min_px"  : min_px,   "max_px"  : max_px,
#             "mean_py"  : mean_py,  "min_py"  : min_py,   "max_py"  : max_py,
#             "mean_pt"  : mean_pt,  "min_pt"  : min_pt,   "max_pt"  : max_pt,
#             "sigma_px" : sigma_px, "sigma_py": sigma_py, "sigma_pt": sigma_pt,
            
#             "emittance_x": emittance_x, "emittance_y": emittance_y, "emittance_t": emittance_t,
#             "alpha_x"    : alpha_x,     "alpha_y"    : alpha_y,     "alpha_t"    : alpha_t,
#             "beta_x"     : beta_x,      "beta_y"     : beta_y,      "beta_t"     : beta_t,
            
#             "dispersion_x" : dispersion_x, "dispersion_px" : dispersion_px,
#             "dispersion_y" : dispersion_y, "dispersion_py" : dispersion_py,
#             "emittance_xn" : emittance_xn, "emittance_yn"  : emittance_yn ,
#             "emittance_tn" : emittance_tn, "charge_C"      : charge_C, 
#         } )
#     ret = pd.DataFrame( stack )
#     return( ret )


# ========================================================= #
# ===  calculate covariance                             === #
# ========================================================= #

def calc__covariance( abpm: pd.DataFrame ) -> pd.DataFrame:

    # abpm : pd.dataframe, xp,yp,tp, px,py,pt, wt

    # ------------------------------------------------- #
    # --- [1] functions                             --- #
    # ------------------------------------------------- #
    def _weighted_mean( val, weights ):
        weight_sum = np.sum(weights)
        if weight_sum == 0.0:
            return np.nan
        return( float( np.sum( val*weights ) / weight_sum ) )
    
    def _weighted_covariance( valX, valY, weights, meanX, meanY ):
        weight_sum = np.sum(weights)
        if weight_sum == 0.0:
            return np.nan
        return( float( np.sum( weights * ( valX-meanX )*( valY-meanY ) ) / weight_sum ) )


    # ------------------------------------------------- #
    # --- [2] prepare variables                     --- #
    # ------------------------------------------------- #
    xp, yp, tp = abpm["xp"].to_numpy(), abpm["yp"].to_numpy(), abpm["tp"].to_numpy()
    px, py, pt = abpm["px"].to_numpy(), abpm["py"].to_numpy(), abpm["pt"].to_numpy()
    weights    = abpm["wt"].to_numpy()
    coords     = np.concatenate( [ xp[:,np.newaxis], px[:,np.newaxis], yp[:,np.newaxis], \
                                   py[:,np.newaxis], tp[:,np.newaxis], pt[:,np.newaxis] ], \
                                 axis=1 )

    # ------------------------------------------------- #
    # --- [3] mean values                           --- #
    # ------------------------------------------------- #
    mean_xp     = _weighted_mean( xp, weights )
    mean_yp     = _weighted_mean( yp, weights )
    mean_tp     = _weighted_mean( tp, weights )
    mean_px     = _weighted_mean( px, weights )
    mean_py     = _weighted_mean( py, weights )
    mean_pt     = _weighted_mean( pt, weights )
    means       = np.array( [ mean_xp, mean_px, mean_yp, mean_py, mean_tp, mean_pt ] )
    
    # ------------------------------------------------- #
    # --- [4] calculate covariance                  --- #
    # ------------------------------------------------- #
    xp_,px_,yp_ = 0, 1, 2
    py_,tp_,pt_ = 3, 4, 5
    covMat      = np.zeros( (6,6) )

    for ik in range( xp_, pt_+1 ):
        for jk in range( ik, pt_+1 ):
            covMat[ik,jk] = _weighted_covariance( coords[:,ik], coords[:,jk], \
                                                  weights, means[ik], means[jk] )
            covMat[jk,ik] = covMat[ik,jk]

    labels = ["xp","px","yp","py","tp","pt"]
    ret    = pd.DataFrame( covMat, index=labels, columns=labels )
    return( ret )


# ========================================================= #
# ===  calculate correlations                           === #
# ========================================================= #

def calc__correlations( bpms: pd.DataFrame ) -> pd.DataFrame:

    def _corr( cov, sigma, v1, v2 ):
        denom = sigma[v1] * sigma[v2]
        if ( ( not np.isfinite( denom ) ) or ( denom == 0.0 ) ): 
            return( np.nan )
        else:
            return( float( cov.loc[ v1,v2 ] / denom ) )
    
    stack   = []
    for step_index,abpm in bpms.groupby( "step", sort=True ):

        ref_s = float( ( abpm.iloc[0] )["ref_s"] )
        cov   = calc__covariance( abpm=abpm )
        sigma = pd.Series( np.sqrt( np.diag( cov ) ).astype( float ), \
                           index=cov.index )

        xp_yp = _corr( cov, sigma, "xp", "yp" )
        xp_tp = _corr( cov, sigma, "xp", "tp" )
        xp_px = _corr( cov, sigma, "xp", "px" )
        xp_py = _corr( cov, sigma, "xp", "py" )
        xp_pt = _corr( cov, sigma, "xp", "pt" )
        
        yp_tp = _corr( cov, sigma, "yp", "tp" )
        yp_px = _corr( cov, sigma, "yp", "px" )
        yp_py = _corr( cov, sigma, "yp", "py" )
        yp_pt = _corr( cov, sigma, "yp", "pt" )
        
        tp_px = _corr( cov, sigma, "tp", "px" )
        tp_py = _corr( cov, sigma, "tp", "py" )
        tp_pt = _corr( cov, sigma, "tp", "pt" )
        
        px_py = _corr( cov, sigma, "px", "py" )
        px_pt = _corr( cov, sigma, "px", "pt" )
        
        py_pt = _corr( cov, sigma, "py", "pt" )

        row   = { "step":step_index, "s":ref_s,
                  "xp-yp" :xp_yp, "xp-tp" :xp_tp, "xp-px" :xp_px, "xp-py" :xp_py, "xp-pt" :xp_pt,
                  "yp-tp" :yp_tp, "yp-px" :yp_px, "yp-py" :yp_py, "yp-pt" :yp_pt, 
                  "tp-px" :tp_px, "tp-py" :tp_py, "tp-pt" :tp_pt, 
                  "px-py" :px_py, "px-pt" :px_pt, 
                  "py-pt" :py_pt, 
                 }
        stack.append( row )
    ret = pd.DataFrame( stack )
    return( ret )
    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    analyze_toolkit()
