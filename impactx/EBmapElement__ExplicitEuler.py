import os, sys, impactx
import numpy  as np
import pandas as pd
import scipy.interpolate as itp

# ========================================================= #
# ===  EBmapElement__ExplicitEuler                      === #
# ========================================================= #
#
#  EB-map Format :
#
#    - B-field 3D Cartesian (xyz)   Coordinate
#      -- ( x, y, z, Bx, By, Bz ) ( .dat or .csv )   unit: [m] and [T]
#
#    - E-field 2D axisymmetric (rz) Coordinate
#      -- ( r, z,    Er, Ez     ) ( .dat or .csv )   unit: [m] and [MV/m]
#
#    - In ImpactX, t-coordinate (tp) denotes the length :: cv *( t-t_ref )
#      -- spped of light cv [times] difference of arrival time ( against reference particle )
#      -- tp > 0 => delayed  arrival to ref.
#      -- tp < 0 => advanced arrival to ref.
#
# --------------------------------------------------------- #

class EBmapElement__ExplicitEuler( impactx.elements.Programmable ):

    # ===================================================== #
    # ===  __init__                                     === #
    # ===================================================== #
    def __init__( self,
                  ds         =0.0     , # element length [m]
                  name       ="EBmap" ,
                  nslice     =1       ,
                  bfieldfile =None    , # (str)
                  efieldfile =None    , # (str)
                  bfactor    =1.0     , # (float)
                  efactor    =1.0     , # (float)
                  freq       =0.0     , # (float) [Hz]
                  phase      =0.0     , # (float) [deg.]
                 ):
        super().__init__( ds=ds, name=name, nslice=nslice )

        self.ds              = ds
        self.name            = name
        self.nslice          = nslice

        self.bfieldfile      = bfieldfile
        self.efieldfile      = efieldfile

        self.bfactor         = float( bfactor )
        self.efactor         = float( efactor )
        self.freq            = float( freq  )
        self.phase           = float( phase )

        # containers for field maps / interpolators
        self.bmap            = None
        self.emap            = None

        self.Bx_interpolator = None
        self.By_interpolator = None
        self.Bz_interpolator = None
        self.Er_interpolator = None
        self.Ez_interpolator = None

        # load field data and prepare interpolators
        self._load__ebfieldfile()
        self._construct__interpolator()

        # assign programmable push
        self.push = self._push__fromfield


    # ===================================================== #
    # === _load__ebfieldfile                            === #
    # ===================================================== #
    def _load__ebfieldfile(self):
        """ Load magnetic/electric field maps from files.

        Formats:
            bfieldfile: ( x, y, z, Bx, By, Bz )
            efieldfile: ( r, z, Er, Ez )
        """

        # ------------------------------------------------- #
        # --- [1] load B-Field                          --- #
        # ------------------------------------------------- #
        if ( self.bfieldfile is not None ):
            if ( os.path.splitext( self.bfieldfile )[1].lower() == ".csv" ):
                bdat = pd.read_csv( self.bfieldfile ).to_numpy()
            else:
                bdat = np.loadtxt( self.bfieldfile )
            if ( bdat.shape[1] < 6 ):
                raise ValueError( "[ERROR] bfieldfile must have 6 columns: x,y,z,Bx,By,Bz" )
            
            xAxis      = np.unique( bdat[:,0] )
            yAxis      = np.unique( bdat[:,1] )
            zAxis      = np.unique( bdat[:,2] )
            nx, ny, nz = len(xAxis), len(yAxis), len(zAxis)
            
            if ( nx*ny*nz != len(bdat) ):
                raise ValueError( "[ERROR] B-field map size is inconsistent with structured grid." )

            Bx = self.bfactor * ( bdat[:,3] ).reshape( (nx,ny,nz), order="C" )
            By = self.bfactor * ( bdat[:,4] ).reshape( (nx,ny,nz), order="C" ) 
            Bz = self.bfactor * ( bdat[:,5] ).reshape( (nx,ny,nz), order="C" )

            self.bmap = { "xAxis": xAxis, "yAxis": yAxis, "zAxis": zAxis,
                          "Bx"   : Bx   , "By"   : By   ,"Bz"    : Bz, }

        # ------------------------------------------------- #
        # --- [2] load E-Field                          --- #
        # ------------------------------------------------- #
        if ( self.efieldfile is not None ):
            if ( os.path.splitext( self.efieldfile )[1].lower() == ".csv" ):
                edat = pd.read_csv( self.efieldfile ).to_numpy()
            else:
                edat = np.loadtxt( self.efieldfile )
            if ( edat.shape[1] < 4 ):
                raise ValueError( "[ERROR] efieldfile must have 4 columns: r,z,Er,Ez" )

            rAxis      = np.unique( edat[:,0] )
            zAxis      = np.unique( edat[:,1] )
            nr, nz     = len(rAxis), len(zAxis)
            if ( nr*nz != len(edat) ):
                raise ValueError( "[ERROR] E-field map size is inconsistent with structured grid." )

            Er = self.efactor * ( edat[:,2] ).reshape( (nr,nz), order="C" )
            Ez = self.efactor * ( edat[:,3] ).reshape( (nr,nz), order="C" )

            self.emap = { "rAxis":rAxis, "zAxis":zAxis, "Er":Er, "Ez":Ez, }


    # ===================================================== #
    # === _construct__interpolator                      === #
    # ===================================================== #
    def _construct__interpolator(self):
        """ construct interpolators for B(x,y,z) and E(r,z).
        """
        # ------------------------------------------------- #
        # --- [1] B-Field interpolator                  --- #
        # ------------------------------------------------- #
        if ( self.bmap is not None ):
            xyzcoord = ( self.bmap["xAxis"], self.bmap["yAxis"], self.bmap["zAxis"] )
            self.Bx_interpolator = itp.RegularGridInterpolator(
                xyzcoord, self.bmap["Bx"], bounds_error=False, fill_value=0.0 )
            self.By_interpolator = itp.RegularGridInterpolator(
                xyzcoord, self.bmap["By"], bounds_error=False, fill_value=0.0 )
            self.Bz_interpolator = itp.RegularGridInterpolator(
                xyzcoord, self.bmap["Bz"], bounds_error=False, fill_value=0.0 )

        # ------------------------------------------------- #
        # --- [2] E-Field interpolator                  --- #
        # ------------------------------------------------- #
        if ( self.emap is not None ):
            rzcoord = ( self.emap["rAxis"], self.emap["zAxis"] )
            self.Er_interpolator = itp.RegularGridInterpolator(
                rzcoord, self.emap["Er"], bounds_error=False, fill_value=0.0 )
            self.Ez_interpolator = itp.RegularGridInterpolator(
                rzcoord, self.emap["Ez"], bounds_error=False, fill_value=0.0 )


    # ===================================================== #
    # === _evaluate_fields                              === #
    # ===================================================== #
    def _evaluate_fields( self, xp, yp, zp, tp=0.0 ):
        """ Evaluate E and B at particle positions.
        
        Args:
            xp, yp, zp (ndarray) : x-, y-, z- positions  [size = npart]
            tp         (ndarray) : time coordinate for RF modulation.
                                   Relative to reference particle. ( optional )
        Returns:
            Ex,Ey,Ez,Bx,By,Bz (ndarray) : 
        """
        cv    = 2.99792458e8

        # ------------------------------------------------- #
        # --- [1] initialization                        --- #
        # ------------------------------------------------- #
        npart = len( xp )
        Ex    = np.zeros( npart )
        Ey    = np.zeros( npart )
        Ez    = np.zeros( npart )
        Bx    = np.zeros( npart )
        By    = np.zeros( npart )
        Bz    = np.zeros( npart )

        # ------------------------------------------------- #
        # --- [2] B-Field                               --- #
        # ------------------------------------------------- #
        if ( self.Bx_interpolator is not None ):
            pts3  = np.column_stack( [xp, yp, zp] )
            Bx[:] = self.Bx_interpolator( pts3 )
            By[:] = self.By_interpolator( pts3 )
            Bz[:] = self.Bz_interpolator( pts3 )

        # ------------------------------------------------- #
        # --- [3] E-Field                               --- #
        # ------------------------------------------------- #
        if ( self.Er_interpolator is not None ):
            rp       = np.sqrt( xp**2 + yp**2 )
            pts2     = np.column_stack( [rp, zp] )
            phi      = np.deg2rad( self.phase ) + ( 2.0*np.pi*self.freq * tp/cv )
            phaseMod = np.cos( phi )

            Er       = self.Er_interpolator( pts2 ) * phaseMod
            Ez[:]    = self.Ez_interpolator( pts2 ) * phaseMod

            # conversion ::  Er => (Ex, Ey)
            mask     = ( rp > 0.0 )
            rpInv    = 1.0 / rp[mask]
            Ex[mask] = Er[mask] * xp[mask] * rpInv
            Ey[mask] = Er[mask] * yp[mask] * rpInv

        return( Ex,Ey,Ez, Bx,By,Bz )


    # ========================================================= #
    # ===  reference particle pusher from eb-field          === #
    # ========================================================= #
    def _push__refp_fromfield( self, refpart ):
        """ push reference particle using EB-map and Explicit Euler Method
        
        Args:
            refpart ( ImpactX.particle_container.refpart ) : reference particle info.
        """
        cv          = 2.99792458e8
        MeV         = 1.0e6
        # ------------------------------------------------- #
        # --- [1] ref_particle info.                    --- #
        # ------------------------------------------------- #
        ref_s_se    = refpart.s - refpart.sedge  # sedge :: s of the element's start point.
        q_sign      = refpart.charge_qe
        mass_MeV    = refpart.mass_MeV
        q_mc        = ( q_sign * cv ) / ( mass_MeV * MeV )
        gammaInv    = 1.0 / np.sqrt( 1.0 + refpart.px**2 + refpart.py**2 + refpart.pz**2 )
        betax       = refpart.px * gammaInv
        betay       = refpart.py * gammaInv
        betaz       = refpart.pz * gammaInv

        # ------------------------------------------------- #
        # --- [2] interpolate EB-fields                 --- #
        # ------------------------------------------------- #
        rx,ry,rs          = np.array( [refpart.x] ),np.array( [refpart.y] ),np.array( [ref_s_se] )
        Ex,Ey,Ez,Bx,By,Bz = self._evaluate_fields( rx,ry,rs,tp=0.0 )
        
        # ------------------------------------------------- #
        # --- [3] 1st-order Euler method integration    --- #
        # ------------------------------------------------- #
        ds_sliced = self.ds / self.nslice
        dt_sec    = ds_sliced / ( betaz * cv )
        
        Fx        = Ex[0] + cv * ( betay * Bz[0] - betaz * By[0] )
        Fy        = Ey[0] + cv * ( betaz * Bx[0] - betax * Bz[0] )
        Fz        = Ez[0] + cv * ( betax * By[0] - betay * Bx[0] )
        px_new    = refpart.px + q_mc * Fx * dt_sec
        py_new    = refpart.py + q_mc * Fy * dt_sec
        pz_new    = refpart.pz + q_mc * Fz * dt_sec
        gamma     = np.sqrt( 1.0 + px_new**2 + py_new**2 + pz_new**2 )
        pt_new    = -1.0 * gamma
        # gammaInv  =  1.0 / gamma       # not needed for 1st-order Euler
        # betax_new = px_new * gammaInv
        # betay_new = py_new * gammaInv
        # betaz_new = pz_new * gammaInv
        
        refpart.x   = refpart.x + cv * betax * dt_sec  # use old betax for 1st-order Euler
        refpart.y   = refpart.y + cv * betay * dt_sec
        refpart.z   = refpart.z + cv * betaz * dt_sec
        refpart.t   = refpart.t + cv *         dt_sec
        refpart.px  = px_new
        refpart.py  = py_new
        refpart.pz  = pz_new
        refpart.pt  = pt_new
        refpart.s   = refpart.s + ds_sliced


    # ========================================================= #
    # ===  beam particle pusher from eb-field               === #
    # ========================================================= #
    def _push__beam_fromfield( self, pc, refpart, ref_old ):
        """ push beam particles using EB-map and Explicit Euler Method
        
        Args:
            pc      ( ImpactX.particle_container )
            refpart ( ImpactX.particle_container.refpart ) :     reference particle info.
            ref_old ( dictionary )                         : old reference particle info.
        """
        cv          = 2.99792458e8
        MeV         = 1.e6
        # ------------------------------------------------- #
        # --- [1] ref_particle info.                    --- #
        # ------------------------------------------------- #
        q_sign      = refpart.charge_qe
        mass_MeV    = refpart.mass_MeV
        q_mc        = ( q_sign * cv ) / ( mass_MeV * MeV )
        
        ref_s_se    = ref_old["s"] - ref_old["sedge"]
        ref_bg_i    = np.sqrt( ref_old["px"]**2 + ref_old["py"]**2 + ref_old["pz"]**2 )
        ref_gamma_i = (-1.0) * ref_old["pt"]

        # ------------------------------------------------- #
        # --- [2] access to the real data of particles  --- #
        # ------------------------------------------------- #
        for lvl in range( pc.finest_level+1 ):
            for pti in impactx.ImpactXParIter( pc, level=lvl ):

                # ------------------------------------------------- #
                # --- [2-1] particle access                     --- #
                # ------------------------------------------------- #
                soa      = pti.soa()
                r_array  = soa.get_real_data()
                xp       = np.array( r_array[0], copy=False )
                yp       = np.array( r_array[1], copy=False )
                tp       = np.array( r_array[2], copy=False )
                px       = np.array( r_array[3], copy=False )
                py       = np.array( r_array[4], copy=False )
                pt       = np.array( r_array[5], copy=False )
                if (len(xp) == 0):
                    continue

                # ------------------------------------------------- #
                # --- [2-2] conversion to absolute coordinates  --- #
                # ------------------------------------------------- #
                xp_abs_i = ref_old["x"]  + xp
                yp_abs_i = ref_old["y"]  + yp
                tp_abs_i = ref_old["t"]  + tp
                px_abs_i = ref_old["px"] + ref_bg_i * px
                py_abs_i = ref_old["py"] + ref_bg_i * py
                
                gm_abs_i = ref_gamma_i - ref_bg_i * pt
                pz_abs_i = np.sqrt( gm_abs_i**2 - px_abs_i**2 - py_abs_i**2 - 1.0 )
                gammaInv = 1.0 / gm_abs_i
                betax    = px_abs_i * gammaInv
                betay    = py_abs_i * gammaInv
                betaz    = pz_abs_i * gammaInv
                sp_abs_i = np.repeat( [ ref_s_se ], len(tp), axis=0 )
                # sp_abs_i = ref_s_se - betaz * tp
                # sp_abs_i = ref_s_se - tp 
                
                # ------------------------------------------------- #
                # --- [2-3] interpolate EB-fields               --- #
                # ------------------------------------------------- #
                Ex,Ey,Ez,Bx,By,Bz = self._evaluate_fields( xp_abs_i, yp_abs_i, sp_abs_i, tp=tp )
                
                # ------------------------------------------------- #
                # --- [2-4] 1st-order Euler method integration  --- #
                # ------------------------------------------------- #
                ds_sliced   = self.ds / self.nslice
                dt_sec      = ds_sliced / ( betaz * cv )

                Fx          = Ex + cv * ( betay * Bz - betaz * By )
                Fy          = Ey + cv * ( betaz * Bx - betax * Bz )
                Fz          = Ez + cv * ( betax * By - betay * Bx )
                px_abs_f    = px_abs_i + q_mc * Fx * dt_sec
                py_abs_f    = py_abs_i + q_mc * Fy * dt_sec
                pz_abs_f    = pz_abs_i + q_mc * Fz * dt_sec
                
                gm_abs_f    = np.sqrt( 1.0 + px_abs_f**2 + py_abs_f**2 + pz_abs_f**2 )
                pt_abs_f    = - gm_abs_f
                
                xp_abs_f    = xp_abs_i + cv * betax * dt_sec
                yp_abs_f    = yp_abs_i + cv * betay * dt_sec
                tp_abs_f    = tp_abs_i + cv *         dt_sec

                ref_bg_f    = np.sqrt( refpart.px**2 + refpart.py**2 + refpart.pz**2 )
                
                xp[:]       =     xp_abs_f - refpart.x
                yp[:]       =     yp_abs_f - refpart.y
                tp[:]       =     tp_abs_f - refpart.t
                px[:]       =   ( px_abs_f - refpart.px ) / ref_bg_f
                py[:]       =   ( py_abs_f - refpart.py ) / ref_bg_f
                pt[:]       =   ( pt_abs_f - refpart.pt ) / ref_bg_f
        

    # ========================================================= #
    # ===  total particle pusher from eb-field              === #
    # ========================================================= #
    def _push__fromfield( self, pc, step, period ):
        """ push beam particles using EB-map and Explicit Euler Method
        
        Args:
            pc      ( ImpactX.particle_container ) : particle_container
            step    ( int ) : required by programmable element
            period  ( ) : required by programmable element
        """
        refpart = pc.ref_particle()
        ref_old = { "x"  :refpart.x,  "y" :refpart.y,  "z" :refpart.z,  "t" :refpart.t,
                    "px" :refpart.px, "py":refpart.py, "pz":refpart.pz, "pt":refpart.pt,
                    "s"  :refpart.s,  "sedge":refpart.sedge }
        self._push__refp_fromfield( refpart )
        self._push__beam_fromfield( pc, refpart, ref_old )
                        

