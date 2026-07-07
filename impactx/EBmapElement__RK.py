import os, sys, impactx
import numpy  as np
import pandas as pd
import scipy.interpolate as itp

# ========================================================= #
# ===  EBmapElement__RK                                 === #
# ========================================================= #
#
#  EB-map Format :
#
#    - B-field 3D Cartesian (xyz)   Coordinate
#      -- ( x, y, z, Bx, By, Bz ) ( .dat or .csv )   unit: [m] and [T]
#
#    - E-field 2D axisymmetric (rz) Coordinate
#      -- ( r, z,    Er, Ez     ) ( .dat or .csv )   unit: [m] and [V/m]
#
#    - In ImpactX, t-coordinate (tp) denotes the length :: cv *( t-t_ref )
#      -- spped of light cv [times] difference of arrival time ( against reference particle )
#      -- tp > 0 => delayed  arrival to ref.
#      -- tp < 0 => advanced arrival to ref.
#
# --------------------------------------------------------- #

class EBmapElement__RK( impactx.elements.Programmable ):

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
                  phase_sync ="none"  , # (str) [ "none", "sync", "forced" ] 
                  int_method ="RK4"   , # (str) [ "ExplicitEuler", "RK2", "RK4" ]
                  aperture_x =0.0     , # (float) ellipse half-aperture [m]
                  aperture_y =0.0     , # (float) ellipse half-aperture [m]
                  aperture_cx=0.0     , # (float) aperture center x in map frame [m]
                  aperture_cy=0.0     , # (float) aperture center y in map frame [m]
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
        self.phase_sync      = phase_sync
        self.int_method      = int_method
        self.aperture_x      = float( aperture_x  )
        self.aperture_y      = float( aperture_y  )
        self.aperture_cx     = float( aperture_cx )
        self.aperture_cy     = float( aperture_cy )

        # -- aperture : particle lost を適用するためのマスク値 -- #
        #      AMReX ParticleIDWrapper::make_invalid() 相当
        self._idcpu_valid_mask = np.uint64( 0x7FFF_FFFF_FFFF_FFFF )

        # 各 slice での除算を避けるため、事前計算
        self._aperture_inv_x2 = 1.0 / ( self.aperture_x * self.aperture_x )
        self._aperture_inv_y2 = 1.0 / ( self.aperture_y * self.aperture_y )
        
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

            zMin       = zAxis[0]
            zAxis      = zAxis - zMin     # z-coord :: [ zMin - zMax] => [ 0.0 - (zMax-zMin) ]
            dz         = zAxis[-1]        # dz must be self.ds
            
            if not( np.isclose( dz, self.ds, rtol=1.e-8 ) ):
                raise ValueError( "[ERROR] ds and loaded B-field-map z span are inconsistent." )
            
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

            zMin       = zAxis[0]
            zAxis      = zAxis - zMin     # z-coord :: [ zMin - zMax] => [ 0.0 - (zMax-zMin) ]
            dz         = zAxis[-1]        # dz must be self.ds
            
            if not( np.isclose( dz, self.ds, rtol=1.e-8 ) ):
                raise ValueError( "[ERROR] ds and loaded B-field-map z span are inconsistent." )
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
    def _evaluate_fields( self, xp, yp, zp, tp, ):
        """ Evaluate E and B at particle positions.
        
        Args:
            xp, yp, zp (ndarray) : x-, y-, s- positions  [size = npart]
            tp         (ndarray) : time coordinate for RF modulation.
                                   Relative to reference particle.
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

        if   ( self.phase_sync.lower() == "none" ):
            tphase = tp
        elif ( self.phase_sync.lower() == "sync" ):
            #  tphase = tp - tref     under construction.
        elif ( self.phase_sync.lower() == "forced" ):
            tphase = 0.0
        else:
            raise ValueError( "[_evaluate_fields]  unknown self.phase_sync argument : {}"
                              .format( self.phase_sync ) )
        
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
            phi      = np.deg2rad( self.phase ) + ( 2.0*np.pi*self.freq * tphase/cv )
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
    # ===  compute velocity ( rhs of dx/dt = v )            === #
    # ========================================================= #
    def _compute__rhs_velocity( self, px, py, pz ):
        """ compute rhs of dx/dt = v
        Args:
            px, py, pz ( ndarray ) : momentum of particle
        Return:
            vx, vy, vz ( ndarray ) : rhs of dx/dt = v, 
        """
        cv       = 2.99792458e8
        gammaInv = 1.0 / np.sqrt( 1.0 + ( px**2 + py**2 + pz**2 ) )
        vx       = cv * px * gammaInv
        vy       = cv * py * gammaInv
        vz       = cv * pz * gammaInv
        return( vx, vy, vz )
    

    # ========================================================= #
    # ===  compute Lorentz force ( rhs of dv/dt = F )       === #
    # ========================================================= #
    def _compute__rhs_LorentzForce( self, xp,yp,tp,sp, px,py,pz, q_mc ):
        """ compute rhs of d(beta gamma)/dt = F/c = q/mc [ E + vxB ]
        Args:
            xp,yp,tp,sp, px,py,pz ( ndarray ) : position of particle,
            q_mc (float) : constant q/(mc)
        Return:
            Fx, Fy, Fz     ( ndarray ) : rhs of d(beta gamma)/dt = F/c, 
        """
        cv                = 2.99792458e8
        Ex,Ey,Ez,Bx,By,Bz = self._evaluate_fields( xp,yp,sp,tp )
        gammaInv          = 1.0 / np.sqrt( 1.0 + px**2 + py**2 + pz**2 )
        betax             = px * gammaInv
        betay             = py * gammaInv
        betaz             = pz * gammaInv
        Fx                = q_mc * ( Ex[:] + cv * ( betay * Bz[:] - betaz * By[:] ) )
        Fy                = q_mc * ( Ey[:] + cv * ( betaz * Bx[:] - betax * Bz[:] ) )
        Fz                = q_mc * ( Ez[:] + cv * ( betax * By[:] - betay * Bx[:] ) )
        return( Fx, Fy, Fz )


    # ========================================================= #
    # ===  compute rhs with independent variable s          === #
    # ========================================================= #
    def _compute__rhs_s( self, xp,yp,tp,sp, px,py,pz, q_mc ):
        """ compute rhs of d[x,y,t,p]/ds for RK integration
        Args:
            xp,yp,tp,sp, px,py,pz ( ndarray ) : position / momentum of particle,
            q_mc (float) : constant q/(mc)
        Return:
            dxds,dyds,dtds,dpxds,dpyds,dpzds ( ndarray ) :
        """
        cv       = 2.99792458e8
        vx,vy,vz = self._compute__rhs_velocity    (                   px,py,pz       )
        Fx,Fy,Fz = self._compute__rhs_LorentzForce( xp,yp,tp,sp, px,py,pz, q_mc )
        dxds     = vx / vz
        dyds     = vy / vz
        dtds     = cv / vz
        dpxds    = Fx / vz
        dpyds    = Fy / vz
        dpzds    = Fz / vz
        return( dxds,dyds,dtds, dpxds,dpyds,dpzds )


    # ========================================================= #
    # ===  RK4 integrator with independent variable s       === #
    # ========================================================= #
    def _integrate__rk4_s( self, xp,yp,tp,sp, px,py,pz, q_mc, ds_sliced ):
        """ advance absolute particle coordinates by 4th-order Runge-Kutta.
        """

        # ------------------------------------------------- #
        # --- [1] k1                                    --- #
        # ------------------------------------------------- #
        k1x,k1y,k1t, k1px,k1py,k1pz = self._compute__rhs_s(
            xp,yp,tp, sp, px,py,pz, q_mc )

        # ------------------------------------------------- #
        # --- [2] k2                                    --- #
        # ------------------------------------------------- #
        h2       = 0.5 * ds_sliced
        xp2      = xp   + h2 * k1x
        yp2      = yp   + h2 * k1y
        tp2      = tp   + h2 * k1t
        sp2      = sp   + h2
        px2      = px   + h2 * k1px
        py2      = py   + h2 * k1py
        pz2      = pz   + h2 * k1pz
        k2x,k2y,k2t, k2px,k2py,k2pz = self._compute__rhs_s(
            xp2,yp2,tp2,sp2, px2,py2,pz2, q_mc )

        # ------------------------------------------------- #
        # --- [3] k3                                    --- #
        # ------------------------------------------------- #
        xp3      = xp   + h2 * k2x
        yp3      = yp   + h2 * k2y
        tp3      = tp   + h2 * k2t
        sp3      = sp   + h2
        px3      = px   + h2 * k2px
        py3      = py   + h2 * k2py
        pz3      = pz   + h2 * k2pz
        k3x,k3y,k3t, k3px,k3py,k3pz = self._compute__rhs_s(
            xp3,yp3,tp3, sp3, px3,py3,pz3, q_mc )

        # ------------------------------------------------- #
        # --- [4] k4                                    --- #
        # ------------------------------------------------- #
        xp4      = xp   + ds_sliced * k3x
        yp4      = yp   + ds_sliced * k3y
        tp4      = tp   + ds_sliced * k3t
        sp4      = sp   + ds_sliced
        px4      = px   + ds_sliced * k3px
        py4      = py   + ds_sliced * k3py
        pz4      = pz   + ds_sliced * k3pz
        k4x,k4y,k4t, k4px,k4py,k4pz = self._compute__rhs_s(
            xp4,yp4,tp4, sp4, px4,py4,pz4, q_mc )

        # ------------------------------------------------- #
        # --- [5] weighted sum                          --- #
        # ------------------------------------------------- #
        coef      = ds_sliced / 6.0
        xp_f      = xp + coef * ( k1x  + 2.0*k2x  + 2.0*k3x  + k4x  )
        yp_f      = yp + coef * ( k1y  + 2.0*k2y  + 2.0*k3y  + k4y  )
        tp_f      = tp + coef * ( k1t  + 2.0*k2t  + 2.0*k3t  + k4t  )
        px_f      = px + coef * ( k1px + 2.0*k2px + 2.0*k3px + k4px )
        py_f      = py + coef * ( k1py + 2.0*k2py + 2.0*k3py + k4py )
        pz_f      = pz + coef * ( k1pz + 2.0*k2pz + 2.0*k3pz + k4pz )
        return( xp_f,yp_f,tp_f, px_f,py_f,pz_f  )


    # ========================================================= #
    # ===  reference particle pusher from eb-field          === #
    # ========================================================= #
    def _push__refp_fromfield( self, refpart ):
        """ push reference particle using EB-map and Explicit Euler Method
        
        Args:
            refpart ( ImpactX.particle_container.refpart ) : reference particle info.
        """
        cv         = 2.99792458e8
        MeV        = 1.0e6
        # ------------------------------------------------- #
        # --- [1] ref_particle info.                    --- #
        # ------------------------------------------------- #
        ref_s_se   = refpart.s - refpart.sedge  # sedge :: s of the element's start point.
        q_sign     = refpart.charge_qe
        mass_MeV   = refpart.mass_MeV
        q_mc       = ( q_sign * cv ) / ( mass_MeV * MeV )
        rx,ry,rt   = np.array( [refpart.x ] ),np.array( [refpart.y ] ),np.array( [refpart.t ] )
        px,py,pt   = np.array( [refpart.px] ),np.array( [refpart.py] ),np.array( [refpart.pt] )
        rs,pz      = np.array( [ref_s_se]   ),np.array( [refpart.pz] )

        # ------------------------------------------------- #
        # --- [2] integration                           --- #
        # ------------------------------------------------- #
        ds_sliced  = self.ds / self.nslice

        if   ( self.int_method == "ExplicitEuler" ):
            vx,vy,vz   = self._compute__rhs_velocity    (              px,py,pz       )
            Fx,Fy,Fz   = self._compute__rhs_LorentzForce( rx,ry,rt,rs, px,py,pz, q_mc )
            dt_sec     = ds_sliced / vz

            refpart.x  = refpart.x  + vx[0] * dt_sec[0]
            refpart.y  = refpart.y  + vy[0] * dt_sec[0]
            refpart.z  = refpart.z  + vz[0] * dt_sec[0]
            refpart.t  = refpart.t  + cv    * dt_sec[0]
            
            refpart.px = refpart.px + Fx[0] * dt_sec[0]
            refpart.py = refpart.py + Fy[0] * dt_sec[0]
            refpart.pz = refpart.pz + Fz[0] * dt_sec[0]
            refpart.pt = -1.0 * np.sqrt( 1.0 + refpart.px**2 + refpart.py**2 + refpart.pz**2 )
            refpart.s  = refpart.s  + ds_sliced

        elif ( self.int_method == "RK2" ):
            vx,vy,vz   = self._compute__rhs_velocity    (              px,py,pz       )
            Fx,Fy,Fz   = self._compute__rhs_LorentzForce( rx,ry,rt,rs, px,py,pz, q_mc )
            dt_sec     = ds_sliced / vz

            # ------------------------------------------------- #
            # --- [3-1] midpoint state                      --- #
            # ------------------------------------------------- #
            x_mid  = refpart.x  + vx[0] * dt_sec[0] * 0.5 
            y_mid  = refpart.y  + vy[0] * dt_sec[0] * 0.5 
            t_mid  = refpart.t  + cv    * dt_sec[0] * 0.5 
            px_mid = refpart.px + Fx[0] * dt_sec[0] * 0.5 
            py_mid = refpart.py + Fy[0] * dt_sec[0] * 0.5 
            pz_mid = refpart.pz + Fz[0] * dt_sec[0] * 0.5 
            s_mid  = ref_s_se   +         ds_sliced * 0.5

            # ------------------------------------------------- #
            # --- [3-2] rhs at midpoint                     --- #
            # ------------------------------------------------- #
            vx_mid, vy_mid, vz_mid = self._compute__rhs_velocity(
                np.array([px_mid]), np.array([py_mid]), np.array( [pz_mid] ) )
            Fx_mid, Fy_mid, Fz_mid = self._compute__rhs_LorentzForce(
                np.array([ x_mid]), np.array([ y_mid]), np.array( [ t_mid] ), np.array( [ s_mid] ), 
                np.array([px_mid]), np.array([py_mid]), np.array( [pz_mid] ), q_mc )
            dt_mid = ds_sliced / vz_mid[0]

            # ------------------------------------------------- #
            # --- [3-3] full update with k2                --- #
            # ------------------------------------------------- #
            refpart.x  = refpart.x  + vx_mid[0] * dt_mid
            refpart.y  = refpart.y  + vy_mid[0] * dt_mid
            refpart.z  = refpart.z  + vz_mid[0] * dt_mid
            refpart.t  = refpart.t  + cv        * dt_mid

            refpart.px = refpart.px + Fx_mid[0] * dt_mid
            refpart.py = refpart.py + Fy_mid[0] * dt_mid
            refpart.pz = refpart.pz + Fz_mid[0] * dt_mid
            refpart.pt = -1.0 * np.sqrt( 1.0 + refpart.px**2 + refpart.py**2 + refpart.pz**2 )
            refpart.s  = refpart.s  + ds_sliced

        elif ( self.int_method == "RK4" ):

            # ------------------------------------------------- #
            # --- [3-1] 4th-order Runge-Kutta integration   --- #
            # ------------------------------------------------- #
            x_f,y_f,t_f, px_f,py_f,pz_f = self._integrate__rk4_s(
                rx,ry,rt,rs, px,py,pz, q_mc, ds_sliced )

            # ------------------------------------------------- #
            # --- [3-2] update reference particle           --- #
            # ------------------------------------------------- #
            refpart.x  = x_f[0]
            refpart.y  = y_f[0]
            refpart.z  = refpart.z + ds_sliced
            refpart.t  = t_f[0]

            refpart.px = px_f[0]
            refpart.py = py_f[0]
            refpart.pz = pz_f[0]
            refpart.pt = -1.0 * np.sqrt( 1.0 + refpart.px**2 + refpart.py**2 + refpart.pz**2 )
            refpart.s  = refpart.s  + ds_sliced

        else:
            raise ValueError( "[ERROR] int_method must be ExplicitEuler, RK2, or RK4." )



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
                idcpu    = np.array( soa.get_idcpu_data(), copy=False )
                
                if (len(xp) == 0):
                    continue

                # ------------------------------------------------- #
                # --- [2-2] conversion to absolute coordinates  --- #
                # ------------------------------------------------- #
                xp_abs_i = ref_old["x"]  + xp
                yp_abs_i = ref_old["y"]  + yp
                tp_abs_i = ref_old["t"]  + tp                # --  global-time
                px_abs_i = ref_old["px"] + ref_bg_i * px
                py_abs_i = ref_old["py"] + ref_bg_i * py
                
                gm_abs_i = ref_gamma_i - ref_bg_i * pt
                pz_abs_i = np.sqrt( gm_abs_i**2 - px_abs_i**2 - py_abs_i**2 - 1.0 )
                sp_abs_i = np.repeat( [ ref_s_se ], len(tp), axis=0 )
                
                # ------------------------------------------------- #
                # --- [2-3] integration                         --- #
                # ------------------------------------------------- #
                ds_sliced = self.ds / self.nslice

                if ( self.int_method == "ExplicitEuler" ):
                    vx,vy,vz = self._compute__rhs_velocity    ( px_abs_i, py_abs_i, pz_abs_i )
                    Fx,Fy,Fz = self._compute__rhs_LorentzForce( xp_abs_i, yp_abs_i, tp_abs_i, sp_abs_i,\
                                                                px_abs_i, py_abs_i, pz_abs_i, q_mc )
                    dt_sec   = ds_sliced / vz

                    px_abs_f  = px_abs_i + Fx * dt_sec
                    py_abs_f  = py_abs_i + Fy * dt_sec
                    pz_abs_f  = pz_abs_i + Fz * dt_sec
                    pt_abs_f  = - np.sqrt( 1.0 + px_abs_f**2 + py_abs_f**2 + pz_abs_f**2 )
                    xp_abs_f  = xp_abs_i + vx * dt_sec
                    yp_abs_f  = yp_abs_i + vy * dt_sec
                    tp_abs_f  = tp_abs_i + cv * dt_sec
                    
                elif ( self.int_method == "RK2" ):
                    vx,vy,vz = self._compute__rhs_velocity    ( px_abs_i, py_abs_i, pz_abs_i )
                    Fx,Fy,Fz = self._compute__rhs_LorentzForce( xp_abs_i, yp_abs_i, tp_abs_i, sp_abs_i,\
                                                                px_abs_i, py_abs_i, pz_abs_i, q_mc )
                    dt_sec   = ds_sliced / vz

                    # ------------------------------------------------- #
                    # --- [2-4-1] midpoint state                    --- #
                    # ------------------------------------------------- #
                    xp_mid = xp_abs_i + vx * dt_sec * 0.5
                    yp_mid = yp_abs_i + vy * dt_sec * 0.5
                    tp_mid = tp_abs_i + cv * dt_sec * 0.5
                    px_mid = px_abs_i + Fx * dt_sec * 0.5
                    py_mid = py_abs_i + Fy * dt_sec * 0.5
                    pz_mid = pz_abs_i + Fz * dt_sec * 0.5
                    sp_mid = sp_abs_i +      ds_sliced * 0.5

                    # ------------------------------------------------- #
                    # --- [2-4-2] rhs at midpoint                   --- #
                    # ------------------------------------------------- #
                    vx_mid, vy_mid, vz_mid = self._compute__rhs_velocity(
                        px_mid, py_mid, pz_mid )
                    Fx_mid, Fy_mid, Fz_mid = self._compute__rhs_LorentzForce(
                        xp_mid, yp_mid, tp_mid, sp_mid,
                        px_mid, py_mid, pz_mid, q_mc )
                    dt_mid = ds_sliced / vz_mid

                    # ------------------------------------------------- #
                    # --- [2-4-3] full update with k2              --- #
                    # ------------------------------------------------- #
                    xp_abs_f = xp_abs_i + vx_mid * dt_mid
                    yp_abs_f = yp_abs_i + vy_mid * dt_mid
                    tp_abs_f = tp_abs_i + cv     * dt_mid

                    px_abs_f = px_abs_i + Fx_mid * dt_mid
                    py_abs_f = py_abs_i + Fy_mid * dt_mid
                    pz_abs_f = pz_abs_i + Fz_mid * dt_mid
                    pt_abs_f = - np.sqrt( 1.0 + px_abs_f**2 + py_abs_f**2 + pz_abs_f**2 )

                elif ( self.int_method == "RK4" ):
                    # ------------------------------------------------- #
                    # --- [2-4-1] 4th-order Runge-Kutta integration --- #
                    # ------------------------------------------------- #
                    xp_abs_f,yp_abs_f,tp_abs_f, px_abs_f,py_abs_f,pz_abs_f = self._integrate__rk4_s(
                        xp_abs_i,yp_abs_i,tp_abs_i,sp_abs_i,
                        px_abs_i,py_abs_i,pz_abs_i, q_mc, ds_sliced )
                    pt_abs_f = - np.sqrt( 1.0 + px_abs_f**2 + py_abs_f**2 + pz_abs_f**2 )

                else:
                    raise ValueError( "[ERROR] int_method must be ExplicitEuler, RK2, or RK4." )

                # ------------------------------------------------- #
                # --- [2-5] update variables                    --- #
                # ------------------------------------------------- #
                ref_bg_f  = np.sqrt( refpart.px**2 + refpart.py**2 + refpart.pz**2 )
                xp[:]     =     xp_abs_f - refpart.x
                yp[:]     =     yp_abs_f - refpart.y
                tp[:]     =     tp_abs_f - refpart.t
                px[:]     =   ( px_abs_f - refpart.px ) / ref_bg_f
                py[:]     =   ( py_abs_f - refpart.py ) / ref_bg_f
                pt[:]     =   ( pt_abs_f - refpart.pt ) / ref_bg_f

                # ------------------------------------------------- #
                # --- [2-6] aperture loss at end of this slice  --- #
                # ------------------------------------------------- #
                self._apply__elliptic_aperture( xp_map=xp, yp_map=yp, idcpu=idcpu )

                
    # ========================================================= #
    # ===  elliptic aperture loss                           === #
    # ========================================================= #
    def _apply__elliptic_aperture( self, xp_map, yp_map, idcpu ):
        """
        Elliptic aperture loss.
        xp_map, yp_map : transverse coordinates [m]
        idcpu          : writable AMReX uint64 particle-id array
        """

        # ------------------------------------------------- #
        # --- [1] elliptical aperture condition         --- #
        # ------------------------------------------------- #
        rho2  = ( xp_map - self.aperture_cx )**2 * self._aperture_inv_x2
        rho2 += ( yp_map - self.aperture_cy )**2 * self._aperture_inv_y2
        lost  = ( rho2 > 1.0 )

        # ------------------------------------------------- #
        # --- [2] invalidate lost particles             --- #
        # ------------------------------------------------- #
        if np.any( lost ):
            idcpu[lost] &= self._idcpu_valid_mask

            
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
                        
