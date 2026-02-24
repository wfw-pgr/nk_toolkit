import os, sys
import meshio
import numpy  as np
import pandas as pd


# ========================================================= #
# ===  map__csvOnMesh                                   === #
# ========================================================= #

def map__csvOnMesh( mshFile=None, outFile=None, cellDataFiles=[] ):
    
    # ------------------------------------------------- #
    # --- [1] Arguments                             --- #
    # ------------------------------------------------- #
    if ( mshFile is None ): sys.exit( "[map__csvOnMesh.py] mshFile == ???" )
    if ( outFile is None ):
        extention = os.path.splitext( mshFile )[1]
        outFile   = mshFile.replace( extention, ".vtu" )

    # ------------------------------------------------- #
    # --- [2] cellDataFiles                         --- #
    # ------------------------------------------------- #
    import nkMeshRoutines.convert__withMeshIO as cwm
    cwm.convert__withMeshIO( mshFile=mshFile, outFile=outFile, cellDataFiles=cellDataFiles, replaceData=False )

    # ------------------------------------------------- #
    # --- [3] add material property                 --- #
    # ------------------------------------------------- #
    # import nkMeshRoutines.assign__materialProperty as amp
    # propertyFile = "dat/materialProperty.conf"
    # amp.assign__materialProperty( inpFile=outFile, outFile=outFile, propertyFile=propertyFile )

    return()
    

# ========================================================= #
# ===  integrate__onVolume                              === #
# ========================================================= #

def integrate__onVolume( mshFile=None, outFile=None, cellDataFiles=[] ):

    # ------------------------------------------------- #
    # --- [4] calculate volume integral             --- #
    # ------------------------------------------------- #
    import nkMeshRoutines.calculate__volumeIntegral as cvi
    target       = "Dose[J/m^3/source]"
    physNum_list = [ 301, 302, 303, 304, 305, 306, 307, 308, 309 ]
    ints         = []
    for ik,physNum in enumerate( physNum_list ):
        ret = cvi.calculate__volumeIntegral( inpFile=outFile, target=target, physNum=physNum )
        print( " (physNum,Heat,Volume,AvgHeat) == ( {0}, {1[0]}, {1[1]}, {1[2]} )".format( physNum, ret ) )
        ints += [ ret ]
    ints   = np.array( ints    )
    volInt = np.sum( ints[:,0] )
    volTot = np.sum( ints[:,1] )
    volAvg = np.sum( ints[:,2] )
    print( "-"*70 )
    print( "--- volume integral summary ---" )
    print( "-"*70 )
    for ik,physNum in enumerate( physNum_list ):
        print( " (physNum,Heat,Volume,AvgHeat) == ( {0}, {1[0]:.6e}, {1[1]:.6e}, {1[2]:.6e} )".format( physNum, ints[ik] ) )
    print( "-"*70 )
    print( " [[Total]] " )
    print( "  volInt :: {} ".format( volInt ) )
    print( "  volTot :: {} ".format( volTot ) )
    print( "  volAvg :: {} ".format( volAvg ) )
    print( "-"*70 )

    # ------------------------------------------------- #
    # --- [5] save volume integral                  --- #
    # ------------------------------------------------- #
    Data = np.concatenate( [ (np.array(physNum_list ))[:,np.newaxis], ints[:,:] ], axis=1 )
    import nkUtilities.save__pointFile as spf
    outFile   = "dat/volume_integral.dat"
    names     = ["physNum","volInt", "volTot", "volAvg"]
    spf.save__pointFile( outFile=outFile, Data=Data, names=names )



# ========================================================= #
# ===  redistribute__cell2point_forFemtet               === #
# ========================================================= #

def redistribute__cell2point_forFemtet( mshFile="msh/model.msh", csvFile="out/heatload.csv",
                                        outFile="dat/output.csv", \
                                        target_physNum=1, key="Dose[J/m^3/source]", shape="tetra" ):

    mm = 1.0e-3
    
    # ------------------------------------------------- #
    # --- [1] routines to use                       --- #
    # ------------------------------------------------- #
    def calculate__tetVolumes( points, tets ):
        # points: (N,3), tets: (M,4) int
        p0  = points[tets[:,0]]
        p1  = points[tets[:,1]]
        p2  = points[tets[:,2]]
        p3  = points[tets[:,3]]
        vol = np.abs( np.einsum("ij,ij->i", np.cross(p1-p0, p2-p0), (p3-p0) ) )/6.0
        return( vol )

    # ------------------------------------------------- #
    # --- [2] bdf file read                         --- #
    # ------------------------------------------------- #
    mesh = meshio.read( mshFile )
    
    if ( key in mesh.cell_data_dict ):
        rdata  = mesh.cell_data_dict[key]
    else:
        rdata  = pd.read_csv( csvFile )[key].to_numpy( dtype=float )
        
    # ------------------------------------------------- #
    # --- [3] 指定する物体IDのcell_dataを取得       --- #
    # ------------------------------------------------- #
    physNums = mesh.cell_data_dict["gmsh:physical"][shape]
    mask     = ( physNums==target_physNum )
    tets_tgt = ( mesh.cells_dict[shape] )[mask]
    w_cell   = rdata[mask].astype(float)
    
    # ------------------------------------------------- #
    # --- [4] redistribute                          --- #
    # ------------------------------------------------- #
    vol      = calculate__tetVolumes( mesh.points, tets_tgt )  # (n_tets_tgt,)
    node_sum = np.zeros( len( mesh.points ), dtype=float )
    node_wgt = np.zeros( len( mesh.points ), dtype=float )
    
    for ax in range(4):
        idx = tets_tgt[:, ax]
        np.add.at( node_sum, idx, w_cell*vol )
        np.add.at( node_wgt, idx, vol )
        
    used_nodes = np.unique( tets_tgt.reshape(-1) )
    w_node     = node_sum[used_nodes] / np.maximum( node_wgt[used_nodes], 1e-300 )

    # ------------------------------------------------- #
    # --- [6] save as csv & return                  --- #
    # ------------------------------------------------- #
    xyzp = mesh.points[ used_nodes ] / mm
    Data = np.c_[ xyzp, w_node ]
    np.savetxt( outFile, Data, delimiter=",", \
                header="xp(mm),yp(mm),zp(mm),{}".format( key ), fmt="%15.8e" )
    return( Data )
    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    mshFile       = "msh/model.msh"
    outFile       = "msh/heatload.vtu"
    cellDataFiles = [ "out/heatload.csv" ]
    map__csvOnMesh( mshFile=mshFile, outFile=outFile, cellDataFiles=cellDataFiles )
    
    
