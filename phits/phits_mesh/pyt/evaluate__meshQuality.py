import numpy  as np
import pandas as pd
import meshio
import nk_toolkit.plot.load__config   as lcf
import nk_toolkit.plot.gplot1D        as gp1


# ========================================================= #
# ===  calculate gamma of tetra                         === #
# ========================================================= #
def calculate__gammaOfTetra( mesh=None, mshFile=None, outFile="dat/mesh_quality.csv", \
                             bins = 20, density=False, eps=1.0e-30 ):

    if ( mesh is None ):
        if ( mshFile is None ):
            raise ValueError("mesh must not be None")
        else:
            mesh = meshio.read( mshFile )
    
    points       = np.asarray( mesh.points[:, :3], dtype=np.float64 )
    tetra_blocks = [ cell_block.data for cell_block in mesh.cells if cell_block.type == "tetra" ]
    if len(tetra_blocks) == 0:
        raise ValueError("No 'tetra' cells were found in the mesh")

    conn  = np.vstack(tetra_blocks)
    X     = points[conn]
    p0    = X[:, 0, :]
    p1    = X[:, 1, :]
    p2    = X[:, 2, :]
    p3    = X[:, 3, :]

    # 6 edge lengths
    e01   = np.linalg.norm(p1 - p0, axis=1)
    e02   = np.linalg.norm(p2 - p0, axis=1)
    e03   = np.linalg.norm(p3 - p0, axis=1)
    e12   = np.linalg.norm(p2 - p1, axis=1)
    e13   = np.linalg.norm(p3 - p1, axis=1)
    e23   = np.linalg.norm(p3 - p2, axis=1)
    L_max = np.maximum.reduce([e01, e02, e03, e12, e13, e23])

    # 4 face areas
    A012  = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    A013  = 0.5 * np.linalg.norm(np.cross(p1 - p0, p3 - p0), axis=1)
    A023  = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0), axis=1)
    A123  = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1), axis=1)
    S_F   = A012 + A013 + A023 + A123

    # Volume
    V     = np.abs( np.einsum( "ij,ij->i", p1-p0, np.cross( p2-p0, p3-p0 ) ) ) / 6.0

    # Gamma
    denom         = S_F * L_max
    gammas        = np.zeros( len(conn), dtype=np.float64 )
    valid         = denom > eps
    gammas[valid] = 6.0 * np.sqrt(6.0) * V[valid] / denom[valid]
    np.clip( gammas, 0.0, 1.0, out=gammas )

    # binning
    hist, bin_edges = np.histogram( gammas, bins=bins,
                                    range=(0.0,1.0), density=density )
    bin_centers     = 0.5 * ( bin_edges[:-1] + bin_edges[1:] )
    hists           = { "histgram":hist, 
                        "bin_edges":bin_edges, "bin_centers":bin_centers }
    if ( outFile is not None ):
        df = pd.DataFrame.from_dict( { "histgram":hist, "bin_centers":bin_centers, \
                                       "bin_edgeL":bin_edges[:-1], "bin_edgeR":bin_edges[1:] } )
        df.to_csv( outFile, index=False )
    
    return( gammas, hists )


# ========================================================= #
# ===  plot histgrams                                   === #
# ========================================================= #

def plot__histgrams( hists, norm=True ):

    config   = lcf.load__config()
    config_  = {
        "figure.size"        : [3.5,3.5],
        "figure.pngFile"     : "png/mesh_histgram.png", 
        "figure.position"    : [ 0.16, 0.16, 0.94, 0.94 ],
        "ax1.y.normalize"    : 1.0e0, 
        "ax1.x.range"        : { "auto":False, "min": 0.0, "max": 1.0, "num":6 },
        "ax1.y.range"        : { "auto":False, "min": 0.0, "max":20.0, "num":3 },
        "ax1.x.label"        : r"$\Gamma$",
        "ax1.y.label"        : "Percentage of elements (%)",
        "ax1.x.minor.nticks" : 1, 
        "plot.marker"        : "o",
        "plot.markersize"    : 3.0,
        "legend.fontsize"    : 9.0, 
    }
    config = { **config, **config_ }
    label1 = "Frontal-Delaunay / HXT"
    label2 = "MeshAdapt / Delaunay"
    
    fig        = gp1.gplot1D( config=config )
    hist_norm  = hists["histgram"] / np.sum( hists["histgram"] ) * 100.0
    fig.add__bar( xAxis=hists["bin_centers"], yAxis=hist_norm, label=label1 )
    fig.set__axis()
    fig.set__legend()
    fig.save__figure()

    

if __name__ == "__main__":
    
    mshFile       = "msh/model_hq.msh"
    # mshFile       = "msh/model_MeshAdapt.msh"
    gammas, hists = calculate__gammaOfTetra( mshFile=mshFile )
    
    

    print(f"number of tetrahedra = {len(gammas)}")
    print(f"gamma min  = {gammas.min():.16f}")
    print(f"gamma max  = {gammas.max():.16f}")
    print(f"gamma mean = {gammas.mean():.16f}")

    # Example: indices of low-quality elements
    threshold = 0.10
    bad = np.where(gammas < threshold)[0]
    print(f"gamma < {threshold} : {len(bad)}")

    # if len(bad) > 0:
    #     print("first 10 low-quality tetrahedra:")
    #     for i in bad[:10]:
    #         print(f"  elem_id={i}, gamma={gammas[i]:.16f}, nodes={conn[i]}")

    plot__histgrams( hists )
