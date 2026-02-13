import os, sys
import invoke

@invoke.task(
    help={
        "stpFile"    : ".stp file ( solidworks )",
        "mshFile"    : ".msh file ( gmsh )",
        "bdfFile"    : ".bdf file ( phits )",
        "configFile" : ".json file",
        "phits_mesh" : "Enable PHITS mesh mode ( true/false )",
    }
)
def mesh( context, stpFile="msh/model.stp", configFile="dat/mesh.json", mshFile=None, bdfFile=None, phits_mesh=True ):
    """
    run mesh__solidworksSTEP( stpFile=..., mshFile=..., bdfFile=..., configFile=.., phits_mesh=... ).
    """
    import nk_toolkit.gmsh.mesh__solidworksSTEP as mss
    mss.mesh__solidworksSTEP( stpFile=stpFile, configFile=configFile, \
                              mshFile=mshFile, bdfFile=bdfFile, phits_mesh=phits_mesh )
