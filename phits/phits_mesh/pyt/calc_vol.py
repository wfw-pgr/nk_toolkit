import sys
import numpy as np
import gmsh

msh_file = "msh/model.msh"
rho = 7.8          # SUS316 [g/cm3]
unit = "m"        # Gmsh座標単位: "mm" を仮定

target_phys_tags = range(1, 10)  # Physical Group 1-9

gmsh.initialize()
gmsh.open(msh_file)

node_tags, coords, _ = gmsh.model.mesh.getNodes()
coords = np.array(coords).reshape(-1, 3)
tag_to_i = {int(tag): i for i, tag in enumerate(node_tags)}

def tet_volume(node_tags4):
    p = coords[[tag_to_i[int(n)] for n in node_tags4]]
    return abs(np.linalg.det(np.array([
        p[1] - p[0],
        p[2] - p[0],
        p[3] - p[0],
    ]))) / 6.0

def volume_of_entity(entity_tag):
    volume = 0.0

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3, entity_tag)

    for etype, nodes in zip(elem_types, elem_node_tags):
        name, dim, order, nnode, _, nprimary = gmsh.model.mesh.getElementProperties(etype)

        if dim != 3:
            continue

        if "Tetrahedron" not in name:
            raise RuntimeError(f"Unsupported 3D element: {name}")

        conn = np.array(nodes, dtype=np.int64).reshape(-1, nnode)

        for elem in conn:
            volume += tet_volume(elem[:4])

    return volume

def to_cm3(v):
    if unit == "mm":
        return v * 1.0e-3
    elif unit == "cm":
        return v
    elif unit == "m":
        return v * 1.0e6
    else:
        raise ValueError(f"Unknown unit: {unit}")

total_volume = 0.0

print("Physical Group volumes")
print("----------------------")

for phys_tag in target_phys_tags:
    try:
        entities = gmsh.model.getEntitiesForPhysicalGroup(3, phys_tag)
    except Exception:
        print(f"Physical Volume {phys_tag}: not found")
        continue

    name = gmsh.model.getPhysicalName(3, phys_tag)
    if name == "":
        name = f"PhysicalVolume_{phys_tag}"

    v = 0.0
    for ent in entities:
        v += volume_of_entity(ent)

    total_volume += v

    v_cm3 = to_cm3(v)
    mass_mg = rho * v_cm3 * 1.0e3

    print(f"[{phys_tag}] {name}")
    print(f"  volume = {v:.8e} {unit}^3")
    print(f"  volume = {v_cm3:.8e} cm^3")
    print(f"  mass   = {mass_mg:.8e} mg")

gmsh.finalize()

total_cm3 = to_cm3(total_volume)
total_mass_mg = rho * total_cm3 * 1.0e3

print("----------------------")
print("[total: Physical Volume 1-9]")
print(f"  volume = {total_volume:.8e} {unit}^3")
print(f"  volume = {total_cm3:.8e} cm^3")
print(f"  mass   = {total_mass_mg:.8e} mg")
