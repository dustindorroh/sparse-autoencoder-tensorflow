import pymesh
mesh = pymesh.load_mesh('bunny.obj')
mesh = pymesh.load_mesh('square_2D.obj')
mesh = pymesh.load_mesh('cube.obj')

mesh.enable_connectivity()

print mesh.num_vertices, mesh.num_faces, mesh.num_voxels
print mesh.dim, mesh.vertex_per_face, mesh.vertex_per_voxel
print mesh.vertices


'''
The following vertex attributes are predifined:

    vertex_normal: A vector field representing surface normals. Zero vectors are assigned to vertices in the interior.
    vertex_volume: A scalar field representing the lumped volume of each vertex (e.g. 1/4 of the total volume of all neighboring tets for tetrahedron mesh.).
    vertex_area: A scalar field representing the lumped surface area of each vertex (e.g. 1/3 of the total face area of its 1-ring neighborhood).
    vertex_laplacian: A vector field representing the discretized Laplacian vector.
    vertex_mean_curvature: A scalar field representing the mean curvature field of the mesh.
    vertex_gaussian_curvature: A scalar field representing the Gaussian curvature field of the mesh.
    vertex_index: A scalar field representing the index of each vertex.
    vertex_valance: A scalar field representing the valance of each vertex.
    vertex_dihedral_angle: A scalar field representing the max dihedral angle of all edges adjacent to this vertex.
'''

mesh.add_attribute('vertex_normal')
#mesh.add_attribute('vertex_volume')
#mesh.add_attribute('vertex_area')
#mesh.add_attribute('vertex_laplacian')
#mesh.add_attribute('vertex_mean_curvature')
#mesh.add_attribute('vertex_gaussian_curvature')
mesh.add_attribute('vertex_index')
#mesh.add_attribute('vertex_valance')
#mesh.add_attribute('vertex_dihedral_angle')

mesh.get_attribute_names()

vertex_normal = mesh.get_attribute('vertex_normal').reshape(-1,mesh.dim)

# Euclidean distance between neighborhood vertices to the tangent plane of the vertex v.
vi=0
neighbor_vertices_indexes = mesh.get_vertex_adjacent_vertices(vi)

# Get the target vertex index and normal  
n = vertex_normal[vi]
v = mesh.vertices[vi]

n_jk = vertex_normal[neighbor_vertices_indexes]
v_jk = mesh.vertices[neighbor_vertices_indexes]


distances = np.abs((v - v_jk).dot(n))/(np.linalg.norm(n)
thetas = np.arccos(np.dot(n_jk_all,n)/(np.linalg.norm(n)*np.linalg.norm(n_jk_all,axis=1)))
thetas = np.arccos(np.dot(n_jk,n))

np.rad2deg(thetas)

