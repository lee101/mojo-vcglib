import numpy as np
import pymeshlab
import pytest
import trimesh

import mojovcglib as mvc


def tetrahedron():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64
    )
    return vertices, faces


def test_mass_properties_matches_trimesh_mirtich_reference():
    vertices, faces = tetrahedron()
    expected = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False
    ).mass_properties
    actual = mvc.mass_properties(vertices, faces)
    np.testing.assert_allclose(actual.mass, expected["mass"], atol=1e-15)
    np.testing.assert_allclose(
        actual.center_mass, expected["center_mass"], atol=1e-15
    )
    np.testing.assert_allclose(
        actual.inertia, expected["inertia"], atol=1e-15
    )


def test_mass_properties_cube_symmetry():
    box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    actual = mvc.mass_properties(box.vertices, box.faces)
    np.testing.assert_allclose(actual.mass, 8.0, atol=1e-14)
    np.testing.assert_allclose(actual.center_mass, 0, atol=1e-14)
    np.testing.assert_allclose(
        actual.inertia, np.eye(3) * (16.0 / 3.0), atol=1e-14
    )


def test_mass_properties_empty():
    result = mvc.mass_properties(
        np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    )
    assert result.mass == 0
    np.testing.assert_array_equal(result.center_mass, 0)
    np.testing.assert_array_equal(result.inertia, 0)


def test_closest_points_matches_trimesh_reference():
    vertices, faces = tetrahedron()
    queries = np.array([[0.2, 0.2, 0.2], [2, 2, 2], [-0.3, 0.2, 0.1]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    expected_points, expected_distance, _ = trimesh.proximity.closest_point_naive(
        mesh, queries
    )
    points, distances, indices = mvc.closest_points(vertices, faces, queries)
    np.testing.assert_allclose(points, expected_points, atol=1e-14)
    np.testing.assert_allclose(distances, expected_distance, atol=1e-14)
    assert np.all(indices >= 0)


def test_closest_point_degenerate_triangle_uses_edges():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    faces = np.array([[0, 1, 2]])
    point, distance, _ = mvc.closest_points(vertices, faces, [[0.5, 1, 0]])
    np.testing.assert_allclose(point[0], [0.5, 0, 0])
    np.testing.assert_allclose(distance[0], 1)


def test_ray_triangle_and_miss():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    origins = np.array([[0.25, 0.25, 1], [2, 2, 1]])
    directions = np.array([[0, 0, -1], [0, 0, -1]])
    distance, barycentric, indices = mvc.intersect_rays(
        vertices, faces, origins, directions
    )
    np.testing.assert_allclose(distance[0], 1)
    np.testing.assert_allclose(barycentric[0], [0.5, 0.25, 0.25])
    assert indices[0] == 0
    assert np.isinf(distance[1])
    assert indices[1] == -1
    assert np.isnan(barycentric[1]).all()


def test_geodesic_matches_pymeshlab_vcglib_on_planar_grid():
    side = 5
    vertices = np.array(
        [[x, y, 0.0] for y in range(side) for x in range(side)]
    )
    faces = []
    for y in range(side - 1):
        for x in range(side - 1):
            a = y * side + x
            faces.extend([[a, a + 1, a + side + 1], [a, a + side + 1, a + side]])
    faces = np.array(faces, dtype=np.int64)
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(vertices, faces))
    mesh_set.compute_scalar_by_geodesic_distance_from_given_point_per_vertex(
        startpoint=vertices[0], maxdistance=pymeshlab.PureValue(100)
    )
    expected = mesh_set.current_mesh().vertex_scalar_array()
    np.testing.assert_allclose(
        mvc.geodesic_distance(vertices, faces, [0]), expected, atol=1e-14
    )


def test_geodesic_disconnected_and_threshold():
    vertices = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    distance = mvc.geodesic_distance(vertices, faces, [0], max_distance=0.5)
    assert distance[0] == 0
    assert np.isinf(distance[3:]).all()


def test_geometry_rejects_nonfinite_and_fractional_inputs():
    vertices, faces = tetrahedron()
    with pytest.raises(ValueError, match="finite"):
        mvc.closest_points(vertices, faces, [[np.nan, 0, 0]])
    with pytest.raises(ValueError, match="finite"):
        mvc.intersect_rays(vertices, faces, [[0, 0, 0]], [[0, np.inf, 0]])
    with pytest.raises(TypeError, match="integers"):
        mvc.geodesic_distance(vertices, faces, [0.5])
    with pytest.raises(ValueError, match="nonnegative"):
        mvc.geodesic_distance(vertices, faces, [0], max_distance=-1)


def test_sampling_exact_count_and_surface_membership():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    for method in ("montecarlo", "stratified"):
        points = mvc.sample_surface(vertices, faces, 1000, method=method, seed=7)
        assert len(points) == 1000
        assert np.all(points[:, 0] >= 0)
        assert np.all(points[:, 1] >= 0)
        assert np.all(points[:, 0] + points[:, 1] <= 1 + 1e-15)
        np.testing.assert_array_equal(points[:, 2], 0)


def test_sampling_area_distribution():
    vertices = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [12, 0, 0], [10, 2, 0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    points = mvc.sample_surface(vertices, faces, 20_000, seed=19)
    large_fraction = np.mean(points[:, 0] > 5)
    np.testing.assert_allclose(large_fraction, 0.8, atol=0.015)


def test_qem_matches_pymeshlab_vcglib_on_tetrahedron():
    vertices, faces = tetrahedron()
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(vertices, faces))
    mesh_set.meshing_decimation_quadric_edge_collapse(
        targetfacenum=2,
        preservetopology=False,
        boundaryweight=0.5,
        autoclean=True,
    )
    actual = mvc.quadric_decimate(
        vertices, faces, 2, preserve_topology=False, boundary_weight=0.5
    )
    np.testing.assert_allclose(
        actual.vertices, mesh_set.current_mesh().vertex_matrix(), atol=2e-8
    )
    np.testing.assert_array_equal(
        actual.faces, mesh_set.current_mesh().face_matrix()
    )
    assert actual.collapsed_edges == 1


def test_qem_preserve_topology_rejects_tetrahedron_collapse():
    vertices, faces = tetrahedron()
    result = mvc.quadric_decimate(vertices, faces, 2, preserve_topology=True)
    assert result.collapsed_edges == 0
    assert len(result.faces) == 4


def test_qem_unconstrained_multi_collapse_is_deterministic():
    side = 6
    vertices = np.array(
        [[x, y, 0.05 * np.sin(x + y)] for y in range(side) for x in range(side)]
    )
    faces = []
    for y in range(side - 1):
        for x in range(side - 1):
            a = y * side + x
            faces.extend(
                [[a, a + 1, a + side + 1], [a, a + side + 1, a + side]]
            )
    faces = np.array(faces, dtype=np.int64)
    first = mvc.quadric_decimate(
        vertices, faces, 30, preserve_topology=False
    )
    second = mvc.quadric_decimate(
        vertices, faces, 30, preserve_topology=False
    )
    np.testing.assert_array_equal(first.faces, second.faces)
    np.testing.assert_allclose(first.vertices, second.vertices, rtol=0, atol=0)
    assert len(first.faces) <= 30
    assert first.collapsed_edges > 1
