import numpy as np
import pymeshlab
import pytest

import mojovcglib as mvc


def tetrahedron():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64
    )
    return vertices, faces


def pymeshlab_result(vertices, faces, filter_name, **parameters):
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(vertices, faces))
    getattr(mesh_set, filter_name)(**parameters)
    return mesh_set.current_mesh().vertex_matrix()


@pytest.mark.parametrize("steps", [1, 3])
def test_laplacian_matches_pymeshlab_vcglib(steps):
    vertices, faces = tetrahedron()
    expected = pymeshlab_result(
        vertices,
        faces,
        "apply_coord_laplacian_smoothing",
        stepsmoothnum=steps,
        cotangentweight=False,
        boundary=True,
    )
    np.testing.assert_allclose(
        mvc.laplacian_smooth(vertices, faces, steps=steps),
        expected,
        rtol=0,
        atol=1e-14,
    )


def test_laplacian_boundary_rule_matches_pymeshlab():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    expected = pymeshlab_result(
        vertices,
        faces,
        "apply_coord_laplacian_smoothing",
        stepsmoothnum=1,
        cotangentweight=False,
        boundary=True,
    )
    np.testing.assert_allclose(mvc.laplacian_smooth(vertices, faces), expected)


def test_smoothing_simd_tail_matches_scalar_boundary_rule():
    vertices = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 0]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    expected = pymeshlab_result(
        vertices,
        faces,
        "apply_coord_laplacian_smoothing",
        stepsmoothnum=1,
        cotangentweight=False,
        boundary=True,
    )
    np.testing.assert_allclose(
        mvc.laplacian_smooth(vertices, faces), expected, rtol=0, atol=1e-14
    )


def test_taubin_matches_pymeshlab_vcglib():
    vertices, faces = tetrahedron()
    expected = pymeshlab_result(
        vertices,
        faces,
        "apply_coord_taubin_smoothing",
        stepsmoothnum=2,
        lambda_=0.5,
        mu=-0.53,
    )
    np.testing.assert_allclose(
        mvc.taubin_smooth(vertices, faces, steps=2),
        expected,
        rtol=0,
        atol=2e-8,
    )


def test_hc_matches_pymeshlab_vcglib():
    vertices, faces = tetrahedron()
    expected = pymeshlab_result(
        vertices, faces, "apply_coord_hc_laplacian_smoothing"
    )
    np.testing.assert_allclose(
        mvc.hc_smooth(vertices, faces), expected, rtol=0, atol=1e-14
    )


def test_scale_dependent_matches_pymeshlab_vcglib():
    vertices, faces = tetrahedron()
    expected = pymeshlab_result(
        vertices,
        faces,
        "apply_coord_laplacian_smoothing_scale_dependent",
        stepsmoothnum=2,
        delta=pymeshlab.PureValue(0.01),
    )
    np.testing.assert_allclose(
        mvc.scale_dependent_smooth(vertices, faces, steps=2, delta=0.01),
        expected,
        rtol=0,
        atol=4e-10,
    )


def test_smoothing_empty_and_zero_steps():
    empty = np.empty((0, 3))
    faces = np.empty((0, 3), dtype=np.int64)
    assert mvc.hc_smooth(empty, faces).shape == (0, 3)
    vertices, tetra_faces = tetrahedron()
    np.testing.assert_array_equal(
        mvc.taubin_smooth(vertices, tetra_faces, steps=0), vertices
    )


def test_loop_matches_pymeshlab_vcglib():
    vertices, faces = tetrahedron()
    expected_set = pymeshlab.MeshSet()
    expected_set.add_mesh(pymeshlab.Mesh(vertices, faces))
    expected_set.meshing_surface_subdivision_loop(
        iterations=1, threshold=pymeshlab.PureValue(0), loopweight=0
    )
    actual_vertices, actual_faces = mvc.loop_subdivide(vertices, faces)
    expected_vertices = expected_set.current_mesh().vertex_matrix()
    expected_order = np.lexsort(expected_vertices.T[::-1])
    actual_order = np.lexsort(actual_vertices.T[::-1])
    np.testing.assert_allclose(
        actual_vertices[actual_order], expected_vertices[expected_order]
    )
    assert actual_faces.shape == expected_set.current_mesh().face_matrix().shape
    assert len(actual_faces) == 4 * len(faces)


def test_loop_boundary_rule_single_triangle():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    refined, refined_faces = mvc.loop_subdivide(vertices, faces)
    expected_even = np.array(
        [[0.125, 0.125, 0], [0.75, 0.125, 0], [0.125, 0.75, 0]]
    )
    np.testing.assert_allclose(refined[:3], expected_even)
    assert refined.shape == (6, 3)
    assert refined_faces.shape == (4, 3)


def test_loop_rejects_nonmanifold_edge():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    with pytest.raises(ValueError, match="edge-manifold"):
        mvc.loop_subdivide(vertices, faces)
