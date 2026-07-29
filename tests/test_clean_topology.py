import numpy as np
import pymeshlab
import pytest

import mojovcglib as mvc


def test_clean_matches_pymeshlab_vcglib_filters():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [9.0, 9.0, 9.0],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 2, 1], [0, 0, 1]], dtype=np.int64)
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(vertices, faces))
    mesh_set.meshing_remove_duplicate_vertices()
    mesh_set.meshing_remove_duplicate_faces()
    mesh_set.meshing_remove_unreferenced_vertices()

    actual_vertices, actual_faces, stats = mvc.clean_mesh(vertices, faces)
    np.testing.assert_allclose(
        actual_vertices, mesh_set.current_mesh().vertex_matrix()
    )
    np.testing.assert_array_equal(
        actual_faces, mesh_set.current_mesh().face_matrix()
    )
    assert stats == mvc.CleanStats(1, 1, 1, 1)


def test_clean_zero_area_and_unreferenced():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [7, 7, 7]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    clean_vertices, clean_faces, stats = mvc.clean_mesh(vertices, faces)
    assert clean_vertices.shape == (3, 3)
    assert clean_faces.shape == (1, 3)
    assert stats.degenerate_faces == 1
    assert stats.unreferenced_vertices == 2


def test_clean_is_idempotent():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    first_v, first_f, first_stats = mvc.clean_mesh(vertices, faces)
    second_v, second_f, second_stats = mvc.clean_mesh(first_v, first_f)
    np.testing.assert_array_equal(first_v, second_v)
    np.testing.assert_array_equal(first_f, second_f)
    assert first_stats == second_stats == mvc.CleanStats(0, 0, 0, 0)


def test_clean_empty_mesh():
    vertices, faces, stats = mvc.clean_mesh(
        np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    )
    assert vertices.shape == (0, 3)
    assert faces.shape == (0, 3)
    assert stats == mvc.CleanStats(0, 0, 0, 0)


def test_mesh_rejects_silent_index_narrowing():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(TypeError, match="integers"):
        mvc.topology_stats(vertices, [[0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="int64"):
        mvc.topology_stats(
            vertices,
            np.array([[0, 1, 2**63]], dtype=np.uint64),
        )


def test_mesh_rejects_unsafe_coordinate_narrowing():
    with pytest.raises(ValueError, match="float64"):
        mvc.topology_stats(
            np.array([[0, 0, 0], [2**53 + 1, 0, 0], [0, 1, 0]]),
            [[0, 1, 2]],
        )


@pytest.mark.parametrize("face_count", [32767, 32768])
def test_clean_face_validity_parallel_threshold(face_count):
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.tile(np.array([[0, 1, 2]], dtype=np.int64), (face_count, 1))
    cleaned_vertices, cleaned_faces, stats = mvc.clean_mesh(vertices, faces)
    np.testing.assert_array_equal(cleaned_vertices, vertices)
    np.testing.assert_array_equal(cleaned_faces, [[0, 1, 2]])
    assert stats.duplicate_faces == face_count - 1


def test_topology_single_triangle():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]])
    assert mvc.topology_stats(vertices, faces) == mvc.TopologyStats(3, 3, 0)


def test_topology_watertight_tetrahedron():
    vertices, faces = tetrahedron()
    stats = mvc.topology_stats(vertices, faces)
    assert stats == mvc.TopologyStats(6, 0, 0)
    assert stats.watertight


def test_topology_nonmanifold_edge():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    stats = mvc.topology_stats(vertices, faces)
    assert stats.nonmanifold_edges == 1
    assert not stats.watertight


def tetrahedron():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64
    )
    return vertices, faces
