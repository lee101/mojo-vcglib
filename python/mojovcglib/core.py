"""Public mesh algorithms backed by the Mojo VCGLib port."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._lib import addr, f64, i64, lib


_VERTEX_KEY_DTYPE = np.dtype(
    [("x", np.float64), ("y", np.float64), ("z", np.float64)]
)


@dataclass(frozen=True)
class CleanStats:
    duplicate_vertices: int
    degenerate_faces: int
    duplicate_faces: int
    unreferenced_vertices: int


@dataclass(frozen=True)
class TopologyStats:
    edges: int
    boundary_edges: int
    nonmanifold_edges: int

    @property
    def watertight(self) -> bool:
        return self.boundary_edges == 0 and self.nonmanifold_edges == 0


@dataclass(frozen=True)
class MassProperties:
    mass: float
    center_mass: np.ndarray
    inertia: np.ndarray


@dataclass(frozen=True)
class DecimationResult:
    vertices: np.ndarray
    faces: np.ndarray
    collapsed_edges: int


def _mesh(vertices, faces, *, allow_invalid: bool = False) -> tuple[np.ndarray, np.ndarray]:
    vertices = f64(vertices)
    faces = i64(faces)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("vertices must have shape (n, 3)")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("faces must have shape (m, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite")
    if not allow_invalid and faces.size:
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("face index is outside the vertex array")
    return vertices, faces


def _topology(faces: np.ndarray, vertex_count: int):
    if len(faces) == 0:
        empty_edges = np.empty((0, 2), dtype=np.int64)
        return (
            empty_edges,
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
        )
    if vertex_count <= np.iinfo(np.int64).max // max(vertex_count, 1):
        key_rows = np.empty_like(faces)
        high = np.empty(len(faces), dtype=np.int64)
        for column, following_column in ((0, 1), (1, 2), (2, 0)):
            np.minimum(
                faces[:, column], faces[:, following_column], out=key_rows[:, column]
            )
            key_rows[:, column] *= vertex_count
            np.maximum(faces[:, column], faces[:, following_column], out=high)
            key_rows[:, column] += high
        keys = key_rows.reshape(-1)
        unique_keys, inverse, counts = np.unique(
            keys, return_inverse=True, return_counts=True
        )
        edges = np.empty((len(unique_keys), 2), dtype=np.int64)
        edges[:, 0] = unique_keys // vertex_count
        edges[:, 1] = unique_keys % vertex_count
    else:
        canonical = np.empty((faces.size, 2), dtype=np.int64)
        canonical_rows = canonical.reshape(-1, 3, 2)
        for column, following_column in ((0, 1), (1, 2), (2, 0)):
            np.minimum(
                faces[:, column],
                faces[:, following_column],
                out=canonical_rows[:, column, 0],
            )
            np.maximum(
                faces[:, column],
                faces[:, following_column],
                out=canonical_rows[:, column, 1],
            )
        edges, inverse, counts = np.unique(
            canonical, axis=0, return_inverse=True, return_counts=True
        )
    return edges, inverse, counts, inverse.reshape(-1, 3)


def _border_mask(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    _, inverse, counts, _ = _topology(faces, vertex_count)
    if not len(faces):
        return np.empty((0, 3), dtype=np.int64)
    return np.ascontiguousarray((counts[inverse] == 1).reshape(-1, 3), dtype=np.int64)


def clean_mesh(
    vertices,
    faces,
    *,
    remove_duplicate_vertices: bool = True,
    remove_degenerate_faces: bool = True,
    remove_duplicate_faces: bool = True,
    remove_unreferenced_vertices: bool = True,
    epsilon: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, CleanStats]:
    """Apply VCGLib-style conservative cleanup and return compact arrays."""
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    vertices, faces = _mesh(vertices, faces, allow_invalid=True)
    original_vertex_count = len(vertices)
    duplicate_vertices = 0

    mapped_faces = faces.copy()
    if remove_duplicate_vertices and len(vertices):
        vertex_keys = vertices.view(_VERTEX_KEY_DTYPE).reshape(-1)
        _, first, inverse = np.unique(
            vertex_keys, return_index=True, return_inverse=True
        )
        representatives = first[inverse]
        in_range = (mapped_faces >= 0) & (mapped_faces < len(vertices))
        mapped_faces[in_range] = representatives[mapped_faces[in_range]]
        duplicate_vertices = len(vertices) - len(first)

    valid = np.empty(len(mapped_faces), dtype=np.int64)
    kept = lib().mvc_face_validity_f64(
        addr(vertices),
        addr(mapped_faces),
        addr(valid),
        len(vertices),
        len(mapped_faces),
        float(epsilon),
    )
    invalid_count = len(mapped_faces) - int(kept)
    if invalid_count and not remove_degenerate_faces:
        raise ValueError("mesh contains invalid or degenerate faces")
    mapped_faces = mapped_faces[valid.astype(bool)]

    duplicate_faces = 0
    if remove_duplicate_faces and len(mapped_faces):
        keys = np.sort(mapped_faces, axis=1)
        vertex_count = len(vertices)
        if vertex_count <= np.iinfo(np.int64).max // max(vertex_count**2, 1):
            face_keys = (
                (keys[:, 0] * vertex_count + keys[:, 1]) * vertex_count
                + keys[:, 2]
            )
            order = np.argsort(face_keys, kind="stable")
            equal_next = face_keys[order[:-1]] == face_keys[order[1:]]
        else:
            order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
            sorted_keys = keys[order]
            equal_next = np.all(sorted_keys[:-1] == sorted_keys[1:], axis=1)
        keep = np.ones(len(mapped_faces), dtype=bool)
        keep[order[:-1][equal_next]] = False
        duplicate_faces = int((~keep).sum())
        mapped_faces = mapped_faces[keep]

    referenced = np.zeros(len(vertices), dtype=bool)
    if mapped_faces.size:
        referenced[mapped_faces.reshape(-1)] = True
    if not remove_unreferenced_vertices:
        return (
            vertices.copy(),
            mapped_faces,
            CleanStats(
                duplicate_vertices, invalid_count, duplicate_faces, 0
            ),
        )

    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[referenced] = np.arange(referenced.sum(), dtype=np.int64)
    compact_faces = remap[mapped_faces]
    compact_vertices = vertices[referenced]
    unreferenced = original_vertex_count - duplicate_vertices - len(compact_vertices)
    return (
        compact_vertices,
        np.ascontiguousarray(compact_faces),
        CleanStats(
            duplicate_vertices,
            invalid_count,
            duplicate_faces,
            max(0, unreferenced),
        ),
    )


def topology_stats(vertices, faces) -> TopologyStats:
    vertices, faces = _mesh(vertices, faces)
    edges, _, counts, _ = _topology(faces, len(vertices))
    if not len(edges):
        return TopologyStats(0, 0, 0)
    expanded = np.repeat(edges, counts, axis=0)
    result = np.empty(3, dtype=np.int64)
    lib().mvc_edge_incidence_stats_i64(addr(expanded), len(expanded), addr(result))
    return TopologyStats(*(int(value) for value in result))


def _smooth(vertices, faces, method: str, steps: int, **kwargs) -> np.ndarray:
    vertices, faces = _mesh(vertices, faces)
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if any(not np.isfinite(value) for value in kwargs.values()):
        raise ValueError("smoothing parameters must be finite")
    result = vertices.copy()
    if not len(vertices) or not len(faces) or steps == 0:
        return result
    border = _border_mask(faces, len(vertices))
    sums = np.empty_like(vertices)
    counts = np.empty(len(vertices), dtype=np.float64)
    library = lib()
    common = (
        addr(vertices),
        addr(faces),
        addr(border),
        addr(result),
        addr(sums),
    )
    if method == "laplacian":
        library.mvc_laplacian_f64(
            *common, addr(counts), len(vertices), len(faces), steps
        )
    elif method == "taubin":
        library.mvc_taubin_f64(
            *common,
            addr(counts),
            len(vertices),
            len(faces),
            steps,
            float(kwargs["lambda_value"]),
            float(kwargs["mu"]),
        )
    elif method == "hc":
        diffs = np.empty_like(vertices)
        library.mvc_hc_f64(
            *common,
            addr(diffs),
            addr(counts),
            len(vertices),
            len(faces),
            steps,
        )
    elif method == "scale_dependent":
        library.mvc_scale_dependent_f64(
            *common,
            addr(counts),
            len(vertices),
            len(faces),
            steps,
            float(kwargs["delta"]),
        )
    else:
        raise ValueError(f"unknown smoothing method {method!r}")
    return result


def laplacian_smooth(vertices, faces, *, steps: int = 1) -> np.ndarray:
    return _smooth(vertices, faces, "laplacian", steps)


def taubin_smooth(
    vertices,
    faces,
    *,
    steps: int = 1,
    lambda_value: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    return _smooth(
        vertices,
        faces,
        "taubin",
        steps,
        lambda_value=lambda_value,
        mu=mu,
    )


def hc_smooth(vertices, faces, *, steps: int = 1) -> np.ndarray:
    return _smooth(vertices, faces, "hc", steps)


def scale_dependent_smooth(
    vertices, faces, *, steps: int = 1, delta: float = 0.01
) -> np.ndarray:
    return _smooth(
        vertices, faces, "scale_dependent", steps, delta=delta
    )


def loop_subdivide(vertices, faces, *, iterations: int = 1):
    """Apply classical Loop odd/even rules to a manifold triangle mesh."""
    vertices, faces = _mesh(vertices, faces)
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    vertices = vertices.copy()
    faces = faces.copy()
    for _ in range(iterations):
        if not len(faces):
            break
        edges, inverse, counts, face_edges = _topology(faces, len(vertices))
        if np.any(counts > 2):
            raise ValueError("Loop subdivision requires edge-manifold input")
        opposites = np.full((len(edges), 2), -1, dtype=np.int64)
        edge_order = np.argsort(inverse, kind="stable")
        starts = np.cumsum(counts) - counts
        opposite_values = faces[:, [2, 0, 1]].reshape(-1)
        opposites[:, 0] = opposite_values[edge_order[starts]]
        paired = counts == 2
        opposites[paired, 1] = opposite_values[
            edge_order[(starts + counts - 1)[paired]]
        ]

        vertex_count = len(vertices)
        adjacency_keys = np.concatenate(
            (
                edges[:, 0] * vertex_count + edges[:, 1],
                edges[:, 1] * vertex_count + edges[:, 0],
            )
        )
        adjacency_keys.sort()
        adjacency_sources = adjacency_keys // vertex_count
        degrees = np.bincount(adjacency_sources, minlength=vertex_count)
        offsets = np.empty(vertex_count + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(degrees, out=offsets[1:])
        neighbors = np.ascontiguousarray(
            adjacency_keys % vertex_count, dtype=np.int64
        )

        boundary_neighbors = np.full((len(vertices), 2), -1, dtype=np.int64)
        boundary_edges = edges[counts == 1]
        if len(boundary_edges):
            boundary_keys = np.concatenate(
                (
                    boundary_edges[:, 0] * vertex_count + boundary_edges[:, 1],
                    boundary_edges[:, 1] * vertex_count + boundary_edges[:, 0],
                )
            )
            boundary_keys.sort()
            boundary_sources = boundary_keys // vertex_count
            boundary_degrees = np.bincount(
                boundary_sources, minlength=vertex_count
            )
            if np.any((boundary_degrees != 0) & (boundary_degrees != 2)):
                raise ValueError("Loop subdivision requires manifold boundaries")
            boundary_neighbors[boundary_sources[::2], 0] = (
                boundary_keys[::2] % vertex_count
            )
            boundary_neighbors[boundary_sources[1::2], 1] = (
                boundary_keys[1::2] % vertex_count
            )
        refined_vertices = np.empty((len(vertices) + len(edges), 3))
        lib().mvc_loop_vertices_f64(
            addr(vertices),
            addr(edges),
            addr(opposites),
            addr(offsets),
            addr(neighbors),
            addr(boundary_neighbors),
            addr(refined_vertices),
            len(vertices),
            len(edges),
        )
        midpoint = len(vertices) + face_edges
        a, b, c = faces.T
        ab, bc, ca = midpoint.T
        refined_faces = np.empty((len(faces), 4, 3), dtype=np.int64)
        refined_faces[:, 0, 0] = a
        refined_faces[:, 0, 1] = ab
        refined_faces[:, 0, 2] = ca
        refined_faces[:, 1, 0] = b
        refined_faces[:, 1, 1] = bc
        refined_faces[:, 1, 2] = ab
        refined_faces[:, 2, 0] = c
        refined_faces[:, 2, 1] = ca
        refined_faces[:, 2, 2] = bc
        refined_faces[:, 3, 0] = ab
        refined_faces[:, 3, 1] = bc
        refined_faces[:, 3, 2] = ca
        refined_faces = refined_faces.reshape(-1, 3)
        vertices = refined_vertices
        faces = np.ascontiguousarray(refined_faces, dtype=np.int64)
    return vertices, faces


def mass_properties(vertices, faces) -> MassProperties:
    vertices, faces = _mesh(vertices, faces)
    result = np.empty(13, dtype=np.float64)
    lib().mvc_mass_properties_f64(
        addr(vertices), addr(faces), addr(result), len(faces)
    )
    return MassProperties(
        float(result[0]), result[1:4].copy(), result[4:13].reshape(3, 3).copy()
    )


def sample_surface(
    vertices,
    faces,
    count: int,
    *,
    method: str = "montecarlo",
    seed: int = 1,
) -> np.ndarray:
    vertices, faces = _mesh(vertices, faces)
    if count < 0:
        raise ValueError("count must be nonnegative")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    result = np.empty((count, 3), dtype=np.float64)
    if count == 0 or len(faces) == 0:
        return result[:0]
    if method == "montecarlo":
        cumulative = np.empty(len(faces) + 1, dtype=np.float64)
        written = lib().mvc_montecarlo_f64(
            addr(vertices),
            addr(faces),
            addr(result),
            addr(cumulative),
            len(faces),
            count,
            seed,
        )
    elif method == "stratified":
        written = lib().mvc_stratified_f64(
            addr(vertices),
            addr(faces),
            addr(result),
            len(faces),
            count,
            seed,
        )
    else:
        raise ValueError("method must be 'montecarlo' or 'stratified'")
    if written < 0 or written > count:
        raise RuntimeError("sampling kernel returned an invalid output length")
    return result[: int(written)].copy()


def closest_points(vertices, faces, queries):
    vertices, faces = _mesh(vertices, faces)
    queries = f64(queries)
    if queries.ndim != 2 or queries.shape[1:] != (3,):
        raise ValueError("queries must have shape (q, 3)")
    if not np.isfinite(queries).all():
        raise ValueError("queries must be finite")
    points = np.empty_like(queries)
    distances = np.empty(len(queries), dtype=np.float64)
    indices = np.empty(len(queries), dtype=np.int64)
    if len(queries):
        lib().mvc_closest_points_f64(
            addr(vertices),
            addr(faces),
            addr(queries),
            addr(points),
            addr(distances),
            addr(indices),
            len(faces),
            len(queries),
        )
    distances[indices < 0] = np.inf
    return points, distances, indices


def intersect_rays(vertices, faces, origins, directions):
    vertices, faces = _mesh(vertices, faces)
    origins = f64(origins)
    directions = f64(directions)
    if origins.ndim != 2 or origins.shape[1:] != (3,):
        raise ValueError("origins must have shape (r, 3)")
    if directions.shape != origins.shape:
        raise ValueError("directions must have the same shape as origins")
    if not np.isfinite(origins).all() or not np.isfinite(directions).all():
        raise ValueError("ray origins and directions must be finite")
    distances = np.empty(len(origins), dtype=np.float64)
    barycentric = np.empty_like(origins)
    indices = np.empty(len(origins), dtype=np.int64)
    if len(origins):
        lib().mvc_ray_mesh_f64(
            addr(vertices),
            addr(faces),
            addr(origins),
            addr(directions),
            addr(distances),
            addr(barycentric),
            addr(indices),
            len(faces),
            len(origins),
        )
    distances[indices < 0] = np.inf
    barycentric[indices < 0] = np.nan
    return distances, barycentric, indices


def geodesic_distance(
    vertices, faces, seeds, *, max_distance: float = np.inf
) -> np.ndarray:
    vertices, faces = _mesh(vertices, faces)
    if np.isnan(max_distance) or max_distance < 0:
        raise ValueError("max_distance must be nonnegative")
    seeds = i64(seeds).reshape(-1)
    if seeds.size and (seeds.min() < 0 or seeds.max() >= len(vertices)):
        raise ValueError("seed is outside the vertex array")
    distances = np.empty(len(vertices), dtype=np.float64)
    sources = np.empty(len(vertices), dtype=np.int64)
    visited = np.empty(len(vertices), dtype=np.int64)
    heap = np.empty(len(vertices), dtype=np.int64)
    if not len(vertices):
        return distances
    incidence_offsets = np.empty(len(vertices) + 1, dtype=np.int64)
    incidence_offsets[0] = 0
    if len(faces):
        flattened_faces = faces.reshape(-1)
        incidence_order = np.argsort(flattened_faces, kind="stable")
        incidence_counts = np.bincount(
            flattened_faces, minlength=len(vertices)
        )
        np.cumsum(incidence_counts, out=incidence_offsets[1:])
        incident_faces = np.ascontiguousarray(
            incidence_order // 3, dtype=np.int64
        )
    else:
        incidence_offsets[1:] = 0
        incident_faces = np.empty(0, dtype=np.int64)
    threshold = (
        float(max_distance)
        if np.isfinite(max_distance)
        else np.finfo(np.float64).max
    )
    lib().mvc_geodesic_f64(
        addr(vertices),
        addr(faces),
        addr(seeds),
        addr(distances),
        addr(sources),
        addr(visited),
        addr(heap),
        addr(incidence_offsets),
        addr(incident_faces),
        len(vertices),
        len(faces),
        len(seeds),
        threshold,
    )
    distances[distances >= np.finfo(np.float64).max * 0.5] = np.inf
    return distances


def _link_condition(faces: np.ndarray, u: int, v: int) -> bool:
    incident_u = faces[np.any(faces == u, axis=1)]
    incident_v = faces[np.any(faces == v, axis=1)]
    neighbors_u = set(incident_u.ravel()) - {u}
    neighbors_v = set(incident_v.ravel()) - {v}
    edge_faces = np.sum(np.any(faces == u, axis=1) & np.any(faces == v, axis=1))
    return neighbors_u.intersection(neighbors_v) == (
        set(incident_u[np.any(incident_u == v, axis=1)].ravel()) - {u, v}
    ) and edge_faces in (1, 2)


def _quadric_decimate_unconstrained(
    vertices: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    target_faces: int,
    quadrics: np.ndarray,
) -> DecimationResult:
    vertex_count = len(vertices)
    result = np.empty(2, dtype=np.int64)
    lib().mvc_qem_decimate_unconstrained_f64(
        addr(vertices),
        addr(faces),
        addr(edges),
        addr(quadrics),
        addr(result),
        vertex_count,
        len(faces),
        len(edges),
        target_faces,
        1.0e-15,
    )
    faces = faces[: int(result[0])]
    collapsed = int(result[1])

    referenced = np.zeros(vertex_count, dtype=bool)
    if faces.size:
        referenced[faces.reshape(-1)] = True
    remap = np.full(vertex_count, -1, dtype=np.int64)
    remap[referenced] = np.arange(referenced.sum())
    return DecimationResult(
        vertices[referenced],
        np.ascontiguousarray(remap[faces]),
        collapsed,
    )


def quadric_decimate(
    vertices,
    faces,
    target_faces: int,
    *,
    preserve_topology: bool = True,
    boundary_weight: float = 0.5,
    use_area: bool = True,
) -> DecimationResult:
    """Decimate with VCGLib plane quadrics and optimal edge placement.

    This uses the `QualityCheck=False`, `ScaleIndependent=False` policy of
    TriEdgeCollapseQuadric and propagates the surviving vertex quadric exactly.
    """
    vertices, faces = _mesh(vertices, faces)
    if target_faces < 0:
        raise ValueError("target_faces must be nonnegative")
    if not np.isfinite(boundary_weight) or boundary_weight < 0:
        raise ValueError("boundary_weight must be finite and nonnegative")
    vertices = vertices.copy()
    faces = faces.copy()
    if len(faces) <= target_faces:
        return DecimationResult(vertices, faces, 0)
    edges, _, edge_counts, face_edges = _topology(faces, len(vertices))
    border = np.ascontiguousarray(edge_counts[face_edges] == 1, dtype=np.int64)
    quadrics = np.empty((len(vertices), 10), dtype=np.float64)
    lib().mvc_qem_init_f64(
        addr(vertices),
        addr(faces),
        addr(border),
        addr(quadrics),
        len(vertices),
        len(faces),
        float(boundary_weight),
        int(use_area),
    )
    if not preserve_topology:
        return _quadric_decimate_unconstrained(
            vertices, faces, edges, target_faces, quadrics
        )
    alive = np.ones(len(vertices), dtype=bool)
    collapsed = 0
    positions = np.empty((len(edges), 3), dtype=np.float64)
    errors = np.empty(len(edges), dtype=np.float64)
    while len(faces) > target_faces:
        if not len(edges):
            break
        lib().mvc_qem_evaluate_f64(
            addr(vertices),
            addr(edges),
            addr(quadrics),
            addr(positions),
            addr(errors),
            len(edges),
            1.0e-15,
        )
        accepted = False
        edge_order = (
            np.argsort(errors[: len(edges)], kind="stable")
            if preserve_topology
            else (int(np.argmin(errors[: len(edges)])),)
        )
        for edge_index in edge_order:
            u, v = (int(x) for x in edges[edge_index])
            if not alive[u] or not alive[v]:
                continue
            if preserve_topology and not _link_condition(faces, u, v):
                continue
            candidate_faces = faces.copy()
            candidate_faces[candidate_faces == u] = v
            candidate_faces = candidate_faces[
                (candidate_faces[:, 0] != candidate_faces[:, 1])
                & (candidate_faces[:, 1] != candidate_faces[:, 2])
                & (candidate_faces[:, 2] != candidate_faces[:, 0])
            ]
            if preserve_topology and len(candidate_faces):
                if len(np.unique(np.sort(candidate_faces, axis=1), axis=0)) != len(
                    candidate_faces
                ):
                    continue
            if len(candidate_faces) == len(faces):
                continue
            vertices[v] = positions[edge_index]
            quadrics[v] += quadrics[u]
            alive[u] = False
            faces = np.ascontiguousarray(candidate_faces)
            collapsed += 1
            accepted = True
            updated_edges = edges.copy()
            updated_edges[updated_edges == u] = v
            updated_edges.sort(axis=1)
            updated_edges = updated_edges[
                updated_edges[:, 0] != updated_edges[:, 1]
            ]
            vertex_count = len(vertices)
            if vertex_count <= np.iinfo(np.int64).max // max(vertex_count, 1):
                edge_keys = (
                    updated_edges[:, 0] * vertex_count + updated_edges[:, 1]
                )
                unique_edge_keys = np.unique(edge_keys)
                edges = np.empty((len(unique_edge_keys), 2), dtype=np.int64)
                edges[:, 0] = unique_edge_keys // vertex_count
                edges[:, 1] = unique_edge_keys % vertex_count
            else:
                edges = np.unique(updated_edges, axis=0)
            break
        if not accepted:
            break
    referenced = np.zeros(len(vertices), dtype=bool)
    if faces.size:
        referenced[faces.reshape(-1)] = True
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[referenced] = np.arange(referenced.sum())
    return DecimationResult(
        vertices[referenced],
        np.ascontiguousarray(remap[faces]),
        collapsed,
    )
