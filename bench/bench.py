"""End-to-end benchmarks against PyMeshLab/VCGLib and trimesh."""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np
import pymeshlab
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojovcglib as mvc


def grid(side: int):
    vertices = np.array(
        [
            [x / (side - 1), y / (side - 1), 0.03 * np.sin(x * 0.3) * np.cos(y * 0.2)]
            for y in range(side)
            for x in range(side)
        ],
        dtype=np.float64,
    )
    faces = []
    for y in range(side - 1):
        for x in range(side - 1):
            a = y * side + x
            faces.extend(
                [[a, a + 1, a + side + 1], [a, a + side + 1, a + side]]
            )
    return vertices, np.array(faces, dtype=np.int64)


def pml_mesh(vertices, faces):
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(pymeshlab.Mesh(vertices, faces))
    return mesh_set


def median_ms(function, repeats=5):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main():
    smooth_v, smooth_f = grid(70)
    geo_v, geo_f = grid(25)
    loop_v, loop_f = grid(20)
    qem_v, qem_f = grid(12)
    rng = np.random.default_rng(42)
    queries = rng.uniform(-0.2, 1.2, size=(400, 3))

    duplicated_v = np.vstack((smooth_v, smooth_v[:1000], [[9, 9, 9]]))
    duplicated_f = np.vstack((smooth_f, smooth_f[:500]))

    def reference_clean():
        mesh_set = pml_mesh(duplicated_v, duplicated_f)
        mesh_set.meshing_remove_duplicate_vertices()
        mesh_set.meshing_remove_duplicate_faces()
        mesh_set.meshing_remove_unreferenced_vertices()

    def reference_smooth():
        mesh_set = pml_mesh(smooth_v, smooth_f)
        mesh_set.apply_coord_taubin_smoothing(
            stepsmoothnum=2, lambda_=0.5, mu=-0.53
        )

    def reference_loop():
        mesh_set = pml_mesh(loop_v, loop_f)
        mesh_set.meshing_surface_subdivision_loop(
            iterations=1, threshold=pymeshlab.PureValue(0), loopweight=0
        )

    def reference_geodesic():
        mesh_set = pml_mesh(geo_v, geo_f)
        mesh_set.compute_scalar_by_geodesic_distance_from_given_point_per_vertex(
            startpoint=geo_v[0], maxdistance=pymeshlab.PureValue(100)
        )

    def reference_qem():
        mesh_set = pml_mesh(qem_v, qem_f)
        mesh_set.meshing_decimation_quadric_edge_collapse(
            targetfacenum=180,
            preservetopology=False,
            boundaryweight=0.5,
            autoclean=True,
        )

    closest_mesh = trimesh.Trimesh(
        vertices=loop_v, faces=loop_f, process=False
    )
    triangles = loop_v[loop_f]

    cases = [
        (
            "clean duplicate mesh (10k faces)",
            lambda: mvc.clean_mesh(duplicated_v, duplicated_f),
            reference_clean,
            5,
        ),
        (
            "Taubin, 2 steps (9.5k faces)",
            lambda: mvc.taubin_smooth(smooth_v, smooth_f, steps=2),
            reference_smooth,
            5,
        ),
        (
            "Loop subdivision (722 faces)",
            lambda: mvc.loop_subdivide(loop_v, loop_f),
            reference_loop,
            5,
        ),
        (
            "geodesic, 625 vertices",
            lambda: mvc.geodesic_distance(geo_v, geo_f, [0]),
            reference_geodesic,
            3,
        ),
        (
            "QEM, 242 to 180 faces",
            lambda: mvc.quadric_decimate(
                qem_v, qem_f, 180, preserve_topology=False
            ),
            reference_qem,
            3,
        ),
        (
            "closest, 400x722",
            lambda: mvc.closest_points(loop_v, loop_f, queries),
            lambda: trimesh.proximity.closest_point_naive(
                closest_mesh, queries
            ),
            5,
        ),
        (
            "mass properties (722 faces)",
            lambda: mvc.mass_properties(loop_v, loop_f),
            lambda: trimesh.triangles.mass_properties(triangles),
            10,
        ),
    ]

    cpu = "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    print(f"Machine: {cpu}")
    print()
    print("| Operation | Mojo ms | Reference ms | Speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo_function, reference_function, repeats in cases:
        mojo_ms = median_ms(mojo_function, repeats)
        reference_ms = median_ms(reference_function, repeats)
        speedup = reference_ms / mojo_ms
        print(
            f"| {name} | {mojo_ms:.3f} | {reference_ms:.3f} | {speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
