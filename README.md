# mojo-vcglib

`mojo-vcglib` is a Mojo/Python port of selected compute-heavy triangle-mesh
algorithms from [VCGLib](https://github.com/cnr-isti-vclab/vcglib), the geometry
library used by MeshLab. It operates on NumPy vertex arrays with shape `(n, 3)`
and triangle index arrays with shape `(m, 3)`.

This is a derived work of VCGLib at upstream revision
`5cae2abc2f9056785b0537dcbe156c40da2aea20`. VCGLib states GPL version 2 or
later in the ported headers; this repository is distributed under
[GPL-3.0](LICENSE). The source comments identify the upstream file and function
above each non-obvious kernel.

## Coverage

This is not a complete VCGLib port. The current release covers only these
algorithm families and policies:

- `clean.h`: exact duplicate vertices, degenerate and duplicate faces,
  unreferenced-vertex compaction, edge counts, boundary edges, non-manifold
  edge detection, and watertightness;
- `smooth.h`: VCGLib classical Laplacian boundary rules, Taubin, HC, and
  Fujiwara scale-dependent smoothing;
- `refine_loop.h`: classical Loop even/odd rules, including boundary rules;
- `geodesic.h`: multi-source Euclidean geodesics with VCGLib's
  triangle-unfolding update;
- `tri_edge_collapse_quadric.h`: area-weighted face quadrics, boundary
  quadrics, optimal 3x3 placement, propagated quadrics, and optional link
  conditions. The exposed policy corresponds to `QualityCheck=False` and
  `ScaleIndependent=False`;
- `point_sampling.h`: exact-count Monte Carlo and stratified surface sampling;
- `inertia.h`: signed volume, center of mass, and the inertia tensor;
- `closest.h` and `distance3.h`: batched closest points on a triangle mesh;
- `intersection.h` and `intersection3.h`: batched nearest ray/mesh hits.

Not yet ported are butterfly subdivision, isotropic remeshing, heat-method
geodesics, Poisson-disk pruning, convex hull, hole filling, curvature fitting,
AABB-grid construction, and general mesh/mesh intersection. No placeholder API
is provided for those algorithms.

## Install

Clone the repository, then run the following from its root. The repository
pins the tested Mojo nightly and all Python reference dependencies; Pixi builds
the shared library in `dist/`.

```bash
pixi install
pixi run build
pixi run test
```

## Usage

Save this as `example.py` in the repository root and run
`pixi run python example.py`:

```python
import numpy as np
from mojovcglib import clean_mesh, mass_properties, taubin_smooth

vertices = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],  # duplicate
])
faces = np.array([
    [0, 2, 1],
    [4, 1, 3],
    [1, 2, 3],
    [2, 0, 3],
], dtype=np.int64)

vertices, faces, stats = clean_mesh(vertices, faces)
smoothed = taubin_smooth(vertices, faces, steps=2)
properties = mass_properties(vertices, faces)

print(stats.duplicate_vertices)  # 1
print(properties.mass)           # 0.16666666666666666
print(smoothed.shape)            # (4, 3)
```

The public package also exports `topology_stats`, `laplacian_smooth`,
`hc_smooth`, `scale_dependent_smooth`, `loop_subdivide`,
`geodesic_distance`, `quadric_decimate`, `sample_surface`,
`closest_points`, and `intersect_rays`.

## Correctness

The test suite has 36 tests. Cleanup, all smoothing schemes, Loop subdivision,
geodesics, and QEM are compared directly with
[PyMeshLab](https://pymeshlab.readthedocs.io/), which invokes MeshLab's VCGLib
algorithm core. Mass properties and closest points are compared with
[trimesh](https://trimesh.org/). Ray intersection and deterministic sampling
use NumPy references or geometric invariants because PyMeshLab does not expose
equivalent deterministic batched calls. Tests cover empty meshes, one
triangle, duplicate vertices and faces, zero-area faces, unreferenced vertices,
disconnected components, and non-manifold edges.

## How it works

`src/vcglib.mojo` is one compilation unit and builds to
`dist/libmojo-vcglib.so`. Concrete Float64 C exports receive NumPy buffer
addresses as 64-bit integers. The ctypes layer keeps arrays alive during each
call, and the library never retains or frees caller memory.

Coordinates use an interleaved `Float64[n, 3]` buffer and faces use
`Int64[m, 3]`. Python builds compact edge incidence and CSR neighbor arrays;
Mojo performs the numerical face/vertex loops. QEM keeps ten coefficients per
vertex for the symmetric 3x3 quadric, linear term, and constant. The Python
topology controller applies collapses while Mojo initializes and evaluates all
candidate quadrics.

## Benchmarks

Run benchmarks only through `pixi run bench`; that task takes a machine-wide
lock. These are end-to-end median Python API timings measured on an Intel Xeon
E5-2697 v4 at 2.30 GHz. The reference is PyMeshLab/VCGLib except for closest
points and mass properties, which use trimesh. Values below `1.00x` mean Mojo
is slower.

| Operation | Mojo ms | Reference ms | Speedup |
|---|---:|---:|---:|
| clean duplicate mesh (10k faces) | 5.303 | 9.736 | 1.84x |
| Taubin, 2 steps (9.5k faces) | 4.107 | 8.651 | 2.11x |
| Loop subdivision (722 faces) | 0.690 | 2.035 | 2.95x |
| geodesic, 625 vertices | 0.950 | 1.747 | 1.84x |
| QEM, 242 to 180 faces | 0.826 | 1.753 | 2.12x |
| closest, 400x722 | 14.626 | 189.027 | 12.92x |
| mass properties (722 faces) | 0.110 | 0.602 | 5.47x |

GPU execution is intentionally not enabled. Cleanup and smoothing are
memory/scatter-bound, geodesic traversal is dependency-heavy, and the QEM
controller operates on small, control-heavy batches. The closest-point kernel
has enough arithmetic intensity to be a plausible GPU candidate, but its CPU
path is already far beyond the optimization target and transferring these
benchmark-sized buffers would not be justified.

The CPU path uses compact integer topology keys, CSR adjacency for geodesic
visits, and an in-kernel QEM collapse controller. Contiguous copies, clears,
geodesic initialization, face-validity reduction, and quadric accumulation use
native-width SIMD with scalar tails. QEM reuses its initial edge topology for
boundary classification and collapse instead of sorting the same edges twice.
No additional threaded path was retained: smoothing uses shared scatter writes,
geodesic and QEM collapse are dependency-heavy, and the independent benchmark
batches are below a size where thread-launch overhead is justified.
