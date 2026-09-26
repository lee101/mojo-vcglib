"""Compute kernels derived from VCGLib's triangle-mesh algorithms.

All arrays are caller-owned C-contiguous buffers. Coordinates are Float64 and
indices/masks are Int64. The Python layer owns topology construction and output
allocation; the numerical loops remain here.
"""

from std.math import sqrt, cos, acos, sin
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime HUGE = 1.7976931348623157e308
comptime PI = 3.1415926535897932384626433832795


def fp(address: Int) -> FPtr:
    return FPtr(unsafe_from_address=address)


def ip(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


struct V3(Copyable, Movable, ImplicitlyCopyable):
    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64, y: Float64, z: Float64):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def zero() -> V3:
        return V3(0.0, 0.0, 0.0)

    def __add__(self, other: V3) -> V3:
        return V3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: V3) -> V3:
        return V3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: Float64) -> V3:
        return V3(self.x * scale, self.y * scale, self.z * scale)

    def __truediv__(self, scale: Float64) -> V3:
        return V3(self.x / scale, self.y / scale, self.z / scale)

    def dot(self, other: V3) -> Float64:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: V3) -> V3:
        return V3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def squared_norm(self) -> Float64:
        return self.dot(self)

    def norm(self) -> Float64:
        return sqrt(self.squared_norm())


@always_inline
def load_v(p: FPtr, i: Int) -> V3:
    return V3(p[3 * i], p[3 * i + 1], p[3 * i + 2])


@always_inline
def store_v(p: FPtr, i: Int, v: V3):
    p[3 * i] = v.x
    p[3 * i + 1] = v.y
    p[3 * i + 2] = v.z


@always_inline
def add_v(p: FPtr, i: Int, v: V3):
    p[3 * i] += v.x
    p[3 * i + 1] += v.y
    p[3 * i + 2] += v.z


@always_inline
def copy_vertices(src: FPtr, dst: FPtr, n: Int):
    comptime W = simdwidthof[DType.float64]()
    var size = 3 * n
    var i = 0
    while i + W <= size:
        dst.store(i, src.load[width=W](i))
        i += W
    while i < size:
        dst[i] = src[i]
        i += 1


@always_inline
def zero_floats(p: FPtr, n: Int):
    comptime W = simdwidthof[DType.float64]()
    var zeros = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        p.store(i, zeros)
        i += W
    while i < n:
        p[i] = 0.0
        i += 1


@always_inline
def fill_floats(p: FPtr, n: Int, value: Float64):
    comptime W = simdwidthof[DType.float64]()
    var values = SIMD[DType.float64, W](value)
    var i = 0
    while i + W <= n:
        p.store(i, values)
        i += W
    while i < n:
        p[i] = value
        i += 1


@always_inline
def fill_ints(p: IPtr, n: Int, value: Int64):
    comptime W = simdwidthof[DType.int64]()
    var values = SIMD[DType.int64, W](value)
    var i = 0
    while i + W <= n:
        p.store(i, values)
        i += W
    while i < n:
        p[i] = value
        i += 1


@always_inline
def face_is_valid(
    vertices: FPtr,
    faces: IPtr,
    f: Int,
    vertex_count: Int,
    epsilon: Float64,
) -> Bool:
    var a = Int(faces[3 * f])
    var b = Int(faces[3 * f + 1])
    var c = Int(faces[3 * f + 2])
    var ok = (
        a >= 0 and b >= 0 and c >= 0
        and a < vertex_count and b < vertex_count and c < vertex_count
        and a != b and b != c and c != a
    )
    if ok:
        var cross = (load_v(vertices, b) - load_v(vertices, a)).cross(
            load_v(vertices, c) - load_v(vertices, a)
        )
        ok = cross.norm() > epsilon
    return ok


# vcglib: vcg/complex/algorithms/clean.h Clean::RemoveDegenerateFace
@export("mvc_face_validity_f64")
def mvc_face_validity_f64(
    vertices_address: Int,
    faces_address: Int,
    valid_address: Int,
    vertex_count: Int,
    face_count: Int,
    epsilon: Float64,
) abi("C") -> Int:
    var vertices = fp(vertices_address)
    var faces = ip(faces_address)
    var valid = ip(valid_address)
    for f in range(face_count):
        var ok = face_is_valid(vertices, faces, f, vertex_count, epsilon)
        valid[f] = 1 if ok else 0

    comptime W = simdwidthof[DType.int64]()
    var kept: Int64 = 0
    var i = 0
    while i + W <= face_count:
        kept += valid.load[width=W](i).reduce_add()
        i += W
    while i < face_count:
        kept += valid[i]
        i += 1
    return Int(kept)


# vcglib: vcg/complex/algorithms/clean.h Clean::CountEdgeNum
@export("mvc_edge_incidence_stats_i64")
def mvc_edge_incidence_stats_i64(
    sorted_edges_address: Int,
    edge_rows: Int,
    result_address: Int,
) abi("C"):
    var edges = ip(sorted_edges_address)
    var result = ip(result_address)
    var unique_count = 0
    var boundary_count = 0
    var nonmanifold_count = 0
    var i = 0
    while i < edge_rows:
        var a = edges[2 * i]
        var b = edges[2 * i + 1]
        var j = i + 1
        while j < edge_rows and edges[2 * j] == a and edges[2 * j + 1] == b:
            j += 1
        var incidence = j - i
        unique_count += 1
        if incidence == 1:
            boundary_count += 1
        elif incidence > 2:
            nonmanifold_count += 1
        i = j
    result[0] = Int64(unique_count)
    result[1] = Int64(boundary_count)
    result[2] = Int64(nonmanifold_count)


# vcglib: vcg/complex/algorithms/smooth.h Smooth::AccumulateLaplacianInfo
def laplacian_accumulate(
    vertices: FPtr,
    faces: IPtr,
    border: IPtr,
    sums: FPtr,
    counts: FPtr,
    vertex_count: Int,
    face_count: Int,
):
    zero_floats(sums, 3 * vertex_count)
    zero_floats(counts, vertex_count)

    for f in range(face_count):
        for j in range(3):
            if border[3 * f + j] == 0:
                var a = Int(faces[3 * f + j])
                var b = Int(faces[3 * f + (j + 1) % 3])
                add_v(sums, a, load_v(vertices, b))
                add_v(sums, b, load_v(vertices, a))
                counts[a] += 1.0
                counts[b] += 1.0

    for f in range(face_count):
        for j in range(3):
            if border[3 * f + j] != 0:
                var a = Int(faces[3 * f + j])
                var b = Int(faces[3 * f + (j + 1) % 3])
                store_v(sums, a, load_v(vertices, a))
                store_v(sums, b, load_v(vertices, b))
                counts[a] = 1.0
                counts[b] = 1.0

    for f in range(face_count):
        for j in range(3):
            if border[3 * f + j] != 0:
                var a = Int(faces[3 * f + j])
                var b = Int(faces[3 * f + (j + 1) % 3])
                add_v(sums, a, load_v(vertices, b))
                add_v(sums, b, load_v(vertices, a))
                counts[a] += 1.0
                counts[b] += 1.0


# vcglib: vcg/complex/algorithms/smooth.h Smooth::VertexCoordLaplacian
@export("mvc_laplacian_f64")
def mvc_laplacian_f64(
    vertices_address: Int,
    faces_address: Int,
    border_address: Int,
    result_address: Int,
    sums_address: Int,
    counts_address: Int,
    vertex_count: Int,
    face_count: Int,
    steps: Int,
) abi("C"):
    var src = fp(vertices_address)
    var faces = ip(faces_address)
    var border = ip(border_address)
    var vertices = fp(result_address)
    var sums = fp(sums_address)
    var counts = fp(counts_address)
    copy_vertices(src, vertices, vertex_count)
    for _ in range(max(steps, 0)):
        laplacian_accumulate(
            vertices, faces, border, sums, counts, vertex_count, face_count
        )
        for i in range(vertex_count):
            if counts[i] > 0.0:
                store_v(
                    vertices,
                    i,
                    (load_v(vertices, i) + load_v(sums, i)) / (counts[i] + 1.0),
                )


# vcglib: vcg/complex/algorithms/smooth.h Smooth::VertexCoordTaubin
@export("mvc_taubin_f64")
def mvc_taubin_f64(
    vertices_address: Int,
    faces_address: Int,
    border_address: Int,
    result_address: Int,
    sums_address: Int,
    counts_address: Int,
    vertex_count: Int,
    face_count: Int,
    steps: Int,
    lambda_value: Float64,
    mu: Float64,
) abi("C"):
    var src = fp(vertices_address)
    var faces = ip(faces_address)
    var border = ip(border_address)
    var vertices = fp(result_address)
    var sums = fp(sums_address)
    var counts = fp(counts_address)
    copy_vertices(src, vertices, vertex_count)
    for _ in range(max(steps, 0)):
        laplacian_accumulate(
            vertices, faces, border, sums, counts, vertex_count, face_count
        )
        for i in range(vertex_count):
            if counts[i] > 0.0:
                var delta = load_v(sums, i) / counts[i] - load_v(vertices, i)
                store_v(vertices, i, load_v(vertices, i) + delta * lambda_value)
        laplacian_accumulate(
            vertices, faces, border, sums, counts, vertex_count, face_count
        )
        for i in range(vertex_count):
            if counts[i] > 0.0:
                var delta = load_v(sums, i) / counts[i] - load_v(vertices, i)
                store_v(vertices, i, load_v(vertices, i) + delta * mu)


# vcglib: vcg/complex/algorithms/smooth.h Smooth::VertexCoordLaplacianHC
@export("mvc_hc_f64")
def mvc_hc_f64(
    vertices_address: Int,
    faces_address: Int,
    border_address: Int,
    result_address: Int,
    sums_address: Int,
    diffs_address: Int,
    counts_address: Int,
    vertex_count: Int,
    face_count: Int,
    steps: Int,
) abi("C"):
    var src = fp(vertices_address)
    var faces = ip(faces_address)
    var border = ip(border_address)
    var vertices = fp(result_address)
    var sums = fp(sums_address)
    var diffs = fp(diffs_address)
    var counts = fp(counts_address)
    copy_vertices(src, vertices, vertex_count)
    for _ in range(max(steps, 0)):
        for i in range(3 * vertex_count):
            sums[i] = 0.0
            diffs[i] = 0.0
        for i in range(vertex_count):
            counts[i] = 0.0
        for f in range(face_count):
            for j in range(3):
                var a = Int(faces[3 * f + j])
                var b = Int(faces[3 * f + (j + 1) % 3])
                var repeats = 2 if border[3 * f + j] != 0 else 1
                for _r in range(repeats):
                    add_v(sums, a, load_v(vertices, b))
                    add_v(sums, b, load_v(vertices, a))
                    counts[a] += 1.0
                    counts[b] += 1.0
        for i in range(vertex_count):
            if counts[i] > 0.0:
                store_v(sums, i, load_v(sums, i) / counts[i])
        for f in range(face_count):
            for j in range(3):
                var a = Int(faces[3 * f + j])
                var b = Int(faces[3 * f + (j + 1) % 3])
                var repeats = 2 if border[3 * f + j] != 0 else 1
                for _r in range(repeats):
                    add_v(diffs, a, load_v(sums, b) - load_v(vertices, b))
                    add_v(diffs, b, load_v(sums, a) - load_v(vertices, a))
        for i in range(vertex_count):
            if counts[i] > 0.0:
                var dif = load_v(diffs, i) / counts[i]
                var sm = load_v(sums, i)
                var old = load_v(vertices, i)
                store_v(vertices, i, sm - (sm - old) * 0.5 + dif * 0.5)


# vcglib: vcg/complex/algorithms/smooth.h Smooth::VertexCoordScaleDependentLaplacian_Fujiwara
@export("mvc_scale_dependent_f64")
def mvc_scale_dependent_f64(
    vertices_address: Int,
    faces_address: Int,
    border_address: Int,
    result_address: Int,
    sums_address: Int,
    lengths_address: Int,
    vertex_count: Int,
    face_count: Int,
    steps: Int,
    delta: Float64,
) abi("C"):
    var src = fp(vertices_address)
    var faces = ip(faces_address)
    var border = ip(border_address)
    var vertices = fp(result_address)
    var sums = fp(sums_address)
    var lengths = fp(lengths_address)
    copy_vertices(src, vertices, vertex_count)
    for _ in range(max(steps, 0)):
        for i in range(3 * vertex_count):
            sums[i] = 0.0
        for i in range(vertex_count):
            lengths[i] = 0.0
        for f in range(face_count):
            for j in range(3):
                if border[3 * f + j] == 0:
                    var a = Int(faces[3 * f + j])
                    var b = Int(faces[3 * f + (j + 1) % 3])
                    var edge = load_v(vertices, b) - load_v(vertices, a)
                    var length = edge.norm()
                    if length > 0.0:
                        edge = edge / length
                        add_v(sums, a, edge)
                        add_v(sums, b, edge * -1.0)
                        lengths[a] += length
                        lengths[b] += length
        for f in range(face_count):
            for j in range(3):
                if border[3 * f + j] != 0:
                    var a = Int(faces[3 * f + j])
                    var b = Int(faces[3 * f + (j + 1) % 3])
                    store_v(sums, a, V3.zero())
                    store_v(sums, b, V3.zero())
                    lengths[a] = 0.0
                    lengths[b] = 0.0
        for f in range(face_count):
            for j in range(3):
                if border[3 * f + j] != 0:
                    var a = Int(faces[3 * f + j])
                    var b = Int(faces[3 * f + (j + 1) % 3])
                    var edge = load_v(vertices, b) - load_v(vertices, a)
                    var length = edge.norm()
                    if length > 0.0:
                        edge = edge / length
                        add_v(sums, a, edge)
                        add_v(sums, b, edge * -1.0)
                        lengths[a] += length
                        lengths[b] += length
        for i in range(vertex_count):
            if lengths[i] > 0.0:
                store_v(
                    vertices,
                    i,
                    load_v(vertices, i) + load_v(sums, i) * (delta / lengths[i]),
                )


# vcglib: vcg/complex/algorithms/refine_loop.h EvenPointLoopGeneric/OddPointLoopGeneric
@export("mvc_loop_vertices_f64")
def mvc_loop_vertices_f64(
    vertices_address: Int,
    edges_address: Int,
    opposites_address: Int,
    offsets_address: Int,
    neighbors_address: Int,
    boundary_neighbors_address: Int,
    result_address: Int,
    vertex_count: Int,
    edge_count: Int,
) abi("C"):
    var vertices = fp(vertices_address)
    var edges = ip(edges_address)
    var opposites = ip(opposites_address)
    var offsets = ip(offsets_address)
    var neighbors = ip(neighbors_address)
    var boundary_neighbors = ip(boundary_neighbors_address)
    var result = fp(result_address)
    for i in range(vertex_count):
        var p = load_v(vertices, i)
        var bl = Int(boundary_neighbors[2 * i])
        var br = Int(boundary_neighbors[2 * i + 1])
        if bl >= 0 and br >= 0:
            store_v(
                result, i,
                p * 0.75 + (load_v(vertices, bl) + load_v(vertices, br)) * 0.125,
            )
        else:
            var begin = Int(offsets[i])
            var end = Int(offsets[i + 1])
            var k = end - begin
            if k > 0:
                var beta: Float64
                if k > 3:
                    var term = 0.375 + cos(2.0 * PI / Float64(k)) * 0.25
                    beta = (0.625 - term * term) / Float64(k)
                else:
                    beta = 3.0 / 16.0
                var value = p * (1.0 - Float64(k) * beta)
                for q in range(begin, end):
                    value = value + load_v(vertices, Int(neighbors[q])) * beta
                store_v(result, i, value)
            else:
                store_v(result, i, p)
    for e in range(edge_count):
        var a = Int(edges[2 * e])
        var b = Int(edges[2 * e + 1])
        var u = Int(opposites[2 * e])
        var d = Int(opposites[2 * e + 1])
        if u < 0 or d < 0:
            store_v(
                result, vertex_count + e,
                (load_v(vertices, a) + load_v(vertices, b)) * 0.5,
            )
        else:
            store_v(
                result, vertex_count + e,
                (load_v(vertices, a) + load_v(vertices, b)) * 0.375
                + (load_v(vertices, u) + load_v(vertices, d)) * 0.125,
            )


@always_inline
def component(v: V3, axis: Int) -> Float64:
    if axis == 0:
        return v.x
    if axis == 1:
        return v.y
    return v.z


# vcglib: vcg/complex/algorithms/inertia.h Inertia::compProjectionIntegrals/Compute
@export("mvc_mass_properties_f64")
def mvc_mass_properties_f64(
    vertices_address: Int,
    faces_address: Int,
    result_address: Int,
    face_count: Int,
) abi("C"):
    var vertices = fp(vertices_address)
    var faces = ip(faces_address)
    var result = fp(result_address)
    var t0 = 0.0
    var t1x = 0.0
    var t1y = 0.0
    var t1z = 0.0
    var t2x = 0.0
    var t2y = 0.0
    var t2z = 0.0
    var tpx = 0.0
    var tpy = 0.0
    var tpz = 0.0
    for f in range(face_count):
        var v0 = load_v(vertices, Int(faces[3 * f]))
        var v1 = load_v(vertices, Int(faces[3 * f + 1]))
        var v2 = load_v(vertices, Int(faces[3 * f + 2]))
        var direction = (v1 - v0).cross(v2 - v0)
        var double_area = direction.norm()
        if double_area <= 1.1754943508222875e-38:
            continue
        var normal = direction / double_area
        var nx = abs(normal.x)
        var ny = abs(normal.y)
        var nz = abs(normal.z)
        var c_axis: Int
        if nx > ny and nx > nz:
            c_axis = 0
        elif ny > nz:
            c_axis = 1
        else:
            c_axis = 2
        var a_axis = (c_axis + 1) % 3
        var b_axis = (a_axis + 1) % 3

        var p1 = 0.0
        var pa = 0.0
        var pb = 0.0
        var paa = 0.0
        var pab = 0.0
        var pbb = 0.0
        var paaa = 0.0
        var paab = 0.0
        var pabb = 0.0
        var pbbb = 0.0
        for j in range(3):
            var e0: V3
            var e1: V3
            if j == 0:
                e0 = v0
                e1 = v1
            elif j == 1:
                e0 = v1
                e1 = v2
            else:
                e0 = v2
                e1 = v0
            var a0 = component(e0, a_axis)
            var b0 = component(e0, b_axis)
            var a1 = component(e1, a_axis)
            var b1 = component(e1, b_axis)
            var da = a1 - a0
            var db = b1 - b0
            var a0_2 = a0 * a0
            var a0_3 = a0_2 * a0
            var a0_4 = a0_3 * a0
            var b0_2 = b0 * b0
            var b0_3 = b0_2 * b0
            var b0_4 = b0_3 * b0
            var a1_2 = a1 * a1
            var a1_3 = a1_2 * a1
            var b1_2 = b1 * b1
            var b1_3 = b1_2 * b1
            var c1 = a1 + a0
            var ca = a1 * c1 + a0_2
            var caa = a1 * ca + a0_3
            var caaa = a1 * caa + a0_4
            var cb = b1 * (b1 + b0) + b0_2
            var cbb = b1 * cb + b0_3
            var cbbb = b1 * cbb + b0_4
            var cab = 3.0 * a1_2 + 2.0 * a1 * a0 + a0_2
            var kab = a1_2 + 2.0 * a1 * a0 + 3.0 * a0_2
            var caab = a0 * cab + 4.0 * a1_3
            var kaab = a1 * kab + 4.0 * a0_3
            var cabb = 4.0 * b1_3 + 3.0 * b1_2 * b0 + 2.0 * b1 * b0_2 + b0_3
            var kabb = b1_3 + 2.0 * b1_2 * b0 + 3.0 * b1 * b0_2 + 4.0 * b0_3
            p1 += db * c1
            pa += db * ca
            paa += db * caa
            paaa += db * caaa
            pb += da * cb
            pbb += da * cbb
            pbbb += da * cbbb
            pab += db * (b1 * cab + b0 * kab)
            paab += db * (b1 * caab + b0 * kaab)
            pabb += da * (a1 * cabb + a0 * kabb)
        p1 /= 2.0
        pa /= 6.0
        paa /= 12.0
        paaa /= 20.0
        pb /= -6.0
        pbb /= -12.0
        pbbb /= -20.0
        pab /= 24.0
        paab /= 60.0
        pabb /= -60.0

        var na = component(normal, a_axis)
        var nb = component(normal, b_axis)
        var nc = component(normal, c_axis)
        var w = -v0.dot(normal)
        var k1 = 1.0 / nc
        var k2 = k1 * k1
        var k3 = k2 * k1
        var k4 = k3 * k1
        var fa = k1 * pa
        var fb = k1 * pb
        var fc = -k2 * (na * pa + nb * pb + w * p1)
        var faa = k1 * paa
        var fbb = k1 * pbb
        var fcc = k3 * (
            na * na * paa + 2.0 * na * nb * pab + nb * nb * pbb
            + w * (2.0 * (na * pa + nb * pb) + w * p1)
        )
        var faaa = k1 * paaa
        var fbbb = k1 * pbbb
        var fccc = -k4 * (
            na * na * na * paaa + 3.0 * na * na * nb * paab
            + 3.0 * na * nb * nb * pabb + nb * nb * nb * pbbb
            + 3.0 * w * (na * na * paa + 2.0 * na * nb * pab + nb * nb * pbb)
            + w * w * (3.0 * (na * pa + nb * pb) + w * p1)
        )
        var faab = k1 * paab
        var fbbc = -k2 * (na * pabb + nb * pbbb + w * pbb)
        var fcca = k3 * (
            na * na * paaa + 2.0 * na * nb * paab + nb * nb * pabb
            + w * (2.0 * (na * paa + nb * pab) + w * pa)
        )
        var f_for_x = fc
        if a_axis == 0:
            f_for_x = fa
        elif b_axis == 0:
            f_for_x = fb
        t0 += normal.x * f_for_x
        if a_axis == 0:
            t1x += normal.x * faa
            t2x += normal.x * faaa
            tpx += normal.x * faab
        elif a_axis == 1:
            t1y += normal.y * faa
            t2y += normal.y * faaa
            tpy += normal.y * faab
        else:
            t1z += normal.z * faa
            t2z += normal.z * faaa
            tpz += normal.z * faab
        if b_axis == 0:
            t1x += normal.x * fbb
            t2x += normal.x * fbbb
            tpx += normal.x * fbbc
        elif b_axis == 1:
            t1y += normal.y * fbb
            t2y += normal.y * fbbb
            tpy += normal.y * fbbc
        else:
            t1z += normal.z * fbb
            t2z += normal.z * fbbb
            tpz += normal.z * fbbc
        if c_axis == 0:
            t1x += normal.x * fcc
            t2x += normal.x * fccc
            tpx += normal.x * fcca
        elif c_axis == 1:
            t1y += normal.y * fcc
            t2y += normal.y * fccc
            tpy += normal.y * fcca
        else:
            t1z += normal.z * fcc
            t2z += normal.z * fccc
            tpz += normal.z * fcca

    t1x /= 2.0
    t1y /= 2.0
    t1z /= 2.0
    t2x /= 3.0
    t2y /= 3.0
    t2z /= 3.0
    tpx /= 2.0
    tpy /= 2.0
    tpz /= 2.0
    result[0] = t0
    if t0 == 0.0:
        for i in range(1, 13):
            result[i] = 0.0
        return
    var center = V3(t1x / t0, t1y / t0, t1z / t0)
    result[1] = center.x
    result[2] = center.y
    result[3] = center.z
    var ixx = t2y + t2z - t0 * (center.y * center.y + center.z * center.z)
    var iyy = t2z + t2x - t0 * (center.z * center.z + center.x * center.x)
    var izz = t2x + t2y - t0 * (center.x * center.x + center.y * center.y)
    var ixy = -tpx + t0 * center.x * center.y
    var iyz = -tpy + t0 * center.y * center.z
    var izx = -tpz + t0 * center.z * center.x
    result[4] = ixx
    result[5] = ixy
    result[6] = izx
    result[7] = ixy
    result[8] = iyy
    result[9] = iyz
    result[10] = izx
    result[11] = iyz
    result[12] = izz


@always_inline
def random01(mut state: Int) -> Tuple[Float64, Int]:
    state = (state * 48271) % 2147483647
    if state <= 0:
        state += 2147483646
    return (Float64(state) / 2147483647.0, state)


@always_inline
def random_barycentric(mut state: Int) -> Tuple[Float64, Float64, Float64, Int]:
    var r1: Float64
    var r2: Float64
    r1, state = random01(state)
    r2, state = random01(state)
    if r1 + r2 > 1.0:
        r1 = 1.0 - r1
        r2 = 1.0 - r2
    return (1.0 - r1 - r2, r1, r2, state)


# vcglib: vcg/complex/algorithms/point_sampling.h SurfaceSampling::Montecarlo
@export("mvc_montecarlo_f64")
def mvc_montecarlo_f64(
    vertices_address: Int,
    faces_address: Int,
    result_address: Int,
    cumulative_address: Int,
    face_count: Int,
    sample_count: Int,
    seed: Int,
) abi("C") -> Int:
    var vertices = fp(vertices_address)
    var faces = ip(faces_address)
    var result = fp(result_address)
    var cumulative = fp(cumulative_address)
    if face_count <= 0 or sample_count <= 0:
        return 0
    cumulative[0] = 0.0
    for f in range(face_count):
        var a = load_v(vertices, Int(faces[3 * f]))
        var b = load_v(vertices, Int(faces[3 * f + 1]))
        var c = load_v(vertices, Int(faces[3 * f + 2]))
        cumulative[f + 1] = cumulative[f] + 0.5 * (b - a).cross(c - a).norm()
    var area = cumulative[face_count]
    if area <= 0.0:
        return 0
    var state = seed % 2147483647
    if state <= 0:
        state += 2147483646
    for i in range(sample_count):
        var r: Float64
        r, state = random01(state)
        var value = area * r
        var lo = 1
        var hi = face_count
        while lo < hi:
            var mid = (lo + hi) // 2
            if cumulative[mid] >= value:
                hi = mid
            else:
                lo = mid + 1
        var f = lo - 1
        var w0: Float64
        var w1: Float64
        var w2: Float64
        w0, w1, w2, state = random_barycentric(state)
        store_v(
            result, i,
            load_v(vertices, Int(faces[3 * f])) * w0
            + load_v(vertices, Int(faces[3 * f + 1])) * w1
            + load_v(vertices, Int(faces[3 * f + 2])) * w2,
        )
    return sample_count


# vcglib: vcg/complex/algorithms/point_sampling.h SurfaceSampling::StratifiedMontecarlo
@export("mvc_stratified_f64")
def mvc_stratified_f64(
    vertices_address: Int,
    faces_address: Int,
    result_address: Int,
    face_count: Int,
    requested_count: Int,
    seed: Int,
) abi("C") -> Int:
    var vertices = fp(vertices_address)
    var faces = ip(faces_address)
    var result = fp(result_address)
    if face_count <= 0 or requested_count <= 0:
        return 0
    var area = 0.0
    for f in range(face_count):
        var a = load_v(vertices, Int(faces[3 * f]))
        var b = load_v(vertices, Int(faces[3 * f + 1]))
        var c = load_v(vertices, Int(faces[3 * f + 2]))
        area += 0.5 * (b - a).cross(c - a).norm()
    if area <= 0.0:
        return 0
    var state = seed % 2147483647
    if state <= 0:
        state += 2147483646
    var fractional = 0.0
    var written = 0
    for f in range(face_count):
        var a = load_v(vertices, Int(faces[3 * f]))
        var b = load_v(vertices, Int(faces[3 * f + 1]))
        var c = load_v(vertices, Int(faces[3 * f + 2]))
        fractional += (
            0.5 * (b - a).cross(c - a).norm()
            * Float64(requested_count) / area
        )
        var count = Int(fractional)
        for _ in range(count):
            var w0: Float64
            var w1: Float64
            var w2: Float64
            w0, w1, w2, state = random_barycentric(state)
            store_v(result, written, a * w0 + b * w1 + c * w2)
            written += 1
        fractional -= Float64(count)
    return written


@always_inline
def closest_on_segment(a: V3, b: V3, q: V3) -> V3:
    var edge = b - a
    var norm2 = edge.squared_norm()
    if norm2 < 2.2250738585072014e-308:
        return (a + b) * 0.5
    var t = (q - a).dot(edge) / norm2
    t = max(0.0, min(1.0, t))
    return a * (1.0 - t) + b * t


# vcglib: vcg/space/distance3.h TrianglePointDistance
@always_inline
def closest_on_triangle(a: V3, b: V3, c: V3, q: V3) -> V3:
    var normal = (b - a).cross(c - a)
    var norm2 = normal.squared_norm()
    if norm2 > 0.0:
        var projected = q - normal * ((q - a).dot(normal) / norm2)
        var n0 = (a - projected).cross(b - projected)
        var n1 = (b - projected).cross(c - projected)
        var n2 = (c - projected).cross(a - projected)
        if normal.dot(n0) >= 0.0 and normal.dot(n1) >= 0.0 and normal.dot(n2) >= 0.0:
            return projected
    var p0 = closest_on_segment(a, b, q)
    var p1 = closest_on_segment(b, c, q)
    var p2 = closest_on_segment(c, a, q)
    var d0 = (p0 - q).squared_norm()
    var d1 = (p1 - q).squared_norm()
    var d2 = (p2 - q).squared_norm()
    if d0 <= d1 and d0 <= d2:
        return p0
    if d1 <= d2:
        return p1
    return p2


# vcglib: vcg/complex/algorithms/closest.h GetClosestFaceBase
@export("mvc_closest_points_f64")
def mvc_closest_points_f64(
    vertices_address: Int,
    faces_address: Int,
    queries_address: Int,
    points_address: Int,
    distances_address: Int,
    face_indices_address: Int,
    face_count: Int,
    query_count: Int,
) abi("C"):
    var vertices = fp(vertices_address)
    var faces = ip(faces_address)
    var queries = fp(queries_address)
    var points = fp(points_address)
    var distances = fp(distances_address)
    var face_indices = ip(face_indices_address)
    for i in range(query_count):
        var q = load_v(queries, i)
        var best = HUGE
        var best_point = V3.zero()
        var best_face = -1
        for f in range(face_count):
            var p = closest_on_triangle(
                load_v(vertices, Int(faces[3 * f])),
                load_v(vertices, Int(faces[3 * f + 1])),
                load_v(vertices, Int(faces[3 * f + 2])),
                q,
            )
            var d2 = (p - q).squared_norm()
            if d2 < best:
                best = d2
                best_point = p
                best_face = f
        store_v(points, i, best_point)
        distances[i] = sqrt(best) if best_face >= 0 else HUGE
        face_indices[i] = Int64(best_face)


# vcglib: vcg/space/intersection3.h IntersectionLineTriangle/IntersectionRayTriangle
@export("mvc_ray_mesh_f64")
def mvc_ray_mesh_f64(
    vertices_address: Int,
    faces_address: Int,
    origins_address: Int,
    directions_address: Int,
    distances_address: Int,
    barycentric_address: Int,
    face_indices_address: Int,
    face_count: Int,
    ray_count: Int,
) abi("C"):
    var vertices = fp(vertices_address)
    var faces = ip(faces_address)
    var origins = fp(origins_address)
    var directions = fp(directions_address)
    var distances = fp(distances_address)
    var barycentric = fp(barycentric_address)
    var face_indices = ip(face_indices_address)
    for i in range(ray_count):
        var origin = load_v(origins, i)
        var direction = load_v(directions, i)
        var best = HUGE
        var best_u = 0.0
        var best_v = 0.0
        var best_face = -1
        for f in range(face_count):
            var a = load_v(vertices, Int(faces[3 * f]))
            var b = load_v(vertices, Int(faces[3 * f + 1]))
            var c = load_v(vertices, Int(faces[3 * f + 2]))
            var edge1 = b - a
            var edge2 = c - a
            var pvec = direction.cross(edge2)
            var det = edge1.dot(pvec)
            if det > 0.000001 or det < -0.000001:
                var tvec = origin - a
                var u_num = tvec.dot(pvec)
                var qvec = tvec.cross(edge1)
                var v_num = direction.dot(qvec)
                var inside = (
                    (u_num >= 0.0 and u_num <= det and v_num >= 0.0 and u_num + v_num <= det)
                    if det > 0.0
                    else (u_num <= 0.0 and u_num >= det and v_num <= 0.0 and u_num + v_num >= det)
                )
                if inside:
                    var inv_det = 1.0 / det
                    var t = edge2.dot(qvec) * inv_det
                    if t >= 0.0 and t < best:
                        best = t
                        best_u = u_num * inv_det
                        best_v = v_num * inv_det
                        best_face = f
        distances[i] = best
        barycentric[3 * i] = 1.0 - best_u - best_v
        barycentric[3 * i + 1] = best_u
        barycentric[3 * i + 2] = best_v
        face_indices[i] = Int64(best_face)


@always_inline
def heap_node_less(
    heap: IPtr, distances: FPtr, left: Int, right: Int
) -> Bool:
    var left_node = Int(heap[left])
    var right_node = Int(heap[right])
    var left_distance = distances[left_node]
    var right_distance = distances[right_node]
    return (
        left_distance < right_distance
        or (left_distance == right_distance and left_node < right_node)
    )


# vcglib: vcg/complex/algorithms/geodesic.h Geodesic::GeoDistance/Visit
@export("mvc_geodesic_f64")
def mvc_geodesic_f64(
    vertices_address: Int,
    faces_address: Int,
    seeds_address: Int,
    distances_address: Int,
    sources_address: Int,
    visited_address: Int,
    heap_address: Int,
    incidence_offsets_address: Int,
    incident_faces_address: Int,
    vertex_count: Int,
    face_count: Int,
    seed_count: Int,
    max_distance: Float64,
) abi("C") -> Int:
    var vertices = fp(vertices_address)
    var faces = ip(faces_address)
    var seeds = ip(seeds_address)
    var distances = fp(distances_address)
    var sources = ip(sources_address)
    var visited = ip(visited_address)
    var heap = ip(heap_address)
    var incidence_offsets = ip(incidence_offsets_address)
    var incident_faces = ip(incident_faces_address)
    fill_floats(distances, vertex_count, HUGE)
    fill_ints(sources, vertex_count, -1)
    fill_ints(visited, vertex_count, -1)
    var heap_size = 0
    for i in range(seed_count):
        var s = Int(seeds[i])
        if s >= 0 and s < vertex_count:
            distances[s] = 0.0
            sources[s] = Int64(s)
            if visited[s] < 0:
                var heap_position = heap_size
                heap[heap_position] = Int64(s)
                visited[s] = Int64(heap_position)
                heap_size += 1
                while heap_position > 0:
                    var parent = (heap_position - 1) // 2
                    if not heap_node_less(
                        heap, distances, heap_position, parent
                    ):
                        break
                    var parent_node = heap[parent]
                    heap[parent] = heap[heap_position]
                    heap[heap_position] = parent_node
                    visited[Int(heap[parent])] = Int64(parent)
                    visited[Int(heap[heap_position])] = Int64(heap_position)
                    heap_position = parent
    var reached = 0
    while heap_size > 0:
        var curr = Int(heap[0])
        heap_size -= 1
        if heap_size > 0:
            heap[0] = heap[heap_size]
            visited[Int(heap[0])] = 0
            var heap_position = 0
            while True:
                var left = 2 * heap_position + 1
                if left >= heap_size:
                    break
                var right = left + 1
                var smallest = left
                if (
                    right < heap_size
                    and heap_node_less(heap, distances, right, left)
                ):
                    smallest = right
                if not heap_node_less(
                    heap, distances, smallest, heap_position
                ):
                    break
                var position_node = heap[heap_position]
                heap[heap_position] = heap[smallest]
                heap[smallest] = position_node
                visited[Int(heap[heap_position])] = Int64(heap_position)
                visited[Int(heap[smallest])] = Int64(smallest)
                heap_position = smallest
        var d_curr = distances[curr]
        if d_curr > max_distance:
            break
        visited[curr] = -2
        reached += 1
        var previous_face = -1
        for incidence in range(
            Int(incidence_offsets[curr]), Int(incidence_offsets[curr + 1])
        ):
            var f = Int(incident_faces[incidence])
            if f == previous_face:
                continue
            previous_face = f
            var z = -1
            for j in range(3):
                if Int(faces[3 * f + j]) == curr:
                    z = j
            if z >= 0:
                for k in range(2):
                    var pw = Int(faces[3 * f + (z + 1 + k) % 3])
                    var pw1 = Int(faces[3 * f + (z + 2 - k) % 3])
                    var edge_curr_pw = (load_v(vertices, pw) - load_v(vertices, curr)).norm()
                    var candidate = d_curr + edge_curr_pw
                    var d_pw1 = distances[pw1]
                    var inter = (load_v(vertices, curr) - load_v(vertices, pw1)).norm()
                    var tol = (inter + d_curr + d_pw1) * 0.0001
                    if (
                        sources[pw1] == sources[curr]
                        and d_pw1 < HUGE * 0.5
                        and not (
                            inter + d_curr < d_pw1 + tol
                            or inter + d_pw1 < d_curr + tol
                            or d_curr + d_pw1 < inter + tol
                        )
                    ):
                        var ew_c = edge_curr_pw
                        var ew_w1 = (load_v(vertices, pw) - load_v(vertices, pw1)).norm()
                        var ec_w1 = inter
                        if ew_c > 0.0 and ew_w1 > 0.0 and ec_w1 > 0.0 and d_curr > 0.0 and d_pw1 > 0.0:
                            var wc = (load_v(vertices, pw) - load_v(vertices, curr)) / ew_c * ew_c
                            var ww1 = (load_v(vertices, pw) - load_v(vertices, pw1)) / ew_w1 * ew_w1
                            var w1c = (load_v(vertices, pw1) - load_v(vertices, curr)) / ec_w1 * ec_w1
                            var ca = max(-1.0, min(1.0, wc.dot(w1c) / (ew_c * ec_w1)))
                            var alpha = acos(ca)
                            var sem = (d_curr + d_pw1 + ec_w1) * 0.5
                            var aa = sem / ec_w1
                            var bb = aa * sem
                            var arg_a = max(0.0, min(1.0, (bb - aa * d_pw1) / d_curr))
                            var alpha2 = 2.0 * acos(min(1.0, sqrt(arg_a)))
                            if alpha + alpha2 <= PI:
                                var arg_b = max(0.0, min(1.0, (bb - aa * d_curr) / d_pw1))
                                var beta2 = 2.0 * acos(min(1.0, sqrt(arg_b)))
                                var cb = max(-1.0, min(1.0, ww1.dot(w1c * -1.0) / (ew_w1 * ec_w1)))
                                var beta = acos(cb)
                                if beta + beta2 > PI:
                                    candidate = d_pw1 + ew_w1
                                else:
                                    var theta = PI - alpha - alpha2
                                    var delta = cos(theta) * ew_c
                                    var height = sin(theta) * ew_c
                                    candidate = sqrt(
                                        height * height
                                        + (d_curr + delta) * (d_curr + delta)
                                    )
                    if candidate < distances[pw]:
                        distances[pw] = candidate
                        sources[pw] = sources[curr]
                        var heap_position = Int(visited[pw])
                        if heap_position == -1:
                            heap_position = heap_size
                            heap[heap_position] = Int64(pw)
                            visited[pw] = Int64(heap_position)
                            heap_size += 1
                        if heap_position >= 0:
                            while heap_position > 0:
                                var parent = (heap_position - 1) // 2
                                if not heap_node_less(
                                    heap, distances, heap_position, parent
                                ):
                                    break
                                var parent_node = heap[parent]
                                heap[parent] = heap[heap_position]
                                heap[heap_position] = parent_node
                                visited[Int(heap[parent])] = Int64(parent)
                                visited[Int(heap[heap_position])] = Int64(
                                    heap_position
                                )
                                heap_position = parent
    return reached


@always_inline
def quadric_add(mut q: FPtr, index: Int, values: FPtr, value_index: Int):
    comptime W = simdwidthof[DType.float64]()
    var j = 0
    while j + W <= 10:
        q.store(
            10 * index + j,
            q.load[width=W](10 * index + j)
            + values.load[width=W](10 * value_index + j),
        )
        j += W
    while j < 10:
        q[10 * index + j] += values[10 * value_index + j]
        j += 1


@always_inline
def add_plane_quadric(q: FPtr, index: Int, n: V3, offset: Float64, scale: Float64):
    q[10 * index] += n.x * n.x * scale
    q[10 * index + 1] += n.y * n.x * scale
    q[10 * index + 2] += n.z * n.x * scale
    q[10 * index + 3] += n.y * n.y * scale
    q[10 * index + 4] += n.z * n.y * scale
    q[10 * index + 5] += n.z * n.z * scale
    q[10 * index + 6] += -2.0 * offset * n.x * scale
    q[10 * index + 7] += -2.0 * offset * n.y * scale
    q[10 * index + 8] += -2.0 * offset * n.z * scale
    q[10 * index + 9] += offset * offset * scale


# vcglib: vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h InitQuadric
@export("mvc_qem_init_f64")
def mvc_qem_init_f64(
    vertices_address: Int,
    faces_address: Int,
    border_address: Int,
    quadrics_address: Int,
    vertex_count: Int,
    face_count: Int,
    boundary_weight: Float64,
    use_area: Int,
) abi("C"):
    var vertices = fp(vertices_address)
    var faces = ip(faces_address)
    var border = ip(border_address)
    var quadrics = fp(quadrics_address)
    zero_floats(quadrics, 10 * vertex_count)
    for f in range(face_count):
        var a_idx = Int(faces[3 * f])
        var b_idx = Int(faces[3 * f + 1])
        var c_idx = Int(faces[3 * f + 2])
        var a = load_v(vertices, a_idx)
        var b = load_v(vertices, b_idx)
        var c = load_v(vertices, c_idx)
        var direction_area = (b - a).cross(c - a)
        var area = direction_area.norm()
        if area > 0.0:
            var normal = direction_area / area
            var offset = normal.dot(a)
            var scale = area if use_area != 0 else 1.0
            add_plane_quadric(quadrics, a_idx, normal, offset, scale)
            add_plane_quadric(quadrics, b_idx, normal, offset, scale)
            add_plane_quadric(quadrics, c_idx, normal, offset, scale)
            for j in range(3):
                if border[3 * f + j] != 0:
                    var u = Int(faces[3 * f + j])
                    var v = Int(faces[3 * f + (j + 1) % 3])
                    var edge = load_v(vertices, v) - load_v(vertices, u)
                    var length = edge.norm()
                    if length > 0.0:
                        var bn = normal.cross(edge / length) * boundary_weight
                        var bo = bn.dot(load_v(vertices, u))
                        add_plane_quadric(quadrics, u, bn, bo, 1.0)
                        add_plane_quadric(quadrics, v, bn, bo, 1.0)


@always_inline
def quadric_apply(q: FPtr, index0: Int, index1: Int, p: V3) -> Float64:
    var a0 = q[10 * index0] + q[10 * index1]
    var a1 = q[10 * index0 + 1] + q[10 * index1 + 1]
    var a2 = q[10 * index0 + 2] + q[10 * index1 + 2]
    var a3 = q[10 * index0 + 3] + q[10 * index1 + 3]
    var a4 = q[10 * index0 + 4] + q[10 * index1 + 4]
    var a5 = q[10 * index0 + 5] + q[10 * index1 + 5]
    var b0 = q[10 * index0 + 6] + q[10 * index1 + 6]
    var b1 = q[10 * index0 + 7] + q[10 * index1 + 7]
    var b2 = q[10 * index0 + 8] + q[10 * index1 + 8]
    var cc = q[10 * index0 + 9] + q[10 * index1 + 9]
    return (
        p.x * p.x * a0 + 2.0 * p.x * p.y * a1 + 2.0 * p.x * p.z * a2 + p.x * b0
        + p.y * p.y * a3 + 2.0 * p.y * p.z * a4 + p.y * b1
        + p.z * p.z * a5 + p.z * b2 + cc
    )


@always_inline
def qem_candidate(
    vertices: FPtr, edges: IPtr, q: FPtr, e: Int, epsilon: Float64
) -> Tuple[V3, Float64]:
    var u = Int(edges[2 * e])
    var v = Int(edges[2 * e + 1])
    var midpoint = (load_v(vertices, u) + load_v(vertices, v)) * 0.5
    var position = midpoint
    if quadric_apply(q, u, v, midpoint) > 2.0 * epsilon:
        var a00 = q[10 * u] + q[10 * v]
        var a01 = q[10 * u + 1] + q[10 * v + 1]
        var a02 = q[10 * u + 2] + q[10 * v + 2]
        var a11 = q[10 * u + 3] + q[10 * v + 3]
        var a12 = q[10 * u + 4] + q[10 * v + 4]
        var a22 = q[10 * u + 5] + q[10 * v + 5]
        var r0 = -0.5 * (q[10 * u + 6] + q[10 * v + 6])
        var r1 = -0.5 * (q[10 * u + 7] + q[10 * v + 7])
        var r2 = -0.5 * (q[10 * u + 8] + q[10 * v + 8])
        var det = (
            a00 * (a11 * a22 - a12 * a12)
            - a01 * (a01 * a22 - a12 * a02)
            + a02 * (a01 * a12 - a11 * a02)
        )
        if abs(det) > 1.0e-15:
            var dx = (
                r0 * (a11 * a22 - a12 * a12)
                - a01 * (r1 * a22 - a12 * r2)
                + a02 * (r1 * a12 - a11 * r2)
            )
            var dy = (
                a00 * (r1 * a22 - a12 * r2)
                - r0 * (a01 * a22 - a12 * a02)
                + a02 * (a01 * r2 - r1 * a02)
            )
            var dz = (
                a00 * (a11 * r2 - r1 * a12)
                - a01 * (a01 * r2 - r1 * a02)
                + r0 * (a01 * a12 - a11 * a02)
            )
            position = V3(dx / det, dy / det, dz / det)
    var priority = max(epsilon, quadric_apply(q, u, v, position))
    if priority <= epsilon:
        priority *= (load_v(vertices, u) - load_v(vertices, v)).norm()
    return (position, priority)


# vcglib: vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h ComputePosition/ComputePriority
@export("mvc_qem_evaluate_f64")
def mvc_qem_evaluate_f64(
    vertices_address: Int,
    edges_address: Int,
    quadrics_address: Int,
    positions_address: Int,
    errors_address: Int,
    edge_count: Int,
    epsilon: Float64,
) abi("C"):
    var vertices = fp(vertices_address)
    var edges = ip(edges_address)
    var q = fp(quadrics_address)
    var positions = fp(positions_address)
    var errors = fp(errors_address)
    for e in range(edge_count):
        var position, priority = qem_candidate(vertices, edges, q, e, epsilon)
        store_v(positions, e, position)
        errors[e] = priority


@export("mvc_qem_decimate_unconstrained_f64")
def mvc_qem_decimate_unconstrained_f64(
    vertices_address: Int,
    faces_address: Int,
    edges_address: Int,
    quadrics_address: Int,
    result_address: Int,
    vertex_count: Int,
    face_count_arg: Int,
    edge_count_arg: Int,
    target_faces: Int,
    epsilon: Float64,
) abi("C"):
    var vertices = fp(vertices_address)
    var faces = ip(faces_address)
    var edges = ip(edges_address)
    var q = fp(quadrics_address)
    var result = ip(result_address)
    var face_count = face_count_arg
    var edge_count = edge_count_arg
    var collapsed = 0
    while face_count > target_faces and edge_count > 0:
        var best_edge = -1
        var best_error = HUGE
        var best_position = V3.zero()
        for e in range(edge_count):
            var position, priority = qem_candidate(vertices, edges, q, e, epsilon)
            if priority < best_error:
                best_edge = e
                best_error = priority
                best_position = position
        if best_edge < 0:
            break

        var u = Int(edges[2 * best_edge])
        var v = Int(edges[2 * best_edge + 1])
        store_v(vertices, v, best_position)
        quadric_add(q, v, q, u)

        var face_write = 0
        for f in range(face_count):
            var a = Int(faces[3 * f])
            var b = Int(faces[3 * f + 1])
            var c = Int(faces[3 * f + 2])
            if a == u:
                a = v
            if b == u:
                b = v
            if c == u:
                c = v
            if a != b and b != c and c != a:
                faces[3 * face_write] = Int64(a)
                faces[3 * face_write + 1] = Int64(b)
                faces[3 * face_write + 2] = Int64(c)
                face_write += 1
        face_count = face_write

        var edge_write = 0
        for e in range(edge_count):
            var a = Int(edges[2 * e])
            var b = Int(edges[2 * e + 1])
            if a == u:
                a = v
            if b == u:
                b = v
            if a != b:
                if a > b:
                    var tmp = a
                    a = b
                    b = tmp
                edges[2 * edge_write] = Int64(a)
                edges[2 * edge_write + 1] = Int64(b)
                edge_write += 1
        edge_count = edge_write

        for i in range(1, edge_count):
            var a = edges[2 * i]
            var b = edges[2 * i + 1]
            var j = i
            while j > 0:
                var previous_a = edges[2 * (j - 1)]
                var previous_b = edges[2 * (j - 1) + 1]
                if previous_a < a or (previous_a == a and previous_b <= b):
                    break
                edges[2 * j] = previous_a
                edges[2 * j + 1] = previous_b
                j -= 1
            edges[2 * j] = a
            edges[2 * j + 1] = b

        if edge_count > 0:
            edge_write = 1
            for e in range(1, edge_count):
                if (
                    edges[2 * e] != edges[2 * (edge_write - 1)]
                    or edges[2 * e + 1] != edges[2 * (edge_write - 1) + 1]
                ):
                    edges[2 * edge_write] = edges[2 * e]
                    edges[2 * edge_write + 1] = edges[2 * e + 1]
                    edge_write += 1
            edge_count = edge_write
        collapsed += 1
    result[0] = Int64(face_count)
    result[1] = Int64(collapsed)


@export("mvc_version")
def mvc_version() abi("C") -> Int:
    return 1
