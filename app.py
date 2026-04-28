import ctypes, os, platform, subprocess, time
from flask import Flask, render_template, request

# ── Compile fastmat.c into a shared library on first run ──────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if platform.system() == "Darwin":
    _lib_file = os.path.join(_here, "fastmat.dylib")
    _compile  = ["gcc", "-O2", "-dynamiclib", "fastmat.c", "-o", _lib_file]
else:
    _lib_file = os.path.join(_here, "fastmat.so")
    _compile  = ["gcc", "-O2", "-shared", "-fPIC", "fastmat.c", "-o", _lib_file]

if not os.path.exists(_lib_file):
    subprocess.run(_compile, cwd=_here, check=True)

# ── Load the library and declare types ───────────────────────────────────────
_lib = ctypes.CDLL(_lib_file)
_FP  = ctypes.POINTER(ctypes.c_float)

_lib.mat_add.argtypes  = [_FP, _FP, _FP, ctypes.c_int]
_lib.mat_add.restype   = None
_lib.mat_mul.argtypes  = [_FP, _FP, _FP, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.mat_mul.restype   = None
_lib.mat_relu.argtypes = [_FP, ctypes.c_int]
_lib.mat_relu.restype  = None

def _arr(lst):
    n = len(lst)
    return (ctypes.c_float * n)(*lst)

def add(a, b, rows, cols):
    n = rows * cols
    out = (ctypes.c_float * n)()
    _lib.mat_add(_arr(a), _arr(b), out, n)
    return list(out)

def multiply(a, b, m, k, n):
    out = (ctypes.c_float * (m * n))()
    _lib.mat_mul(_arr(a), _arr(b), out, m, k, n)
    return list(out)

def relu(a):
    arr = _arr(a)
    _lib.mat_relu(arr, len(a))
    return list(arr)

# Pure-Python multiply used only to show the speed comparison in the UI
def _py_multiply(a, b, m, k, n):
    out = [0.0] * (m * n)
    for i in range(m):
        for p in range(k):
            for j in range(n):
                out[i*n + j] += a[i*k + p] * b[p*n + j]
    return out

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    a_val, b_val, results = "3", "-2", None

    if request.method == "POST":
        try:
            a = float(request.form["a"])
            b = float(request.form["b"])
            a_val, b_val = request.form["a"], request.form["b"]

            def fmt(v):  return int(v) if v == int(v) else round(v, 4)
            def mfmt(v): return int(v) if v == int(v) else round(v, 2)

            # Addition
            add_res   = add([a, b], [b, a], 1, 2)
            add_trace = (
                f'C runs the loop: '
                f'i=0: <code>{fmt(a)} + {fmt(b)} = {fmt(add_res[0])}</code>, '
                f'i=1: <code>{fmt(b)} + {fmt(a)} = {fmt(add_res[1])}</code>'
            )

            # Multiply
            mat = [a, b, b, a]
            mr  = multiply(mat, mat, 2, 2, 2)
            mul_steps = (
                f'<li><span class="step-num">1</span><span class="step-text">C zeros out the result array, then runs 3 nested loops: i → k → j</span></li>'
                f'<li><span class="step-num">2</span><span class="step-text">out[0][0] = <code>{fmt(a)}×{fmt(a)} + {fmt(b)}×{fmt(b)}</code> = <strong>{mfmt(mr[0])}</strong></span></li>'
                f'<li><span class="step-num">3</span><span class="step-text">out[0][1] = <code>{fmt(a)}×{fmt(b)} + {fmt(b)}×{fmt(a)}</code> = <strong>{mfmt(mr[1])}</strong></span></li>'
                f'<li><span class="step-num">4</span><span class="step-text">out[1][0] = <code>{fmt(b)}×{fmt(a)} + {fmt(a)}×{fmt(b)}</code> = <strong>{mfmt(mr[2])}</strong></span></li>'
                f'<li><span class="step-num">5</span><span class="step-text">out[1][1] = <code>{fmt(b)}×{fmt(b)} + {fmt(a)}×{fmt(a)}</code> = <strong>{mfmt(mr[3])}</strong></span></li>'
                f'<li><span class="step-num">6</span><span class="step-text">Loop order i→k→j keeps memory accesses sequential — cache-friendly, no wasted reads</span></li>'
            )

            # ReLU
            relu_vals  = [a, b, -abs(a) if a != 0 else -1.0, -abs(b) if b != 0 else -3.0]
            relu_res   = relu(relu_vals)
            relu_items = [{"before": f"in: {fmt(v)}", "after": fmt(r), "clamped": v < 0}
                          for v, r in zip(relu_vals, relu_res)]

            # Benchmark: compare C (via ctypes) against pure Python
            size  = 32
            mat32 = [float((i + abs(a) + 1) % 7) for i in range(size * size)]

            t0 = time.perf_counter()
            for _ in range(3): multiply(mat32, mat32, size, size, size)
            c_ms = round((time.perf_counter() - t0) / 3 * 1000, 2)

            t0 = time.perf_counter()
            for _ in range(3): _py_multiply(mat32, mat32, size, size, size)
            py_ms = round((time.perf_counter() - t0) / 3 * 1000, 2)

            results = {
                "a": fmt(a), "b": fmt(b),
                "add_result": fmt(add_res[0]),
                "add_trace":  add_trace,
                "mul_steps":  mul_steps,
                "r00": mfmt(mr[0]), "r01": mfmt(mr[1]),
                "r10": mfmt(mr[2]), "r11": mfmt(mr[3]),
                "relu_items": relu_items,
                "py_ms": py_ms,
                "c_ms":  c_ms,
            }
        except (ValueError, ZeroDivisionError):
            pass

    return render_template("index.html", a_val=a_val, b_val=b_val, results=results)


if __name__ == "__main__":
    app.run(debug=True)
