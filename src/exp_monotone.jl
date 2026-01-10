using Random
using MAT
using CSV
using DataFrames

include("fw.jl")
include("util.jl")

# -------------------------
# Problem data
# -------------------------
n = 63
s = 20
t_vals = [1, 5, 10, 15]

matfile = matopen("data$n.mat")
C = n == 63 ? read(matfile, "A") : read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)

tau_vals = range(0, eigmin(C)-1e-10, length = 100)

# -------------------------
# Tolerance
# -------------------------
tol = 1e-3

# -------------------------
# Warm-up phase (CRUCIAL)
# -------------------------
println("Warming up (compiling methods)...")

t_warm = t_vals[1]
tau_warm = tau_vals[1]
At_warm = compute_At(C, tau_warm)

fw_gaug_fact(C, tau_warm, s, t_warm; tol=tol)
fw_gaug_fact(C, tau_warm, s, t_warm; tol=tol, line_search=true)
afw_gaug_fact(C, tau_warm, s, t_warm; tol=tol, line_search=false)
afw_gaug_fact(C, tau_warm, s, t_warm; tol=tol, line_search=true)
pairwise_fw_gaug_fact(C, tau_warm, s, t_warm; tol=tol, line_search=false)
pairwise_fw_gaug_fact(C, tau_warm, s, t_warm; tol=tol, line_search=true)

# also warm up objective / bounds
x_dummy = fill(s / n, n)
gaug_fact_objective(x_dummy, At_warm, t_warm, tau_warm)
spectral_bound(C, t_warm)

println("Warm-up complete.\n")

# -------------------------
# Method Parameters
# -------------------------
# method_names = ["FW", "AFW", "PFW"]
method_names = ["FW"]
methods = Dict(
    "FW" => fw_gaug_fact,
    "AFW" => afw_gaug_fact,
    "PFW" => pairwise_fw_gaug_fact,
)
line_search_vals = [false]

# -------------------------
# Data Collection
# -------------------------
df = DataFrame()
results_filepath = "results_monotone_n$(n)_s$(s).csv"
results = []

for t in t_vals
    println("--------------------")
    println("t: $t")

    for tau in tau_vals
        result = []
        append!(result, [n, s, t])
        At = compute_At(C, tau)

        runtime = @elapsed begin
            x, gap, k = fw_gaug_fact(
                C, tau, s, t;
                tol = tol,
                line_search = false,
            )
        end
        obj = gaug_fact_objective(x, At, t, tau)
        spectral_bound_val = spectral_bound(C, t)
        append!(result, [tau, obj, runtime, spectral_bound_val])

        push!(results, result)
    end
end

results_matrix = hcat(results...)'
cols = [
    :n, :s, :t,
    :tau, :fw_obj, :fw_runtime,
    :spectral_bound_val
]

df = DataFrame(results_matrix, cols)

CSV.write(results_filepath, df)