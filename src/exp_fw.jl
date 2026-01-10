using Random
using MAT
using CSV
using DataFrames

include("fw.jl")
include("util.jl")

# -------------------------
# Problem data
# -------------------------
n = 90
s = 20
t_vals = [i for i in 1:s]

matfile = matopen("data$n.mat")
C = read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)

# Choose t_a < λ_min(C)
t_a = minimum(eigvals(Symmetric(C))) - 1e-10

# -------------------------
# Compute A(t_a)
# -------------------------
At = compute_At(C, t_a)

# -------------------------
# Tolerance
# -------------------------
tol = 1e-3

# -------------------------
# Warm-up phase (CRUCIAL)
# -------------------------
println("Warming up (compiling methods)...")

t_warm = t_vals[1]

fw_gaug_fact(C, t_a, s, t_warm; tol=tol)
fw_gaug_fact(C, t_a, s, t_warm; tol=tol, line_search=true)
afw_gaug_fact(C, t_a, s, t_warm; tol=tol, line_search=false)
afw_gaug_fact(C, t_a, s, t_warm; tol=tol, line_search=true)
pairwise_fw_gaug_fact(C, t_a, s, t_warm; tol=tol, line_search=false)
pairwise_fw_gaug_fact(C, t_a, s, t_warm; tol=tol, line_search=true)

# also warm up objective / bounds
x_dummy = fill(s / n, n)
gaug_fact_objective(x_dummy, At, t_warm, t_a)
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
line_search_vals = [false, true]

# -------------------------
# Data Collection
# -------------------------
df = DataFrame()
results_filepath = "results_fw_n$(n)_s$(s).csv"
results = []

for t in t_vals
    println("--------------------")
    println("t: $t")

    result = []
    append!(result, [n, s, t])

    for method_name in method_names
        for line_search_val in line_search_vals
            runtime = @elapsed begin
                x, gap, k = fw_gaug_fact(
                    C, t_a, s, t;
                    tol = tol,
                    line_search = line_search_val,
                )
            end
            obj = gaug_fact_objective(x, At, t, t_a)
            append!(result, [obj, gap, k, runtime])
        end
    end

    simplex_x = simplex_sol(At, s)
    simplex_obj = gaug_fact_objective(simplex_x, At, t, t_a)
    push!(result, simplex_obj)

    spectral_bound_val = spectral_bound(C, t)
    push!(result, spectral_bound_val)

    push!(results, result)
end

results_matrix = hcat(results...)'
cols = [
    :n, :s, :t,
    :fw_obj, :fw_gap, :fw_k, :fw_runtime,
    :fwls_obj, :fwls_gap, :fwls_k, :fwls_runtime,
    # :afw_obj, :afw_gap, :afw_k, :afw_runtime,
    # :afwls_obj, :afwls_gap, :afwls_k, :afwls_runtime,
    # :pfw_obj, :pfw_gap, :pfw_k, :pfw_runtime,
    # :pfwls_obj, :pfwls_gap, :pfwls_k, :pfwls_runtime,
    :simplex_obj, :spectral_bound_val
]

df = DataFrame(results_matrix, cols)

CSV.write(results_filepath, df)