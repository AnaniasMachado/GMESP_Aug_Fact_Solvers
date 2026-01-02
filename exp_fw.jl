using Random
using CSV
using DataFrames

include("fw.jl")
include("util.jl")

# -------------------------
# Problem data
# -------------------------
n = 100
s = 50
t_vals = [5*i for i in 1:10]

Random.seed!(2)
C = randn(n, n)
C = C' * C + I

# Choose t_a < λ_min(C)
t_a = 0.9 * minimum(eigvals(Symmetric(C)))

# -------------------------
# Compute A(t_a)
# -------------------------
At = compute_At(C, t_a)

# -------------------------
# Tolerance
# -------------------------

tol = 1e-3

# -------------------------
# Data Collection
# -------------------------

df = DataFrame()
results_filepath = "results.csv"

for t in t_vals
    println("--------------------")
    println("t: $t")

    # -------------------------
    # Run Frank-Wolfe
    # -------------------------
    fw_runtime = @elapsed begin
        fw_x, fw_gap, fw_k = fw_gaug_fact_paper(
            C, t_a, s, t;
            tol = tol
        )
    end
    fw_obj = gaug_fact_objective(fw_x, At, t, t_a)

    println("--------------------")
    println("Frank-Wolfe stats:")
    println("time: $fw_runtime")
    println("obj: $fw_obj")
    println("gap: $fw_gap")
    println("k: $fw_k")
    println("rp: $(abs(sum(fw_x) - s))")

    # -------------------------
    # Run Frank-Wolfe Exact LS
    # -------------------------
    fwls_runtime = @elapsed begin
        fwls_x, fwls_gap, fwls_k = fw_gaug_fact_exact_ls(
            C, t_a, s, t;
            tol = tol
        )
    end
    fwls_obj = gaug_fact_objective(fwls_x, At, t, t_a)
    spectral_bound = gaug_fact_objective(fwls_x, At, t, t_a)

    println("--------------------")
    println("Frank-Wolfe Exact LS stats:")
    println("time: $fwls_runtime")
    println("obj: $fwls_obj")
    println("gap: $fwls_gap")
    println("k: $fwls_k")
    println("rp: $(abs(sum(fwls_x) - s))")
    println("spectral bound: $spectral_bound")

    result = DataFrame(
        n = [n],
        s = [s],
        t = [t],
        fw_obj = [fw_obj],
        fw_gap = [fw_gap],
        fw_k = [fw_k],
        fw_runtime = [fw_runtime],
        fwls_obj = [fwls_obj],
        fwls_gap = [fwls_gap],
        fwls_k = [fwls_k],
        fwls_runtime = [fwls_runtime],
        spectral_bound = [spectral_bound],
    )

    append!(df, result)
end

CSV.write(results_filepath, df)