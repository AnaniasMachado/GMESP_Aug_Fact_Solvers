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
t_vals = [i for i in 1:s]

matfile = matopen("data$n.mat")
C = read(matfile, "C")
close(matfile)

C = Matrix{Float64}(C)

# Choose t_a < λ_min(C)
t_a = minimum(eigvals(Symmetric(C))) - 1e-10

# -------------------------
# Tolerance
# -------------------------
tol = 1e-3

# -------------------------
# Data Collection
# -------------------------
df = DataFrame()
results_filepath = "results_gap_n$(n)_s$(s).csv"
results = []

for t in t_vals
    println("--------------------")
    println("t: $t")

    result = []
    append!(result, [n, s, t])

    runtime_ddgfact = @elapsed begin
        x_ddgfact, res_ddgfact, k_ddgfact = fw_gaug_fact(
            C, 0, s, t;
            tol = tol,
            line_search = false,
        )
    end
    z_ddgfact = gaug_fact_objective(x, t, t_a)

    runtime_ddgfact_plus = @elapsed begin
        x_ddgfact_plus, res_ddgfact_plus, k_ddgfact_plus = fw_gaug_fact(
            C, t_a, s, t;
            tol = tol,
            line_search = false,
        )
    end
    z_ddgfact_plus = gaug_fact_objective(x, t, t_a)

    x_ls, z_ls = run_all_LS(C, s, t)

    z_spec = spectral_bound(C, t)

    append!(result, [z_ddgfact-z_ls, z_ddgfact_plus-z_ls, z_spec-z_ls])

    push!(results, result)
end

results_matrix = hcat(results...)'
cols = [
    :n, :s, :t,
    :ddgfact_gap, :ddgfact_plus_gap, :spec_gap,
]

df = DataFrame(results_matrix, cols)

CSV.write(results_filepath, df)