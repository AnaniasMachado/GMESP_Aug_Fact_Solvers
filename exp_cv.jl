using Random
using CSV
using DataFrames

include("cv.jl")

Random.seed!(2)

function random_pd_matrix(n::Int)
    X = randn(n, n)
    return X' * X + 0.1I
end

n = 100
C = random_pd_matrix(n)

λmin, _ = eigs(Symmetric(C), nev=1, which=:SR)
λmin = λmin[1]

s = 50                   # sum(x) = s
ta = 0.9 *  λmin         # shift for C - tI

L = n
τ = 0.99 / sqrt(L)
σ = 0.99 / sqrt(L)

df = DataFrame()
results_filepath = "results.csv"

for t in 1:s
    cv_runtime = @elapsed begin
        xsol = condat_vu_gmesp(C, t, ta; s=s, τ=τ, σ=σ)
    end

    Ata = cholesky(C - ta * I).L
    a_cols = [Ata[:, i] for i in 1:n]

    g, grad = spectral_obj_grad(xsol, a_cols, t, ta)

    println("\n================ Condat–Vũ Results ================\n")
    println("  s = $s")
    println("  t = $t")

    println("Objective value:")
    println("  f(x) = ", -g)

    println("\nRuntime:")
    println("  elapsed time = ", round(cv_runtime, digits=4), " seconds")

    println("\nSolution statistics:")
    println("  sum(x)           = ", sum(xsol))
    println("  min(x), max(x)   = ", minimum(xsol), ", ", maximum(xsol))

    println("\nConstraint violations:")
    println("  equality (|sum(x) - s|) = ", abs(sum(xsol) - s))

    println("\nSolution vector (rounded):")
    println(round.(xsol, digits=4))

    println("\n===================================================\n")

    result = DataFrame(
        n = [n],
        s = [s],
        t = [t],
        ta = [ta],
        min_eigval = [λmin],
        obj = [-g],
        time = [cv_runtime],
    )

    append!(df, result)
end

CSV.write(results_filepath, df)