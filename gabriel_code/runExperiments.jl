using LinearAlgebra, JuMP, Ipopt, MAT
using DataFrames, CSV

include("heuristics.jl")
include("relaxations.jl")

# Choose data_size to be either 63, 90, 124
data_size = 63
file = matopen("data$(data_size).mat");
data_size == 63 ? C = Symmetric(read(file, string("A"))) : C = Symmetric(read(file, string("C")));

n = size(C,1);
global s = 20; 

results = [];
results_filepath = "gabriel_results.csv"

for t = 1:s
    # local search procedure
    x_ls,z_ls = run_all_LS(C,s,t);
    # factorization bound
    x_ddfact, z_ddfact = ddfact_gmesp(C,s,t);
    # augmented factorization bound
    x_aug_ddfact, z_aug_ddfact = aug_ddfact_gmesp(C,s,t);
    # spectral bound 
    z_spec = spectral_bound(C,t);
    # results for iteration t
    arr = [n,s,t, z_ls, z_spec, z_ddfact, z_aug_ddfact];
    push!(results,arr)
    # show results: n, s, gap spectral, gap ddfact, gap augmented ddfact
    @info n,s,t,z_spec-z_ls,z_ddfact-z_ls,z_aug_ddfact-z_ls
end

results_matrix = hcat(results...)'

df = DataFrame(results_matrix, [:n, :s, :t, :z_ls, :z_spec, :z_ddfact, :z_aug_ddfact])

CSV.write(results_filepath, df)