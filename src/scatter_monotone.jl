using CSV
using DataFrames
using Plots

n = 63
s = 20
t_vals = [1, 5, 10, 15]

# Read CSV
df = CSV.read("results_monotone_n$(n)_s$(s).csv", DataFrame)

# Loop over each t value
for t_val in t_vals
    # Filter the dataframe for this t
    df_t = filter(row -> row.t == t_val, df)

    # Create scatter plot
    scatter(
        df_t.tau,
        df_t.fw_obj,
        label = "FW",
        # yscale = :log10,
        xlabel = "Tau",
        ylabel = "FW objective",
        title = "t = $t_val"
    )

    # Save figure
    savefig("scatter_s$(s)_t$(t_val).png")
end
