
# debug why discontinuous at border.



include("../examples/helpers/visualization.jl")


PyPlot.close("all")

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])


Random.seed!(25)


fig_num = 1

radius = 0.3 #3.0
δ = 1e-5

#xq = [-2.58, 9.13]
#xq =  [-2.038690328209032, 9.387816600304607]
#xq = X_parts[2][3]

#xq =  [-2.468; -6.576]
#xq =  [-2.331; -6.462]

#xq =  [-2.491; -6.688]
#xq =  [-2.375; -6.574]

#xq = reverse(xq)

#xq =  [-0.766; -5.782] # dark.
#xq = [-0.775; -5.671] # brighter.


# linee across boundary.
xq_st = [0.16; -4.441]
xq_fin = [1.053; -4.513]
t_xq = LinRange(0, 1.0, 100)
dir_xq = xq_fin-xq_st
Xq = collect( xq_st + t_xq[i] .* dir_xq for i = 1:length(t_xq) )

# individual models..
q_all = xx->collect( PatchMixtureKriging.queryinner(xx, X_set[m], θ, η.c_set[m]) for m = 1:length(X_set) )
q_all_evals_tuple = q_all.(Xq)

s_select_a = 3
q_evals_a = collect(q_all_evals_tuple[n][s_select_a][1] for n = 1:length(t_xq))

s_select_b = 4
q_evals_b = collect(q_all_evals_tuple[n][s_select_b][1] for n = 1:length(t_xq))

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(q_evals_a, label = "$(s_select_a)")
PyPlot.plot(q_evals_b, "--", label = "$(s_select_b)")

PyPlot.title("queries from GPs")
PyPlot.legend()


 # I am here. figure out why the weighted query has a dip.
 
# the weighted models.
q = xx->PatchMixtureKriging.querymixtureGP(xx, η, root, levels, radius, δ, θ, σ², weight_θ;
debug_flag = true)

q_Xq_tuple = q.(Xq)

q_evals = collect( q_Xq_tuple[n][1] for n = 1:length(t_xq))


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(q_evals)

PyPlot.title("weighted query vs. t index")
PyPlot.legend()


## point evaluate.
xq = [0.536; 4.494]
Yq1, Vq1 =  q(xq)

println("xq = ", xq)
println("Yq1 = ", Yq1)
println("Vq1 = ", Vq1)
println()


#### debug.

debug_vars_set = collect( q_Xq_tuple[n][end] for n = 1:length(t_xq))

N_regions = length(X_parts)


Ws_tilde, Us, Ps, Rs = parsedebugvarsset(N_regions, debug_vars_set, 1.0)


# diagnose weights.
Ws_tilde_display = collect( collect( Ws_tilde[n][i] for n = 1:length(t_xq) ) for i = 1:N_regions )

PyPlot.figure(fig_num)
fig_num += 1

for i = 1:N_regions
    PyPlot.plot(Ws_tilde_display[i], label = "region $(i)")
end

PyPlot.title("weights vs t index")
PyPlot.legend()

# diagnose u.
Us_display = collect( collect( Us[n][i] for n = 1:length(t_xq) ) for i = 1:N_regions )

PyPlot.figure(fig_num)
fig_num += 1

for i = 1:N_regions
    PyPlot.plot(Us_display[i], label = "region $(i)")
end

PyPlot.title("u vs t index")
PyPlot.legend()



println("q_all(Xq[86]) = ")
display(q_all(Xq[86]))

println("Us[86] = ")
display(Us[86])

println("Ps[86] = ")
display(Ps[86])

println("Rs[86] = ")
display(Rs[86])

println()

# figure out why blending is not available for one side.
# next, try adaptive RKHS.
 
@assert 1==2

#### visualize.
println("Global scope for debug")
p = xq

# find the region for p.
p_region_ind = PatchMixtureKriging.findpartition(p, root, levels)
println("p_region_ind = ", p_region_ind)
println()

# get all hyperplanes.
hps = PatchMixtureKriging.fetchhyperplanes(root)
@assert length(hps) == length(X_parts) - 1 # sanity check.

# radius = 0.2
# δ = 1e-5
#region_inds, ts, zs, hps_keep_flags = PatchMixtureKriging.findneighbourpartitions(p, radius, root, levels, hps, p_region_ind; δ = δ)
region_inds, ts, zs, hps_keep_flags = findneighbourpartitions(p, radius, root, levels, hps, p_region_ind; δ = δ)

# debug.
hps_kept = hps[hps_keep_flags]
hp = hps_kept[1]
u = hp.v
c = hp.c

z_kept = zs[hps_keep_flags]
t_kept = ts[hps_keep_flags]
#z = z_kept[1]

dists = collect( norm(z_kept[i]-p) for i = 1:length(z_kept) )
@assert norm( dists - abs.(t_kept) ) < 1e-10 # sanity check.

# visualize on fresh plot.
fig_num, ax = visualize2Dpartition(X_set, y_set, t_set, fig_num, "levels = $(levels)")

PyPlot.scatter([p[1];], [p[2];], marker = "x", s = 600.0, label = "p")

for i = 1:length(z_kept)
    PyPlot.scatter([z_kept[i][1];], [z_kept[i][2];], marker = "d", s = 600.0, label = "z = $(i)")
end

PyPlot.axis([p[1]-radius; p[1]+radius; p[2]-radius; p[2]+radius])
PyPlot.axis("scaled")

PyPlot.legend()
