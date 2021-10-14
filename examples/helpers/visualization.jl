function generatecolours(N::Int)
    # oversample since the HSV starts from red, and ends at red.
    M = round(Int, 1.3*N)

    colours = range(Colors.HSV(0,1,1), stop = Colors.HSV(-360,1,1), length = M)
    colours = convert(Vector{Colors.RGB}, colours)
    colours = collect( [colours[n].r; colours[n].g; colours[n].b] for n = 1:N )

    return colours
end

function visualize2Dpartition(X_parts::Vector{Vector{Vector{T}}},
    y_set,
    t_set,
    fig_num::Int,
    title_string::String) where T <: Real
    
    colours_for_pts = generatecolours(length(X_parts))

    PyPlot.figure(fig_num)
    ax = PyPlot.axes()
    fig_num += 1

    # partitions of points.
    for i = 1:length(X_parts)

        # points.
        X = X_parts[i]

        x1 = collect( X[n][1] for n = 1:length(X))
        x2 = collect( X[n][2] for n = 1:length(X))

        PyPlot.scatter(x1, x2, label = "$(i)", color = colours_for_pts[i])

        # annotate centroid.
        z1, z2 = Statistics.mean(X)
        # PyPlot.annotate("$(i)",
        # xy=[z1;z2],
        # xycoords="data",
        # fontsize=20.0,
        # #bbox = ["boxstyle"=>"rarrow,pad=0.3", "fc"=>"cyan", "ec"=>"b", "lw"=>2])
        # bbox = (z1,z2, z1+0.5, z2+0.5) )

        #PyPlot.text(z1, z2, "$(i)", backgroundcolor = "white")
        bbox = Dict("boxstyle"=>"round, pad=0.3", "fc"=>"cyan", "ec"=>"b", "lw"=>2)
        PyPlot.text(z1, z2, "$(i)", bbox = bbox)
    end

    ### boundaries of partitions.
    @assert length(t_set) == length(y_set)
    @assert !isempty(X_parts[1][1])

    #colours_for_boundaries = generatecolours(length(t_set))

    for i = 1:length(t_set)
        
       # PyPlot.plot(t_set[i], y_set[i], color = colours_for_boundaries[i])
       PyPlot.plot(t_set[i], y_set[i], color = "black")
    end
    
    PyPlot.axis("scaled")
    PyPlot.title(title_string)
    PyPlot.legend()

    return fig_num, ax
end


function visualizemeshgridpcolorx1horizontal( x_ranges::Vector{LinRange{T}},
    Y::Matrix{T},
    marker_locations::Vector,
    marker_symbol::String,
    fig_num::Int,
    title_string::String;
    x1_title_string::String = "Dimension 1",
    x2_title_string::String = "Dimension 2",
    cmap = "Greens_r") where T <: Real

    #
    @assert length(x_ranges) == 2
    x_coords = collect( collect(x_ranges[d]) for d = 1:2 )

    PyPlot.figure(fig_num)
    fig_num += 1
    PyPlot.pcolormesh(x_coords[1], x_coords[2], Y, cmap = cmap, shading = "auto")
    PyPlot.xlabel(x1_title_string)
    PyPlot.ylabel(x2_title_string)
    PyPlot.title(title_string)

    for i = 1:length(marker_locations)
        #pt = reverse(marker_locations[i])
        pt = marker_locations[i]
        PyPlot.annotate(marker_symbol, xy=pt, xycoords="data")
    end

    PyPlot.plt.colorbar()
    PyPlot.axis("scaled")

    return fig_num
end