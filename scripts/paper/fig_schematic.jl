# julia --project=scripts/paper scripts/paper/fig_schematic.jl [outdir]

isdefined(@__MODULE__, :SIM) || include(joinpath(@__DIR__, "setup.jl"))

function arrow!(ax, p, q; color = INK, linewidth = 1.2, headsize = 8)
    d = q - p
    lines!(ax, [Point2f(p), Point2f(q)]; color, linewidth)
    scatter!(ax, [Point2f(q)]; marker = :rtriangle, rotation = atan(d[2], d[1]), markersize = headsize, color)
end

# agents are dots; a velocity or force is an arrow with its head at the front
agent!(ax, x; color = INK, size = 7) = scatter!(ax, [Point2f(x)]; marker = :circle, markersize = size, color, strokewidth = 0)

let
    BLUE = COLOR["d3"]   # relative position and the coupled groups; PURPLE: relative velocity; FORCE: forces

    # (a) local environment of agent i
    fig = Figure(size = (0.34WIDTH, 0.34WIDTH), figure_padding = 2)
    ax = Axis(fig[1, 1]; aspect = AxisAspect(1), limits = (-6.2, 6.2, -6.2, 6.2))
    hidedecorations!(ax)
    hidespines!(ax)
    R = 5.0
    θs = range(0, 2π; length = 200)
    lines!(ax, R .* cos.(θs), R .* sin.(θs); color = INK2, linewidth = 0.75, linestyle = :dash)
    text!(ax, R / sqrt(2), -R / sqrt(2); text = "R", align = (:left, :top), offset = (3, -2), fontsize = 8, color = INK2)
    neighbours = [([2.6, 1.6], [0.9, 0.5]), ([-2.2, 2.4], [0.8, -0.2]), ([1.2, -3.0], [1.0, 0.3]),
        ([-3.4, -1.2], [0.6, 0.7]), ([3.8, -1.0], [0.7, 0.6])]
    vi = [1.0, 0.25]
    outside = ([-4.8, 4.0], [1.0, -0.3])
    agent!(ax, outside[1]; color = (INK2, 0.3), size = 6)
    arrow!(ax, outside[1], outside[1] .+ 1.1 .* outside[2]; color = (INK2, 0.3), linewidth = 0.8, headsize = 5)
    # r_ij and u_ij for one neighbour
    xj, vj = neighbours[1]
    arrow!(ax, [0.0, 0.0], xj; color = BLUE, linewidth = 1.2, headsize = 7)
    text!(ax, xj[1] / 2, xj[2] / 2; text = rich("r", subscript("ij")), offset = (-12, 4), fontsize = 9, color = BLUE)
    uij = vj .- vi
    for (x, v) in neighbours
        arrow!(ax, x, x .+ 1.1 .* v; color = INK2, linewidth = 0.9, headsize = 6)
        agent!(ax, x; color = INK2, size = 6)
    end
    arrow!(ax, xj, xj .+ 4 .* uij; color = PURPLE, linewidth = 1.2, headsize = 7)
    text!(ax, xj[1] + 4uij[1], xj[2] + 4uij[2]; text = rich("u", subscript("ij")), offset = (4, 0), fontsize = 9, color = PURPLE)
    text!(ax, xj[1], xj[2]; text = "j", offset = (4, -11), fontsize = 8, color = INK2)
    arrow!(ax, [0.0, 0.0], 1.6 .* vi; color = INK, linewidth = 1.4, headsize = 8)
    agent!(ax, [0.0, 0.0]; size = 9)
    text!(ax, 0.0, 0.0; text = "i", offset = (-4, -12), fontsize = 9, color = INK)
    text!(ax, 1.6vi[1], 1.6vi[2]; text = rich("v", subscript("i")), offset = (4, -2), fontsize = 9, color = INK)

    save(joinpath(OUT, "fig_schematic_env.pdf"), fig)

    # (b) pure body-ordered expansion
    fig = Figure(size = (0.62WIDTH, 0.2WIDTH), figure_padding = 2)
    gl = fig[1, 1] = GridLayout()
    # each term: the agents it couples (i and n - 1 neighbours, one group drawn) and its force on i (orange)
    others = ([2.0, 1.3], [0.2, 2.2], [1.9, -1.2])
    names = ["j", "k", "l"]
    forces = ([-0.2, -1.0], [0.35, -0.95], [0.4, 0.95], [1.0, -0.35])
    titles = ["F⁽¹⁾: 1-body", "F⁽²⁾: 2-body", "F⁽³⁾: 3-body", "F⁽⁴⁾: 4-body"]
    for N in 0:3
        a = Axis(gl[1, N + 1]; aspect = AxisAspect(1), limits = (-1.6, 2.8, -2.2, 2.8), title = titles[N + 1], titlesize = TITLE)
        hidedecorations!(a)
        hidespines!(a)
        pts = [[0.0, 0.0], others[1:N]...]
        N >= 2 && poly!(a, Point2f.(N == 2 ? pts : pts[[1, 3, 2, 4]]); color = (BLUE, 0.15), strokewidth = 0)
        for j in 1:length(pts), k in (j + 1):length(pts)
            lines!(a, [Point2f(pts[j]), Point2f(pts[k])]; color = (BLUE, j == 1 ? 1.0 : 0.5), linewidth = j == 1 ? 1.0 : 0.6)
        end
        for (j, p) in enumerate(pts[2:end])
            agent!(a, p; color = INK2, size = 6)
            text!(a, p[1], p[2]; text = names[j], offset = (4, 2), fontsize = 8, color = INK2)
        end
        arrow!(a, [0.0, 0.0], forces[N + 1]; color = FORCE, linewidth = 1.3, headsize = 6)
        agent!(a, [0.0, 0.0]; size = 8)
        text!(a, 0.0, 0.0; text = "i", offset = (-9, 1), fontsize = 8, color = INK)
    end
    colgap!(gl, 2)
    save(joinpath(OUT, "fig_schematic_orders.pdf"), fig)
end
