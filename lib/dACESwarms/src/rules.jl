"""
    InteractionRule

Implements `velocity_update(rule, vᵢ, neighbours, dt) -> vᵢ′` and
`interaction_radius(rule) -> R`, with `neighbours = [(xⱼ - xᵢ, vⱼ) : |xⱼ - xᵢ| ≤ R]`
(minimum image).
"""
abstract type InteractionRule end

function velocity_update end

interaction_radius(rule::InteractionRule) = rule.radius

"""
    ForceRule <: InteractionRule

`vᵢ′ = vᵢ + dt aᵢ` with `aᵢ = acceleration(rule, vᵢ, neighbours)`.
"""
abstract type ForceRule <: InteractionRule end

function acceleration end

velocity_update(rule::ForceRule, v, neighbours, dt) = v + dt * acceleration(rule, v, neighbours)

smooth_cutoff(ρ, R) = ρ < R ? (1 - (ρ / R)^2)^2 : zero(ρ)

# C² step from 0 (x ≤ 0) to 1 (x ≥ w); Heaviside for w = 0
function smoothstep(x, w)
    w == 0 && return float(x > 0)
    t = clamp(x / w, 0, 1)
    return t^3 * (10 - 15t + 6t^2)
end

"""
    Boids(; radius, separation, cohere, separate, align, speed, smoothing = 0)

Reynolds boids: `vᵢ′ = speed · normalize(vᵢ + dt (cohere Σ rᵢⱼ - separate Σ_{ρ≤separation} rᵢⱼ + align Σ vⱼ) / Nᵢ)`.
`smoothing = w > 0`: weights `sⱼ = S(radius - ρ)` in the sums (a C² step of width `w` inside the radius),
`Nᵢ = √(w² + (Σ sⱼ)²)`, and the separation weight `1 - S(ρ - separation + w/2)`, a step of width `w` centred on
the separation radius, with `S(x)` rising from 0 at `x ≤ 0` to 1 at `x ≥ w`.
"""
Base.@kwdef struct Boids <: InteractionRule
    radius::Float64 = 5.0
    separation::Float64 = 2.0
    cohere::Float64 = 0.1
    separate::Float64 = 0.25
    align::Float64 = 0.04
    speed::Float64 = 1.0
    smoothing::Float64 = 0.0
end

function velocity_update(r::Boids, v, neighbours, dt)
    isempty(neighbours) && return v
    c = s = a = zero(v)
    W = 0.0
    for (dx, vj) in neighbours
        ρ = norm(dx)
        w = r.smoothing == 0 ? 1.0 : smoothstep(r.radius - ρ, r.smoothing)
        c += w * dx
        a += w * vj
        W += w
        s -= w * (1 - smoothstep(ρ - r.separation + r.smoothing / 2, r.smoothing)) * dx
    end
    W == 0 && return v
    r.smoothing > 0 && (W = sqrt(r.smoothing^2 + W^2))
    vnew = v + dt * (r.cohere * c + r.separate * s + r.align * a) / W
    return r.speed * vnew / norm(vnew)
end

"""
    DOrsogna(; radius, α, β, Cr, lr, Ca, la, γ, lv, κ, κv, κ4, l3, averaged, w0, chirality, ω, lateral)

Self-propulsion, Morse, Cucker–Smale and flanking/encirclement terms (body orders 1–4):

    aᵢ = (α - β|vᵢ|²) vᵢ + Σⱼ fc U'(ρ) r̂ᵢⱼ + γ Σⱼ fc e^{-ρ/lv} uᵢⱼ
         + Σ_{j<k} gⱼ gₖ (1 - r̂ᵢⱼ⋅r̂ᵢₖ)/2 (κ (rᵢⱼ + rᵢₖ) + κv (uᵢⱼ + uᵢₖ))/2
         + κ4 Σ_{j<k<l} gⱼ gₖ gₗ (1 - |r̂ᵢⱼ + r̂ᵢₖ + r̂ᵢₗ|²/9) (uᵢⱼ + uᵢₖ + uᵢₗ)/3,

`fc = (1 - ρ²/R²)²`, `U = Cr e^{-ρ/lr} - Ca e^{-ρ/la}`, `g = fc e^{-ρ/l3}`.
`averaged`: alignment `γ Σⱼ wⱼ uᵢⱼ / (w0 + Σⱼ wⱼ)`, `wⱼ = fc e^{-ρ/lv}`.

Chiral options (2D only, all 0 by default, which keeps the rule mirror symmetric):
`chirality = θ` turns the pairwise terms (Morse and alignment) by the angle `θ`, `ω` adds a constant turning
`ω (-vᵢʸ, vᵢˣ)`, and `lateral = b` (`|b| < 1`) weights the alignment by `1 + b sin φᵢⱼ`, with `φᵢⱼ` the bearing of
neighbour `j` from the heading of `i` (positive to the left).
"""
Base.@kwdef struct DOrsogna <: ForceRule
    radius::Float64 = 5.0
    α::Float64 = 1.0
    β::Float64 = 0.5
    Cr::Float64 = 1.0
    lr::Float64 = 0.8
    Ca::Float64 = 0.5
    la::Float64 = 2.0
    γ::Float64 = 0.5
    lv::Float64 = 2.0
    κ::Float64 = 0.0
    κv::Float64 = 0.0
    κ4::Float64 = 0.0
    l3::Float64 = 1.5
    averaged::Bool = false
    w0::Float64 = 1.0
    chirality::Float64 = 0.0
    ω::Float64 = 0.0
    lateral::Float64 = 0.0
end

chiral(r::DOrsogna) = r.chirality != 0 || r.ω != 0 || r.lateral != 0
turn(x, θ) = SVector(cos(θ) * x[1] - sin(θ) * x[2], sin(θ) * x[1] + cos(θ) * x[2])
# sine of the bearing of the neighbour at dx (distance ρ) from the heading v, positive to the left
bearing_sine(v, dx, ρ) = (n = norm(v); n == 0 ? 0.0 : (v[1] * dx[2] - v[2] * dx[1]) / (n * ρ))

function acceleration(r::DOrsogna, v, neighbours)
    chiral(r) && length(v) != 2 && throw(ArgumentError("the chiral options of DOrsogna are two-dimensional"))
    a = (r.α - r.β * dot(v, v)) * v
    r.ω != 0 && (a += r.ω * SVector(-v[2], v[1]))
    pair, align, W = zero(v), zero(v), 0.0
    for (dx, vj) in neighbours
        ρ = norm(dx)
        fc = smooth_cutoff(ρ, r.radius)
        dU = -r.Cr / r.lr * exp(-ρ / r.lr) + r.Ca / r.la * exp(-ρ / r.la)
        w = fc * exp(-ρ / r.lv) * (r.lateral == 0 ? 1.0 : 1 + r.lateral * bearing_sine(v, dx, ρ))
        pair += fc * dU * dx / ρ
        align += w * (vj - v)
        W += w
    end
    social = pair + r.γ * (r.averaged ? align / (r.w0 + W) : align)
    a += r.chirality == 0 ? social : turn(social, r.chirality)
    (r.κ != 0 || r.κv != 0 || r.κ4 != 0) || return a
    # the many-body terms: distances, weights and directions once per neighbour, not once per pair or triple
    ρs = [norm(dx) for (dx, _) in neighbours]
    gs = [smooth_cutoff(ρ, r.radius) * exp(-ρ / r.l3) for ρ in ρs]
    n = length(neighbours)
    if r.κ != 0 || r.κv != 0
        for j in 1:n, k in (j + 1):n
            (xj, vj), (xk, vk) = neighbours[j], neighbours[k]
            flank = (1 - dot(xj, xk) / (ρs[j] * ρs[k])) / 2
            a += gs[j] * gs[k] * flank * (r.κ * (xj + xk) + r.κv * (vj + vk - 2v)) / 2
        end
    end
    if r.κ4 != 0
        units = [dx / ρ for ((dx, _), ρ) in zip(neighbours, ρs)]
        for j in 1:n, k in (j + 1):n, l in (k + 1):n
            vj, vk, vl = neighbours[j][2], neighbours[k][2], neighbours[l][2]
            s = units[j] + units[k] + units[l]
            a += r.κ4 * gs[j] * gs[k] * gs[l] * (1 - dot(s, s) / 9) * (vj + vk + vl - 3v) / 3
        end
    end
    return a
end

"""
    Zonal(; radius, repulsion, orientation, turning, speed, smoothing = 0)

Simplified zonal model: `d = -Σ_{ρ<repulsion} r̂ᵢⱼ` if non-empty, else
`Σ_{repulsion≤ρ<orientation} v̂ⱼ + Σ_{orientation≤ρ<radius} r̂ᵢⱼ`;
`v̂ᵢ′ ∝ v̂ᵢ + dt turning (d̂ - (d̂⋅v̂ᵢ) v̂ᵢ)`, `|vᵢ′| = speed`.
`smoothing = w > 0`: the zone indicators become C² steps of width `w` centred on the zone boundaries (weights
`ωʳⱼ`, `ωᵒⱼ`, `ωᵃⱼ` of the repulsion, orientation and attraction zones, summing to 1 inside `radius - w`), the
cutoff a step of width `w` inside the radius, the priority of repulsion a continuous weight,
`d = -Σ ωʳⱼ r̂ᵢⱼ + (1 - min(1, Σ ωʳⱼ)) (Σ ωᵒⱼ v̂ⱼ + Σ ωᵃⱼ r̂ᵢⱼ)`, and `d̂ = d / √(w² + |d|²)`.
"""
Base.@kwdef struct Zonal <: InteractionRule
    radius::Float64 = 5.0
    repulsion::Float64 = 1.0
    orientation::Float64 = 3.0
    turning::Float64 = 2.0
    speed::Float64 = 1.0
    smoothing::Float64 = 0.0
end

function velocity_update(r::Zonal, v, neighbours, dt)
    v̂ = v / norm(v)
    w = r.smoothing
    away, d, nrep = zero(v), zero(v), 0.0
    for (dx, vj) in neighbours
        ρ = norm(dx)
        if w == 0
            if ρ < r.repulsion
                away -= dx / ρ
                nrep += 1
            elseif ρ < r.orientation
                d += vj / norm(vj)
            else
                d += dx / ρ
            end
        else
            S1 = smoothstep(ρ - r.repulsion + w / 2, w)     # 0 inside the repulsion zone, 1 outside
            S3 = smoothstep(ρ - r.orientation + w / 2, w)   # 0 inside the orientation zone, 1 outside
            ωr = 1 - S1
            ωo = S1 * (1 - S3)
            ωa = S3 * smoothstep(r.radius - ρ, w)
            away -= ωr * dx / ρ
            nrep += ωr
            d += ωo * vj / norm(vj) + ωa * dx / ρ
        end
    end
    d = w == 0 ? (nrep > 0 ? away : d) : away + (1 - min(1.0, nrep)) * d
    norm(d) < 1.0e-12 && return r.speed * v̂
    d̂ = d / sqrt(w^2 + norm(d)^2)
    w = v̂ + dt * r.turning * (d̂ - dot(d̂, v̂) * v̂)
    return r.speed * w / norm(w)
end
