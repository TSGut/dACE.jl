# dACE.jl

Directional atomic cluster expansion (dACE) learns rotation-equivariant,
body-ordered acceleration rules from collective trajectory data. Each agent's
acceleration is expanded in pure (self-interaction-free) basis functions of its
own velocity and its neighbours' relative positions and velocities. Contributions
from each body order can be read off the fitted model.

The 2D basis is equivariant under joint rotations. In 3D, the model uses the
agent's heading frame, or a fixed vertical frame for systems with gravity.

## Installation

Requires Julia 1.12 or later. The package is not yet registered. Install it
directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/TSGut/dACE.jl")
```

For experiment scripts and tracking-data setup, see [Reproducing the paper](reproducing.md).

## Two dimensions

Build a local environment for each agent from positions and velocities at one
time step, then evaluate a dACE basis for one agent:

```@example two_d
using dACE

positions = [0.0 1.0 -1.0; 0.0 0.5 0.0]
velocities = [1.0 0.8 0.9; 0.0 0.1 -0.1]
envs = environments(positions, velocities; cutoff = 2.0)

basis = dACEBasis(maxdeg = (2, 3, 3), cutoff = 2.0)
features = evaluate(basis, envs[1])
nothing # hide
```

For repeated evaluation, reuse a workspace and output buffer:

```@example two_d
workspace = EvaluationWorkspace(basis)
features = zeros(ComplexF64, length(basis))
evaluate!(features, basis, envs[1], workspace)
nothing # hide
```

A workspace belongs to one basis and should not be shared by concurrent tasks.
Batch prediction and fitting manage their own reusable workspaces.
Environment construction uses `NeighbourLists.jl` cell lists, retaining one
minimum-image displacement per neighbour and the strict `distance < cutoff`
condition in both 2D and 3D.

## Three dimensions

In 3D a model has two parts, each a `FrameBasis` in a frame `(e₁, e₂, e₃)` of the
agent: `plane` for the acceleration in the plane across `e₃` and `axis` for the part
along it. `heading_bases` uses the heading frame `e₃ = v̂`, `vertical_bases` the fixed
frame `e₃ = ẑ`, for systems with a preferred vertical such as motion under gravity:

```@example three_d
using dACE

positions = [0.0 1.0 -1.0; 0.0 0.5 0.0; 0.0 0.0 0.5]
velocities = [1.0 0.8 0.9; 0.0 0.1 -0.1; 0.0 0.0 0.1]
envs = environments(positions, velocities; cutoff = 2.0)

plane, axis = heading_bases(maxdeg = (2, 3), cutoff = 2.0)
features = (evaluate(plane, envs[1]), evaluate(axis, envs[1]))
nothing # hide
```

The degree keywords per neighbour, in 2D and 3D:

| keyword | `dACEBasis` (2D) | `FrameBasis` (3D) |
| --- | --- | --- |
| `maxdeg[n]` | total degree of the `n`-body terms | the same |
| `maxm` | angular indices of the neighbour's position and velocity | the same, in the plane |
| `maxl` | degree in the neighbour's relative speed | the same, in the plane |
| `maxa` | — | degree in the neighbour's direction along the axis |
| `maxb` | — | degree in its relative velocity along the axis |

## Fit from trajectories

For trajectory arrays with dimensions `(dimension, agent, time)`, derive
environments and acceleration targets, then fit:

```julia
envs, accs = trajectory_data(positions2d, velocities2d, dt; cutoff = 2.0)
model = fit(basis, envs, accs; reflections = true)
acceleration = predict(model, envs[1])
contributions = body_order_contributions(model, envs[1])   # [1]: 1-body, [2]: 2-body, …

envs3, accs3 = trajectory_data(positions3d, velocities3d, dt; cutoff = 2.0)
model3d = fit((plane, axis), envs3, accs3; reflections = true)
```

`fit` minimises the squared error plus a ridge penalty `λ² Σₖ γₖ |c̃ₖ|²` on the
coefficients of the column-normalised basis.

- **Prior.** The weights `γ` default to the algebraic smoothness prior of ACE,
  `smoothness_prior(basis)`. `prior = nothing` gives equal weights.
- **Regularisation.** By default `λ` maximises the evidence (marginal likelihood) of
  the fit, as Bayesian ridge regression does in ACE, from the same SVD that gives the
  fit, so it needs no validation data and no grid. A number, `fit(...; λ = 1e-3)`,
  fixes it. `model.λ` is the value used.
- **Reflections.** `reflections = true` restricts the coefficients to real values,
  which makes the model equivariant under reflections as well (in 3D, through planes
  containing the axis). Most collective-motion rules are. The default `false` allows
  chiral rules.
- **Solver.** By default `fit` streams the design through a QR factorisation, one
  chunk of environments at a time, so memory depends on the basis size only and the
  condition number is that of the design, not its square. `method = :normal_equations`
  accumulates the normal equations instead, which is cheaper per environment but loses
  accuracy at weak regularisation and needs a numeric `λ`, since the evidence needs the
  QR factor.
- **Paths.** `fit_path` returns the fits for a list of `λ` from a single pass over the
  data.

### Choosing λ on held-out data instead

`design_statistics` streams one data set (a video, a trial, a group) into QR
statistics once. Statistics pool with `+`, so cross-validation needs no second pass
over the data. `select_lambda` picks the regularisation on validation statistics,
and `fit` accepts statistics in place of data:

```julia
S = Dict(g => design_statistics(basis, envs[g], accs[g]) for g in groups)
train, validation = groups[1:end-1], groups[end:end]
choice = select_lambda(sum(S[g] for g in train), sum(S[g] for g in validation), basis;
                       λs = 10.0 .^ (-8:0.5:0), reflections = true)
model = fit(basis, sum(S[g] for g in groups); choice..., reflections = true)
```

`choice` holds `λ` and, for a basis with a self term, `λself` (see below). For the
3D pair `(plane, axis)`, the statistics, the choice and the fit come per part.

### Choosing the degree on held-out data

The total degrees `maxdeg` are chosen the same way. `select_basis` takes candidate
bases, chooses `λ` for each on the validation data and returns the one with the
lowest validation error. A higher degree resolves a smooth law better when the data
constrain it and overfits when they do not. Passing a single basis fixes the degree.

```julia
ladder = [dACEBasis(maxdeg = d, maxl = 0, maxm = 1) for d in ((4, 8, 6), (4, 10, 8), (6, 12, 10))]
choice = select_basis(ladder, envs[train], accs[train], envs[val], accs[val]; λs = 10.0 .^ (-8:0.5:0))
model = fit(choice.basis, envs, accs; λ = choice.λ, λself = choice.λself)
```

`choice.errors` holds the validation error of every candidate.

## Memory, speed maps and normalisation

Animal trajectories often need more than positions and velocities. All options
keep the basis exactly equivariant and are available in 2D and 3D:

- `history = History(...)`: which past velocities each part of the model sees, in
  sampling steps (one step is the environments' `dt`, so `1:4` means 1 to 4 steps
  ago). Anything left out sees no past, and the current velocities are always used.

  | part of the model | keyword | sees |
  | --- | --- | --- |
  | self term | `selfterm = 1:4` | the agent's own velocities at those steps |
  | reactions to neighbours | `interactions = (own = 1,)` | also the agent's own state one step ago |
  | reactions to neighbours | `interactions = (neighbours = 1,)` | also each neighbour's state one step ago |

  - `selfterm`: a temporal body-order expansion of the agent's own past in the frame
    of its current heading (functions of one past state and products of two), part
    of the 1-body term, sized by `selfterm_degree` and `selfterm_maxm`. It acts
    whether or not there are neighbours. Animals and drones carry much of their
    behaviour here (speed regulation, burst-and-coast swimming, turning bouts),
    which atoms, without self-propulsion, do not need.
  - `own`: the reaction to a neighbour may depend on what the agent was just doing
    (speeding up, slowing down, turning). It acts only through neighbours, and in
    data where agents nearly always have neighbours it is hard to separate from the
    self term.
  - `use = :velocity` (default), `:acceleration` or `:both`: what a past state is,
    for all parts. Past velocities are scaled by the basis `vscale`, past
    accelerations by the basis `ascale`.

  Environments carry past velocities only, from frames up to the current one, so a
  past acceleration is a difference of velocities and shares no position with a
  finite-difference target ahead of `t` (a past acceleration computed from the
  previous target would leak tracking noise into the prediction). Fill them with
  `trajectory_data(...; history = basis.history)`, or for one snapshot with
  `environments(...; history, past_velocities, dt)`. `CompositeBasis(interaction,
  selfterm)` combines an interaction basis with another self model.
- `inputmap = :tanh`: velocities and accelerations are mapped to
  `tanh(|x| / scale) x / |x|` before the polynomials, which resolves heavy-tailed
  speeds (bursts and glides) and keeps tracking glitches bounded. The default is the
  linear `x / scale`.
- `normaliser = (c, p)`: the 2-body and higher terms are divided by `(c + S₀)^p`,
  `S₀ = Σⱼ fc(ρⱼ)`, a fixed normalisation by crowding. Use it to test whether a
  response is normalised (averaged over neighbours), not as a default: it helps on
  rules that average over neighbours but biases rules that sum over them, most for
  many-body terms. Compare fits with and without it on held-out data.
- `inner_envelope = q`: radial functions vanish at distance 0 like `ρ^q`.

```@example options
using dACE

positions = [0.0 1.0 -1.0; 0.0 0.5 0.0]
velocities = [1.0 0.8 0.9; 0.0 0.1 -0.1]
past_velocities = [[0.9 0.8 1.0; 0.1 0.0 -0.2], [0.8 0.9 1.0; 0.2 0.1 -0.2]]   # one and two steps earlier

history = History(selfterm = 1:2, interactions = (neighbours = 1,), use = :both)
basis = dACEBasis(maxdeg = (4, 4), cutoff = 2.0, history = history, ascale = 2.0, inputmap = :tanh)
envs = environments(positions, velocities; cutoff = 2.0, history, past_velocities, dt = 0.1)
features = evaluate(basis, envs[1])
nothing # hide
```

### Fitting a basis with a self term

A self term with velocity history and the interaction terms are not identified
against each other: the self term can absorb part of the interaction signal, which
raises the one-step prediction while the interactions get worse. `fit` therefore
takes `self_fit`:

| `self_fit` | what is fitted | use |
| --- | --- | --- |
| `:staged` (default) | the interaction functions alone (regularisation `λ`), then the self term on their residual (`λself`, default `λ`), then the `own` terms on what is left (`λ`) | interaction laws: the interactions are those of the fit without any past |
| `:self_first` | the self term alone (`λself`), then the interactions, then the `own` terms (`λ`) | for comparison, since the self term takes whatever interaction signal it can, so expect worse interactions |
| `:joint` | all columns in one least-squares problem, as ACE fits a potential | when only the prediction matters, or the self term is known not to absorb interactions |

`self_fit` applies to `dACEBasis`, the 3D bases and `CompositeBasis` (whose
`interaction` part goes first), to fits from data or from statistics, with
`method = :qr` or `:normal_equations`, and in `fit_path` (`λself`, default the same
`λ` as the interactions). `select_lambda` chooses `λ` and `λself` stage by stage.
Fitting the `own` terms last protects the plain interactions but recovers an
own-state-dependent response only partly when the two overlap. `:joint` recovers it
fully, at the cost of that protection. A basis with only one kind of column fits the
same either way. An ACEfit `solver` fits jointly only and needs `self_fit = :joint`.

```@example options
accs = [[0.1, 0.0], [0.0, -0.1], [0.05, 0.05]]
model = fit(basis, envs, accs; λ = 1e-3, λself = 1e-4)                    # staged
model_joint = fit(basis, envs, accs; λ = 1e-3, self_fit = :joint)         # ACE-like
```

## Rollouts

`rollout(model, pos0, vel0, dt, nsteps; past_velocities)` integrates a fitted 2D or
3D model forward with explicit Euler steps. Models with a `History` carry their past
velocities along (`dt` must be the sampling step of the training data).

## AR(1) diagnostic

`fit_ar1(basis, envs, accs, lagged_envs, lagged_accs)` fits the force law jointly with
persistent individual noise `η(t) = ρ η(t - Δ) + ξ` and returns an `ARModel` (2D
only, with the lagged environments and accelerations one step earlier from
`trajectory_data(...; lagged = true)`). The fitted `ρ` shows whether residuals persist. This is currentl mainly intended for diagnostic use. On tracked positions the previous
finite-difference acceleration often shares frames with the target, so the one-step
prediction `predict(model, env, lagged_env, lagged_acc)` partly predicts tracking
error, and as `ρ → 1` the force law is no longer identified. Evaluate the force law,
`predict(model.model, env)`.

See the [API reference](api.md) for all exported functions and options.
