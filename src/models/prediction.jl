# prediction over many environments, errors and rollouts, for every model type

# Prediction scratch is local to a batch or rollout, never stored in a shared model.
prediction_workspace(m) = nothing

prediction_workspace(m::dACEModel) = EvaluationWorkspace(m.basis)

prediction_workspace(m::FrameModel) = (EvaluationWorkspace(m.plane), EvaluationWorkspace(m.axis))

prediction_workspace(m::ARModel) = prediction_workspace(m.model)

predict_with_workspace(m, env, work) = predict(m, env)

predict_with_workspace(m::dACEModel, env, work) = tovec(contract(evaluate!(work, m.basis, env), m.c))

predict_with_workspace(m::ARModel, env, work) = predict_with_workspace(m.model, env, work)

function predict_with_workspace(m::FrameModel, env, work)
    e1, e2, e3 = basis_frame(m.plane, env)
    ap = contract(evaluate!(work[1], m.plane, env), m.c_plane)
    az = real(contract(evaluate!(work[2], m.axis, env), m.c_axis))
    return real(ap) * e1 + imag(ap) * e2 + az * e3
end

function predict(m, envs::AbstractVector)
    work = prediction_workspace(m)
    return [predict_with_workspace(m, env, work) for env in envs]
end

"""
    rmse(model, envs, accs; relative = false)

RMS acceleration error, relative to the RMS of `accs` if `relative`.
"""
function rmse(m, envs, accs; relative = false)
    check_lengths(envs, accs)
    isempty(accs) && throw(ArgumentError("no data to score"))
    work = prediction_workspace(m)
    e = sqrt(sum(norm(predict_with_workspace(m, env, work) - a)^2 for (env, a) in zip(envs, accs)) / length(accs))
    relative || return e
    scale = sqrt(sum(a -> norm(a)^2, accs) / length(accs))
    scale > 0 || throw(ArgumentError("relative error of all-zero targets"))
    return e / scale
end

"""
    RolloutDivergence(step)

Thrown by `rollout` when the velocities become non-finite at `step`: the model has left the region where it
predicts finite accelerations (typically states unlike its training data).
"""
struct RolloutDivergence <: Exception
    step::Int
end

Base.showerror(io::IO, e::RolloutDivergence) = print(io, "RolloutDivergence: the rollout produced non-finite velocities at step ", e.step)

"""
    rollout(model, pos0, vel0, dt, nsteps; extent = nothing, periodic = false, past_velocities = nothing) -> (pos, vel)

Synchronous explicit Euler: `vᵢ += dt aᵢ`, `xᵢ += dt vᵢ`. Models with a `History` see the velocities of
the previous steps (step `dt`, so `dt` must be the sampling step of the training data); `past_velocities[k]`:
the velocities `k` steps before the start, as for `environments` (default: `vel0`, i.e. no past acceleration).
Throws a `RolloutDivergence` if the velocities become non-finite.
"""
function rollout(m, pos0, vel0, dt, nsteps; extent = nothing, periodic = false, past_velocities = nothing)
    positive_dt(dt)
    check_snapshot(pos0, vel0)
    D = size(pos0, 1)
    pos = zeros(D, size(pos0, 2), nsteps + 1)
    vel = zeros(D, size(vel0, 2), nsteps + 1)
    pos[:, :, 1] .= pos0
    vel[:, :, 1] .= vel0
    history = history_of(m)
    H = history_length(history)
    before(k) = isnothing(past_velocities) ? vel0 : past_velocities[k]   # k ≥ 1 steps before the start
    work = prediction_workspace(m)
    for t in 1:nsteps
        envs = environments(view(pos, :, :, t), view(vel, :, :, t); cutoff = cutoff(m), extent, periodic, history,
            past_velocities = [t - k >= 1 ? view(vel, :, :, t - k) : before(k - t + 1) for k in 1:H], dt)
        for (i, env) in enumerate(envs)
            a = predict_with_workspace(m, env, work)
            for k in 1:D
                vel[k, i, t + 1] = vel[k, i, t] + dt * a[k]
                x = pos[k, i, t] + dt * vel[k, i, t + 1]
                pos[k, i, t + 1] = periodic ? mod(x, extent[k]) : x
            end
        end
        all(isfinite, view(vel, :, :, t + 1)) || throw(RolloutDivergence(t))
    end
    return pos, vel
end
