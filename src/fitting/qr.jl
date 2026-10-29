# streamed QR least squares: the triangular factor R of the augmented design [B t] (or, for the
# AR(1) noise model, [B Bₚ t tₚ]) is updated block by block of rows, so that ‖Bc - t‖² = ‖R [c; -1]‖²
# for every c without forming BᴴB, whose condition number is the square of that of B

using LinearAlgebra: BlasInt

using LinearAlgebra.BLAS: @blasfunc

# LAPACK xTPQRT with a square upper-triangular A on top of a full block B (L = 0): A ← R of [A; B]
for (fn, T) in ((:dtpqrt_, Float64), (:ztpqrt_, ComplexF64))
    @eval function tpqrt!(A::StridedMatrix{$T}, B::StridedMatrix{$T}, Tm::Matrix{$T}, work::Vector{$T})
        n = size(A, 2)
        size(A, 1) == n && size(B, 2) == n || throw(DimensionMismatch("A must be square with as many columns as B"))
        stride(A, 1) == 1 && stride(B, 1) == 1 || throw(ArgumentError("columns must be contiguous"))
        nb = size(Tm, 1)
        (n == 0 || 1 <= nb <= n) && size(Tm, 2) >= n && length(work) >= nb * n || throw(ArgumentError("bad tpqrt buffers"))
        info = Ref{BlasInt}(0)
        ccall(
            (@blasfunc($fn), LinearAlgebra.libblastrampoline), Cvoid,
            (
                Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
            ),
            size(B, 1), n, 0, nb, A, max(1, stride(A, 2)), B, max(1, stride(B, 2)), Tm, max(1, stride(Tm, 2)), work, info,
        )
        info[] == 0 || error("LAPACK tpqrt failed with info = $(info[])")
        return A
    end
end

# reusable buffers for row-block updates of a p × p factor
struct QRUpdater{T}
    Tm::Matrix{T}
    work::Vector{T}
end

QRUpdater(T, p; nb = 64) = (b = max(1, min(nb, p)); QRUpdater(zeros(T, b, p), zeros(T, b * p)))

function update!(R, rows, u::QRUpdater)
    size(rows, 1) == 0 && return R
    # one non-finite entry would silently turn the whole factor into NaN
    all(isfinite, rows) || throw(ArgumentError("non-finite entries in the design rows (check the environments and targets for NaN or Inf)"))
    return tpqrt!(R, rows, u.Tm, u.work)
end

"""
    QRStatistics

Upper-triangular factor `R` of the augmented design, columns `[B t]` (`K + 1`) or, for the AR(1)
noise model, `[B Bₚ t tₚ]` (`2K + 2`) with the lagged design and targets, accumulated over `n`
rows (`qr_statistics`, or `add_rows!` for a design built elsewhere). Statistics of separate data
sets pool with `+`; `select_columns` keeps a subset of the design columns.
"""
mutable struct QRStatistics{T}
    R::Matrix{T}
    K::Int
    lagged::Bool
    n::Int
end

"""
    QRStatistics(T, K; lagged = false)

Empty statistics for `K` design columns (`T = Float64` or `ComplexF64`), filled by `add_rows!`.
"""
QRStatistics(T::Type, K::Integer; lagged = false) =
    (p = lagged ? 2K + 2 : K + 1; QRStatistics(zeros(T, p, p), Int(K), lagged, 0))

"""
    add_rows!(S::QRStatistics, W) -> S

Add the rows `W = [B t]` (or `[B Bₚ t tₚ]` for lagged statistics) to `S`. Not thread-safe:
callers that build rows concurrently must serialise the calls.
"""
function add_rows!(S::QRStatistics{T}, W::AbstractMatrix) where {T}
    size(W, 2) == size(S.R, 1) || throw(DimensionMismatch("rows have $(size(W, 2)) columns, expected $(size(S.R, 1))"))
    eltype(W) <: Complex && T <: Real && throw(ArgumentError("complex rows for real statistics"))
    update!(S.R, Matrix{T}(W), QRUpdater(T, size(S.R, 1)))
    S.n += size(W, 1)
    return S
end

"""
    select_columns(S::QRStatistics, cols) -> QRStatistics

Statistics of the design restricted to the columns `cols` (targets kept, and for lagged
statistics the same columns of the lagged design), re-triangularised.
"""
function select_columns(S::QRStatistics, cols)
    K = S.K
    all(c -> 1 <= c <= K, cols) || throw(ArgumentError("columns must lie in 1:$K"))
    idx = S.lagged ? [cols; K .+ cols; 2K + 1; 2K + 2] : [cols; K + 1]
    F = qr(S.R[:, idx])
    return QRStatistics(Matrix(UpperTriangular(F.R)), length(cols), S.lagged, S.n)
end

"""
    target_norm2(S::QRStatistics)

`Σₛ |tₛ|²` of the (current, not lagged) targets.
"""
target_norm2(S::QRStatistics) = sum(abs2, view(S.R, :, S.lagged ? 2S.K + 1 : S.K + 1))

"""
    qr_statistics(basis, envs, targets; past = nothing, chunksize = 2000) -> QRStatistics

Streamed QR of `[B t]`, `B[s, k] = Bₖ(envs[s])`, `t` complex targets, one chunk of rows at a time
(memory independent of the number of environments); `past = (penvs, ptargets)`: also the lagged
design and targets of the AR(1) noise model.
"""
function qr_statistics(basis, envs, targets; past = nothing, chunksize = 2000)
    check_lengths(envs, targets)
    lagged = !isnothing(past)
    lagged && (length(past[1]) == length(past[2]) == length(envs) || throw(DimensionMismatch("past differs in length")))
    chunksize > 0 || throw(ArgumentError("chunksize must be positive"))
    K = length(basis)
    p = lagged ? 2K + 2 : K + 1
    R = zeros(ComplexF64, p, p)
    W = Matrix{ComplexF64}(undef, min(chunksize, max(length(envs), 1)), p)
    u = QRUpdater(ComplexF64, p)
    works = workspace_pool(basis)
    for chunk in Iterators.partition(eachindex(envs), chunksize)
        Wc = view(W, 1:length(chunk), :)
        foreach_workspace(basis, length(chunk); works) do work, i
            s = chunk[i]
            evaluate!(view(Wc, i, 1:K), basis, envs[s], work)
            lagged && evaluate!(view(Wc, i, (K + 1):(2K)), basis, past[1][s], work)
        end
        if lagged
            Wc[:, 2K + 1] .= view(targets, chunk)
            Wc[:, 2K + 2] .= view(past[2], chunk)
        else
            Wc[:, K + 1] .= view(targets, chunk)
        end
        update!(R, Wc, u)
    end
    return QRStatistics(R, K, lagged, length(envs))
end

# statistics of the rows of a given augmented matrix (tests, external designs)
function qr_statistics(W::AbstractMatrix, K::Integer; lagged = false, chunksize = 2000)
    size(W, 2) == (lagged ? 2K + 2 : K + 1) || throw(DimensionMismatch("W has $(size(W, 2)) columns"))
    T = eltype(W) <: Real ? Float64 : ComplexF64
    R = zeros(T, size(W, 2), size(W, 2))
    u = QRUpdater(T, size(W, 2))
    for chunk in Iterators.partition(axes(W, 1), chunksize)
        update!(R, Matrix{T}(W[chunk, :]), u)
    end
    return QRStatistics(R, Int(K), lagged, size(W, 1))
end

function Base.:+(a::QRStatistics, b::QRStatistics)
    a.K == b.K && a.lagged == b.lagged || throw(ArgumentError("statistics of different designs"))
    T = promote_type(eltype(a.R), eltype(b.R))
    R = Matrix{T}(a.R)
    update!(R, Matrix{T}(b.R), QRUpdater(T, size(R, 1)))
    return QRStatistics(R, a.K, a.lagged, a.n + b.n)
end

# real coefficients (reflections): for real x, ‖Re(R)x - Re r‖² + ‖Im(R)x - Im r‖² = ‖R x - r‖², so the
# real factor is that of [Re R; Im R]
realfactor(R::AbstractMatrix{<:Real}) = Matrix{Float64}(R)

function realfactor(R::AbstractMatrix{<:Complex})
    F = qr!([real(R); imag(R)])
    return Matrix(UpperTriangular(F.R))
end

factor(S::QRStatistics, reflections) = reflections ? realfactor(S.R) : S.R

function check_plain(S::QRStatistics)
    S.lagged && throw(ArgumentError("lagged (AR(1)) statistics: use ar_solve and ar_sse"))
    return S
end

"""
    lsq_sse(S::QRStatistics, c)

`Σₛ |tₛ - Σₖ cₖ Bₖ(envₛ)|²` over the data of `S` (e.g. a validation set).
"""
lsq_sse(S::QRStatistics, c) = sum(abs2, check_plain(S).R * [c; -1])
