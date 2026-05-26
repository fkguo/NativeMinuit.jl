"""
    quadgauss(f, x::T, w::T) where {T<:Vector{Float64}}

Integration of function `f` using the Gaussian quadratures `x` with weights `w`.
`x` and `w` can be generated using, e.g., `gauss(N, a, b)` in the package `QuadGK`.
Using `quadgk` directly from that package causes memory allocation.
However, if the integration region `[a, b]` is fixed, this function does not lead to any allocation and thus is much faster.
"""
function quadgauss(f, x::T, w::T) where {T<:Vector{Float64}}
    res = zero(f(x[1]))  # zero of the same type as f(x[1]), to avoid type instability
    for i in eachindex(x)
        res += f(x[i]) * w[i]
    end
    return res
end

const x10, w10 = gauss(10, -1.0, 1.0);
const x20, w20 = gauss(20, -1.0, 1.0);
const x32, w32 = gauss(32, -1.0, 1.0);
