# SPDX-License-Identifier: LGPL-2.1-or-later
using Test, NativeMinuit, DataFrames, ComponentArrays

struct InterfaceShiftedVector{T} <: AbstractVector{T}
    data::Vector{T}
    reads::Base.RefValue{Int}
end
InterfaceShiftedVector(v::Vector{T}) where {T} = InterfaceShiftedVector{T}(v, Ref(0))
Base.size(v::InterfaceShiftedVector) = size(v.data)
Base.axes(v::InterfaceShiftedVector) = (0:length(v.data)-1,)
Base.IndexStyle(::Type{<:InterfaceShiftedVector}) = IndexLinear()
function Base.getindex(v::InterfaceShiftedVector, i::Int)
    v.reads[] += 1
    0 <= i < length(v) || throw(BoundsError(v, i))
    return v.data[i+1]
end
function Base.setindex!(v::InterfaceShiftedVector, value, i::Int)
    0 <= i < length(v) || throw(BoundsError(v, i))
    v.data[i+1] = value
end
Base.similar(v::InterfaceShiftedVector, ::Type{T}) where {T} =
    InterfaceShiftedVector(Vector{T}(undef, length(v)))

@testset "DataFrame diagnostic names preserve parameter samples" begin
    for parameter_names in (["a", "b"], ["delta_chisq", "delta_chisq_"],
                            ["delta_chisq", "delta_chisq_", "delta_chisq__", "delta_chisq___"])
        n = length(parameter_names)
        m = Minuit(x -> sum(abs2, x), zeros(n); names=parameter_names)
        migrad!(m)
        kwargs = (; nsamples=64, seed=113)
        samples = get_contours_samples(m; kwargs...)
        original = copy(samples.samples)
        df = contour_df_samples(m; kwargs...)
        @test nrow(df) > 0
        @test ncol(df) == n + 1
        @test Matrix(df[:, 1:n]) == original
        @test samples.samples == original
        @test df[:, end] == samples.delta_chisq_values
        @test length(unique(names(df))) == n + 1
        @test names(df)[1:n] == parameter_names
        @test names(df)[end] == "delta_chisq" * repeat("_", parameter_names[1] == "a" ? 0 : n)
    end
end

@testset "One-based coordinate boundary" begin
    shifted = InterfaceShiftedVector([0.0, 0.0])
    records = [MinuitParameter("a", 0.0, 0.1), MinuitParameter("b", 0.0, 0.1)]
    @test_throws ArgumentError Parameters(records; values=shifted)
    @test_throws ArgumentError Minuit(sum, shifted)
    @test shifted.reads[] == 0
    for field in (:names, :errors, :fixed, :limits)
        values = field === :names ? ["a", "b"] : field === :errors ? [0.1, 0.1] :
                 field === :fixed ? [false, false] : [(NaN, NaN), (NaN, NaN)]
        @test_throws ArgumentError Minuit(sum, zeros(2); (field => InterfaceShiftedVector(values),)...)
    end
    for (names0, values0, errors0) in ((InterfaceShiftedVector(["a", "b"]), zeros(2), ones(2)),
                                      (["a", "b"], shifted, ones(2)),
                                      (["a", "b"], zeros(2), InterfaceShiftedVector(ones(2))))
        @test_throws ArgumentError Parameters(names0, values0, errors0)
    end
    @test_throws ArgumentError MinimumParameters(shifted, 0.0)
    @test_throws ArgumentError FunctionGradient(shifted, zeros(2), zeros(2))
    @test_throws ArgumentError migrad(sum, shifted, ones(2))
    @test_throws ArgumentError migrad(sum, zeros(2), shifted)
    valid_fit = Minuit(x -> sum(abs2, x), zeros(2))
    for field in (:values, :errors, :fixed, :limits)
        input = field === :fixed ? [false, false] : field === :limits ? [(NaN, NaN), (NaN, NaN)] : ones(2)
        @test_throws ArgumentError setproperty!(valid_fit, field, InterfaceShiftedVector(input))
    end

    cf = CostFunctionWithGradient(x -> sum(abs2, x), x -> InterfaceShiftedVector(2 .* x))
    par = MinimumParameters([1.0, 2.0], [0.1, 0.1], 5.0)
    @test_throws ArgumentError analytical_gradient(par, cf)
    for x0 in ([0, 0], range(0.0, 0.0; length=2), view(zeros(5), 1:2:3), ComponentArray(a=0.0, b=0.0))
        snapshot = copy(x0)
        m = Minuit(x -> sum(abs2, x), x0)
        migrad!(m)
        @test m.valid
        @test x0 == snapshot
    end
end
