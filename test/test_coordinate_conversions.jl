# SPDX-License-Identifier: LGPL-2.1-or-later
using NativeMinuit, Test
# The standalone invocation also exercises the shared array-interface controls.
if !isdefined(@__MODULE__, :InterfaceShiftedVector)
    include("test_interface_boundaries.jl")
end
@testset "Direct coordinate conversion axes" begin
    p = Parameters(["a", "b"], zeros(2), ones(2))
    for operation in (x -> int_to_ext_vector(p, x),
                      x -> ext_to_int_vector(p, x),
                      x -> NativeMinuit.int_to_ext_vector!(zeros(2), p, x))
        shifted = InterfaceShiftedVector([1.0, 2.0])
        @test_throws ArgumentError operation(shifted)
        @test shifted.reads[] == 0
    end
    output = InterfaceShiftedVector([91.0, 92.0])
    @test_throws ArgumentError NativeMinuit.int_to_ext_vector!(output, p, [1.0, 2.0])
    @test output.reads[] == 0
    @test output.data == [91.0, 92.0]
    for input in ([1.0, 2.0], view([1.0, -9.0, 2.0], 1:2:3))
        @test int_to_ext_vector(p, input) == [1.0, 2.0]
        @test ext_to_int_vector(p, input) == [1.0, 2.0]
        backing = fill(-9.0, 3)
        destination = view(backing, 1:2:3)
        @test NativeMinuit.int_to_ext_vector!(destination, p, input) === destination
        @test backing == [1.0, -9.0, 2.0]
    end
end
