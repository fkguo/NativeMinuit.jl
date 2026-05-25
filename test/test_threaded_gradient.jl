# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "numerical_gradient! — Phase 2.2 threaded path" begin

    @testset "Threaded result matches serial — Quad-4D" begin
        cf = CostFunction(x -> sum(abs2, x .- [1.0, 2.0, 3.0, 4.0]))
        par = MinimumParameters([0.5, 1.5, 2.5, 3.5], [0.1, 0.1, 0.1, 0.1], cf([0.5, 1.5, 2.5, 3.5]))
        prev = JuMinuit.initial_gradient(par, par.dirin, cf)
        strategy = Strategy(0)

        # Serial
        out_serial = FunctionGradient(zeros(4), zeros(4), zeros(4))
        x_work_serial = similar(par.x)
        numerical_gradient!(out_serial, x_work_serial, par, prev, cf, strategy;
                             threaded = false)

        # Threaded — same FCN, fresh state
        cf2 = CostFunction(x -> sum(abs2, x .- [1.0, 2.0, 3.0, 4.0]))
        par2 = MinimumParameters(copy(par.x), copy(par.dirin), par.fval)
        prev2 = JuMinuit.initial_gradient(par2, par2.dirin, cf2)
        out_threaded = FunctionGradient(zeros(4), zeros(4), zeros(4))
        x_work_threaded = similar(par2.x)
        numerical_gradient!(out_threaded, x_work_threaded, par2, prev2, cf2, strategy;
                             threaded = true)

        # Threaded result should match serial to bit precision (same FCN
        # evaluations, just dispatched across threads).
        for i in 1:4
            @test out_threaded.grad[i] ≈ out_serial.grad[i] atol = 1e-12
            @test out_threaded.g2[i] ≈ out_serial.g2[i] atol = 1e-12
            @test out_threaded.gstep[i] ≈ out_serial.gstep[i] atol = 1e-12
        end
    end

    @testset "Threaded falls back to serial when nthreads == 1" begin
        # If Threads.nthreads() == 1, the threaded path should still
        # work (just serially in effect). Verifies no crash on
        # single-thread systems.
        cf = CostFunction(x -> sum(abs2, x))
        par = MinimumParameters([1.0, 2.0], [0.1, 0.1], cf([1.0, 2.0]))
        prev = JuMinuit.initial_gradient(par, par.dirin, cf)
        out = FunctionGradient(zeros(2), zeros(2), zeros(2))
        # threaded=true; if nthreads==1 we skip the threaded branch
        numerical_gradient!(out, similar(par.x), par, prev, cf, Strategy(0);
                             threaded = true)
        @test out.grad[1] ≈ 2.0 atol = 1e-6
        @test out.grad[2] ≈ 4.0 atol = 1e-6
    end
end
