# SPDX-License-Identifier: LGPL-2.1-or-later

@testset "Strategy" begin
    @testset "Level 0 — Low (Phase 0 default)" begin
        s = Strategy(0)
        # C++ MnStrategy.cxx:33–44 (SetLowStrategy)
        @test s.level == 0
        @test s.grad_ncycles == 2
        @test s.grad_step_tolerance == 0.5
        @test s.grad_tolerance == 0.1
        @test s.hessian_ncycles == 3
        @test s.hessian_step_tolerance == 0.5
        @test s.hessian_g2_tolerance == 0.1
        @test s.hessian_grad_ncycles == 1
    end

    @testset "Level 1 — Medium (C++ Minuit2 default)" begin
        s = Strategy(1)
        # C++ MnStrategy.cxx:46–57 (SetMediumStrategy)
        @test s.level == 1
        @test s.grad_ncycles == 3
        @test s.grad_step_tolerance == 0.3
        @test s.grad_tolerance == 0.05
        @test s.hessian_ncycles == 5
        @test s.hessian_step_tolerance == 0.3
        @test s.hessian_g2_tolerance == 0.05
        @test s.hessian_grad_ncycles == 2
    end

    @testset "Level 2 — High" begin
        s = Strategy(2)
        # C++ MnStrategy.cxx:59–70 (SetHighStrategy)
        @test s.level == 2
        @test s.grad_ncycles == 5
        @test s.grad_step_tolerance == 0.1
        @test s.grad_tolerance == 0.02
        @test s.hessian_ncycles == 7
        @test s.hessian_step_tolerance == 0.1
        @test s.hessian_g2_tolerance == 0.02
        @test s.hessian_grad_ncycles == 6
    end

    @testset "Invalid levels" begin
        @test_throws ArgumentError Strategy(3)
        @test_throws ArgumentError Strategy(-1)
        @test_throws ArgumentError Strategy(10)
    end

    @testset "Performance hygiene" begin
        s = Strategy(0)
        @test isbits(s)  # zero-alloc when passed through call chain
        @test (@inferred Strategy(0)) isa Strategy
        @test (@inferred Strategy(1)) isa Strategy
        @test (@inferred Strategy(2)) isa Strategy
    end
end
