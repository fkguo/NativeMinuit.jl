# Bounded gradient reference

`cpp_gradient_witness.cxx` evaluates the Minuit2 initial and numerical gradients
and minimizes well-conditioned quadratic objectives with lower, upper and double
bounds. Near-boundary cases exercise recovery from negative internal curvature.
The reference uses [the pinned upstream source](../docs/UPSTREAM.md).

```sh
cmake -S tools -B tools/gradient-build -DCMAKE_BUILD_TYPE=Release
cmake --build tools/gradient-build --target cpp_gradient_witness
./tools/gradient-build/cpp_gradient_witness > bounded_gradient_reference.txt
```

Compare the output with `test/reference_data/bounded_gradient_reference.txt`
before updating the fixture. The columns are case name, internal start, initial
first derivative, initial diagonal curvature, initial step, numerical first
derivative/curvature/step, numerical first derivative/curvature/step with a forced
large previous step, final validity flag, external coordinate, function value,
EDM and external covariance. The acceptance budgets are specified in
`test/test_bounded_gradient_oracle.jl`.
