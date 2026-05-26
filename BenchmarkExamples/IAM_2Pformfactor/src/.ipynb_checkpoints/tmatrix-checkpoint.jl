
# zero allocation can be achieved by using SMatrix from StaticArrys

# partial-wave projections
swave(f::Function) = 0.5quadgauss(f, x10, w10)
pwave(f::Function) = 0.5quadgauss(x -> f(x) * x, x10, w10)
dwave(f::Function) = 0.5quadgauss(x -> f(x) * 0.5(3x^2-1), x10, w10);


# partial-wave amplitudes from CHPT
# IJ = 00, channels: ππ, KKbar, ηη
function St2_00!(tm, w)
    _s = w^2
    tm[1, 1] = swave(x->V2_I0_ππππ(_s, x))
    tm[2, 1] = swave(x->V2_I0_ππkk(_s, x))
    tm[3, 1] = swave(x->V2_I0_ππηη(_s, x))
    tm[1, 2] = tm[2, 1]
    tm[2, 2] = swave(x->V2_I0_kkkk(_s, x))
    tm[3, 2] = swave(x->V2_I0_kkηη(_s, x))
    tm[1, 3] = tm[3, 1]
    tm[2, 3] = tm[3, 2]
    tm[3, 3] = swave(x->V2_I0_ηηηη(_s, x))
    return SMatrix{3,3}(tm)
end
function St4_00!(tm, w, lecr)
    _s = w^2
    tm[1, 1] = swave(x->V4_I0_ππππ(_s, x, lecr))
    tm[2, 1] = swave(x->V4_I0_ππkk(_s, x, lecr))
    tm[3, 1] = swave(x->V4_I0_ππηη(_s, x, lecr))
    tm[1, 2] = tm[2, 1]
    tm[2, 2] = swave(x->V4_I0_kkkk(_s, x, lecr))
    tm[3, 2] = swave(x->V4_I0_kkηη(_s, x, lecr))
    tm[1, 3] = tm[3, 1]
    tm[2, 3] = tm[3, 2]
    tm[3, 3] = swave(x->V4_I0_ηηηη(_s, x, lecr))
    return SMatrix{3,3}(tm)
end

# IJ = 10, channels: πη, KKbar
function St2_10!(tm, w)
    _s = w^2
    tm[1, 1] = swave(x->V2_I1_πηπη(_s, x))
    tm[2, 1] = swave(x->V2_I1_πηkk(_s, x))
    tm[1, 2] = tm[2, 1]
    tm[2, 2] = swave(x->V2_I1_kkkk(_s, x))
    return SMatrix{2,2}(tm)
end
function St4_10!(tm, w, lecr)
    _s = w^2
    tm[1, 1] = swave(x->V4_I1_πηπη(_s, x, lecr))
    tm[2, 1] = swave(x->V4_I1_πηkk(_s, x, lecr))
    tm[1, 2] = tm[2, 1]
    tm[2, 2] = swave(x->V4_I1_kkkk(_s, x, lecr))
    return SMatrix{2,2}(tm)
end

# IJ = 11, channels: ππ, KKbar; note πη have negative G-parity and decouples
# this indeed happens to be zero, as a check of the code
function St2_11!(tm, w)
    _s = w^2
    tm[1, 1] = pwave(x->V2_I1_ππππ(_s, x))
    tm[2, 1] = pwave(x->V2_I1_ππkk(_s, x))
    tm[1, 2] = tm[2, 1]
    tm[2, 2] = pwave(x->V2_I1_kkkk(_s, x))
    return SMatrix{2,2}(tm)
end
function St4_11!(tm, w, lecr)
    _s = w^2
    tm[1, 1] = pwave(x->V4_I1_ππππ(_s, x, lecr))
    tm[2, 1] = pwave(x->V4_I1_ππkk(_s, x, lecr))
    tm[1, 2] = tm[2, 1]
    tm[2, 2] = pwave(x->V4_I1_kkkk(_s, x, lecr))
    return SMatrix{2,2}(tm)
end


# IJ = 20, channel: ππ
t2_20(w) = swave(x->V2_I2_ππππ(w^2, x))
t4_20(w, lecr) = swave(x->V4_I2_ππππ(w^2, x, lecr))

# IJ = 1/2 0, channels: Kπ, Kη
function St2_strange_10!(tm, w)
    _s = w^2
    tm[1, 1] = swave(x->V2_I1_kπkπ(_s, x))
    tm[2, 1] = swave(x->V2_I1_kπkη(_s, x))
    tm[1, 2] = tm[2, 1]
    tm[2, 2] = swave(x->V2_I1_kηkη(_s, x))
    return SMatrix{2,2}(tm)
end
function St4_strange_10!(tm, w, lecr)
    _s = w^2
    tm[1, 1] = swave(x->V4_I1_kπkπ(_s, x, lecr))
    tm[2, 1] = swave(x->V4_I1_kπkη(_s, x, lecr))
    tm[1, 2] = tm[2, 1]
    tm[2, 2] = swave(x->V4_I1_kηkη(_s, x, lecr))
    return SMatrix{2,2}(tm)
end

# IJ = 1/2 1, channels: Kπ, Kη
function St2_strange_11!(tm, w)
    _s = w^2
    tm[1, 1] = pwave(x->V2_I1_kπkπ(_s, x))
    tm[2, 1] = pwave(x->V2_I1_kπkη(_s, x))
    tm[1, 2] = tm[2, 1]
    tm[2, 2] = pwave(x->V2_I1_kηkη(_s, x))
    return SMatrix{2,2}(tm)
end
function St4_strange_11!(tm, w, lecr)
    _s = w^2
    tm[1, 1] = pwave(x->V4_I1_kπkπ(_s, x, lecr))
    tm[2, 1] = pwave(x->V4_I1_kπkη(_s, x, lecr))
    tm[1, 2] = tm[2, 1]
    tm[2, 2] = pwave(x->V4_I1_kηkη(_s, x, lecr))
    return SMatrix{2,2}(tm)
end


# IJ = 3/2 0, channel: Kπ
t2_strange_30(w) = swave(x->V2_I3_kπkπ(w^2, x))
t4_strange_30(w, lecr) = swave(x->V4_I3_kπkπ(w^2, x, lecr));


# amplitudes from IAM

# function Stiam_00(w, lecr)
#     s2 = St2_00!(c_00_2, w)
#     s4 = St4_00!(c_00_4, w, lecr)
#     s2 * inv(s2 - s4) * s2
# end

#=
this code-generation block generates IAM amplitudes:
Stiam_00(w, lecr), Stiam_10(w, lecr), Stiam_11(w, lecr), Stiam_strange_10(w, lecr), Stiam_strange_11(w, lecr):
=#
for str in ["00", "10", "11", "strange_10", "strange_11"]
    _fnames = Symbol("Stiam_$(str)")
    _t2 = Symbol("St2_$(str)!")
    _t4 = Symbol("St4_$(str)!")
    _c2 = Symbol("c_$(str)_2")
    _c4 = Symbol("c_$(str)_4")
    _expr = quote
        function ($_fnames)(w, lecr)
           s2 = ($_t2)(($_c2), w);
           s4 = ($_t4)(($_c4), w, lecr);
           s2 * inv(s2 - s4) * s2
        end
    end
    eval(_expr)
end

function tiam_20(w, lecr)
    _t2 = t2_20(w); _t4 = t4_20(w, lecr)
    return _t2^2/(_t2 - _t4)
end

function tiam_strange_30(w, lecr)
    _t2 = t2_strange_30(w); _t4 = t4_strange_30(w, lecr)
    return _t2^2/(_t2 - _t4)
end;


# IJ = 01, channel: KKbar
t2_01(w) = pwave(x->V2_I0_kkkk(w^2, x))
t4_01(w, lecr) = pwave(x->V4_I0_kkkk(w^2, x, lecr))

function tiam_01(w, lecr)
    _t2 = t2_01(w); _t4 = t4_01(w, lecr)
    return _t2^2/(_t2 - _t4)
end;
