# modifed NLO amplitudes to remove the unitarity violating LHC imaginary part.

function V4_I0_kkkk_nonj(s, x, lecr)
    t, u = tu(KK, KK, s, x)
    _jbar(z,m1,m2) = 0; _jbbar(z,m1,m2) = 0
    return ((mueta * (14 * mk^2 - mpi^2 - ( - 2 * mpi^2 + 9 * s)^2 / (12.0 * meta^2) + (8 * mk^2 * (2 * mk^2 + 3 * s - 6 * t) - 9 * (s^2 - 2 * t^2)) / (2.0 * ( - meta^2 + mpi^2)) + 9 * (s - u))) / (12.0 * fpi^2) + ( - 1648 * mk^4 + 186 * s * t + 1032 * mk^2 * u - 177 * u^2) / (2304.0 * fpi^4 * pi^2) - (muk * ( - 8 * mk^4 + 5 * (s^2 + s * t + t^2) - 13 * mk^2 * u + 6 * u^2)) / (6.0 * fpi^2 * mk^2) - (muk * (36 * mk^4 + 5 * s^2 - s * u + 8 * u^2 - 2 * mk^2 * (s + 16 * u))) / (12.0 * fpi^2 * mk^2) + (mupi * (5 * ( - 2 * mk^2 + u) - ( - 32 * mk^4 + 11 * s^2 + 8 * s * t + 11 * t^2 + 8 * mk^2 * u) / (24.0 * mpi^2) + ( - 64 * mk^4 + 9 * (s^2 + t^2) + 24 * mk^2 * u) / (16.0 * (mk^2 - mpi^2)))) / (2.0 * fpi^2) + (mueta * (64 * mk^2 - 2 * mpi^2 - (8 * mpi^4 - 36 * mpi^2 * (s + t) + 81 * (s^2 + t^2)) / (12.0 * meta^2) - 27 * u + ( - 64 * mk^4 + 9 * (s^2 + t^2) + 24 * mk^2 * u) / (2.0 * ( - meta^2 + mpi^2)))) / (12.0 * fpi^2) - ( - 4 * mk^2 * ( - 80 * mk^2 + 30 * s + 57 * u) + 3 * (31 * s^2 + 4 * s * u + 16 * u^2)) / (2304.0 * fpi^4 * pi^2) + (mupi * ((16 * mk^4 + 24 * mk^2 * (s - 2 * t) - 9 * (s^2 - 2 * t^2)) / (8.0 * (mk^2 - mpi^2)) + 5 * ( - 2 * mk^2 + u) - (11 * s^2 + 4 * t^2 - 8 * mk^2 * (s + 2 * t) + 4 * s * (2 * t + u)) / (12.0 * mpi^2))) / (4.0 * fpi^2)
    + (( - 6 * meta^2 - 2 * mpi^2 + 9 * s)^2 * jbar(s, meta, meta)) / (288.0 * fpi^4) + ((4 * mk^2 * (2 * mk^2 - t) + s * (s - u)) * jbar(s, mk, mk)) / (6.0 * fpi^4) - (( - 3 * meta^2 - 8 * mk^2 - mpi^2 + 9 * s)^2 * jbar(s, mpi, meta)) / (432.0 * fpi^4) + ((s * (8 * mk^2 + 7 * s - 4 * t) + 8 * mpi^2 * ( - 4 * mk^2 + s + 2 * t)) * jbar(s, mpi, mpi)) / (96.0 * fpi^4) + (( - 6 * meta^2 - 2 * mpi^2 + 9 * s)^2 * jbar(s, meta, meta) + 60 * ( - 8 * mk^4 + s * (2 * s + t) + 4 * mk^2 * u) * jbar(s, mk, mk) + (2 * ( - 3 * meta^2 - 8 * mk^2 - mpi^2 + 9 * s)^2 * jbar(s, mpi, meta)) / 3.0 + 3 * ( - 8 * mpi^2 * ( - 4 * mk^2 + s + 2 * t) + s * ( - 8 * mk^2 + 11 * s + 4 * t)) * jbar(s, mpi, mpi)) / (288.0 * fpi^4)
    + (( - 8 * mk^4 + t * (s + 2 * t) + 4 * mk^2 * u) * jbar(t, mk, mk)) / (24.0 * fpi^4)
    + (( - 3 * meta^2 - 8 * mk^2 - mpi^2 + 9 * t)^2 * _jbar(t, mpi, meta)) / (216.0 * fpi^4)
    + (( - 4 * mpi^2 + t) * ( - 4 * mk^2 + 2 * s + t) * _jbar(t, mpi, mpi)) / (24.0 * fpi^4)
    + (( - 6 * meta^2 - 2 * mpi^2 + 9 * t)^2 * jbar(t, meta, meta)
        + 60 * ( - 8 * mk^4 + t * (s + 2 * t) + 4 * mk^2 * u) * jbar(t, mk, mk)
        + (2 * ( - 3 * meta^2 - 8 * mk^2 - mpi^2 + 9 * t)^2 * _jbar(t, mpi, meta)) / 3.0 + 3 * ( - 8 * mpi^2 * ( - 4 * mk^2 + 2 * s + t)
            + t * ( - 8 * mk^2 + 4 * s + 11 * t)) * _jbar(t, mpi, mpi)) / (288.0 * fpi^4)
    + (3 * ( - 2 * mk^2 + u)^2 * jbar(u, mk, mk)) / (4.0 * fpi^4) + (2 * (2 * ( - 2 * mk^2 + u)^2 * lecr[2] + ( - 2 * mk^2 + s)^2 * (4 * lecr[1] + lecr[3]) + ( - 2 * mk^2 + t)^2 * (2 * lecr[2] + lecr[3]))) / fpi^4 + (4 * (2 * ( - 2 * mk^2 + u)^2 * lecr[2] + (( - 2 * mk^2 + s)^2 + ( - 2 * mk^2 + t)^2) * (2 * lecr[1] + lecr[2] + lecr[3]) - 4 * mk^2 * u * lecr[4] - 2 * mpi^2 * ( - 2 * mk^2 + u) * lecr[5] - 4 * mk^4 * (lecr[5] - 2 * (2 * lecr[6] + lecr[8])))) / fpi^4 + (4 * (4 * mk^2 * s * lecr[4] - mpi^2 * ( - 2 * mk^2 + u) * lecr[5] - 2 * mk^4 * (4 * lecr[4] + lecr[5] - 2 * (2 * lecr[6] + lecr[8])))) / fpi^4
    )
end

V4_I0_kkkk_j(s, x, lecr) = V4_I0_kkkk(s, x, lecr) - V4_I0_kkkk_nonj(s, x, lecr)

# removing also KKbar t and u-channel leds to a worse fit.
function V4_I0_ηηηη_nonj(s, x, lecr)
    t, u = tu(ηη, ηη, s, x)
    _jbar(z,m1,m2) = 0; _jbbar(z,m1,m2) = 0
    return ( - ((24 * meta^4 - 8 * meta^2 * mpi^2 + mpi^4) * mueta) / (3.0 * fpi^2 * meta^2) + (( - 48 * meta^2 + 7 * mpi^2) * mupi) / (9.0 * fpi^2) - (16 * (23 * meta^4 - 22 * meta^2 * mk^2 + 10 * mk^4) + 27 * ( - 4 * meta^2 * t + t^2 - s * u)) / (192.0 * fpi^4 * pi^2) - (muk * (378 * meta^4 - 48 * meta^2 * mpi^2 + 14 * mpi^4 + 81 * ( - 4 * meta^2 * t + t^2 - s * u))) / (18.0 * fpi^2 * mk^2)
    + (((16 * mk^2 - 7 * mpi^2)^2 * jbar(s, meta, meta)) / 27.0 + (( - 6 * meta^2 - 2 * mpi^2 + 9 * s)^2 * jbar(s, mk, mk)) / 12.0 + mpi^4 * jbar(s, mpi, mpi)) / (6.0 * fpi^4) + (((16 * mk^2 - 7 * mpi^2)^2 * jbar(t, meta, meta)) / 27.0
    + (( - 6 * meta^2 - 2 * mpi^2 + 9 * t)^2 * _jbar(t, mk, mk)) / 12.0
            + mpi^4 * _jbar(t, mpi, mpi)) / (6.0 * fpi^4) + (((16 * mk^2 - 7 * mpi^2)^2 * jbar(u, meta, meta)) / 27.0
            + (( - 6 * meta^2 - 2 * mpi^2 + 9 * u)^2 * _jbar(u, mk, mk)) / 12.0
            + mpi^4 * _jbar(u, mpi, mpi)) / (6.0 * fpi^4) + (4 * ( - 4 * meta^4 + s^2 + t^2 + u^2) * (2 * lecr[1] + 2 * lecr[2] + lecr[3])) / fpi^4 - (8 * (12 * meta^4 * lecr[4] + (13 * meta^4 - 10 * meta^2 * mpi^2 + 3 * mpi^4) * lecr[5] - 36 * meta^4 * lecr[6] - 24 * (2 * meta^4 - 3 * meta^2 * mpi^2 + mpi^4) * lecr[7]
            - 6 * (7 * meta^4 - 6 * meta^2 * mpi^2 + 2 * mpi^4) * lecr[8])) / (3.0 * fpi^4)) / 2.0
end
V4_I0_ηηηη_j(s, x, lecr) = V4_I0_ηηηη(s, x, lecr) - V4_I0_ηηηη_nonj(s, x, lecr)

function V4_I0_kkηη_nonj(s, x, lecr)
    t, u = tu(KK, ηη, s, x)
    _jbar(z,m1,m2) = 0; _jbbar(z,m1,m2) = 0
    return ((mupi * ( - 84 * mk^4 + 2 * mk^2 * (26 * meta^2 + 69 * s) + 3 * (16 * meta^4 - 50 * meta^2 * s + (t - u)^2))) / (48.0 * fpi^2 * ( - meta^2 + mk^2)) - (mueta * (144 * meta^6 - 128 * mk^6 + 144 * mk^4 * s + 2 * meta^4 * ( - 166 * mk^2 + 153 * s) + meta^2 * (428 * mk^4 - 486 * mk^2 * s + 27 * (t - u)^2))) / (144.0 * fpi^2 * meta^2 * ( - meta^2 + mk^2)) + (muk * (92 * mk^6 - 81 * meta^2 * s^2 - 60 * mk^4 * (meta^2 + 3 * s) + 18 * mk^2 * (8 * meta^4 + 6 * meta^2 * s + 5 * s^2 - 2 * t * u))) / (72.0 * fpi^2 * mk^2 * ( - meta^2 + mk^2)) - (116 * mk^4 + mk^2 * (184 * meta^2 - 153 * s) - 9 * (4 * meta^4 - 3 * meta^2 * s + 10 * s^2 + 2 * t * u)) / (2304.0 * fpi^4 * pi^2) - ((16 * mk^2 - 7 * mpi^2) * ( - 6 * meta^2 - 2 * mpi^2 + 9 * s) * jbar(s, meta, meta)) / (216.0 * fpi^4) - (s * ( - 6 * meta^2 - 2 * mpi^2 + 9 * s) * jbar(s, mk, mk)) / (16.0 * fpi^4) - (mpi^2 * s * jbar(s, mpi, mpi)) / (8.0 * fpi^4)
    - ((( - meta^2 + mk^2)^2 * ( - 6 * meta^2 - 18 * mk^2 + 4 * mpi^2 - 3 * u) * jbar(t, mk, meta)) / t + ((45 * meta^4 + 189 * mk^4 + 8 * mpi^4 + 12 * mpi^2 * ( - 2 * meta^2 + 3 * t)
    - 18 * mk^2 * (9 * meta^2 + 4 * mpi^2 + 6 * t - 3 * u) + 27 * t * (t - u) + 54 * meta^2 * u) * jbar(t, mk, meta)) / 9.0 + ((18 * meta^4 + 29 * mk^4 + 11 * mpi^4 + 27 * t * (t - u) + 6 * mpi^2 * (8 * meta^2 - 6 * t + 9 * u)
    + 2 * mk^2 * ( - 78 * meta^2 - 47 * mpi^2 + 18 * t + 27 * u)) * _jbar(t, mpi, mk)) / 9.0 - ((2 * mk^6 - 2 * mk^2 * mpi^2 * (4 * meta^2 + 5 * mpi^2 + 3 * u)
    + mk^4 * ( - 8 * meta^2 + 14 * mpi^2 + 3 * u) + mpi^2 * (6 * meta^4 + mpi^2 * (4 * meta^2 + 3 * u))) * _jbar(t, mpi, mk)) / t
    + (6 * ( - meta^2 + mk^2)^2 * (( - meta^2 + mk^2)^2 * jbbar(t, mk, meta) + (mk^2 - mpi^2)^2 * _jbbar(t, mpi, mk))) / t^2) / (32.0 * fpi^4)
    - ((( - meta^2 + mk^2)^2 * ( - 6 * meta^2 - 18 * mk^2 + 4 * mpi^2 - 3 * t) * jbar(u, mk, meta)) / u + ((45 * meta^4 + 189 * mk^4 + 8 * mpi^4 + 54 * meta^2 * t + 27 * u * ( - t + u) + 12 * mpi^2 * ( - 2 * meta^2 + 3 * u)
    - 18 * mk^2 * (9 * meta^2 + 4 * mpi^2 - 3 * t + 6 * u)) * jbar(u, mk, meta)) / 9.0 - ((2 * mk^6 - 2 * mk^2 * mpi^2 * (4 * meta^2 + 5 * mpi^2 + 3 * t) + mk^4 * ( - 8 * meta^2 + 14 * mpi^2 + 3 * t)
    + mpi^2 * (6 * meta^4 + mpi^2 * (4 * meta^2 + 3 * t))) * _jbar(u, mpi, mk)) / u + ((18 * meta^4 + 29 * mk^4 + 11 * mpi^4 + 6 * mpi^2 * (8 * meta^2 + 9 * t - 6 * u) + 27 * u * ( - t + u)
    + 2 * mk^2 * ( - 78 * meta^2 - 47 * mpi^2 + 27 * t + 18 * u)) * _jbar(u, mpi, mk)) / 9.0
    + (6 * ( - meta^2 + mk^2)^2 * (( - meta^2 + mk^2)^2 * jbbar(u, mk, meta) + (mk^2 - mpi^2)^2 * _jbbar(u, mpi, mk))) / u^2) / (32.0 * fpi^4) - ((( - meta^2 - mk^2 + t)^2 + ( - meta^2 - mk^2 + u)^2) * (12 * lecr[2] + lecr[3]) + 2 * (2 * meta^2 - s) * (2 * mk^2 - s) * (12 * lecr[1] + 5 * lecr[3])) / (3.0 * fpi^4) + (2 * (12 * meta^4 + 3 * mpi^4 + mpi^2 * (5 * meta^2 - 9 * s)) * lecr[5]) / (3.0 * fpi^4) - (4 * (2 * (meta^2 + mk^2) * s * lecr[4] + 8 * meta^2 * mk^2 * ( - lecr[4] + lecr[6]) + 2 * (3 * meta^4 - 4 * meta^2 * mpi^2 + mpi^4) * lecr[7] + (6 * meta^4 - 3 * meta^2 * mpi^2 + mpi^4) * lecr[8])) / fpi^4
    )
end
V4_I0_kkηη_j(s, x, lecr) = V4_I0_kkηη(s, x, lecr) - V4_I0_kkηη_nonj(s, x, lecr)


function St4_00!(tm, w, lecr)
    _s = w^2
    tm[1, 1] = swave(x->V4_I0_ππππ(_s, x, lecr))
    tm[2, 1] = swave(x->V4_I0_ππkk(_s, x, lecr))
    tm[3, 1] = swave(x->V4_I0_ππηη(_s, x, lecr))
    tm[1, 2] = tm[2, 1]
#     tm[2, 2] = swave(x->V4_I0_kkkk(_s, x, lecr))
    if w >= 2*KK.m1
        tm[2, 2] = swave(x->V4_I0_kkkk(_s, x, lecr))
        tm[3, 2] = swave(x->V4_I0_kkηη(_s, x, lecr))
    else
        tm[2, 2] = swave(x->V4_I0_kkkk_nonj(_s, x, lecr) ) + real( swave(x->V4_I0_kkkk_j(_s, x, lecr)) )
        tm[3, 2] = swave(x->V4_I0_kkηη_nonj(_s, x, lecr) )  + real( swave(x->V4_I0_kkηη_j(_s, x, lecr)) )
    end
    tm[1, 3] = tm[3, 1]
    tm[2, 3] = tm[3, 2]
#     tm[3, 3] = swave(x->V4_I0_ηηηη(_s, x, lecr))
    if w >= 2*ηη.m1
        tm[3, 3] = swave(x->V4_I0_ηηηη(_s, x, lecr))
    else
        tm[3, 3] = swave(x->V4_I0_ηηηη_nonj(_s, x, lecr) ) + real( swave(x->V4_I0_ηηηη_j(_s, x, lecr)) )
    end
    return SMatrix{3,3}(tm)
end



function V4_I1_kkkk_nonj(s, x, lecr)
    t, u = tu(KK, KK, s, x)
    _jbar(z,m1,m2) = 0; _jbbar(z,m1,m2) = 0
    return (- (mueta * (14 * mk^2 - mpi^2 - ( - 2 * mpi^2 + 9 * s)^2 / (12.0 * meta^2) + (8 * mk^2 * (2 * mk^2 + 3 * s - 6 * t) - 9 * (s^2 - 2 * t^2)) / (2.0 * ( - meta^2 + mpi^2)) + 9 * (s - u))) / (12.0 * fpi^2) + ( - 1648 * mk^4 + 186 * s * t + 1032 * mk^2 * u - 177 * u^2) / (2304.0 * fpi^4 * pi^2) - (muk * ( - 8 * mk^4 + 5 * (s^2 + s * t + t^2) - 13 * mk^2 * u + 6 * u^2)) / (6.0 * fpi^2 * mk^2) + (muk * (36 * mk^4 + 5 * s^2 - s * u + 8 * u^2 - 2 * mk^2 * (s + 16 * u))) / (12.0 * fpi^2 * mk^2) + (mupi * (5 * ( - 2 * mk^2 + u) - ( - 32 * mk^4 + 11 * s^2 + 8 * s * t + 11 * t^2 + 8 * mk^2 * u) / (24.0 * mpi^2) + ( - 64 * mk^4 + 9 * (s^2 + t^2) + 24 * mk^2 * u) / (16.0 * (mk^2 - mpi^2)))) / (2.0 * fpi^2) + (mueta * (64 * mk^2 - 2 * mpi^2 - (8 * mpi^4 - 36 * mpi^2 * (s + t) + 81 * (s^2 + t^2)) / (12.0 * meta^2) - 27 * u + ( - 64 * mk^4 + 9 * (s^2 + t^2) + 24 * mk^2 * u) / (2.0 * ( - meta^2 + mpi^2)))) / (12.0 * fpi^2) + ( - 4 * mk^2 * ( - 80 * mk^2 + 30 * s + 57 * u) + 3 * (31 * s^2 + 4 * s * u + 16 * u^2)) / (2304.0 * fpi^4 * pi^2) - (mupi * ((16 * mk^4 + 24 * mk^2 * (s - 2 * t) - 9 * (s^2 - 2 * t^2)) / (8.0 * (mk^2 - mpi^2)) + 5 * ( - 2 * mk^2 + u) - (11 * s^2 + 4 * t^2 - 8 * mk^2 * (s + 2 * t) + 4 * s * (2 * t + u)) / (12.0 * mpi^2))) / (4.0 * fpi^2) - (( - 6 * meta^2 - 2 * mpi^2 + 9 * s)^2 * jbar(s, meta, meta)) / (288.0 * fpi^4) - ((4 * mk^2 * (2 * mk^2 - t) + s * (s - u)) * jbar(s, mk, mk)) / (6.0 * fpi^4) + (( - 3 * meta^2 - 8 * mk^2 - mpi^2 + 9 * s)^2 * jbar(s, mpi, meta)) / (432.0 * fpi^4) - ((s * (8 * mk^2 + 7 * s - 4 * t) + 8 * mpi^2 * ( - 4 * mk^2 + s + 2 * t)) * jbar(s, mpi, mpi)) / (96.0 * fpi^4) + (( - 6 * meta^2 - 2 * mpi^2 + 9 * s)^2 * jbar(s, meta, meta) + 60 * ( - 8 * mk^4 + s * (2 * s + t) + 4 * mk^2 * u) * jbar(s, mk, mk) + (2 * ( - 3 * meta^2 - 8 * mk^2 - mpi^2 + 9 * s)^2 * jbar(s, mpi, meta)) / 3.0 + 3 * ( - 8 * mpi^2 * ( - 4 * mk^2 + s + 2 * t) + s * ( - 8 * mk^2 + 11 * s + 4 * t)) * jbar(s, mpi, mpi)) / (288.0 * fpi^4) - (( - 8 * mk^4 + t * (s + 2 * t) + 4 * mk^2 * u) * jbar(t, mk, mk)) / (24.0 * fpi^4)
    - (( - 3 * meta^2 - 8 * mk^2 - mpi^2 + 9 * t)^2 * _jbar(t, mpi, meta)) / (216.0 * fpi^4)
    - (( - 4 * mpi^2 + t) * ( - 4 * mk^2 + 2 * s + t) * _jbar(t, mpi, mpi)) / (24.0 * fpi^4)
    + (( - 6 * meta^2 - 2 * mpi^2 + 9 * t)^2 * jbar(t, meta, meta)
      + 60 * ( - 8 * mk^4 + t * (s + 2 * t) + 4 * mk^2 * u) * jbar(t, mk, mk)
      + (2 * ( - 3 * meta^2 - 8 * mk^2 - mpi^2 + 9 * t)^2 * _jbar(t, mpi, meta)) / 3.0 + 3 * ( - 8 * mpi^2 * ( - 4 * mk^2 + 2 * s + t)
      + t * ( - 8 * mk^2 + 4 * s + 11 * t)) * _jbar(t, mpi, mpi)) / (288.0 * fpi^4)
    + (( - 2 * mk^2 + u)^2 * jbar(u, mk, mk)) / (4.0 * fpi^4) - (2 * (2 * ( - 2 * mk^2 + u)^2 * lecr[2] + ( - 2 * mk^2 + s)^2 * (4 * lecr[1] + lecr[3]) + ( - 2 * mk^2 + t)^2 * (2 * lecr[2] + lecr[3]))) / fpi^4 + (4 * (2 * ( - 2 * mk^2 + u)^2 * lecr[2] + (( - 2 * mk^2 + s)^2 + ( - 2 * mk^2 + t)^2) * (2 * lecr[1] + lecr[2] + lecr[3]) - 4 * mk^2 * u * lecr[4] - 2 * mpi^2 * ( - 2 * mk^2 + u) * lecr[5] - 4 * mk^4 * (lecr[5] - 2 * (2 * lecr[6] + lecr[8])))) / fpi^4 - (4 * (4 * mk^2 * s * lecr[4] - mpi^2 * ( - 2 * mk^2 + u) * lecr[5] - 2 * mk^4 * (4 * lecr[4] + lecr[5] - 2 * (2 * lecr[6] + lecr[8])))) / fpi^4)
end

V4_I1_kkkk_j(s, x, lecr) = V4_I1_kkkk(s, x, lecr) - V4_I1_kkkk_nonj(s, x, lecr)


function St4_10!(tm, w, lecr)
    _s = w^2
    tm[1, 1] = swave(x->V4_I1_πηπη(_s, x, lecr))
    tm[2, 1] = swave(x->V4_I1_πηkk(_s, x, lecr))
    tm[1, 2] = tm[2, 1]
    if w >= 2*KK.m1
        tm[2, 2] = swave(x->V4_I1_kkkk(_s, x, lecr))
    else
        tm[2, 2] = swave(x->V4_I1_kkkk_nonj(_s, x, lecr) ) + real(swave(x->V4_I1_kkkk_j(_s, x, lecr)) )
    end
    return SMatrix{2,2}(tm)
end

function St4_11!(tm, w, lecr)
    _s = w^2
    tm[1, 1] = pwave(x->V4_I1_ππππ(_s, x, lecr))
    tm[2, 1] = pwave(x->V4_I1_ππkk(_s, x, lecr))
    tm[1, 2] = tm[2, 1]
    # to remedy the unitarity breaking
#     tm[2, 2] = pwave(x->V4_I1_kkkk(_s, x, lecr))
    if w >= 2*KK.m1
        tm[2, 2] = pwave(x->V4_I1_kkkk(_s, x, lecr))
    else
        tm[2, 2] = pwave(x->V4_I1_kkkk_nonj(_s, x, lecr) ) + real( pwave(x->V4_I1_kkkk_j(_s, x, lecr)) )
    end
    return SMatrix{2,2}(tm)
end



# remove the im part of t-channel pipi and u-channel piK
function V4_I1_kηkη_nonj(s, x, lecr)
    t, u = tu(Kη, Kη, s, x)
    _jbar(z,m1,m2) = 0; _jbbar(z,m1,m2) = 0
    return (- (mupi * ( - 84 * mk^4 + 2 * mk^2 * (26 * meta^2 + 69 * t) + 3 * (16 * meta^4 - 50 * meta^2 * t + (s - u)^2))) / (48.0 * fpi^2 * ( - meta^2 + mk^2)) + (mueta * (144 * meta^6 - 128 * mk^6 + 144 * mk^4 * t + 2 * meta^4 * ( - 166 * mk^2 + 153 * t) + meta^2 * (428 * mk^4 - 486 * mk^2 * t + 27 * (s - u)^2))) / (144.0 * fpi^2 * meta^2 * ( - meta^2 + mk^2)) - (muk * (92 * mk^6 - 81 * meta^2 * t^2 - 60 * mk^4 * (meta^2 + 3 * t) + 18 * mk^2 * (8 * meta^4 + 6 * meta^2 * t + 5 * t^2 - 2 * s * u))) / (72.0 * fpi^2 * mk^2 * ( - meta^2 + mk^2)) + (116 * mk^4 + mk^2 * (184 * meta^2 - 153 * t) - 9 * (4 * meta^4 - 3 * meta^2 * t + 10 * t^2 + 2 * s * u)) / (2304.0 * fpi^4 * pi^2) + ((16 * mk^2 - 7 * mpi^2) * ( - 6 * meta^2 - 2 * mpi^2 + 9 * t) * jbar(t, meta, meta)) / (216.0 * fpi^4) + (t * ( - 6 * meta^2 - 2 * mpi^2 + 9 * t) * jbar(t, mk, mk)) / (16.0 * fpi^4)
    + (mpi^2 * t * _jbar(t, mpi, mpi)) / (8.0 * fpi^4) + ((( - meta^2 + mk^2)^2 * ( - 6 * meta^2 - 18 * mk^2 + 4 * mpi^2 - 3 * u) * jbar(s, mk, meta)) / s + ((45 * meta^4 + 189 * mk^4 + 8 * mpi^4 + 12 * mpi^2 * ( - 2 * meta^2 + 3 * s) - 18 * mk^2 * (9 * meta^2 + 4 * mpi^2 + 6 * s - 3 * u) + 27 * s * (s - u) + 54 * meta^2 * u) * jbar(s, mk, meta)) / 9.0 + ((18 * meta^4 + 29 * mk^4 + 11 * mpi^4 + 27 * s * (s - u) + 6 * mpi^2 * (8 * meta^2 - 6 * s + 9 * u) + 2 * mk^2 * ( - 78 * meta^2 - 47 * mpi^2 + 18 * s + 27 * u)) * jbar(s, mpi, mk)) / 9.0 - ((2 * mk^6 - 2 * mk^2 * mpi^2 * (4 * meta^2 + 5 * mpi^2 + 3 * u) + mk^4 * ( - 8 * meta^2 + 14 * mpi^2 + 3 * u) + mpi^2 * (6 * meta^4 + mpi^2 * (4 * meta^2 + 3 * u))) * jbar(s, mpi, mk)) / s + (6 * ( - meta^2 + mk^2)^2 * (( - meta^2 + mk^2)^2 * jbbar(s, mk, meta) + (mk^2 - mpi^2)^2 * jbbar(s, mpi, mk))) / s^2) / (32.0 * fpi^4)
    + ((( - meta^2 + mk^2)^2 * ( - 6 * meta^2 - 18 * mk^2 + 4 * mpi^2 - 3 * s) * jbar(u, mk, meta)) / u + ((45 * meta^4 + 189 * mk^4 + 8 * mpi^4 + 54 * meta^2 * s + 27 * u * ( - s + u) + 12 * mpi^2 * ( - 2 * meta^2 + 3 * u)
    - 18 * mk^2 * (9 * meta^2 + 4 * mpi^2 - 3 * s + 6 * u)) * jbar(u, mk, meta)) / 9.0 - ((2 * mk^6 - 2 * mk^2 * mpi^2 * (4 * meta^2 + 5 * mpi^2 + 3 * s) + mk^4 * ( - 8 * meta^2 + 14 * mpi^2 + 3 * s)
    + mpi^2 * (6 * meta^4 + mpi^2 * (4 * meta^2 + 3 * s))) * _jbar(u, mpi, mk)) / u + ((18 * meta^4 + 29 * mk^4 + 11 * mpi^4 + 6 * mpi^2 * (8 * meta^2 + 9 * s - 6 * u) + 27 * u * ( - s + u)
    + 2 * mk^2 * ( - 78 * meta^2 - 47 * mpi^2 + 27 * s + 18 * u)) * _jbar(u, mpi, mk)) / 9.0
    + (6 * ( - meta^2 + mk^2)^2 * (( - meta^2 + mk^2)^2 * jbbar(u, mk, meta)
    + (mk^2 - mpi^2)^2 * _jbbar(u, mpi, mk))) / u^2) / (32.0 * fpi^4) + ((( - meta^2 - mk^2 + s)^2 + ( - meta^2 - mk^2 + u)^2) * (12 * lecr[2] + lecr[3]) + 2 * (2 * meta^2 - t) * (2 * mk^2 - t) * (12 * lecr[1] + 5 * lecr[3])) / (3.0 * fpi^4) - (2 * (12 * meta^4 + 3 * mpi^4 + mpi^2 * (5 * meta^2 - 9 * t)) * lecr[5]) / (3.0 * fpi^4) + (4 * (2 * (meta^2 + mk^2) * t * lecr[4] + 8 * meta^2 * mk^2 * ( - lecr[4] + lecr[6]) + 2 * (3 * meta^4 - 4 * meta^2 * mpi^2 + mpi^4) * lecr[7] + (6 * meta^4 - 3 * meta^2 * mpi^2 + mpi^4) * lecr[8])) / fpi^4
        )
end
V4_I1_kηkη_j(s, x, lecr) = V4_I1_kηkη(s, x, lecr) - V4_I1_kηkη_nonj(s, x, lecr)

function St4_strange_10!(tm, w, lecr)
    _s = w^2
    tm[1, 1] = swave(x->V4_I1_kπkπ(_s, x, lecr))
    tm[2, 1] = swave(x->V4_I1_kπkη(_s, x, lecr))
    tm[1, 2] = tm[2, 1]
    if w >=  (Kη.m1 + Kη.m2)
        tm[2, 2] = swave(x->V4_I1_kηkη(_s, x, lecr))
    else
        tm[2, 2] = swave(x->V4_I1_kηkη_nonj(_s, x, lecr) ) + real( swave(x->V4_I1_kηkη_j(_s, x, lecr)) )
    end
    return SMatrix{2,2}(tm)
end

function St4_strange_11!(tm, w, lecr)
    _s = w^2
    tm[1, 1] = pwave(x->V4_I1_kπkπ(_s, x, lecr))
    tm[2, 1] = pwave(x->V4_I1_kπkη(_s, x, lecr))
    tm[1, 2] = tm[2, 1]
    # to remedy the unitarity breaking
#     tm[2, 2] = pwave(x->V4_I1_kηkη(_s, x, lecr))
    if w >= (Kη.m1 + Kη.m2)
        tm[2, 2] = pwave(x->V4_I1_kηkη(_s, x, lecr))
    else
        tm[2, 2] = pwave(x->V4_I1_kηkη_nonj(_s, x, lecr) ) + real( pwave(x->V4_I1_kηkη_j(_s, x, lecr)) )
    end
    return SMatrix{2,2}(tm)
end;
