qon(s, m1, m2) = sqrt((s - (m1+m2)^2) * (s - (m1-m2)^2))/(2sqrt(s))

function δ00_0(w, lec) # without considering continuity
    _m1, _m2 = ππ.m1, ππ.m2
    if w ≥ _m1 + _m2
        _q = qon(w^2, _m1, _m2)
        _tm = Stiam_00(w, lec)[1, 1]
        res = 0.5atand(real(_tm), -imag(_tm) + 4π*w/_q )
    else
        res = 0.0
    end
    return  res
end

# for easy plot
function δ00(w, lec; jump = 900, jumpd = 80)
    res = δ00_0(w, lec)
    return  (w>jump && res <jumpd) ? res + 180 : res
end

function η00(w, lec)
    _m1 = _m2 = ππ.m1;  _m3 = _m4 = KK.m1
    if w ≥ _m3 + _m4
        _s = w^2
        _q1 = qon(_s, _m1, _m2)
        _q2 = qon(_s, _m3, _m4)
        _tm = abs2(Stiam_00(w, lec)[2, 1])
        res = sqrt( 1 - _q1 * _q2/(16π^2*_s) * _tm )
    else
        res = 1.0
    end
    return  res
end

function δππkk00(w, lec)
    if w < KK.m1 + KK.m2
        res = δ00(w, lec)
    else
        _tm = Stiam_00(w, lec)[1, 2]
        res = atand(imag(_tm), real(_tm))
    end
    return res < 0 ? res + 360 : res
end

function δ11_0(w, lec)
    _m1, _m2 = ππ.m1, ππ.m2
    if w ≥ _m1 + _m2
        _q = qon(w^2, _m1, _m2)
        _tm = Stiam_11(w, lec)[1, 1]
        res = 0.5atand(real(_tm), -imag(_tm) + 4π*w/_q )
    else
        res = 0.0
    end
    return  res
end
function δ11(w, lec)
    res = δ11_0(w, lec)
    return res < 0 ? res + 180 : res
end


function δ10_πη_0(w, lec)
    _m1, _m2 = πη.m1, πη.m2
    if w ≥ _m1 + _m2
        _q = qon(w^2, _m1, _m2)
        _tm = Stiam_10(w, lec)[1, 1]
        res = 0.5atand(real(_tm), -imag(_tm) + 4π*w/_q )
    else
        res = 0.0
    end
    return  res
end
function δ10_πη(w, lec)
    res = δ10_πη_0(w, lec)
    return res < 0 ? res + 180 : res
end

function δ10_kk(w, lec)
    _m1, _m2 = KK.m1, KK.m2
    if w ≥ _m1 + _m2
        _q = qon(w^2, _m1, _m2)
        _tm = Stiam_10(w, lec)[2, 2]
        res = 0.5atand(real(_tm), -imag(_tm) + 4π*w/_q )
    else
        res = 0.0
    end
    return  res
end


function δ20(w, lec)
    _m1, _m2 = ππ.m1, ππ.m2
    if w ≥ _m1 + _m2
        _q = qon(w^2, _m1, _m2)
        _tm = tiam_20(w, lec)
        res = 0.5atand(real(_tm), -imag(_tm) + 4π*w/_q )
    else
        res = 0.0
    end
    return  res
end

function δhalf0_0(w, lec)
    _m1, _m2 = Kπ.m1, Kπ.m2
    if w ≥ _m1 + _m2
        _q = qon(w^2, _m1, _m2)
        _tm = Stiam_strange_10(w, lec)[1, 1]
        res = 0.5atand(real(_tm), -imag(_tm) + 4π*w/_q )
    else
        res = 0.0
    end
    return  res
end
function δhalf0(w, lec)
    res = δhalf0_0(w, lec)
    return res < 0 ? res + 180 : res
end

function δhalf1_0(w, lec)
    _m1, _m2 = Kπ.m1, Kπ.m2
    if w ≥ _m1 + _m2
        _q = qon(w^2, _m1, _m2)
        _tm = Stiam_strange_11(w, lec)[1, 1]
        res = 0.5atand(real(_tm), -imag(_tm) + 4π*w/_q )
    else
        res = 0.0
    end
    return  res
end
function δhalf1(w, lec)
    res = δhalf1_0(w, lec)
    return res < 0 ? res + 180 : res
end

function δ3half0(w, lec)
    _m1, _m2 = Kπ.m1, Kπ.m2
    if w ≥ _m1 + _m2
        _q = qon(w^2, _m1, _m2)
        _tm = tiam_strange_30(w, lec)
        res = 0.5atand(real(_tm), -imag(_tm) + 4π*w/_q )
    else
        res = 0.0
    end
    return  res
end
