
"""
Black model function for testing.
"""
function BlackOverK(x)
    @assert size(x)==(3,)
    moneyness = x[1]
    stdDev    = x[2]
    callOrPut = x[3]
    d1 = log(moneyness) / stdDev + stdDev / 2.0
    d2 = d1 - stdDev
    norm = Normal()
    return callOrPut * (moneyness*cdf(norm,callOrPut*d1)-cdf(norm,callOrPut*d2))
end
