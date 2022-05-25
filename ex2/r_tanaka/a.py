
    # plot
    formula = str()
    round(coef, 2)
    for i in range(deg + 1):
        for k in range(deg + 1 - i):
            formula += (str(coef[w])+(x1 ** str(i)) * (x2 ** str(k)))
            w += 1