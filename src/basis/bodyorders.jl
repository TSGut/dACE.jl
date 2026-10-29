# the body order of every column of a basis

# correlation order N (an (N + 1)-body term) of every column, 0 for the self term (internal)
correlation_orders(basis) = [[correlation_order(f) for f in basis.funcs]; zeros(Int, nself(basis))]

"""
    body_orders(basis)

Body order of every column of `basis`: 1 for the 1-body terms (including the self term), 2 for the pair terms, ….
"""
body_orders(basis) = correlation_orders(basis) .+ 1

"""
    max_body_order(basis)

The highest body order of `basis`: `length(maxdeg)`, unless `filter` removed every function of that order.
"""
max_body_order(basis) = max_order(basis) + 1
