def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    
    for i in range(steps):
        grad = (2*a*x) + b # derivative of ax^2 + bx + c -> 2ax + b
        x -= lr*grad   # gradient descent update rule

    return x