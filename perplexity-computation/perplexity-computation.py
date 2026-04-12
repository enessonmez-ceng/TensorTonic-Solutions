def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    import math
    H = 0
    N = len(actual_tokens)
    for i in range(len(prob_distributions)):
        H += (-1/N)*(math.log(prob_distributions[i][actual_tokens[i]]))
        
    return math.e**H
    
          