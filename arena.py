def integer_to_one_hot(integer, max_value=4):
    one_hot = [0] * max_value
    one_hot[integer] = 1
    return one_hot