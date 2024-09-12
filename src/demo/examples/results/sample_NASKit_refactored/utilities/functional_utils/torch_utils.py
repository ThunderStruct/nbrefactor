
def transforms_equal(compose_a, compose_b):
    if len(compose_a.transforms) != len(compose_b.transforms):
        return False

    for idx, _ in compose_a.transforms:
        t_a = compose_a.transforms[idx]
        t_b = compose_b.transforms[idx]

        if type(t_a) != type(t_b):
            return False

        for param in t_a.__dict__:
            if param not in t_b.__dict__:
                # parameters not identical
                return False

            # check equality only on public parameters
            if param[0] != '_':
                if t_a.__dict__[param] != t_b.__dict__[param]:
                    return False

    return True


