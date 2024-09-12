

def flatten_dict(d, current_key='', sep='.'):

    if d is None:
        return {}

    items = []
    for k, v in d.items():
        new_key = f'{current_key}{sep}{k}' if current_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


