from .utilities.logger import Logger

def default_img_classification_scoring(models, task):
    """
    Default candidate scoring function for image classification tasks.
    Since the scoring is relative, more than 1 model (typically best model
    + currently evaluated model) should be provided.

    The calculation involves normalizing each metric feature relative to
    the same feature in other models (i.e. each column in the metrics
    dataset).

    Args:
        models (:obj:`list`): list of :class:`~Network` objects to score
        task (:class:`~BaseTask`): the task to score the given models based \
        upon. We use the task's ID and version to filter the model metrics, \
        and the objectives/score weights to proportionally aggregate all given \
        objectives into a single-objective (scalar) problem.

    Returns:
        :obj:`list`: list of normalized scalar scores for each given model \
        (order is preserved and ensured to be the same as the passed list \
        of models).
    """

    task_id = task.id
    task_version = task.version
    objective = task.objective
    weights = objective.score_weights

    # normalize weights (whilst preserving inversions)
    total_weight = sum(abs(v) for v in weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    # structure model metrics;
    # filter for given task id and get last epoch's results
    metrics = [model.metrics.aggregate(on_task_id=task_id,
                                       on_task_version=task.version,
                                       epoch=-1)\
               for model in models]

    for i, metric in enumerate(metrics):
        # filter metrics based on weights' key-set intersection
        metrics[i] = { key: metric[key] \
                      for key in metric if key in normalized_weights }

    # scale model metrics to conserve proportionality, then apply weights
    for k in normalized_weights:
        if not any([True for metric in metrics if k in metric]):
            # weight key not found
            continue

        max_val = max([abs(metric[k]) for metric in metrics if k in metric])
        for i, metric in enumerate(metrics):
            if k not in metric:
                # raise KeyError(f'Key {k} not found in a models\' metrics')
                # ^commented out, sometimes `metric` is `{}`
                continue

            if max_val == 0:
                # avoid division by 0
                Logger.debug('Scoring Function - Absolute Max Val is `0`', k)
                metrics[i][k] = 0
                continue

            metrics[i][k] = metric[k] / max_val * normalized_weights[k]

    scalar_scores = [sum(list(metric.values())) for metric in metrics]

    return scalar_scores   # scalar scores' list; 1 for each model


