def compute_accuracy_by_group(predictions, val_y, demographic_groups):
    """
    Compute accuracy for different demographic subgroups.
    """
    subgroup_accuracies = []

    for group in demographic_groups:
        for toxic_label in [0, 1]:  # 0 = not toxic, 1 = toxic
            subset = val_y[(val_y[group] == 1) & (val_y['y'] == toxic_label)]
            if len(subset) > 0:
                accuracy = (predictions[subset.index] == subset['y'].values).mean()
                subgroup_accuracies.append(accuracy)

    worst_group_accuracy = min(subgroup_accuracies)
    return worst_group_accuracy
