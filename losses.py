from torch.nn import functional as F

smoothing = 0.1

def smooth_one_hot(labels, epsilon=smoothing, num_classes=0):
    """
    Applies label smoothing to one-hot encoded labels.

    Args:
        labels (torch.Tensor): One-hot encoded labels with shape (batch_size, num_classes).
        epsilon (float, optional): Smoothing factor. Defaults to 0.1.

    Returns:
        torch.Tensor: Smoothed labels with the same shape as input.
    """
    return (1 - epsilon) * labels + epsilon * labels.new_full((labels.size()), 1 / num_classes)

def smooth_labels(labels, epsilon=smoothing, num_classes=0):
    """
    Applies label smoothing to raw integer labels.

    Args:
        labels (torch.Tensor): Raw integer labels with shape (batch_size,).
        epsilon (float, optional): Smoothing factor. Defaults to 0.1.

    Returns:
        torch.Tensor: Smoothed one-hot encoded labels with shape (batch_size, num_classes).
    """
    one_hot = F.one_hot(labels, num_classes=num_classes)
    return smooth_one_hot(one_hot, epsilon, num_classes=num_classes)

# For loss_function_1 (with ignore_index):
def loss_function_1_smoothed(outputs, targets, num_classes):
    smoothed_targets = smooth_labels(targets.masked_fill_(targets == 0, -100), epsilon=smoothing, num_classes=num_classes)  # Ignore index handling
    return F.cross_entropy(outputs, smoothed_targets, ignore_index=-100)

# For loss_function_2 (without ignore_index):
def loss_function_2_smoothed(outputs, targets, num_classes):
    smoothed_targets = smooth_labels(targets, epsilon=smoothing, num_classes=num_classes)
    return F.cross_entropy(outputs, smoothed_targets)

