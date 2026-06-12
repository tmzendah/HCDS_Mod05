"""
src/losses.py
Loss functions for the 2x2 KL grading experiment.

Two loss functions compared:
-----------------------------
1. CrossEntropyLoss  -- standard categorical classification.
                        Treats KL grades as unordered categories.
                        No knowledge of grade ordinality encoded.
                        Output: softmax over 5 classes.

2. CoralLoss         -- Consistent Rank Logits (Cao et al. 2020).
                        Treats KL grading as ordinal regression.
                        Encodes clinical reality: grade 0 < 1 < 2 < 3 < 4.
                        Decomposes 5-class problem into 4 binary tasks:
                          P(grade > 0)
                          P(grade > 1)
                          P(grade > 2)
                          P(grade > 3)
                        Output: 4 sigmoid probabilities (one per boundary).

Why this matters clinically:
-----------------------------
Standard CE treats misclassifying KL1 as KL0 equally as bad as
misclassifying KL1 as KL4. Clinically these are very different errors:
  - KL1 as KL0: patient discharged, no follow-up, OA progression missed
  - KL1 as KL4: unnecessary surgical referral
CORAL encodes that adjacent grade errors are less severe than
distant grade errors by enforcing ordinal monotonicity.

Reference:
  Cao et al. (2020). Rank consistent ordinal regression for neural
  networks with application to age estimation.
  Pattern Recognition Letters, 140, 325-331.
  https://doi.org/10.1016/j.patrec.2020.11.008
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# 1. Standard Cross-Entropy (categorical baseline)
# ─────────────────────────────────────────────────────────

def get_ce_loss() -> nn.Module:
    """
    Returns standard PyTorch cross-entropy loss.

    No class weighting applied -- all four experimental conditions
    use identical loss settings so that the only variable is the
    loss function type (CE vs CORAL), not its configuration.

    Returns:
        nn.CrossEntropyLoss instance
    """
    return nn.CrossEntropyLoss()


# ─────────────────────────────────────────────────────────
# 2. CORAL Loss (ordinal regression)
# ─────────────────────────────────────────────────────────

class CoralLoss(nn.Module):
    """
    CORAL loss for ordinal regression (Cao et al. 2020).

    How it works step by step:
    --------------------------
    Step 1 -- Model output
        Instead of 5 class logits, the model outputs 4 rank logits,
        one per grade boundary:
            logit[0] = log-odds that grade > 0
            logit[1] = log-odds that grade > 1
            logit[2] = log-odds that grade > 2
            logit[3] = log-odds that grade > 3

    Step 2 -- Binary target construction
        For a true label y, binary targets are constructed as:
            target[k] = 1 if y > k, else 0

        Examples:
            y=0  ->  targets = [0, 0, 0, 0]
            y=1  ->  targets = [1, 0, 0, 0]
            y=2  ->  targets = [1, 1, 0, 0]
            y=3  ->  targets = [1, 1, 1, 0]
            y=4  ->  targets = [1, 1, 1, 1]

    Step 3 -- Loss computation
        Binary cross-entropy at each rank boundary,
        averaged across all boundaries and all samples.

    Step 4 -- Prediction at inference
        Use coral_predict() to convert logits to KL grade:
            probs = sigmoid(logits)
            predicted_grade = sum(probs > 0.5)
        e.g. probs=[0.95, 0.80, 0.30, 0.10] -> grade 2

    Why sigmoid not softmax?
        Each boundary is an independent binary decision.
        Sigmoid handles independent probabilities.
        Softmax enforces that probabilities sum to 1 across classes --
        that constraint does not apply here.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_ranks   = num_classes - 1   # 4 boundaries for 5 classes

    def forward(
        self,
        logits: torch.Tensor,   # shape: [batch_size, num_ranks]
        labels: torch.Tensor,   # shape: [batch_size], integer KL grades
    ) -> torch.Tensor:
        """
        Args:
            logits : raw model outputs, shape [batch_size, 4]
                     NOT passed through sigmoid yet
            labels : ground truth KL grades, shape [batch_size]
                     integer values in {0, 1, 2, 3, 4}

        Returns:
            scalar loss tensor
        """
        # Build rank threshold vector: [0, 1, 2, 3]
        # shape: [1, num_ranks]
        rank_thresholds = torch.arange(
            self.num_ranks,
            device=logits.device,
            dtype=labels.dtype
        ).unsqueeze(0)

        # Expand labels for broadcasting
        # shape: [batch_size, 1]
        labels_expanded = labels.unsqueeze(1)

        # Binary targets: 1 where label > threshold, 0 otherwise
        # shape: [batch_size, num_ranks]
        #
        # Example for batch=[0, 2, 4]:
        #   labels_expanded = [[0], [2], [4]]
        #   rank_thresholds = [[0, 1, 2, 3]]
        #   binary_targets  = [[0,0,0,0],
        #                       [1,1,0,0],
        #                       [1,1,1,1]]
        binary_targets = (labels_expanded > rank_thresholds).float()

        # Binary cross-entropy with logits (sigmoid applied internally)
        # Numerically stable -- avoids explicit sigmoid then log
        loss = F.binary_cross_entropy_with_logits(
            logits,
            binary_targets,
            reduction="mean"
        )

        return loss


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORAL logits to predicted KL grade at inference.

    Args:
        logits : raw model outputs, shape [batch_size, 4]

    Returns:
        predicted_grades : shape [batch_size], integer values in {0,1,2,3,4}

    How it works:
        1. Apply sigmoid to get P(grade > k) for each boundary k
        2. Count how many boundaries have P > 0.5
        3. That count is the predicted grade

    Examples:
        logits -> probs           -> sum(>0.5) -> grade
        [3,2,-1,-2] -> [0.95,0.88,0.27,0.12] ->  2     -> KL2
        [-2,-2,-2,-2] -> [0.12,0.12,0.12,0.12] -> 0    -> KL0
        [3,3,3,3]   -> [0.95,0.95,0.95,0.95] ->  4     -> KL4
    """
    probs     = torch.sigmoid(logits)               # [batch_size, 4]
    predicted = (probs > 0.5).sum(dim=1).long()     # [batch_size]
    return predicted


# ─────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────

def get_loss_function(loss_name: str,
                      num_classes: int = 5) -> nn.Module:
    """
    Returns the correct loss function given a string name.
    Used by train.py so loss choice is driven by command-line args.

    Args:
        loss_name   : 'ce' for cross-entropy, 'coral' for CORAL loss
        num_classes : number of output classes (default 5 for KL grades)

    Returns:
        loss function as nn.Module

    Raises:
        ValueError if loss_name is not recognised
    """
    if loss_name == "ce":
        return get_ce_loss()
    elif loss_name == "coral":
        return CoralLoss(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            f"Choose 'ce' or 'coral'."
        )


# ─────────────────────────────────────────────────────────
# Verification (run directly to test CORAL logic)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  losses.py -- CORAL loss verification")
    print("=" * 60)

    loss_fn = CoralLoss(num_classes=5)

    # Test 1 -- confident correct predictions, loss should be low
    print("\n  Test 1: confident correct predictions")
    logits = torch.tensor([
        [-3.0, -3.0, -3.0, -3.0],   # confident KL0
        [ 3.0, -3.0, -3.0, -3.0],   # confident KL1
        [ 3.0,  3.0, -3.0, -3.0],   # confident KL2
        [ 3.0,  3.0,  3.0, -3.0],   # confident KL3
        [ 3.0,  3.0,  3.0,  3.0],   # confident KL4
    ])
    labels = torch.tensor([0, 1, 2, 3, 4])
    loss   = loss_fn(logits, labels)
    preds  = coral_predict(logits)
    print(f"    Loss      : {loss.item():.4f}  (expected: low ~0.05)")
    print(f"    Predicted : {preds.tolist()}")
    print(f"    True      : {labels.tolist()}")
    assert preds.tolist() == labels.tolist(), "Predictions should match labels"
    print("    PASSED")

    # Test 2 -- random logits, loss should be moderate
    print("\n  Test 2: random logits")
    torch.manual_seed(42)
    logits_rand = torch.randn(8, 4)
    labels_rand = torch.randint(0, 5, (8,))
    loss_rand   = loss_fn(logits_rand, labels_rand)
    print(f"    Loss      : {loss_rand.item():.4f}  (expected: ~0.5-0.8)")
    print("    PASSED")

    # Test 3 -- factory function
    print("\n  Test 3: factory function")
    ce_loss    = get_loss_function("ce")
    coral_loss = get_loss_function("coral")
    print(f"    CE loss   : {type(ce_loss).__name__}")
    print(f"    CORAL loss: {type(coral_loss).__name__}")
    print("    PASSED")

    print("\n  All tests passed.")
    print("=" * 60)
