"""
Prototypical Network for STUNT.

Supports both:

Single task:
    support_x = [N_support, features]
    support_y = [N_support]
    query_x   = [N_query, features]

Batched tasks:
    support_x = [B, N_support, features]
    support_y = [B, N_support]
    query_x   = [B, N_query, features]

For each task, the prototype of each class is
the mean embedding of its support examples.

Query examples are classified using squared
Euclidean distance to the prototypes.
"""

import torch
from torch import nn


class ProtoNet(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder

    # =========================================================
    # Compute prototypes for ONE task
    # =========================================================

    def _compute_prototypes_single(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ):

        support_embeddings = self.encoder(
            support_x
        )

        classes = torch.unique(
            support_y,
            sorted=True,
        )

        prototypes = []

        for cls in classes:

            class_embeddings = (
                support_embeddings[
                    support_y == cls
                ]
            )

            prototype = class_embeddings.mean(
                dim=0
            )

            prototypes.append(
                prototype
            )

        prototypes = torch.stack(
            prototypes,
            dim=0,
        )

        return classes, prototypes

    # =========================================================
    # Compute prototypes for BATCH of tasks
    # =========================================================

    def _compute_prototypes_batch(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ):

        batch_size = support_x.shape[0]

        # Flatten the task dimension temporarily.
        flat_support_x = support_x.reshape(
            -1,
            support_x.shape[-1],
        )

        flat_embeddings = self.encoder(
            flat_support_x
        )

        embeddings = flat_embeddings.reshape(
            support_x.shape[0],
            support_x.shape[1],
            -1,
        )

        # The STUNT tasks are n-way tasks with
        # labels 0, 1, ..., n_way-1.
        n_way = int(
            support_y.max().item()
        ) + 1

        prototypes = []

        for cls in range(n_way):

            mask = (
                support_y == cls
            )

            # mask:
            # [B, N_support]

            mask = mask.unsqueeze(-1)

            # [B, N_support, 1]

            masked_embeddings = (
                embeddings * mask
            )

            class_count = mask.sum(
                dim=1
            ).clamp_min(1)

            class_prototype = (
                masked_embeddings.sum(
                    dim=1
                )
                / class_count
            )

            prototypes.append(
                class_prototype
            )

        prototypes = torch.stack(
            prototypes,
            dim=1,
        )

        # [B, n_way, embedding_dim]

        return prototypes

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ):

        # -----------------------------------------------------
        # Single task
        # -----------------------------------------------------

        if support_x.dim() == 2:

            classes, prototypes = (
                self._compute_prototypes_single(
                    support_x,
                    support_y,
                )
            )

            query_embeddings = self.encoder(
                query_x
            )

            distances = torch.sum(
                (
                    query_embeddings.unsqueeze(1)
                    - prototypes.unsqueeze(0)
                ) ** 2,
                dim=2,
            )

            logits = -distances

            return logits, classes

        # -----------------------------------------------------
        # Batched tasks
        # -----------------------------------------------------

        if support_x.dim() == 3:

            prototypes = (
                self._compute_prototypes_batch(
                    support_x,
                    support_y,
                )
            )

            batch_size = query_x.shape[0]

            flat_query_x = query_x.reshape(
                -1,
                query_x.shape[-1],
            )

            flat_query_embeddings = self.encoder(
                flat_query_x
            )

            query_embeddings = (
                flat_query_embeddings.reshape(
                    batch_size,
                    query_x.shape[1],
                    -1,
                )
            )

            # -------------------------------------------------
            # prototypes:
            # [B, n_way, embedding_dim]
            #
            # query:
            # [B, n_query, embedding_dim]
            # -------------------------------------------------

            distances = torch.sum(
                (
                    query_embeddings.unsqueeze(2)
                    - prototypes.unsqueeze(1)
                ) ** 2,
                dim=-1,
            )

            # [B, n_query, n_way]

            logits = -distances

            return logits

        raise ValueError(
            "support_x must have either "
            "2 or 3 dimensions."
        )

    # =========================================================
    # Loss
    # =========================================================

    def loss(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ):

        # -----------------------------------------------------
        # Single task
        # -----------------------------------------------------

        if support_x.dim() == 2:

            logits, classes = self.forward(
                support_x,
                support_y,
                query_x,
            )

            target = torch.zeros_like(
                query_y
            )

            for i, cls in enumerate(
                classes
            ):

                target[
                    query_y == cls
                ] = i

            return nn.functional.cross_entropy(
                logits,
                target,
            )

        # -----------------------------------------------------
        # Batched tasks
        # -----------------------------------------------------

        if support_x.dim() == 3:

            logits = self.forward(
                support_x,
                support_y,
                query_x,
            )

            # query_y is already mapped to
            # 0 ... n_way-1 by STUNTTaskGenerator.

            return nn.functional.cross_entropy(
                logits.reshape(
                    -1,
                    logits.shape[-1],
                ),
                query_y.reshape(-1),
            )

        raise ValueError(
            "support_x must have either "
            "2 or 3 dimensions."
        )

    # =========================================================
    # Prediction
    # =========================================================

    def predict(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ):

        # -----------------------------------------------------
        # Single task
        # -----------------------------------------------------

        if support_x.dim() == 2:

            logits, classes = self.forward(
                support_x,
                support_y,
                query_x,
            )

            predicted_indices = logits.argmax(
                dim=1
            )

            predictions = classes[
                predicted_indices
            ]

            return predictions

        # -----------------------------------------------------
        # Batched tasks
        # -----------------------------------------------------

        if support_x.dim() == 3:

            logits = self.forward(
                support_x,
                support_y,
                query_x,
            )

            return logits.argmax(
                dim=-1
            )

        raise ValueError(
            "support_x must have either "
            "2 or 3 dimensions."
        )