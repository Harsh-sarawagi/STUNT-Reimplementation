import torch

from src.encoder import (
    TabularEncoder
)

from src.protonet import (
    ProtoNet
)


def test_protonet_forward():

    torch.manual_seed(
        42
    )

    encoder = TabularEncoder(
        input_dim=10,
        hidden_dim=32,
        embedding_dim=16,
    )

    model = ProtoNet(
        encoder
    )

    # 3-way, 2-shot
    support_x = torch.randn(
        6,
        10
    )

    support_y = torch.tensor([
        0, 0,
        1, 1,
        2, 2,
    ])

    # 3-way, 3-query
    query_x = torch.randn(
        9,
        10
    )

    logits, classes = model(
        support_x,
        support_y,
        query_x
    )

    assert logits.shape == (
        9,
        3
    )

    assert classes.tolist() == [
        0,
        1,
        2
    ]


def test_protonet_loss():

    torch.manual_seed(
        42
    )

    encoder = TabularEncoder(
        input_dim=10,
        hidden_dim=32,
        embedding_dim=16,
    )

    model = ProtoNet(
        encoder
    )

    support_x = torch.randn(
        6,
        10
    )

    support_y = torch.tensor([
        0, 0,
        1, 1,
        2, 2,
    ])

    query_x = torch.randn(
        9,
        10
    )

    query_y = torch.tensor([
        0, 0, 0,
        1, 1, 1,
        2, 2, 2,
    ])

    loss = model.loss(
        support_x,
        support_y,
        query_x,
        query_y
    )

    assert loss.ndim == 0

    assert torch.isfinite(
        loss
    )


def test_protonet_prediction():

    torch.manual_seed(
        42
    )

    encoder = TabularEncoder(
        input_dim=10,
        hidden_dim=32,
        embedding_dim=16,
    )

    model = ProtoNet(
        encoder
    )

    support_x = torch.randn(
        6,
        10
    )

    support_y = torch.tensor([
        0, 0,
        1, 1,
        2, 2,
    ])

    query_x = torch.randn(
        9,
        10
    )

    predictions = model.predict(
        support_x,
        support_y,
        query_x
    )

    assert predictions.shape == (
        9,
    )

    assert all(
        prediction.item()
        in [0, 1, 2]
        for prediction in predictions
    )