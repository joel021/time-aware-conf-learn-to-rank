import unittest
import torch
from recsysconfident.ml.models.learn_rank.ua_lightgcn import UALightGCN


class TestUALightGCN(unittest.TestCase):

    def test_ua_lightgcn_forward_and_loss(self):
        n_users = 5
        n_items = 5
        emb_dim = 16

        # Create a simple sparse GCN Graph
        indices = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        values = torch.ones(5)
        graph = torch.sparse_coo_tensor(indices, values, (10, 10))

        items_per_user = {
            0: [0],
            1: [1],
            2: [2],
            3: [3],
            4: [4]
        }

        model = UALightGCN(
            items_per_user=items_per_user,
            Graph=graph,
            n_users=n_users,
            n_items=n_items,
            emb_dim=emb_dim,
            n_layers=2,
            keep_prob=1.0,
            A_split=False,
            rmin=1.0,
            rmax=5.0,
            dropout=False
        )

        users_ids = torch.tensor([0, 1, 2])
        items_ids = torch.tensor([0, 1, 2])

        # Test forward pass shape
        output = model(users_ids, items_ids)
        self.assertEqual(output.shape, (3, 2), "Forward output should stack mean and std (shape: batch_size, 2)")

        # Test nll_bpr_loss computation
        loss = model.nll_bpr_loss(users_ids, items_ids)
        self.assertTrue(loss.item() > 0, "Loss value should be positive")

        # Test predict output shape and boundaries
        mu, confidence = model.predict(users_ids, items_ids)
        self.assertEqual(mu.shape, (3,), "mu shape should match batch size")
        self.assertEqual(confidence.shape, (3,), "confidence shape should match batch size")
        self.assertTrue(torch.all(confidence >= 0.0) and torch.all(confidence <= 1.0), "Confidence must be a valid CDF probability [0, 1]")
