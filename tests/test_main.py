import unittest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms # type: ignore[import]
from client import Client
from group import Group
from utils import create_model, split_indices, filter_test, ensemble_eval

class TestComponents(unittest.TestCase):
    def setUp(self):
        X = torch.randn(10, 1, 28, 28)
        y = torch.randint(0, 10, (10,))
        loader = DataLoader(list(zip(X, y)), batch_size=5)
        self.client = Client(1, loader)
        self.group = Group(1, [self.client])

    def test_create_model_output_shape(self):
        model = create_model()
        x = torch.randn(1, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (1, 10))

    def test_client_and_group_flags(self):
        self.assertFalse(self.client.is_mal)
        self.group.compute_has_mal()
        self.assertFalse(self.group.has_mal)
        self.client.is_mal = True
        self.group.compute_has_mal()
        self.assertTrue(self.group.has_mal)
        self.group.assign_affected()
        self.assertTrue(self.client.is_affected)

    def test_split_and_filter(self):
        tf = transforms.ToTensor()
        ds = datasets.MNIST('.', train=True, download=True, transform=tf)
        pi = [0.1]*10
        idx = split_indices(ds, pi, n=50)
        self.assertEqual(len(idx), 50)
        loader = filter_test(ds, pi, n=50)
        for Xb, yb in loader:
            self.assertEqual(Xb.shape[0], 50)

    def test_ensemble_eval_identity(self):
        self.client.is_affected = False
        loader = self.client.loader
        res = ensemble_eval([self.client], loader)
        self.assertAlmostEqual(res[1] + res[2], 1.0)

if __name__ == '__main__':
    unittest.main()
