import unittest

try:
    import torch
    from src.models import LeNet5
except ModuleNotFoundError:  # Makes repository inspection possible without ML dependencies installed.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class LeNet5Tests(unittest.TestCase):
    def test_output_shape_for_62_classes(self):
        model = LeNet5()
        self.assertEqual(tuple(model(torch.randn(4, 1, 32, 32)).shape), (4, 62))

    def test_backward_and_update_are_finite(self):
        model = LeNet5()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        images, labels = torch.randn(3, 1, 32, 32), torch.tensor([0, 10, 61])
        before = model.c1.weight.detach().clone()
        loss = torch.nn.CrossEntropyLoss()(model(images), labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.c1.weight.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, model.c1.weight))
