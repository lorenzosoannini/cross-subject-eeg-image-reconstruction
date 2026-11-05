import unittest
import torch
from torch import nn
from model.network import ReconstructionModel, EEGModel
from model.decomposition import DSNLatentDecomposition
from model.dsn_losses import (
    latent_reconstruction_loss,
    cross_covariance_loss,
    variance_floor_loss,
)
from model.encoder import LoRALinear, GradReverse
from model.aggregation import aggregate_conditions


class Encoder(nn.Module):
    def forward_encoder(self, x):
        return x


class Branch(nn.Module):
    def forward(self, x):
        return x * 3, x


class Representation(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg_encoder = Encoder()
        self.decomposition = nn.Module()
        self.decomposition.invariant_branch = Branch()
        self.visual_resampler = nn.Identity()
        self.full_projection = nn.Linear(4, 4, bias=False)
        self.semantic_projection = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.full_projection.weight.copy_(torch.eye(4) * 2)
            self.semantic_projection.weight.copy_(torch.eye(4) * 5)

    condition = EEGModel.condition


class Prior(nn.Module):
    def forward(self, *, text_embed, image_embed):
        self.received = text_embed
        return (text_embed - image_embed).square().mean(), text_embed


class RoutesTests(unittest.TestCase):
    def test_prior_uses_Fs_in_training_and_inference(self):
        model = ReconstructionModel.__new__(ReconstructionModel)
        nn.Module.__init__(model)
        model.representation = Representation()
        model.diffusion_prior = Prior()
        for name in [
            "image_classifier",
            "image_discriminator",
            "text_projection",
            "blurry_decoder",
        ]:
            setattr(model, name, nn.Identity())
        eeg = torch.ones(2, 3, 4)
        out = model(eeg, torch.zeros_like(eeg))
        torch.testing.assert_close(model.diffusion_prior.received, out["F_s"])
        self.assertFalse(torch.equal(out["F"], out["F_s"]))
        out["prior_loss"].backward()
        self.assertIsNotNone(model.representation.semantic_projection.weight.grad)
        self.assertIsNone(model.representation.full_projection.weight.grad)
        torch.testing.assert_close(model.condition(eeg), out["F_s"])

    def test_average_is_after_nonlinearity_before_prior(self):
        class Nonlinear(nn.Module):
            def condition(self, x):
                return x.square()

        loader = [
            dict(
                eeg=torch.tensor([[[1.0]], [[3.0]], [[4.0]]]),
                visual_index=torch.tensor([7, 7, 8]),
                trial_row=torch.tensor([2, 5, 9]),
            )
        ]
        values, counts, events = aggregate_conditions(
            Nonlinear(), loader, torch.device("cpu"), {"precision": "fp32"}
        )
        self.assertEqual(values[7].item(), 5.0)
        self.assertEqual(counts[7], 2)
        self.assertEqual(events[7], [2, 5])

    def test_joint_decoder_gradients_reach_both_learned_branches(self):
        model = DSNLatentDecomposition(16, 8, 4, 1, 2, 16, 0.0)
        x = torch.randn(3, 5, 16, requires_grad=True)
        ei, es, recon, zi, zs = model(x)
        self.assertFalse(torch.allclose(es, x - ei))
        (
            latent_reconstruction_loss(recon, x)
            + cross_covariance_loss(zi, zs)
            + variance_floor_loss(zi, zs)
        ).backward()
        self.assertIsNotNone(model.invariant_branch.down_project.weight.grad)
        self.assertIsNotNone(model.subject_branch.down_project.weight.grad)

    def test_lora_gate_and_subject_gradient_reversal(self):
        layer = LoRALinear(nn.Linear(4, 4), rank=2, alpha=4, dropout=0.0, enabled=True)
        with torch.no_grad():
            layer.lora_B.fill_(1.0)
        x = torch.ones(2, 4)
        on = layer(x)
        layer.set_lora_enabled(False)
        self.assertFalse(torch.equal(on, layer(x)))
        x = torch.ones(2, requires_grad=True)
        GradReverse.lambd = 1.0
        GradReverse.apply(x).sum().backward()
        torch.testing.assert_close(x.grad, -torch.ones(2))


if __name__ == "__main__":
    unittest.main()
