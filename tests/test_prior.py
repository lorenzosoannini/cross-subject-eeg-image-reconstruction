import unittest
import torch
from model.prior import BrainDiffusionPrior, PriorNetwork


class PriorTests(unittest.TestCase):
    def test_forward_backward_and_twenty_corrected_ddim_steps(self):
        net = PriorNetwork(
            dim=52,
            depth=1,
            dim_head=52,
            heads=1,
            causal=False,
            num_tokens=2,
            learned_query_mode="pos_emb",
        )
        prior = BrainDiffusionPrior(
            net=net,
            image_embed_dim=52,
            condition_on_text_encodings=False,
            timesteps=100,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )
        fs = torch.randn(2, 2, 52, requires_grad=True)
        loss, pred = prior(text_embed=fs, image_embed=torch.randn_like(fs))
        loss.backward()
        self.assertIsNotNone(fs.grad)
        calls = []
        original = net.forward

        def tracked(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        net.forward = tracked
        sampled = prior.p_sample_loop_ddim(
            fs.shape, {"text_embed": fs.detach()}, timesteps=20, cond_scale=1.0
        )
        net.forward = original
        self.assertTrue(torch.isfinite(sampled).all())
        self.assertEqual(tuple(sampled.shape), tuple(fs.shape))
        self.assertEqual(len(calls), 20)


if __name__ == "__main__":
    unittest.main()
