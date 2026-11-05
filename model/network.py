"""Learned EEG decomposition and the sole F_s-conditioned generative route."""

import torch
from torch import nn
from .encoder import (
    REVEEncoder,
    VisualTokenResampler,
    SparseAwareVisionProjector,
    EEGReconDecoder,
    SubjClassifier,
    SubjDiscriminator,
)
from .decomposition import DSNLatentDecomposition


class EEGModel(nn.Module):
    def __init__(self, positions):
        super().__init__()
        self.eeg_encoder = REVEEncoder(
            reve_positions_path=str(positions), n_channels=63, reve_adaptation="lora"
        )
        self.decomposition = DSNLatentDecomposition(1216, dropout=0.0)
        self.visual_resampler = VisualTokenResampler(
            dim=1216, out_tokens=256, num_heads=19, depth=2, dropout=0.0
        )
        self.full_projection = SparseAwareVisionProjector(1216, 1664, 256)
        self.semantic_projection = SparseAwareVisionProjector(1216, 1664, 256)
        self.eeg_decoder = EEGReconDecoder()
        self.subject_classifier = SubjClassifier(1216, 63, 9)
        self.subject_discriminator = SubjDiscriminator(1216, 63, 9)

    def encode(self, eeg):
        e = self.eeg_encoder.forward_encoder(eeg)
        ei, es, reconstruction, zi, zs = self.decomposition(e)
        f = self.full_projection(self.visual_resampler(e))
        fs = self.semantic_projection(self.visual_resampler(ei))
        return dict(
            E=e,
            E_i=ei,
            E_s=es,
            F=f,
            F_s=fs,
            latent_reconstruction=reconstruction,
            z_i=zi,
            z_s=zs,
        )

    def condition(self, eeg):
        e = self.eeg_encoder.forward_encoder(eeg)
        ei, _ = self.decomposition.invariant_branch(e)
        return self.semantic_projection(self.visual_resampler(ei))

    def forward(self, eeg):
        out = self.encode(eeg)
        out.update(
            eeg_reconstruction=self.eeg_decoder(out["E"]),
            subject_logits=self.subject_classifier(out["E_s"]),
            subject_adversarial_logits=self.subject_discriminator(out["E_i"]),
        )
        return out


class ReconstructionModel(nn.Module):
    def __init__(self, representation, num_concepts, text_projection_checkpoint=None):
        super().__init__()
        from .heads import (
            ImageClassifier,
            ImageDiscriminator,
            CLIPProj,
            BlurryReconDecoder,
        )
        from .prior import PriorNetwork, BrainDiffusionPrior

        self.representation = representation
        # Preserve the representation learned in pretraining, as in ZEBRA downstream.
        for module in (
            representation.eeg_encoder,
            representation.decomposition,
            representation.eeg_decoder,
            representation.subject_classifier,
            representation.subject_discriminator,
        ):
            module.requires_grad_(False)
        self.image_classifier = ImageClassifier(1664, num_concepts)
        self.image_discriminator = ImageDiscriminator(1664, num_concepts)
        self.text_projection = CLIPProj()
        if text_projection_checkpoint is not None:
            self.text_projection.load_state_dict(
                torch.load(
                    text_projection_checkpoint, map_location="cpu", weights_only=True
                )
            )
        self.blurry_decoder = BlurryReconDecoder()
        self.diffusion_prior = BrainDiffusionPrior(
            net=PriorNetwork(
                dim=1664,
                depth=6,
                dim_head=52,
                heads=32,
                causal=False,
                num_tokens=256,
                learned_query_mode="pos_emb",
            ),
            image_embed_dim=1664,
            condition_on_text_encodings=False,
            timesteps=100,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )

    def train(self, mode=True):
        super().train(mode)
        for module in (
            self.representation.eeg_encoder,
            self.representation.decomposition,
            self.representation.eeg_decoder,
            self.representation.subject_classifier,
            self.representation.subject_discriminator,
        ):
            module.eval()
        return self

    def condition(self, eeg):
        return self.representation.condition(eeg)

    def forward(self, eeg, target):
        with torch.no_grad():
            e = self.representation.eeg_encoder.forward_encoder(eeg)
            ei, _ = self.representation.decomposition.invariant_branch(e)
        out = {
            "F": self.representation.full_projection(
                self.representation.visual_resampler(e)
            ),
            "F_s": self.representation.semantic_projection(
                self.representation.visual_resampler(ei)
            ),
        }
        prior_loss, prediction = self.diffusion_prior(
            text_embed=out["F_s"], image_embed=target
        )
        out.update(
            prior_loss=prior_loss,
            prior_prediction=prediction,
            image_logits_all=self.image_classifier(out["F"]),
            image_logits=self.image_classifier(out["F_s"]),
            image_adversarial_logits=self.image_discriminator(out["F"] - out["F_s"]),
            text_prediction=self.text_projection(out["F"]),
            blurry_latent=self.blurry_decoder(prediction),
        )
        return out


class SemanticGenerator(nn.Module):
    """Inference contains only REVE, the shared branch, F_s projection and prior."""

    def __init__(self, positions):
        super().__init__()
        from .decomposition import DSNLatentBranch
        from .prior import PriorNetwork, BrainDiffusionPrior

        self.eeg_encoder = REVEEncoder(
            reve_positions_path=str(positions), n_channels=63, reve_adaptation="lora"
        )
        self.invariant_branch = DSNLatentBranch(1216, 512, 2, 8, 0.0)
        self.visual_resampler = VisualTokenResampler(
            dim=1216, out_tokens=256, num_heads=19, depth=2, dropout=0.0
        )
        self.semantic_projection = SparseAwareVisionProjector(1216, 1664, 256)
        self.diffusion_prior = BrainDiffusionPrior(
            net=PriorNetwork(
                dim=1664,
                depth=6,
                dim_head=52,
                heads=32,
                causal=False,
                num_tokens=256,
                learned_query_mode="pos_emb",
            ),
            image_embed_dim=1664,
            condition_on_text_encodings=False,
            timesteps=100,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )

    def load_downstream_state(self, state):
        mapping = {
            "representation.eeg_encoder.": "eeg_encoder.",
            "representation.decomposition.invariant_branch.": "invariant_branch.",
            "representation.visual_resampler.": "visual_resampler.",
            "representation.semantic_projection.": "semantic_projection.",
            "diffusion_prior.": "diffusion_prior.",
        }
        selected = {}
        for key, value in state.items():
            for source, target in mapping.items():
                if key.startswith(source):
                    selected[target + key[len(source) :]] = value
                    break
        self.load_state_dict(selected, strict=True)

    def condition(self, eeg):
        ei, _ = self.invariant_branch(self.eeg_encoder.forward_encoder(eeg))
        return self.semantic_projection(self.visual_resampler(ei))
