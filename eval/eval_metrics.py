import torch
import numpy as np
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
import scipy as sp
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights


@torch.no_grad()
def _batched_features(
    images,
    model,
    preprocess,
    feature_layer=None,
    device="cuda",
    batch_size=32,
):
    features = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(
            [preprocess(image) for image in images[start : start + batch_size]],
            dim=0,
        ).to(device)
        output = model(batch)
        if feature_layer is not None:
            output = output[feature_layer]
        features.append(output.float().flatten(1).cpu())
    return torch.cat(features)


@torch.no_grad()
def two_way_identification(
    all_recons,
    all_images,
    model,
    preprocess,
    feature_layer=None,
    return_avg=True,
    device="cuda",
    batch_size=32,
):
    preds = _batched_features(
        all_recons, model, preprocess, feature_layer, device, batch_size
    )
    reals = _batched_features(
        all_images, model, preprocess, feature_layer, device, batch_size
    )

    if len(reals) < 2:
        raise ValueError("Two-way identification requires at least two images.")
    correlation_device = torch.device(device)
    reals = reals.float()
    reals = reals - reals.mean(dim=1, keepdim=True)
    reals = torch.nn.functional.normalize(reals, dim=1).to(correlation_device)
    success_chunks = []
    for start in range(0, len(preds), batch_size):
        stop = min(start + batch_size, len(preds))
        predicted = preds[start:stop].float()
        predicted = predicted - predicted.mean(dim=1, keepdim=True)
        predicted = torch.nn.functional.normalize(predicted, dim=1).to(
            correlation_device
        )
        correlations = reals @ predicted.T
        local_indices = torch.arange(stop - start, device=correlation_device)
        congruents = correlations[start + local_indices, local_indices]
        better_than_others = correlations < congruents.unsqueeze(0)
        better_than_others[start + local_indices, local_indices] = False
        success_chunks.append(better_than_others.sum(dim=0).cpu())
    success_cnt = torch.cat(success_chunks).numpy()

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images) - 1)
        return perf
    else:
        return success_cnt, len(all_images) - 1


def cal_pixcorr(all_recons, all_images, batch_size=32):
    preprocess = transforms.Compose(
        [
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
    )

    corrsum = 0
    for start in range(0, len(all_images), batch_size):
        images = (
            preprocess(all_images[start : start + batch_size])
            .reshape(len(all_images[start : start + batch_size]), -1)
            .cpu()
        )
        recons = (
            preprocess(all_recons[start : start + batch_size])
            .reshape(len(all_recons[start : start + batch_size]), -1)
            .cpu()
        )
        for image, recon in zip(images, recons):
            if recon.max() == 0:
                continue
            corrsum += np.corrcoef(image, recon)[0][1]
    corrmean = corrsum / len(all_images)
    pixcorr = corrmean
    return pixcorr


def cal_ssim(all_recons, all_images, batch_size=32):
    # see https://github.com/zijin-gu/meshconv-decoding/issues/3
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim

    preprocess = transforms.Compose(
        [
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
    )

    ssim_score = []
    for start in range(0, len(all_images), batch_size):
        img_gray = rgb2gray(
            preprocess(all_images[start : start + batch_size])
            .permute((0, 2, 3, 1))
            .cpu()
        )
        recon_gray = rgb2gray(
            preprocess(all_recons[start : start + batch_size])
            .permute((0, 2, 3, 1))
            .cpu()
        )
        for im, rec in zip(img_gray, recon_gray):
            ssim_score.append(
                ssim(
                    rec,
                    im,
                    multichannel=True,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                    data_range=1.0,
                )
            )
    ssim = np.mean(ssim_score)
    return ssim


def cal_alexnet(all_recons, all_images, device="cuda", batch_size=32):
    alex_weights = AlexNet_Weights.IMAGENET1K_V1

    alex_model = create_feature_extractor(
        alexnet(weights=alex_weights), return_nodes=["features.4", "features.11"]
    ).to(device)
    alex_model.eval().requires_grad_(False)

    # see alex_weights.transforms()
    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # layer = 'early, AlexNet(2)'
    # print(f"\n---{layer}---")
    all_per_correct = two_way_identification(
        all_recons,
        all_images,
        alex_model,
        preprocess,
        "features.4",
        device=device,
        batch_size=batch_size,
    )
    alexnet2 = np.mean(all_per_correct)
    # print(f"2-way Percent Correct: {alexnet2:.4f}")

    # layer = 'mid, AlexNet(5)'
    # print(f"\n---{layer}---")
    all_per_correct = two_way_identification(
        all_recons,
        all_images,
        alex_model,
        preprocess,
        "features.11",
        device=device,
        batch_size=batch_size,
    )
    alexnet5 = np.mean(all_per_correct)
    # print(f"2-way Percent Correct: {alexnet5:.4f}")

    return alexnet2, alexnet5


def cal_inceptionv3(all_recons, all_images, device="cuda", batch_size=32):

    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(
        inception_v3(weights=weights), return_nodes=["avgpool"]
    ).to(device)
    inception_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose(
        [
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    all_per_correct = two_way_identification(
        all_recons,
        all_images,
        inception_model,
        preprocess,
        "avgpool",
        device=device,
        batch_size=batch_size,
    )

    inception_score = np.mean(all_per_correct)
    # print(f"2-way Percent Correct: {inception:.4f}")

    return inception_score


def cal_clip(all_recons, all_images, device="cuda", batch_size=32):
    import clip

    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    all_per_correct = two_way_identification(
        all_recons,
        all_images,
        clip_model.encode_image,
        preprocess,
        None,
        device=device,
        batch_size=batch_size,
    )
    clip_ = np.mean(all_per_correct)
    # print(f"2-way Percent Correct: {clip_:.4f}")
    return clip_


def cal_efficientnet(all_recons, all_images, device="cuda", batch_size=32):

    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(
        efficientnet_b1(weights=weights), return_nodes=["avgpool"]
    ).to(device)
    eff_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose(
        [
            transforms.Resize(432, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gt = _batched_features(
        all_images, eff_model, preprocess, "avgpool", device, batch_size
    ).numpy()
    fake = _batched_features(
        all_recons, eff_model, preprocess, "avgpool", device, batch_size
    ).numpy()

    effnet = np.array(
        [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
    ).mean()
    # print("Distance:", effnet)

    return effnet


def cal_swav(all_recons, all_images, device="cuda", batch_size=32):
    swav_model = torch.hub.load("facebookresearch/swav:main", "resnet50", verbose=False)
    swav_model = create_feature_extractor(swav_model, return_nodes=["avgpool"]).to(
        device
    )
    swav_model.eval().requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(240, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gt = _batched_features(
        all_images, swav_model, preprocess, "avgpool", device, batch_size
    ).numpy()
    fake = _batched_features(
        all_recons, swav_model, preprocess, "avgpool", device, batch_size
    ).numpy()

    swav = np.array(
        [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
    ).mean()
    return swav
