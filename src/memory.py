import random

import torch


class PrototypeMemory:
    def __init__(
        self,
        num_classes,
        hidden_size,
        device,
        num_centroids=1,
        prototype_std_scale=0.5,
        min_points_per_centroid=8,
    ):
        self.prototypes = {}
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.device = device
        self.num_centroids = max(1, int(num_centroids))
        self.prototype_std_scale = max(0.0, float(prototype_std_scale))
        self.min_points_per_centroid = max(1, int(min_points_per_centroid))


    def _run_kmeans(self, features, num_centroids, num_iters=10):
        num_points = features.size(0)
        k = min(num_centroids, num_points)
        if k <= 1:
            assignments = torch.zeros(num_points, dtype=torch.long, device=features.device)
            return features[:1].clone(), assignments

        init_indices = torch.randperm(num_points, device=features.device)[:k]
        centroids = features[init_indices].clone()

        for _ in range(num_iters):
            distances = torch.cdist(features, centroids)
            assignments = distances.argmin(dim=1)

            new_centroids = []
            for centroid_idx in range(k):
                cluster_points = features[assignments == centroid_idx]
                if cluster_points.numel() == 0:
                    refill_idx = torch.randint(0, num_points, (1,), device=features.device)
                    new_centroids.append(features[refill_idx].squeeze(0))
                else:
                    new_centroids.append(cluster_points.mean(dim=0))

            new_centroids = torch.stack(new_centroids, dim=0)
            if torch.allclose(new_centroids, centroids, atol=1e-4, rtol=1e-4):
                centroids = new_centroids
                break
            centroids = new_centroids

        distances = torch.cdist(features, centroids)
        assignments = distances.argmin(dim=1)
        return centroids, assignments

    def _build_class_prototypes(self, feats):
        feats = feats.to(self.device)
        class_mean = feats.mean(dim=0)
        class_var = torch.clamp((feats - class_mean).pow(2).mean(dim=0), min=1e-6)
        class_std = torch.sqrt(class_var)

        max_centroids_by_count = max(1, feats.size(0) // self.min_points_per_centroid)
        effective_num_centroids = min(self.num_centroids, max_centroids_by_count)
        centroids, assignments = self._run_kmeans(feats, effective_num_centroids)
        means, stds, weights = [], [], []

        for centroid_idx in range(centroids.size(0)):
            cluster_points = feats[assignments == centroid_idx]
            if cluster_points.numel() == 0:
                means.append(centroids[centroid_idx])
                stds.append(class_std)
                weights.append(1.0)
                continue

            cluster_mean = cluster_points.mean(dim=0)
            if cluster_points.size(0) < self.min_points_per_centroid:
                cluster_std = class_std
            else:
                cluster_var = torch.clamp(
                    (cluster_points - cluster_mean).pow(2).mean(dim=0),
                    min=1e-6,
                )
                cluster_std = torch.sqrt(cluster_var)

            means.append(cluster_mean)
            stds.append(cluster_std)
            weights.append(float(cluster_points.size(0)))

        means = torch.stack(means, dim=0)
        stds = torch.stack(stds, dim=0)
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-8)

        return {
            "means": means,
            "stds": stds,
            "weights": weights,
            "class_mean": class_mean,
            "mean": means[0],
            "std": stds[0],
        }

    def update_prototypes(self, model, loader, device):
        model.eval()
        class_features = {}
        activation = {}

        def get_activation(name):
            def hook(_module, input_tensor, _output):
                activation[name] = input_tensor[0].detach()

            return hook

        target_layer = (
            model.classifier[0]
            if isinstance(model.classifier, torch.nn.Sequential)
            else model.classifier
        )
        handle = target_layer.register_forward_hook(get_activation("feat"))

        try:
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    _ = model(input_ids, mask)

                    feats = activation.get("feat")
                    if feats is None:
                        continue

                    for feat, label in zip(feats, labels):
                        class_features.setdefault(label.item(), []).append(feat.detach())
        finally:
            handle.remove()

        for label, feat_list in class_features.items():
            feats = torch.stack(feat_list, dim=0)
            self.prototypes[label] = self._build_class_prototypes(feats)


        total_centroids = sum(
            prototype["means"].size(0) for prototype in self.prototypes.values()
        )
        print(
            f"[Memory] prototypes updated: "
            f"classes={len(self.prototypes)}, centroids={total_centroids}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



    def class_mean(self, label):
        proto = self.prototypes[label]
        if "class_mean" in proto:
            return proto["class_mean"]
        means = proto.get("means")
        weights = proto.get("weights")
        if means is not None and weights is not None:
            return (means * weights[:, None]).sum(dim=0)
        return proto["mean"]

    def _sample_feature_for_label(self, label, feature_mode="gaussian"):
        proto = self.prototypes[label]
        if feature_mode == "mean":
            return self.class_mean(label).clone()
        if feature_mode != "gaussian":
            raise ValueError(
                f"Unsupported prototype feature_mode={feature_mode!r}"
            )
        centroid_idx = torch.multinomial(proto["weights"], num_samples=1).item()
        mean = proto["means"][centroid_idx]
        std = proto["stds"][centroid_idx] * self.prototype_std_scale
        noise = torch.randn_like(mean)
        return mean + noise * std

    def get_prototype_batch(self, batch_size=32, feature_mode="gaussian"):
        if len(self.prototypes) == 0:
            return None, None

        labels = list(self.prototypes.keys())
        sampled_labels = [random.choice(labels) for _ in range(batch_size)]

        batch_feats = [
            self._sample_feature_for_label(label, feature_mode=feature_mode)
            for label in sampled_labels
        ]

        batch_feats = torch.stack(batch_feats, dim=0)
        batch_labels = torch.tensor(sampled_labels, device=self.device)
        return batch_feats, batch_labels
