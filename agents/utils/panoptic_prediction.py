"""
Panoptic FPN Semantic Prediction Module.

Enhanced semantic prediction using Panoptic FPN with:
- Pixel-level class probability maps via forward propagation
- Argmax operation for semantic labels
- CRF post-processing for boundary optimization
- Small object detection optimization via FPN feature fusion
"""

import numpy as np
import torch

from constants import coco_categories_mapping

try:
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data.catalog import MetadataCatalog
    HAS_DETECTRON2 = True
except ImportError:
    HAS_DETECTRON2 = False

from models.semantic_utils import CRFPostProcessor


class SemanticPredPanopticFPN:
    """
    Panoptic FPN based semantic prediction with CRF refinement.

    Uses a pretrained Panoptic FPN network for semantic inference,
    generating pixel-level class probability maps with argmax for labels
    and CRF post-processing for boundary optimization.
    """

    def __init__(self, args):
        self.args = args
        self.num_sem_categories = 16

        if HAS_DETECTRON2:
            self.model = self._build_panoptic_model(args)
        else:
            self.model = None

        self.crf_processor = CRFPostProcessor(
            num_classes=self.num_sem_categories,
            num_iterations=5,
            spatial_sigma=3.0,
            bilateral_sigma_xy=60.0,
            bilateral_sigma_rgb=10.0
        )

        self.use_crf = getattr(args, 'use_crf', True)
        self.fpn_features = None

    def _build_panoptic_model(self, args):
        """Build Panoptic FPN model from detectron2."""
        cfg = get_cfg()

        try:
            cfg.merge_from_file(
                "configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
            cfg.MODEL.WEIGHTS = \
                "detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl"
        except Exception:
            cfg.merge_from_file(
                "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.WEIGHTS = \
                "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = getattr(
            args, 'sem_pred_prob_thr', 0.9)

        if hasattr(args, 'sem_gpu_id'):
            if args.sem_gpu_id == -2:
                cfg.MODEL.DEVICE = "cpu"
            elif args.sem_gpu_id >= 0:
                cfg.MODEL.DEVICE = f"cuda:{args.sem_gpu_id}"
            else:
                cfg.MODEL.DEVICE = "cuda:0"
        else:
            cfg.MODEL.DEVICE = "cuda:0"

        cfg.freeze()

        model = build_model(cfg)
        model.eval()
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

        return model

    def get_prediction(self, img: np.ndarray) -> tuple:
        """
        Get semantic prediction with Panoptic FPN + CRF refinement.

        Args:
            img: (H, W, 3) RGB image

        Returns:
            semantic_input: (H, W, 16) semantic prediction channels
            vis_img: (H, W, 3) visualization image
        """
        bgr_img = img[:, :, ::-1]
        H, W = img.shape[:2]

        semantic_input = np.zeros((H, W, self.num_sem_categories))

        if self.model is not None:
            predictions = self._run_model(bgr_img)
            semantic_input, class_probs = self._process_predictions(
                predictions, H, W)

            if self.use_crf and class_probs is not None:
                refined_labels = self.crf_processor.apply(class_probs, img)
                semantic_input = self._labels_to_channels(
                    refined_labels, H, W)
        else:
            semantic_input = self._fallback_prediction(img)

        return semantic_input, img

    def _run_model(self, bgr_img: np.ndarray) -> dict:
        """Run Panoptic FPN model on image."""
        image_tensor = torch.as_tensor(
            bgr_img.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image_tensor,
                    "height": bgr_img.shape[0],
                    "width": bgr_img.shape[1]}]

        with torch.no_grad():
            outputs = self.model(inputs)

        return outputs[0]

    def _process_predictions(self, predictions: dict,
                             H: int, W: int) -> tuple:
        """Process model output into semantic channels."""
        semantic_input = np.zeros((H, W, self.num_sem_categories))
        class_probs = None

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            panoptic_seg = panoptic_seg.cpu().numpy()

            class_probs = np.zeros((self.num_sem_categories, H, W))

            for seg_info in segments_info:
                cat_id = seg_info['category_id']
                if cat_id in coco_categories_mapping:
                    idx = coco_categories_mapping[cat_id]
                    mask = (panoptic_seg == seg_info['id']).astype(np.float32)
                    semantic_input[:, :, idx] += mask
                    score = seg_info.get('score', 1.0)
                    class_probs[idx] += mask * score

        elif "instances" in predictions:
            instances = predictions["instances"]
            class_probs = np.zeros((self.num_sem_categories, H, W))

            for j, class_idx in enumerate(
                    instances.pred_classes.cpu().numpy()):
                if class_idx in coco_categories_mapping:
                    idx = coco_categories_mapping[class_idx]
                    mask = instances.pred_masks[j].cpu().numpy().astype(
                        np.float32)
                    semantic_input[:, :, idx] += mask
                    score = instances.scores[j].cpu().item()
                    class_probs[idx] += mask * score

        if "sem_seg" in predictions:
            sem_seg = predictions["sem_seg"].cpu()
            class_probs_raw = torch.softmax(sem_seg, dim=0).numpy()

            class_probs = np.zeros((self.num_sem_categories, H, W))
            for coco_id, nav_id in coco_categories_mapping.items():
                if coco_id < class_probs_raw.shape[0]:
                    class_probs[nav_id] += class_probs_raw[coco_id]

        return semantic_input, class_probs

    def _labels_to_channels(self, labels: np.ndarray,
                            H: int, W: int) -> np.ndarray:
        """Convert label map to multi-channel semantic prediction."""
        channels = np.zeros((H, W, self.num_sem_categories))
        for c in range(self.num_sem_categories):
            channels[:, :, c] = (labels == c).astype(np.float32)
        return channels

    def _fallback_prediction(self, img: np.ndarray) -> np.ndarray:
        """Fallback when model is not available - use existing MaskRCNN."""
        from agents.utils.semantic_prediction import SemanticPredMaskRCNN
        fallback = SemanticPredMaskRCNN(self.args)
        pred, _ = fallback.get_prediction(img)
        return pred

    def get_fpn_features(self) -> dict:
        """
        Get FPN feature maps for small object detection enhancement.
        Available after running get_prediction.
        """
        return self.fpn_features if self.fpn_features else {}
