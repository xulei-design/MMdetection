import cv2
import numpy as np
from typing import List, Tuple

from mmcv.transforms import BaseTransform, Compose

from mmdet.registry import TRANSFORMS


def get_transforms(version: str) -> List[Compose]:
    if version == 'color':
        transforms = [
            dict(type='AutoContrast'), dict(type='Brightness'), dict(type='Color'),
            dict(type='Contrast'), dict(type='Equalize'), dict(type='Invert4Mix'),
            dict(type='Posterize'), dict(type='Sharpness'), dict(type='Solarize'),
            dict(type='SolarizeAdd')
        ]
    elif version == 'geo':
        transforms = [
            dict(type='BgShearX'), dict(type='BgShearY'), dict(type='BgRotate'),
            dict(type='BgTranslateX'), dict(type='BgTranslateY'),
            dict(type='BBoxShearX'), dict(type='BBoxShearY'), dict(type='BBoxRotate'),
            dict(type='BBoxTranslateX'), dict(type='BBoxTranslateY'),
        ]
    elif version == 'oamix':
        transforms = [
            dict(type='AutoContrast'), dict(type='Brightness'), dict(type='Color'),
            dict(type='Contrast'), dict(type='Equalize'), dict(type='Invert4Mix'),
            dict(type='Posterize'), dict(type='Sharpness'),
            dict(type='BgShearX'), dict(type='BgShearY'), dict(type='BgRotate'),
            dict(type='BgTranslateX'), dict(type='BgTranslateY'),
            dict(type='BBoxShearX'), dict(type='BBoxShearY'), dict(type='BBoxRotate'),
            dict(type='BBoxTranslateX'), dict(type='BBoxTranslateY'),
        ]
    else:
        raise TypeError(f"Invalid version: {version}. Please add the version to the get_transforms function.")
    transforms = [Compose(transforms) for transforms in transforms]
    return transforms


def bbox_overlaps_np(bboxes1: np.ndarray, bboxes2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Calculate overlap between two set of bboxes.

    Args:
        bboxes1 (ndarray): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (ndarray): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        ndarray: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    if len(bboxes2) == 0:
        return np.zeros((bboxes1.shape[-2], 0))
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]

    if rows * cols == 0:
        return np.zeros(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    lt = np.maximum(bboxes1[..., :, None, :2],
                    bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = np.minimum(bboxes1[..., :, None, 2:],
                    bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    overlap = wh[..., 0] * wh[..., 1]

    union = area1[..., None] + area2[..., None, :] - overlap

    union = np.maximum(union, eps)
    ious = overlap / union
    return ious


@TRANSFORMS.register_module()
class OAMix(BaseTransform):
    r"""Data augmentation method in `Object-Aware Domain Generalization for Object Detection
    <https://arxiv.org/abs/2312.12133>`_.

    Refer to https://github.com/woojulee24/OA-DG for implementation details.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32])

    Modified Keys:

    - img
    - gt_bboxes

    Args:
        version (str): The version of the augmentation method.
            Defaults to 'oamix'.
        aug_prob_coeff (float): The coefficient of the augmentation probability.
            Defaults to 1.0.
        mixture_width (int): The number of augmentation operations in the mixture.
            Defaults to 3.
        mixture_depth (int): The depth of augmentation operations in the mixture.
            If mixture_depth is -1, the depth is randomly sampled from [1, 4].
            Defaults to -1.
        box_scale (tuple): The scale of the random bounding boxes.
            Defaults to (0.01, 0.1).
        box_ratio (tuple): The aspect ratio of the random bounding boxes.
            Defaults to (3, 1/3).
        sigma_ratio (float): The ratio of the sigma for the Gaussian blur.
            Defaults to 0.3.
        score_thresh (float): The threshold of the saliency score.
            Defaults to 10.
    """
    def __init__(self,
                 version: str = "oamix",
                 aug_prob_coeff: float = 1.0,
                 mixture_width: int = 3,
                 mixture_depth: int = -1,
                 box_scale: tuple = (0.01, 0.1),
                 box_ratio: tuple = (3, 0.33),
                 sigma_ratio: float = 0.3,
                 score_thresh: float = 10.0) -> None:
        assert version in ['color', 'geo', 'oamix'], "The version should be either 'color', 'geo', or 'oamix'." \
                                                     "Please add the version to the get_transforms function."
        assert aug_prob_coeff > 0, "The augmentation probability coefficient should be greater than 0."
        assert isinstance(mixture_width, int) and mixture_width > 0, "The mixture width should be greater than 0."
        assert isinstance(mixture_depth, int) and mixture_depth >= -1, "The mixture depth should be greater than or equal to -1."
        assert isinstance(box_scale, tuple) and len(box_scale) == 2, "The box scale should be a tuple of 2 elements."
        assert isinstance(box_ratio, tuple) and len(box_ratio) == 2, "The box ratio should be a tuple of 2 elements."
        assert 0 <= sigma_ratio <= 1, "The sigma ratio should be in the range [0, 1]."
        assert score_thresh >= 0, "The score threshold should be greater than or equal to 0."
        super(OAMix, self).__init__()

        self.version = version
        self.transforms = get_transforms(version)
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.box_scale = box_scale
        self.box_ratio = box_ratio
        self.sigma_ratio = sigma_ratio
        self.score_thresh = score_thresh

    def transform(self, results) -> dict:
        """ The transform function. """
        img = results['img']
        gt_bboxes = results['gt_bboxes'].numpy()
        gt_masks = self.get_masks(gt_bboxes, img.shape, use_blur=True)

        results['img'] = self.oamix(img.copy(), gt_bboxes, gt_masks)

        return results

    def oamix(self, img_orig: np.ndarray, gt_bboxes: np.ndarray, gt_masks: List[np.ndarray]) -> np.ndarray:
        ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        img_mix = np.zeros_like(img_orig, dtype=np.float32)
        for i in range(self.mixture_width):
            """ Multi-level transformation """
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            img_aug = img_orig.copy()
            for _ in range(depth):
                img_aug = self.multilivel_transform(img_aug, gt_bboxes, gt_masks)
            img_mix += ws[i] * img_aug

        """ Object-aware mixing """
        img_oamix = self.object_aware_mixing(img_orig, img_mix, gt_bboxes, gt_masks)

        return np.asarray(img_oamix, dtype=img_orig.dtype)

    def multilivel_transform(self, img: np.ndarray, gt_bboxes: np.ndarray, gt_masks: List[np.ndarray]) -> np.ndarray:
        rand_bboxes = self.get_random_bboxes(img.shape, num_bboxes=(1, 3))
        rand_masks = self.get_masks(rand_bboxes, img.shape)

        img_tmp = np.zeros_like(img, dtype=np.float32)
        for rand_mask in rand_masks:
            img_tmp += rand_mask * self.aug(img, gt_bboxes, gt_masks)
        union_mask = np.max(rand_masks, axis=0)
        img_aug = np.asarray(
            img_tmp + (1.0 - union_mask) * self.aug(img, gt_bboxes, gt_masks), dtype=img.dtype
        )
        return img_aug

    def get_random_bboxes(self, img_shape: Tuple, num_bboxes: Tuple[int, int],
                          max_iters: int = 50, eps: float = 1e-6) -> np.ndarray:
        assert max_iters > 0, "The maximum number of iterations should be greater than 0."

        h_img, w_img, _ = img_shape
        num_target_bboxes = np.random.randint(*num_bboxes)
        rand_bboxes = np.zeros((0, 4))
        for i in range(max_iters):
            if len(rand_bboxes) >= num_target_bboxes:
                break
            scale = np.random.uniform(*self.box_scale)
            aspect_ratio = np.random.uniform(*self.box_ratio)

            height = scale * h_img
            width = height * aspect_ratio
            if width > w_img or height > h_img:
                continue # Invalid bbox (out of the image)

            xmin = np.random.uniform(0, w_img - width)
            ymin = np.random.uniform(0, h_img - height)
            xmax = xmin + width
            ymax = ymin + height

            rand_bbox = np.array([[xmin, ymin, xmax, ymax]])
            ious = bbox_overlaps_np(rand_bbox, rand_bboxes)
            if np.sum(ious) > eps:
                continue # Invalid bbox (overlapping with existing bboxes)

            rand_bboxes = np.concatenate([rand_bboxes, rand_bbox], axis=0)

        return rand_bboxes

    def get_masks(self, bboxes: np.ndarray, img_shape: tuple, use_blur: bool = False) -> List[np.ndarray]:
        """ Get the masks of the bounding boxes. """
        mask_list = []
        for bbox in bboxes:
            if len(bbox.shape) == 2 and bbox.shape[1] == 4:
                bbox = bbox[0]
            mask = np.zeros(img_shape, dtype=np.float32)
            mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1.0
            if use_blur:
                sigma_x = (bbox[2] - bbox[0]) * self.sigma_ratio / 3 * 2
                sigma_y = (bbox[3] - bbox[1]) * self.sigma_ratio / 3 * 2
                if not (sigma_x <= 0 or sigma_y <= 0):
                    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma_x.item(), sigmaY=sigma_y.item())
                mask = cv2.resize(mask, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_list.append(mask)
        return mask_list

    def aug(self, img: np.ndarray, gt_bboxes: np.ndarray, gt_masks: List[np.ndarray]):
        op = np.random.choice(self.transforms)
        op_kwargs = {'img': img, 'bboxes': gt_bboxes, 'masks': gt_masks, 'img_shape': img.shape}
        img_aug = op(op_kwargs)['img']
        return img_aug

    def object_aware_mixing(self,
                            img_orig: np.ndarray,
                            img_mix: np.ndarray,
                            gt_bboxes: np.ndarray,
                            gt_masks: List[np.ndarray]) -> np.ndarray:
        gt_scores = self.get_saliency_scores(img_orig, gt_bboxes)

        target_indices = gt_scores < self.score_thresh
        target_bboxes = gt_bboxes[target_indices]
        target_masks = [gt_masks[i] for i in np.where(target_indices)[0]]
        target_m = np.random.uniform(0.0, 0.5, len(target_bboxes)).astype(np.float32)

        rand_bboxes = self.get_random_bboxes(img_orig.shape, num_bboxes=(3, 5))
        rand_masks = self.get_masks(rand_bboxes, img_orig.shape, use_blur=True)
        rand_m = np.random.uniform(0.0, 1.0, len(rand_bboxes)).astype(np.float32)

        target_bboxes = np.vstack((target_bboxes, rand_bboxes))
        target_masks.extend(rand_masks)
        target_m = np.concatenate((target_m, rand_m))

        orig = np.zeros_like(img_orig, dtype=np.float32)
        aug = np.zeros_like(img_orig, dtype=np.float32)
        mask_sum = np.zeros_like(img_orig, dtype=np.float32)

        for bbox, mask, m in zip(target_bboxes, target_masks, target_m):
            mask_sum += mask
            mask_max = np.maximum(mask_sum, mask)
            mask_overlap = mask_sum - mask_max
            overlap_factor = (mask - mask_overlap * 0.5)

            orig += (1.0 - m) * img_orig * overlap_factor
            aug += m * img_mix * overlap_factor
            mask_sum = mask_max

        img_oamix = orig + aug

        m = np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff)

        img_oamix += (1.0 - m) * img_orig * (1.0 - mask_sum)
        img_oamix += m * img_mix * (1.0 - mask_sum)
        img_oamix = np.clip(img_oamix, 0, 255)

        return img_oamix

    @staticmethod
    def get_saliency_scores(img: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        saliency_scores = []
        for bbox in np.asarray(bboxes, dtype=np.int32):
            bbox_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliency_map) = saliency.computeSaliency(bbox_img)
            score = np.mean((saliency_map * 255).astype("uint8"))
            saliency_scores.append(score)
        return np.asarray(saliency_scores, dtype=np.uint8)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(version={self.version}, aug_prob_coeff={self.aug_prob_coeff}, ' \
                    f'mixture_width={self.mixture_width}, mixture_depth={self.mixture_depth}, ' \
                    f'box_scale={self.box_scale}, box_ratio={self.box_ratio}, ' \
                    f'sigma_ratio={self.sigma_ratio}, score_thresh={self.score_thresh})'
        return repr_str

    @staticmethod
    def _load_example_data() -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread("../demo/demo.jpg")
        gt_bboxes = np.array(
            [[609.2460327148438, 111.9759292602539, 635.9223022460938, 137.42437744140625],
             [480.33782958984375, 110.44952392578125, 521.1831665039062, 129.6164093017578],
             [295.34356689453125, 116.82196807861328, 379.7244873046875, 149.78955078125],
             [219.83975219726562, 177.6780548095703, 455.0238952636719, 382.3981628417969],
             [0.2255704551935196, 111.30818176269531, 62.27484130859375, 145.1354522705078],
             [191.24462890625, 108.73335266113281, 297.60186767578125, 155.57919311523438],
             [431.61810302734375, 105.31916046142578, 482.30120849609375, 132.21238708496094],
             [589.4951171875, 111.1348648071289, 616.8546752929688, 126.33065032958984],
             [167.97503662109375, 106.92251586914062, 211.0978546142578, 140.34495544433594],
             [270.0672912597656, 104.84465789794922, 326.28662109375, 128.14691162109375],
             [395.87152099609375, 111.33557891845703, 433.2410583496094, 132.824462890625],
             [60.50261306762695, 94.38719940185547, 85.39842224121094, 105.71919250488281],
             [373.8731384277344, 136.39341735839844, 434.0091857910156, 187.2471466064453],
             [141.07984924316406, 96.27764129638672, 166.647705078125, 105.06587982177734],
             [224.4158477783203, 97.63524627685547, 249.95281982421875, 107.63406372070312],
             [556.1692504882812, 110.58447265625, 588.9140014648438, 127.4863052368164],
             [77.04759979248047, 90.36402130126953, 97.74053192138672, 98.54667663574219]]
        )
        return img, gt_bboxes

    def _test_transformations(self) -> None:
        img, gt_bboxes = self._load_example_data()
        gt_masks = self.get_masks(gt_bboxes, img.shape, use_blur=True)

        transform_type_list = ['AutoContrast', 'Brightness', 'Color', 'Contrast', 'Equalize', 'Invert4Mix',
                               'Posterize', 'Sharpness', 'Solarize', 'SolarizeAdd',
                               'BgShearX', 'BgShearY', 'BgRotate', 'BgTranslateX', 'BgTranslateY',
                               'BBoxShearX', 'BBoxShearY', 'BBoxRotate', 'BBoxTranslateX', 'BBoxTranslateY']
        for type in transform_type_list:
            op = Compose(dict(type=type))
            op_kwargs = {'img': img, 'bboxes': gt_bboxes, 'masks': gt_masks, 'img_shape': img.shape}
            _ = op(op_kwargs)['img']

        return

    def _test_multilevel_transformations(self) -> None:
        img_orig, gt_bboxes = self._load_example_data()
        gt_masks = self.get_masks(gt_bboxes, img_orig.shape, use_blur=True)

        for idx in range(10):
            ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
            img_mix = np.zeros_like(img_orig, dtype=np.float32)
            for i in range(self.mixture_width):
                """ Multi-level transformation """
                depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
                img_aug = img_orig.copy()
                for _ in range(depth):
                    img_aug = self.multilivel_transform(img_aug, gt_bboxes, gt_masks)
                img_mix += ws[i] * img_aug

        return

    def _test_objectaware_mixing(self) -> None:
        img_orig, gt_bboxes = self._load_example_data()
        gt_masks = self.get_masks(gt_bboxes, img_orig.shape, use_blur=True)

        ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        img_mix = np.zeros_like(img_orig, dtype=np.float32)
        for i in range(self.mixture_width):
            """ Multi-level transformation """
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            img_aug = img_orig.copy()
            for _ in range(depth):
                img_aug = self.multilivel_transform(img_aug, gt_bboxes, gt_masks)
            img_mix += ws[i] * img_aug

        """ Object-aware mixing """
        img_oamix = self.object_aware_mixing(img_orig, img_mix, gt_bboxes, gt_masks)

        return
