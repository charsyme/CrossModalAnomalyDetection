import os

import cv2
import numpy as np
import torch


class DepthAnythingV2Estimator:
    """
    Depth Anything V2 inference wrapper.

    Modes
    -----
    relative:
        Relative / non-metric monocular depth.

    metric:
        Metric monocular depth in metres.

    Encoders
    --------
    vits
    vitb
    vitl

    Metric datasets
    ---------------
    hypersim:
        Indoor metric model, max depth = 20 m.

    vkitti:
        Outdoor metric model, max depth = 80 m.
    """

    MODEL_CONFIGS = {
        'vits': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384],
        },

        'vitb': {
            'encoder': 'vitb',
            'features': 128,
            'out_channels': [96, 192, 384, 768],
        },

        'vitl': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024],
        },
    }

    def __init__(
        self,
        mode='relative',
        encoder='vits',
        metric_dataset='hypersim',
        input_size=518,
        checkpoint_dir=None,
        device=None,
    ):
        # ====================================================
        # Configuration
        # ====================================================

        self.mode = str(mode).lower()
        self.encoder = str(encoder).lower()
        self.metric_dataset = str(metric_dataset).lower()
        self.input_size = int(input_size)

        if self.mode not in (
            'relative',
            'metric',
        ):
            raise ValueError(
                "mode must be either "
                "'relative' or 'metric'"
            )

        if self.encoder not in self.MODEL_CONFIGS:
            raise ValueError(
                f'Unsupported encoder: {self.encoder}. '
                f'Supported encoders: '
                f'{list(self.MODEL_CONFIGS.keys())}'
            )

        if (
            self.mode == 'metric'
            and self.metric_dataset not in (
                'hypersim',
                'vkitti',
            )
        ):
            raise ValueError(
                "metric_dataset must be either "
                "'hypersim' or 'vkitti'"
            )

        self.is_metric = (
            self.mode == 'metric'
        )

        # ====================================================
        # Device
        # ====================================================

        if device is None:

            if torch.cuda.is_available():
                device = 'cuda:0'

            elif (
                hasattr(torch.backends, 'mps')
                and torch.backends.mps.is_available()
            ):
                device = 'mps'

            else:
                device = 'cpu'

        self.device = torch.device(
            device
        )

        print(
            f'Depth Anything V2 device: '
            f'{self.device}'
        )

        # ====================================================
        # Checkpoint directory
        # ====================================================

        if checkpoint_dir is None:

            package_dir = os.path.dirname(
                os.path.abspath(__file__)
            )

            checkpoint_dir = os.path.join(
                package_dir,
                'checkpoints'
            )

        self.checkpoint_dir = os.path.abspath(
            checkpoint_dir
        )

        # ====================================================
        # Load model
        # ====================================================

        self.model = self._load_model()

    # ========================================================
    # Model loading
    # ========================================================

    def _load_model(self):

        import sys

        config = self.MODEL_CONFIGS[
            self.encoder
        ].copy()

        repo_dir = os.path.expanduser(
            '~/ros2_jazzy/src/Depth-Anything-V2'
        )

        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(
                f'Depth Anything V2 repository not found: '
                f'{repo_dir}'
            )

        # ====================================================
        # Relative model
        # ====================================================

        if self.mode == 'relative':

            import_root = repo_dir

            checkpoint_name = (
                f'depth_anything_v2_'
                f'{self.encoder}.pth'
            )

            self.max_depth = None

        # ====================================================
        # Metric model
        # ====================================================

        else:

            import_root = os.path.join(
                repo_dir,
                'metric_depth'
            )

            if self.metric_dataset == 'hypersim':

                self.max_depth = 20.0

            elif self.metric_dataset == 'vkitti':

                self.max_depth = 80.0

            else:

                raise ValueError(
                    f'Unsupported metric dataset: '
                    f'{self.metric_dataset}'
                )

            checkpoint_name = (
                f'depth_anything_v2_metric_'
                f'{self.metric_dataset}_'
                f'{self.encoder}.pth'
            )

        # ====================================================
        # Import official Depth Anything package
        # ====================================================

        if import_root not in sys.path:
            sys.path.insert(
                0,
                import_root
            )

        from depth_anything_v2.dpt import (
            DepthAnythingV2
        )

        # ====================================================
        # Create model
        # ====================================================

        if self.mode == 'relative':

            model = DepthAnythingV2(
                **config
            )

        else:

            model = DepthAnythingV2(
                **config,
                max_depth=self.max_depth
            )

        # ====================================================
        # Load checkpoint
        # ====================================================

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            checkpoint_name
        )

        if not os.path.isfile(
                checkpoint_path
        ):
            raise FileNotFoundError(
                f'Checkpoint not found: '
                f'{checkpoint_path}'
            )

        print(
            f'Depth Anything repository: '
            f'{import_root}'
        )

        print(
            f'Loading checkpoint: '
            f'{checkpoint_path}'
        )

        state_dict = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=True
        )

        model.load_state_dict(
            state_dict
        )

        model = model.to(
            self.device
        )

        model.eval()

        return model

    # ========================================================
    # Inference
    # ========================================================

    @torch.inference_mode()
    def predict(
        self,
        image,
        input_format='bgr',
    ):
        """
        Run Depth Anything V2 inference.

        Parameters
        ----------
        image:
            NumPy H x W x 3 uint8 image.

        input_format:
            'bgr' or 'rgb'.

        Returns
        -------
        depth:
            H x W float32 NumPy array.

            Relative mode:
                Relative/non-metric depth values.

            Metric mode:
                Metric camera-axis depth in metres.
        """

        if image is None:
            raise ValueError(
                'Input image is None'
            )

        image = np.asarray(
            image
        )

        if image.ndim != 3:
            raise ValueError(
                f'Expected HxWx3 image, '
                f'got shape {image.shape}'
            )

        if image.shape[2] != 3:
            raise ValueError(
                f'Expected three image channels, '
                f'got shape {image.shape}'
            )

        # ====================================================
        # Input format
        # ====================================================

        input_format = str(
            input_format
        ).lower()

        if input_format == 'rgb':

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

        elif input_format != 'bgr':

            raise ValueError(
                "input_format must be "
                "'bgr' or 'rgb'"
            )

        image = np.ascontiguousarray(
            image,
            dtype=np.uint8
        )

        # ====================================================
        # Depth inference
        # ====================================================

        depth = self.model.infer_image(
            image,
            input_size=self.input_size
        )

        depth = np.asarray(
            depth,
            dtype=np.float32
        )

        if depth.ndim != 2:
            raise RuntimeError(
                'Depth Anything returned an '
                f'unexpected shape: {depth.shape}'
            )

        if not np.any(
            np.isfinite(depth)
        ):
            raise RuntimeError(
                'Depth Anything returned no '
                'finite depth values'
            )

        return np.ascontiguousarray(
            depth,
            dtype=np.float32
        )
