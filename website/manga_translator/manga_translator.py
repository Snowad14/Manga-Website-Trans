from PIL import Image
import cv2
import numpy as np
import os
import torch
from typing import List
import time
import logging

from .utils import (
    BASE_PATH,
    LANGAUGE_ORIENTATION_PRESETS,
    ModelWrapper,
    TextBlock,
    Context,
    load_image,
    dump_image,
    replace_prefix,
    visualize_textblocks,
    add_file_logger,
    remove_file_logger,
    count_valuable_text,
)

from .detection import dispatch as dispatch_detection, prepare as prepare_detection
from .ocr import dispatch as dispatch_ocr, prepare as prepare_ocr
from .mask_refinement import dispatch as dispatch_mask_refinement
from .inpainting import dispatch as dispatch_inpainting, prepare as prepare_inpainting

# Will be overwritten by __main__.py if module is being run directly (with python -m)
logger = logging.getLogger('manga_translator')

DEFAULT_PARAMATERS = {
    "mode": "demo",
    "input": "13.jpg",
    "dest": "",
    "target_lang": "ENG",
    "verbose": False,
    "detector": "default",
    "ocr": "48px_ctc",
    "inpainter": "lama_mpe",
    "upscaler": "esrgan",
    "translator": "google",
    "translator_chain": None,
    "use_cuda": True,
    "use_cuda_limited": False,
    "model_dir": None,
    "detection_size": 1536,
    "detection_auto_orient": False,
    "det_rearrange_max_batches": 4,
    "inpainting_size": 2048,
    "unclip_ratio": 2.3,
    "box_threshold": 0.7,
    "text_threshold": 0.5,
    "text_mag_ratio": 1,
    "font_size_offset": 0,
    "font_size_minimum": -1,
    "force_horizontal": False,
    "force_vertical": False,
    "align_left": False,
    "align_center": False,
    "align_right": False,
    "upscale_ratio": None,
    "downscale": False,
    "manga2eng": False,
    "capitalize": False,
    "mtpe": False,
    "font_path": "",
    "host": "",
    "port": 5003,
    "nonce": "",
    "ws_url": "ws://localhost:5000",
    "enlarge_border": False
}



def set_main_logger(l):
    global logger
    logger = l

class MangaTranslator():

    def __init__(self, params: dict = None):

        self._progress_hooks = []
        self._add_logger_hook()

        params = params or {}
        self.verbose = params.get('verbose', False)
        self.ignore_errors = params.get('ignore_errors', False if params.get('mode', 'demo') == 'demo' else True)

        self.device = 'cuda' if params.get('use_cuda', False) else 'cpu'
        self._cuda_limited_memory = params.get('use_cuda_limited', False)
        if self._cuda_limited_memory and not self.using_cuda:
            self.device = 'cuda'
        if self.using_cuda and not torch.cuda.is_available():
            raise Exception('CUDA compatible device could not be found whilst --use-cuda args was set...')

        self.result_sub_folder = ''

        print("Loading Models")
        prepare_detection(params.get("detector"))
        prepare_ocr(params.get("ocr"), self.device)
        prepare_inpainting(params.get("inpainter"), self.device)
        print("Models Loaded")



    @property
    def using_cuda(self):
        return self.device.startswith('cuda')

    def translate(self, image: Image.Image, params: dict = None) -> Image.Image:
        params = params or {}
        params = Context(**params)

        if 'direction' not in params:
            if params.force_horizontal:
                params.direction = 'h'
            elif params.force_vertical:
                params.direction = 'v'
            else:
                params.direction = 'auto'
        if 'alignment' not in params:
            if params.align_left:
                params.alignment = 'left'
            elif params.align_center:
                params.alignment = 'center'
            elif params.align_right:
                params.alignment = 'right'
            else:
                params.alignment = 'auto'
        params.setdefault('renderer', 'manga2eng' if params['manga2eng'] else 'default')

        return self._translate(image, params)

    def _translate(self, image: Image.Image, params: Context) -> Image.Image:
        # TODO: Split up into self sufficient functions that call what they need automatically

        # The default text detector doesn't work very well on smaller images, might want to
        # consider adding automatic upscaling on certain kinds of small images.
        if params.upscale_ratio:
            self._report_progress('upscaling')
            image_upscaled = (self._run_upscaling(params.upscaler, [image], params.upscale_ratio))[0]
        else:
            image_upscaled = image

        img_rgb, img_alpha = load_image(image_upscaled)

        self._report_progress('detection')
        text_regions, mask_raw, mask = self._run_detection(params.detector, img_rgb, params.detection_size, params.text_threshold,
                                                                 params.box_threshold, params.unclip_ratio, params.det_rearrange_max_batches,
                                                                 params.detection_auto_orient)
        if self.verbose:
            cv2.imwrite(self._result_path('mask_raw.png'), mask_raw)
            bboxes = visualize_textblocks(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB), text_regions)
            cv2.imwrite(self._result_path('bboxes.png'), bboxes)

        if not text_regions:
            self._report_progress('skip-no-regions', True)
            return image

        self._report_progress('ocr')
        text_regions = self._run_ocr(params.ocr, img_rgb, text_regions)

        if not text_regions:
            self._report_progress('skip-no-text', True)
            return image

        # Delayed mask refinement to take advantage of the region filtering done by ocr
        if mask is None:
            self._report_progress('mask-generation')
            mask = self._run_mask_refinement(text_regions, img_rgb, mask_raw)

        if self.verbose:
            inpaint_input_img = self._run_inpainting('none', img_rgb, mask)
            cv2.imwrite(self._result_path('inpaint_input.png'), cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('mask_final.png'), mask)

        self._report_progress('inpainting')
        img_inpainted = self._run_inpainting(params.inpainter, img_rgb, mask, params.inpainting_size)
        # cv2.imshow("inpainted", img_inpainted)
        # cv2.waitKey(0)

        final_regions = []

        for region in text_regions:
            Fulltext = " ".join(region.text)
            Coord = region.xyxy
            if params.enlarge_border:
                from .enlarger import Rectangle
                onlyRegions = [region.xyxy for region in text_regions]
                newRect = Rectangle.expand_rectangle(Image.fromarray(img_inpainted), Rectangle(Coord), onlyRegions)
                Coord = newRect.area
            final_regions.append((Coord, Fulltext))

        return img_inpainted, final_regions




    def _result_path(self, path: str) -> str:
        return os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)

    def _report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            ph(state, finished)

    def _add_logger_hook(self):
        LOG_MESSAGES = {
            'upscaling':            'Running upscaling',
            'detection':            'Running text detection',
            'ocr':                  'Running OCR',
            'mask-generation':      'Running mask refinement',
            'translating':          'Translating',
            'rendering':            'Rendering translated text',
            'downscaling':          'Running downscaling',
            'saved':                'Saving results',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-regions':      'No text regions! - Skipping',
            'skip-no-text':         'No text regions with text! - Skipping',
        }
        LOG_MESSAGES_ERROR = {
            'error-translating':    'Text translator returned empty queries',
            # 'error-lang':           'Target language not supported by chosen translator',
        }

        def ph(state, finished):
            if state in LOG_MESSAGES:
                logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                logger.warn(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)

    def _run_detection(self, key: str, img: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                             unclip_ratio: float, det_rearrange_max_batches: int, auto_orient: bool):
        return dispatch_detection(key, img, detect_size, text_threshold, box_threshold, unclip_ratio, det_rearrange_max_batches,
                                        auto_orient, self.device, self.verbose)

    def _run_ocr(self, key: str, img: np.ndarray, text_regions: List[TextBlock]):
        text_regions = dispatch_ocr(key, img, text_regions, self.device, self.verbose)

        # Filter regions by their text
        text_regions = list(filter(lambda r: count_valuable_text(r.get_text()) > 1 and not r.get_text().isnumeric(), text_regions))
        return text_regions

    def _run_mask_refinement(self, text_regions: List[TextBlock], raw_image: np.ndarray, raw_mask: np.ndarray, method: str = 'fit_text'):
        return dispatch_mask_refinement(text_regions, raw_image, raw_mask, method, self.verbose)

    def _run_inpainting(self, key: str, img: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024):
        return dispatch_inpainting(key, img, mask, inpainting_size, self.using_cuda, self.verbose)


    # # def _run_text_rendering(self, key: str, img: np.ndarray, text_regions: List[TextBlock], text_mag_ratio: np.integer,
    # #                               text_direction: str = 'auto', font_path: str = '', font_size_offset: int = 0, font_size_minimum: int = 0,
    # #                               original_img: np.ndarray = None, mask: np.ndarray = None, rearrange_regions: bool = False):
    # #     # manga2eng currently only supports horizontal rendering
    # #     if key == 'manga2eng' and text_regions and LANGAUGE_ORIENTATION_PRESETS.get(text_regions[0].target_lang) == 'h':
    # #         output = dispatch_eng_render(img, original_img, text_regions, font_path)
    # #     else:
    # #         output = dispatch_rendering(img, text_regions, text_mag_ratio, font_path, font_size_offset, font_size_minimum, rearrange_regions, mask)
    # #     return output
