import numpy as np

from .common import CommonInpainter, OfflineInpainter
from .inpainting_aot import AotInpainter
from .inpainting_lama_mpe import LamaMPEInpainter
from .inpainting_sd import StableDiffusionInpainter
from .none import NoneInpainter
from .original import OriginalInpainter

INPAINTERS = {
    'default': AotInpainter,
    'lama_mpe': LamaMPEInpainter,
    'sd': StableDiffusionInpainter,
    'none': NoneInpainter,
    'original': OriginalInpainter,
}
inpainter_cache = {}

def get_inpainter(key: str, *args, **kwargs) -> CommonInpainter:
    if key not in INPAINTERS:
        raise ValueError(f'Could not find inpainter for: "{key}". Choose from the following: %s' % ','.join(INPAINTERS))
    if not inpainter_cache.get(key):
        inpainter = INPAINTERS[key]
        inpainter_cache[key] = inpainter(*args, **kwargs)
    return inpainter_cache[key]

def prepare(inpainter_key: str, device: str = 'cpu'):
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        inpainter.download()
        inpainter.load(device)

def dispatch(inpainter_key: str, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, device: str = 'cpu', verbose: bool = False) -> np.ndarray:
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        inpainter.load(device)
    return inpainter.inpaint(image, mask, inpainting_size, verbose)
