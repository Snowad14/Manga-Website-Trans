import cv2
import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon, MultiPoint
from functools import cached_property
import copy

from .general import color_difference
# from ..detection.ctd_utils.utils.imgproc_utils import union_area, xywh2xyxypoly

# LANG_LIST = ['eng', 'ja', 'unknown']
# LANGCLS2IDX = {'eng': 0, 'ja': 1, 'unknown': 2}

# determines render direction
LANGAUGE_ORIENTATION_PRESETS = {
    'CHS': 'auto',
    'CHT': 'auto',
    'CSY': 'h',
    'NLD': 'h',
    'ENG': 'h',
    'FRA': 'h',
    'DEU': 'h',
    'HUN': 'h',
    'ITA': 'h',
    'JPN': 'auto',
    'KOR': 'auto',
    'PLK': 'h',
    'PTB': 'h',
    'ROM': 'h',
    'RUS': 'h',
    'ESP': 'h',
    'TRK': 'h',
    'VIN': 'h',
}

class TextBlock(object):
    """
    Object that stores a block of text made up of textlines.
    """
    def __init__(self, lines: List,
                 text: List[str] = None,
                 language: str = 'unknown',
                 font_size: float = -1,
                 angle: int = 0,
                 translation: str = "",
                 fg_color: Tuple[float] = (0, 0, 0),
                 bg_color: Tuple[float] = (0, 0, 0),
                 line_spacing = 1.,
                 letter_spacing = 1.,
                 font_family: str = "",
                 bold: bool = False,
                 underline: bool = False,
                 italic: bool = False,
                 direction: str = 'auto',
                 alignment: str = 'auto',
                 rich_text: str = "",
                 _bounding_rect: List = None,
                 accumulate_color = True,
                 default_stroke_width = 0.2,
                 font_weight = 50,
                 target_lang: str = "",
                 opacity: float = 1.,
                 shadow_radius: float = 0.,
                 shadow_strength: float = 1.,
                 shadow_color: Tuple = (0, 0, 0),
                 shadow_offset: List = [0, 0],
                 prob: float = 1,
                 **kwargs) -> None:
        self.lines = np.array(lines, dtype=np.int32)
        # self.lines.sort()
        self.language = language
        self.font_size = round(font_size)
        self.angle = angle
        self._direction = direction

        self.text = text if text is not None else []
        self.prob = prob

        self.translation = translation

        # note they're accumulative rgb values of textlines
        self.fg_colors = fg_color
        self.bg_colors = bg_color

        # self.stroke_width = stroke_width
        self.font_family: str = font_family
        self.bold: bool = bold
        self.underline: bool = underline
        self.italic: bool = italic
        self.rich_text = rich_text
        self.line_spacing = line_spacing
        self.letter_spacing = letter_spacing
        self._alignment = alignment
        self.target_lang = target_lang

        self._bounding_rect = _bounding_rect
        self.default_stroke_width = default_stroke_width
        self.font_weight = font_weight
        self.accumulate_color = accumulate_color

        self.opacity = opacity
        self.shadow_radius = shadow_radius
        self.shadow_strength = shadow_strength
        self.shadow_color = shadow_color
        self.shadow_offset = shadow_offset

    @cached_property
    def xyxy(self):
        x1 = self.lines[..., 0].min()
        y1 = self.lines[..., 1].min()
        x2 = self.lines[..., 0].max()
        y2 = self.lines[..., 1].max()
        return [x1, y1, x2, y2]

    @cached_property
    def xywh(self):
        x, y, w, h = self.xyxy
        return [x, y, w-x, h-y]

    @cached_property
    def center(self) -> np.ndarray:
        xyxy = np.array(self.xyxy)
        return (xyxy[:2] + xyxy[2:]) / 2

    @cached_property
    def unrotated_polygons(self) -> np.ndarray:
        polygons = self.lines.reshape(-1, 8)
        if self.angle != 0:
            polygons = rotate_polygons(self.center, polygons, self.angle)
        return polygons

    @cached_property
    def min_rect(self) -> np.ndarray:
        polygons = self.unrotated_polygons
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        if self.angle != 0:
            min_bbox = rotate_polygons(self.center, min_bbox, -self.angle)
        return min_bbox.reshape(-1, 4, 2).astype(np.int64)

    @cached_property
    def polygon_aspect_ratio(self) -> float:
        """width / height"""
        polygons = self.unrotated_polygons.reshape(-1, 4, 2)
        middle_pts = (polygons[:, [1, 2, 3, 0]] + polygons) / 2
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
        return np.mean(norm_h / norm_v)

    @cached_property
    def aspect_ratio(self) -> float:
        """width / height"""
        middle_pts = (self.min_rect[:, [1, 2, 3, 0]] + self.min_rect) / 2
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0])
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3])
        return norm_h / norm_v

    @property
    def polygon_object(self) -> Polygon:
        min_rect = self.min_rect[0]
        return MultiPoint([tuple(min_rect[0]), tuple(min_rect[1]), tuple(min_rect[2]), tuple(min_rect[3])]).convex_hull

    @property
    def area(self) -> float:
        return self.polygon_object.area

    def normalizd_width_list(self) -> List[float]:
        polygons = self.unrotated_polygons
        width_list = []
        for polygon in polygons:
            width_list.append((polygon[[2, 4]] - polygon[[0, 6]]).sum())
        width_list = np.array(width_list)
        width_list = width_list / np.sum(width_list)
        return width_list.tolist()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

    def to_dict(self):
        blk_dict = copy.deepcopy(vars(self))
        return blk_dict

    def get_transformed_region(self, img: np.ndarray, line_idx: int, textheight: int, maxwidth: int = None) -> np.ndarray:
        src_pts = np.array(self.lines[line_idx], dtype=np.float64)

        middle_pnt = (src_pts[[1, 2, 3, 0]] + src_pts) / 2
        vec_v = middle_pnt[2] - middle_pnt[0]   # vertical vectors of textlines
        vec_h = middle_pnt[1] - middle_pnt[3]   # horizontal vectors of textlines
        ratio = np.linalg.norm(vec_v) / np.linalg.norm(vec_h)

        if ratio < 1:
            h = int(textheight)
            w = int(round(textheight / ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img, M, (w, h))
        else:
            w = int(textheight)
            h = int(round(textheight * ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img, M, (w, h))
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if maxwidth is not None:
            h, w = region.shape[: 2]
            if w > maxwidth:
                region = cv2.resize(region, (maxwidth, h))
        return region

    def get_text(self):
        if isinstance(self.text, str):
            return self.text
        return ' '.join(self.text).strip()

    def set_font_colors(self, fg_colors, bg_colors, accumulate=True):
        self.accumulate_color = accumulate
        num_lines = len(self.lines) if accumulate and len(self.lines) > 0 else 1
        # set font color
        fg_colors = np.array(fg_colors) * num_lines
        self.fg_colors = fg_colors
        # set stroke color  
        bg_colors = np.array(bg_colors) * num_lines
        self.bg_colors = bg_colors

    def get_font_colors(self, bgr=False):
        num_lines = len(self.lines)
        frgb = np.array(self.fg_colors)
        brgb = np.array(self.bg_colors)
        if self.accumulate_color:
            if num_lines > 0:
                frgb = (frgb / num_lines).astype(np.int32)
                brgb = (brgb / num_lines).astype(np.int32)
                if bgr:
                    return frgb[::-1], brgb[::-1]
                else:
                    return frgb, brgb
            else:
                return [0, 0, 0], [0, 0, 0]
        else:
            return frgb, brgb

    @property
    def direction(self):
        """Render direction determined through used language or aspect ratio."""
        if self._direction not in ('h', 'v'):
            d = LANGAUGE_ORIENTATION_PRESETS.get(self.target_lang)
            if d in ('h', 'v'):
                return d

            if self.aspect_ratio < 1:
                return 'v'
            else:
                return 'h'
        return self._direction

    @property
    def vertical(self):
        return self.direction == 'v'

    @property
    def horizontal(self):
        return self.direction == 'h'

    @property
    def alignment(self):
        """Render alignment determined through used language."""
        if self._alignment in ('left', 'center', 'right'):
            return self._alignment
        if len(self.lines) == 1:
            return 'center'

        if LANGAUGE_ORIENTATION_PRESETS.get(self.target_lang) == 'h':
            return 'center'
        else:
            return 'left'

    @property
    def stroke_width(self):
        diff = color_difference(*self.get_font_colors())
        if diff > 15:
            return self.default_stroke_width
        return 0


def rotate_polygons(center, polygons, rotation, new_center=None, to_int=True):
    if new_center is None:
        new_center = center
    rotation = np.deg2rad(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    polygons = polygons.astype(np.float32)

    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += new_center[1]
    rotated[:, ::2] += new_center[0]
    if to_int:
        return rotated.astype(np.int64)
    return rotated

def visualize_textblocks(canvas, blk_list: List[TextBlock]):
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)  # line width
    for i, blk in enumerate(blk_list):
        bx1, by1, bx2, by2 = blk.xyxy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (127, 255, 127), lw)
        for j, line in enumerate(blk.lines):
            cv2.putText(canvas, str(j), line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,127,0), 1)
            cv2.polylines(canvas, [line], True, (0,127,255), 2)
        cv2.polylines(canvas, [blk.min_rect], True, (127,127,0), 2)
        cv2.putText(canvas, str(i), (bx1, by1 + lw), 0, lw / 3, (255,127,127), max(lw-1, 1), cv2.LINE_AA)
        center = [int((bx1 + bx2)/2), int((by1 + by2)/2)]
        cv2.putText(canvas, 'a: %.2f' % blk.angle, [bx1, center[1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
        cv2.putText(canvas, 'x: %s' % bx1, [bx1, center[1] + 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
        cv2.putText(canvas, 'y: %s' % by1, [bx1, center[1] + 60], cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
    return canvas

