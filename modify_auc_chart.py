# -*- coding: utf-8 -*-
"""直接修改原始PNG：换柱状图颜色 + 去掉差异值标注"""
from PIL import Image
import numpy as np

img = Image.open(r'data\WOE验证_AUC对比.png')
arr = np.array(img).copy()

# 原始柱状图颜色 → 目标颜色
# 绿色  (40, 167, 69)  → #3498db (52, 152, 219)  78维
# 橙色  (253, 126, 20) → #e74c3c (231, 76, 60)   13维
# 绿色2 (49, 170, 77)  → #3498db

# —— 绿色系容差掩码 ——
green_mask = (
    (arr[:,:,0] >= 35) & (arr[:,:,0] <= 60) &
    (arr[:,:,1] >= 155) & (arr[:,:,1] <= 185) &
    (arr[:,:,2] >= 50) & (arr[:,:,2] <= 95)
)

# —— 橙色系容差掩码 ——
orange_mask = (
    (arr[:,:,0] >= 240) & (arr[:,:,0] <= 255) &
    (arr[:,:,1] >= 110) & (arr[:,:,1] <= 145) &
    (arr[:,:,2] >= 5) & (arr[:,:,2] <= 45)
)

# 柱状图区域整体掩码（用于判断哪些文字属于柱状图上的标注）
bar_region_mask = green_mask | orange_mask

# 替换绿色 → #3498db
arr[green_mask, 0] = 52
arr[green_mask, 1] = 152
arr[green_mask, 2] = 219

# 替换橙色 → #e74c3c
arr[orange_mask, 0] = 231
arr[orange_mask, 1] = 76
arr[orange_mask, 2] = 60

# 去掉柱状图上的"差异值"标注文字（如 -0.0001）
# 标注文字通常是深色小字，位于柱状图上方附近
# 通过形态学膨胀柱状图区域，覆盖紧邻的小块深色文字
from scipy.ndimage import binary_dilation
bar_region_2d = bar_region_mask[:,:,0]  # 取单通道用于膨胀
# 仅在 y 方向膨胀更多，以覆盖柱子上方的差异标注
struct = np.ones((12, 3), dtype=bool)
bar_expanded = binary_dilation(bar_region_2d, structure=struct, iterations=1)

# 在膨胀区域内，将深色/红色像素替换为白色背景
dark_text_mask = (
    (arr[:,:,0] <= 55) & (arr[:,:,1] <= 55) & (arr[:,:,2] <= 55) &
    bar_expanded & ~bar_region_2d
)
red_text_mask = (
    (arr[:,:,0] >= 200) & (arr[:,:,0] <= 235) &
    (arr[:,:,1] >= 35) & (arr[:,:,1] <= 75) &
    (arr[:,:,2] >= 40) & (arr[:,:,2] <= 90) &
    bar_expanded
)
# 灰色差异文字
gray_text_mask = (
    (arr[:,:,0] >= 85) & (arr[:,:,0] <= 170) &
    (arr[:,:,1] >= 85) & (arr[:,:,1] <= 170) &
    (arr[:,:,2] >= 85) & (arr[:,:,2] <= 170) &
    (np.abs(arr[:,:,0].astype(int) - arr[:,:,1].astype(int)) < 8) &
    (np.abs(arr[:,:,0].astype(int) - arr[:,:,2].astype(int)) < 8) &
    bar_expanded & ~bar_region_2d
)

# 白色覆盖
arr[dark_text_mask] = [255, 255, 255, 255]
arr[red_text_mask] = [255, 255, 255, 255]
arr[gray_text_mask] = [255, 255, 255, 255]

Image.fromarray(arr).save(r'data\WOE验证_AUC对比.png')
print('Done.')
print(f'Green pixels replaced: {green_mask.sum()}')
print(f'Orange pixels replaced: {orange_mask.sum()}')
print(f'Dark text pixels removed: {dark_text_mask.sum()}')
print(f'Red text pixels removed: {red_text_mask.sum()}')
print(f'Gray text pixels removed: {gray_text_mask.sum()}')
