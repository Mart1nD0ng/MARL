"""
IEEE期刊论文风格的多子图拼接脚本

将6张图片拼接成2行3列的布局：
- 第一行：logical_ep0, logical_ep100, logical_ep200
- 第二行：topology_ep0, topology_ep100, topology_ep200

设计特点：
1. 使用统一的子图标签 (a), (b), (c), (d), (e), (f)
2. 共享图例放置在图片右侧或底部
3. 使用Times New Roman字体（IEEE标准）
4. 适合单栏或双栏期刊排版
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

# IEEE 期刊标准设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8


def create_ieee_composite_figure(
    input_dir: str,
    output_path: str,
    dpi: int = 300,
    single_column: bool = False,
):
    """
    创建IEEE期刊风格的复合图像
    
    Args:
        input_dir: 包含图像文件的目录
        output_path: 输出路径
        dpi: 输出DPI（IEEE推荐300-600）
        single_column: 是否为单栏格式（True: ~3.5in宽，False: ~7in宽）
    """
    
    # 定义要拼接的图像文件
    logical_files = ['logical_ep0.png', 'logical_ep100.png', 'logical_ep200.png']
    topology_files = ['topology_ep0.png', 'topology_ep100.png', 'topology_ep200.png']
    
    # 子图标签
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    # 子图标题（可选，IEEE通常用简短描述）
    subtitles = ['Episode 0', 'Episode 100', 'Episode 200',
                 'Episode 0', 'Episode 100', 'Episode 200']
    
    # 加载图像
    images = []
    for f in logical_files + topology_files:
        img_path = os.path.join(input_dir, f)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
        else:
            print(f"[Warning] Image not found: {img_path}")
            return
    
    # IEEE期刊尺寸规范
    # 单栏: 3.5 inches, 双栏: 7.16 inches
    if single_column:
        fig_width = 3.5  # inches
    else:
        fig_width = 7.16  # inches (IEEE double column)
    
    # 计算合适的高度
    # 2行3列布局，保持子图的宽高比
    # 由于原图比例不同，我们统一裁剪/调整到相同比例
    aspect_ratio = 1.0  # 目标宽高比（正方形便于比较）
    
    # 子图之间的间距
    h_spacing = 0.02  # 水平间距（相对于图宽）
    v_spacing = 0.08  # 垂直间距（相对于图高），留出标签空间
    
    # 计算每个子图的尺寸
    subplot_width = (1.0 - 4 * h_spacing) / 3  # 3列
    subplot_height = subplot_width * aspect_ratio * (fig_width / (fig_width * 0.75))  # 调整高度
    
    # 创建图形 - 调整高度以适应2行 + 共享图例
    fig_height = fig_width * 0.72  # 适合2行3列 + 底部图例
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor('white')
    
    # 定义子图位置 [left, bottom, width, height]
    # 第一行（logical topology）
    row1_bottom = 0.42
    row2_bottom = 0.08
    
    subplot_w = 0.29
    subplot_h = 0.40
    
    positions = [
        # Row 1: Logical Topology
        [0.04, row1_bottom, subplot_w, subplot_h],
        [0.36, row1_bottom, subplot_w, subplot_h],
        [0.68, row1_bottom, subplot_w, subplot_h],
        # Row 2: Physical Topology
        [0.04, row2_bottom, subplot_w, subplot_h],
        [0.36, row2_bottom, subplot_w, subplot_h],
        [0.68, row2_bottom, subplot_w, subplot_h],
    ]
    
    # 绘制每个子图
    for idx, (img, pos, label, subtitle) in enumerate(zip(images, positions, labels, subtitles)):
        ax = fig.add_axes(pos)
        
        # 将PIL图像转换为numpy数组
        img_array = np.array(img)
        
        # 显示图像（移除原图的边框和标题区域）
        # 裁剪掉边缘的空白区域
        img_cropped = _smart_crop(img_array)
        
        ax.imshow(img_cropped)
        ax.axis('off')
        
        # 添加子图标签 - 放在左下角
        ax.text(0.02, -0.02, label, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='left',
                fontfamily='serif')
        
        # 添加子图标题 - 放在标签右侧
        ax.text(0.15, -0.02, subtitle, transform=ax.transAxes,
                fontsize=9, va='top', ha='left', style='italic',
                fontfamily='serif')
    
    # 添加行标签
    fig.text(0.01, 0.70, 'Logical\nTopology', fontsize=10, fontweight='bold',
             va='center', ha='left', rotation=90, fontfamily='serif')
    fig.text(0.01, 0.30, 'Physical\nTopology', fontsize=10, fontweight='bold',
             va='center', ha='left', rotation=90, fontfamily='serif')
    
    # 添加共享图例 - 放在图片下方
    _add_shared_legend(fig)
    
    # 保存
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] IEEE composite figure saved to: {output_path}")


def _smart_crop(img_array: np.ndarray, margin_ratio: float = 0.02, 
                remove_title: bool = True, remove_legend: bool = True,
                remove_stats: bool = True, is_logical: bool = False) -> np.ndarray:
    """
    智能裁剪图像，移除标题、图例和统计信息区域
    
    Args:
        img_array: 输入图像数组
        margin_ratio: 边距比例
        remove_title: 是否移除顶部标题区域
        remove_legend: 是否移除右上角图例
        remove_stats: 是否移除底部统计信息
        is_logical: 是否是logical拓扑图（有灰色背景）
    """
    h, w = img_array.shape[:2]
    img_work = img_array.copy()
    
    if is_logical:
        # Logical图 - 直接裁剪掉图例区域
        top_crop = int(h * 0.045) if remove_title else 0
        bottom_crop = int(h * 0.065) if remove_stats else 0
        right_crop = int(w * 0.28) if remove_legend else 0
        left_crop = int(w * 0.02)
        
        img_cropped = img_work[top_crop:h-bottom_crop, left_crop:w-right_crop]
        
        # 进一步裁剪白边
        h2, w2 = img_cropped.shape[:2]
        if len(img_cropped.shape) == 3:
            gray = np.mean(img_cropped[:, :, :3], axis=2)
        else:
            gray = img_cropped
        
        threshold = 245
        non_white = gray < threshold
        
        if np.any(non_white):
            rows = np.any(non_white, axis=1)
            cols = np.any(non_white, axis=0)
            
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            margin_h = int(h2 * margin_ratio)
            margin_w = int(w2 * margin_ratio)
            
            rmin = max(0, rmin - margin_h)
            rmax = min(h2, rmax + margin_h)
            cmin = max(0, cmin - margin_w)
            cmax = min(w2, cmax + margin_w)
            
            img_cropped = img_cropped[rmin:rmax, cmin:cmax]
        
        return img_cropped
    else:
        # Physical Topology图 - 保留原图完整性，不做任何处理
        return img_work


def _add_shared_legend(fig):
    """添加共享图例（Agent颜色标识）"""
    
    # Agent颜色列表（与visualize.py一致）
    colors_list = [
        "#E74C3C", "#F39C12", "#F1C40F", "#2ECC71", "#3498DB",
        "#9B59B6", "#1ABC9C", "#34495E", "#E67E22", "#95A5A6"
    ]
    
    # 创建图例元素
    legend_elements = []
    for i in range(10):
        element = Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=colors_list[i],
                         markeredgecolor='white',
                         markersize=8, label=f'Agent {i}')
        legend_elements.append(element)
    
    # 在底部添加图例
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=10, fontsize=7, frameon=True,
               framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.01),
               handletextpad=0.3, columnspacing=0.8)


def create_ieee_figure_v2(
    input_dir: str,
    output_path: str,
    dpi: int = 300,
):
    """
    创建IEEE期刊风格的复合图像 - 版本2
    
    特点：
    - 图例移到右侧
    - 更紧凑的布局
    - 适合期刊双栏格式
    """
    
    # 定义要拼接的图像文件
    logical_files = ['logical_ep0.png', 'logical_ep100.png', 'logical_ep200.png']
    topology_files = ['topology_ep0.png', 'topology_ep100.png', 'topology_ep200.png']
    
    # 子图标签
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    # 加载图像
    images = []
    for f in logical_files + topology_files:
        img_path = os.path.join(input_dir, f)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
        else:
            print(f"[Warning] Image not found: {img_path}")
            return
    
    # IEEE双栏宽度
    fig_width = 7.16  # inches
    fig_height = 5.0  # inches
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor('white')
    
    # 创建GridSpec布局 - 留出右侧图例空间
    from matplotlib.gridspec import GridSpec
    
    # 主图区域占85%宽度，图例占15%
    gs = GridSpec(2, 4, figure=fig, 
                  width_ratios=[1, 1, 1, 0.15],
                  height_ratios=[1, 1],
                  wspace=0.05, hspace=0.15,
                  left=0.02, right=0.98, top=0.92, bottom=0.08)
    
    # 子图标题
    col_titles = ['Initial (Ep. 0)', 'Mid-training (Ep. 100)', 'Trained (Ep. 200)']
    row_titles = ['Logical Topology', 'Physical Topology']
    
    # 绘制子图
    for row in range(2):
        for col in range(3):
            idx = row * 3 + col
            ax = fig.add_subplot(gs[row, col])
            
            img_array = np.array(images[idx])
            img_cropped = _smart_crop(img_array)
            
            ax.imshow(img_cropped)
            ax.axis('off')
            
            # 添加子图标签
            ax.text(0.02, 0.98, labels[idx], transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top', ha='left',
                    fontfamily='serif',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # 第一行添加列标题
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, fontweight='bold',
                            fontfamily='serif', pad=3)
            
            # 第一列添加行标题
            if col == 0:
                ax.text(-0.08, 0.5, row_titles[row], transform=ax.transAxes,
                       fontsize=9, fontweight='bold', va='center', ha='right',
                       rotation=90, fontfamily='serif')
    
    # 在右侧添加图例
    ax_legend = fig.add_subplot(gs[:, 3])
    ax_legend.axis('off')
    
    colors_list = [
        "#E74C3C", "#F39C12", "#F1C40F", "#2ECC71", "#3498DB",
        "#9B59B6", "#1ABC9C", "#34495E", "#E67E22", "#95A5A6"
    ]
    
    legend_elements = []
    for i in range(10):
        element = Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=colors_list[i],
                         markeredgecolor='gray', markeredgewidth=0.5,
                         markersize=7, label=f'Agent {i}')
        legend_elements.append(element)
    
    ax_legend.legend(handles=legend_elements, loc='center left',
                     ncol=1, fontsize=7, frameon=True,
                     framealpha=0.9, edgecolor='lightgray',
                     handletextpad=0.4, labelspacing=0.6,
                     title='Agents', title_fontsize=8)
    
    # 保存
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.03)
    plt.close(fig)
    print(f"[OK] IEEE composite figure (v2) saved to: {output_path}")


def create_ieee_figure_v3(
    input_dir: str,
    output_path: str,
    dpi: int = 300,
):
    """
    创建IEEE期刊风格的复合图像 - 版本3（推荐）
    
    特点：
    - 更精细的布局控制
    - 统一的子图大小
    - 底部共享图例（单行）
    - 简洁的标签设计
    """
    
    logical_files = ['logical_ep0.png', 'logical_ep100.png', 'logical_ep200.png']
    topology_files = ['topology_ep0.png', 'topology_ep100.png', 'topology_ep200.png']
    
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    # 加载并预处理图像
    images = []
    all_files = logical_files + topology_files
    for idx, f in enumerate(all_files):
        img_path = os.path.join(input_dir, f)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # 根据图片类型调整裁剪参数
            is_logical = idx < 3  # 前3张是logical图
            img_cropped = _smart_crop(
                img_array, 
                margin_ratio=0.01,
                remove_title=True,
                remove_legend=True,
                remove_stats=True,
                is_logical=is_logical
            )
            images.append(img_cropped)
        else:
            print(f"[Warning] Image not found: {img_path}")
            return
    
    # 统一所有图像到相同大小
    target_size = (800, 800)  # 统一的目标尺寸（像素）
    images_resized = []
    for img in images:
        img_pil = Image.fromarray(img)
        # 保持宽高比居中填充
        img_resized = _resize_with_padding(img_pil, target_size)
        images_resized.append(np.array(img_resized))
    
    # IEEE尺寸
    fig_width = 7.16  # inches (双栏)
    aspect = 0.72  # 高度/宽度比
    fig_height = fig_width * aspect
    
    fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor('white')
    
    # 调整子图间距
    plt.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.12,
                        wspace=0.05, hspace=0.08)
    
    # 列标题和行标题
    col_titles = ['Initial', 'Mid-training', 'Final']
    row_labels = ['Logical\nTopology', 'Physical\nTopology']
    
    for row in range(2):
        for col in range(3):
            idx = row * 3 + col
            ax = axes[row, col]
            
            ax.imshow(images_resized[idx])
            ax.axis('off')
            
            # 子图标签（左上角，白底）
            ax.text(0.03, 0.97, labels[idx], transform=ax.transAxes,
                    fontsize=9, fontweight='bold', va='top', ha='left',
                    fontfamily='serif',
                    bbox=dict(boxstyle='round,pad=0.15', 
                              facecolor='white', alpha=0.85,
                              edgecolor='gray', linewidth=0.5))
            
            # 列标题（仅第一行）
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, fontweight='bold',
                            fontfamily='serif', pad=4)
    
    # 行标签（左侧）
    for row, label in enumerate(row_labels):
        fig.text(0.01, 0.72 - row * 0.40, label, fontsize=9, fontweight='bold',
                va='center', ha='left', rotation=90, fontfamily='serif')
    
    # 底部共享图例
    colors_list = [
        "#E74C3C", "#F39C12", "#F1C40F", "#2ECC71", "#3498DB",
        "#9B59B6", "#1ABC9C", "#34495E", "#E67E22", "#95A5A6"
    ]
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=colors_list[i],
               markeredgecolor='gray', markeredgewidth=0.3,
               markersize=6, label=f'Agent {i}')
        for i in range(10)
    ]
    
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=10, fontsize=6.5, frameon=True,
               framealpha=0.95, edgecolor='lightgray',
               bbox_to_anchor=(0.52, 0.01),
               handletextpad=0.2, columnspacing=0.5)
    
    # 保存
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] IEEE composite figure (v3-recommended) saved to: {output_path}")


def _resize_with_padding(img: Image.Image, target_size: tuple, 
                         bg_color: tuple = (255, 255, 255)) -> Image.Image:
    """
    调整图像大小，保持宽高比，用背景色填充
    """
    target_w, target_h = target_size
    
    # 计算缩放比例
    ratio = min(target_w / img.width, target_h / img.height)
    new_w = int(img.width * ratio)
    new_h = int(img.height * ratio)
    
    # 缩放图像
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 创建背景
    if img.mode == 'RGBA':
        bg = Image.new('RGBA', target_size, bg_color + (255,))
    else:
        bg = Image.new('RGB', target_size, bg_color)
    
    # 居中粘贴
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    
    if img.mode == 'RGBA':
        bg.paste(img_resized, (offset_x, offset_y), img_resized)
    else:
        bg.paste(img_resized, (offset_x, offset_y))
    
    return bg


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create IEEE-style composite figure')
    parser.add_argument('--input_dir', type=str, 
                        default='result_save/run_vis_v11',
                        help='Directory containing the source images')
    parser.add_argument('--output', type=str,
                        default='result_save/run_vis_v11/ieee_composite_figure.png',
                        help='Output file path')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (300 for IEEE journals)')
    parser.add_argument('--version', type=int, default=3, choices=[1, 2, 3],
                        help='Figure layout version (1, 2, or 3). v3 is recommended.')
    
    args = parser.parse_args()
    
    if args.version == 1:
        create_ieee_composite_figure(args.input_dir, args.output, args.dpi)
    elif args.version == 2:
        create_ieee_figure_v2(args.input_dir, args.output, args.dpi)
    else:
        create_ieee_figure_v3(args.input_dir, args.output, args.dpi)
