import json
import seaborn as sns
import scienceplots
import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path
import sys


plt.style.use(['science'])

if len(sys.argv) != 2:
    print("Need benchmark output file: cargo criterion --message-format=json > FILE", file=sys.stderr)
    sys.exit(1)

output = [json.loads(line) for line in open(sys.argv[1])]

rows = []
for res in output:
    if res['reason'] != 'benchmark-complete':
        continue
    split = res['id'].split('/')
    facet = split[0]
    method = split[1]
    params = split[2].split(',')
    (width, height, max_features) = tuple(params)
    time = float(res['mean']['estimate']) * 1e-6
    mean_dev = float(res['mean_abs_dev']['estimate'] if 'mean_abs_dev' in res else 0)
    rows.append({ 'facet': facet, 'Method': method, 'width': int(width), 'height': int(height), 'max_features': int(max_features), 'time': time, 'mean_dev': mean_dev})


df = pl.DataFrame(rows).with_columns(resolution=pl.format('{}x{}', pl.col('width'), pl.col('height'))) \
    .sort(pl.col('width').mul(pl.col('height')))

out_dir = Path('images')
out_dir.mkdir(exist_ok=True)

order = ["OpenCV SIFT", "VulkanSIFT", "VulkanSIFT NoUpscale", "Ours"]

g = sns.catplot(df.filter(pl.col('facet').eq('Image Size'), pl.col('Method').ne('OpenCV SIFT')), kind='bar', x='resolution', y='time', hue='Method', hue_order=order)
g.ax.set_title('Detection Time by Image Size')
g.ax.set(xlabel='Image Size', ylabel='Time (ms)')
for ax in g.axes.flatten():
    for c in ax.containers:
        ax.bar_label(c, label_type='edge', fmt='{:.1f}')

inset_rect = [0.77, 0.7, 0.25, 0.25]

inset_ax = g.figure.add_axes(inset_rect)
sns.barplot(
    data=df.filter(pl.col('facet').eq('Image Size')),
    x="resolution",
    y="time",
    hue="Method",
    ax=inset_ax,
    legend=False,
    hue_order=order
)
inset_ax.set_xlabel('')
inset_ax.set_ylabel('')
inset_ax.tick_params(axis='both', which='major', labelsize=6)

g.savefig(out_dir / "image_size.svg")





g = sns.catplot(df.filter(pl.col('facet').eq('Feature Count'), pl.col('Method').ne('OpenCV SIFT')), kind='bar', x='max_features', y='time', hue='Method', hue_order=order)
g.ax.set_title('Detection Time by Feature Count (4096x3072 image)')
g.ax.set(xlabel='Feature Count ', ylabel='Time (ms)')
for ax in g.axes.flatten():
    for c in ax.containers:
        ax.bar_label(c, label_type='edge', fmt='{:.1f}')
inset_ax = g.figure.add_axes(inset_rect)
sns.barplot(
    data=df.filter(pl.col('facet').eq('Feature Count')),
    x="max_features",
    y="time",
    hue="Method",
    ax=inset_ax,
    legend=False,
    hue_order=order
)
inset_ax.set_xlabel('')
inset_ax.set_ylabel('')
inset_ax.tick_params(axis='both', which='major', labelsize=6)
g.savefig(out_dir / "feature_count.svg")
