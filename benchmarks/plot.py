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

g = sns.catplot(df.filter(pl.col('facet').eq('Image Size')), kind='bar', x='resolution', y='time', hue='Method')
g.ax.set_title('Detection Time by Image Size')
g.ax.set(xlabel='Image Size', ylabel='Time (ms)')
for ax in g.axes.flatten():
    for c in ax.containers:
        ax.bar_label(c, label_type='edge', fmt='{:.1f}')

g.savefig("image_size.svg")

g = sns.catplot(df.filter(pl.col('facet').eq('Feature Count')), kind='bar', x='max_features', y='time', hue='Method')
g.ax.set_title('Detection Time by Feature Count (4096x3072 image)')
g.ax.set(xlabel='Image Size', ylabel='Time (ms)')
for ax in g.axes.flatten():
    for c in ax.containers:
        ax.bar_label(c, label_type='edge', fmt='{:.1f}')
g.savefig("feature_count.svg")
