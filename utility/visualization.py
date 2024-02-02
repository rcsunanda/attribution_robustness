import logging
from pprint import pprint

import matplotlib.pyplot as plt
import torchvision
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from utility import common


# plot_args_list contains a list of dicts for to be given as kwargs to each plot_func call
# An example plot_func maybe --> my_line_plot(ax, x, y, title)
# A corresponding dict in plot_args_list may be --> {'x': x_data, 'y': y_data, 'title': AAA}
def plot_grid(rows, cols, fig_title, default_plot_func, plot_args_list):
    assert rows * cols >= len(plot_args_list)   # Check for adequate grid size
    fig, axes = create_subplot_grid(rows, cols, fig_title)
    axes = axes.flatten()

    for i, elem in enumerate(plot_args_list):
        # np.set_printoptions(threshold=10)
        # print('Plotting element: ', i, elem)
        plot_func = elem.get('plot_func', default_plot_func)    # To support specific plot funcs for each plot
        elem.pop('plot_func', None)                    # Remove plot_func from kwargs
        plot_func(axes[i], **elem)

    return fig


# Helper function to create a grid of subplots
def create_subplot_grid(rows, cols, title):
    fig = plt.figure(figsize=(2 * rows, 2 * cols))
    common.add_figure_to_save(fig, title)
    fig.suptitle(title)

    axes = fig.subplots(rows, cols)
    return fig, axes


def plot_histogram(ax, hist, bins, title=None):
    if title:
        ax.set_xlabel(title, fontsize=6)

    ax.hist(hist, bins=bins)


def plot_ig_maps_sequence(ig_maps_seq):
    # pprint(ig_maps_seq)

    max_cols = 5

    for i, entry in enumerate(ig_maps_seq):
        fig = plt.figure()
        title = f'IG maps during training - sample {i}'
        common.add_figure_to_save(fig, title)
        fig.suptitle(title)
        fig.tight_layout(h_pad=5, w_pad=5)

        image = entry['image'].squeeze().cpu()
        ig_maps = entry['ig_maps']

        num_maps = len(ig_maps)
        num_maps = num_maps + 1     # Extra subplot for original image

        rows = num_maps // max_cols + 1
        cols = min(num_maps, max_cols)

        # print(f'rows: {rows}, cols: {cols}')

        axes = fig.subplots(rows, cols)
        axes = np.ravel(axes)

        for ax in axes:
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

        # Plot the original image in the first subplot
        ax = axes[0]
        ax.axis('on')
        ax.set_xlabel('Original', fontsize=8)
        ax.imshow(image, cmap='gray')

        # pprint(ig_maps)

        j = 1
        for epoch, ig_map in ig_maps.items():
            # print(f'j: {j}, epoch: {epoch}, ig_map: {ig_map.shape}')
            saliency = torch.sum(torch.abs(ig_map), 0)  # Sum across channels
            saliency = saliency / torch.sum(saliency)
            saliency = saliency.cpu()

            ax = axes[j]
            ax.axis('on')
            ax.set_xlabel(f'Epoch {epoch}', fontsize=8)
            ax.imshow(saliency, cmap='hot')

            j += 1


def plot_feature_sig(ax, x, title=None, cmap='hot'):
    x = x.detach().numpy()

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_xlabel(title, fontsize=6)

    xmin = np.min(x)
    xmax = np.max(x)

    if x.shape[0] == 3:     # Channel first format (3 channels)
        x = np.moveaxis(x, 0, -1)   # Convert to channel last format

    x = (x - xmin) / (xmax - xmin)

    if x.shape[0] == 1:     # Monochrome image
        x = np.squeeze(x, 0)  # Remove the first dimension (no. of channels = 1)

    hm = ax.imshow(x, plt.get_cmap(cmap))      # Color map is applied only for 1-channel images (eg: attributions)

    # plt.colorbar(hm, cax=ax)


def plot_grayscale(ax, x, title=None):
    plot_feature_sig(ax, x, title, cmap='gray')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def visualize_data_samples(dataset, num_samples, tb_writer, tb_tag):
    # print('Adding data samples to Tensorboard')

    X_samples, y = next(iter(dataset))
    X_samples = X_samples[0:num_samples]

    X_samples = X_samples * 0.2 + 0.5     # unnormalize

    img_grid = torchvision.utils.make_grid(X_samples)
    tb_writer.add_image(tb_tag, img_grid)
    tb_writer.close()

    return None


def visualize_data_projections(dataset, num_samples, tb_writer):
    print('Adding data projections to Tensorboard')

    X_samples, y_samples = [], []
    sample_count = 0
    for X, y in dataset:
        X_samples.extend([e for e in X])
        y_samples.extend([e for e in y])
        sample_count += X.shape[0]

        if sample_count >= num_samples:
            break

    X_samples = torch.stack(X_samples)
    y_samples = torch.stack(y_samples)

    X_samples = X_samples[0:num_samples]
    y_samples = y_samples[0:num_samples]

    # log embeddings
    features = X_samples.view(num_samples, -1)  # Flatten images
    tb_writer.add_embedding(features, metadata=y_samples, label_img=X_samples)

    return None


