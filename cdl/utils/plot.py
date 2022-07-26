import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import to_numpy


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


# ---------------------------------- TO CREATE A SERIES OF PICTURES ---------------------------------- #
# from https://zulko.wordpress.com/2012/09/29/animate-your-3d-plots-with-pythons-matplotlib/

def make_views(ax, angles, elevation=None, width=4, height=3,
               prefix='tmprot_', **kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created.

    Returns: the list of files created (for later removal)
    """

    files = []
    ax.figure.set_size_inches(width, height)

    for i, angle in enumerate(angles):
        ax.view_init(elev=elevation, azim=angle)
        fname = '%s%03d.jpeg' % (prefix, i)
        ax.figure.savefig(fname)
        files.append(fname)

    return files


# ----------------------- TO TRANSFORM THE SERIES OF PICTURE INTO AN ANIMATION ----------------------- #

def make_movie(files, output, fps=10, bitrate=1800, **kwargs):
    """
    Uses mencoder, produces a .mp4/.ogv/... movie from a list of
    picture files.
    """

    output_name, output_ext = os.path.splitext(output)
    command = {'.mp4': 'mencoder "mf://%s" -mf fps=%d -o %s.mp4 -ovc lavc\
                         -lavcopts vcodec=msmpeg4v2:vbitrate=%d'
                       % (",".join(files), fps, output_name, bitrate)}

    command['.ogv'] = command['.mp4'] + '; ffmpeg -i %s.mp4 -r %d %s' % (output_name, fps, output)

    print(command[output_ext])
    output_ext = os.path.splitext(output)[1]
    os.system(command[output_ext])


def make_gif(files, output, delay=100, repeat=True, **kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s' % (delay, loop, " ".join(files), output))


def make_strip(files, output, **kwargs):
    """
    Uses imageMagick to produce a .jpeg strip from a list of
    picture files.
    """

    os.system('montage -tile 1x -geometry +0+0 %s %s' % (" ".join(files), output))


# ---------------------------------------------- MAIN FUNCTION ---------------------------------------------- #

def rotanimate(ax, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """

    output_ext = os.path.splitext(output)[1]

    files = make_views(ax, angles, **kwargs)

    D = {'.mp4': make_movie,
         '.ogv': make_movie,
         '.gif': make_gif,
         '.jpeg': make_strip,
         '.png': make_strip}

    D[output_ext](files, output, **kwargs)

    for f in files:
        os.remove(f)


def plot_adjacency_intervention_mask(params, model, writer, step):
    adjacency = model.get_adjacency()
    intervention_mask = model.get_intervention_mask()
    adjacency = to_numpy(adjacency)
    intervention_mask = to_numpy(intervention_mask)
    adjacency_intervention = np.concatenate([adjacency, intervention_mask], axis=-1)

    obs_keys = params.obs_keys
    obs_spec = params.obs_spec
    feature_dim, action_dim = intervention_mask.shape

    fig = plt.figure(figsize=((feature_dim + action_dim) * 0.45 + 2, feature_dim * 0.45 + 2))

    use_cmi = params.training_params.inference_algo == "cmi"
    vmax = params.inference_params.cmi_params.CMI_threshold if use_cmi else 1.0
    if vmax < 0.01:
        vmax = vmax * 100
        adjacency_intervention = adjacency_intervention * 100
    sns.heatmap(adjacency_intervention, linewidths=1, vmin=0, vmax=vmax, square=True, annot=True, fmt='.2f', cbar=False)

    ax = plt.gca()
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    if params.encoder_params.encoder_type == "identity":
        tick_loc = []
        cum_idx = 0
        for k in obs_keys:
            obs_dim = obs_spec[k].shape[0]
            tick_loc.append(cum_idx + obs_dim * 0.5)
            cum_idx += obs_dim
            plt.vlines(cum_idx, ymin=0, ymax=feature_dim, colors='blue', linewidths=3)
            if k != obs_keys[-1]:
                plt.hlines(cum_idx, xmin=0, xmax=feature_dim + action_dim, colors='blue', linewidths=3)

        plt.xticks(tick_loc + [feature_dim + 0.5 * action_dim], obs_keys + ["action"], rotation=90)
        plt.yticks(tick_loc, obs_keys, rotation=0)
    else:
        plt.vlines(feature_dim, ymin=0, ymax=feature_dim, colors='blue', linewidths=3)
        plt.xticks([0.5 * feature_dim, feature_dim + 0.5 * action_dim], ["feature", "action"], rotation=90)
    fig.tight_layout()
    writer.add_figure("adjacency", fig, step + 1)
    plt.close("all")


def plot_disentanglement(params, encoder, decoder, replay_buffer, writer, step):
    assert params.encoder_params.encoder_type == "conv" and decoder is not None

    feature_dim = encoder.feature_dim
    num_interpolation = 2

    obs_1_and_2 = replay_buffer.sample_distinct_obses()                  # (2, obs, spec)

    with torch.no_grad():
        image_1_and_2 = torch.cat([obs_1_and_2[key] for key in encoder.image_keys], dim=-3)
        enc_1_and_2 = encoder(obs_1_and_2).mean                          # (2, feature_dim)
        enc_1, enc_2 = torch.unbind(enc_1_and_2)
        interpolation = enc_1.repeat(feature_dim, num_interpolation, 1)  # (feature_dim, num_interpolation, feature_dim)
        for i in range(feature_dim):
            interpolation[i, :, i] += torch.linspace(enc_1[i], enc_2[i],
                                                     steps=num_interpolation + 2, device=interpolation.device)[1:-1]

        recon_1_and_2 = decoder(enc_1_and_2).mean                        # (2, c, h, w)
        recon_interpolation = decoder(interpolation).mean                # (feature_dim, num_interpolation, c, h, w)

        # assume both RGB and depth are normalized to [0, 1]
        recon_1_and_2 = torch.clip(recon_1_and_2, min=0, max=1)
        recon_interpolation = torch.clip(recon_interpolation, min=0, max=1)

    image_1_and_2 = to_numpy(image_1_and_2)
    recon_1_and_2 = to_numpy(recon_1_and_2)
    recon_interpolation = to_numpy(recon_interpolation)

    # (2, h, w, c)
    image_1_and_2 = np.transpose(image_1_and_2, (0, 2, 3, 1))
    recon_1_and_2 = np.transpose(recon_1_and_2, (0, 2, 3, 1))

    # (feature_dim, num_interpolation, h, w, c)
    recon_interpolation = np.transpose(recon_interpolation, (0, 1, 3, 4, 2))

    image_1_and_2 = np.flip(image_1_and_2, axis=-3)
    recon_1_and_2 = np.flip(recon_1_and_2, axis=-3)
    recon_interpolation = np.flip(recon_interpolation, axis=-3)

    image_keys = encoder.image_keys
    num_sub_image = len(image_keys)
    image_channels = [params.obs_spec[key].shape[-3] for key in image_keys]
    image_channels_start_end = [(sum(image_channels[:i]), sum(image_channels[:i + 1])) for i in range(num_sub_image)]

    image_1, image_2 = image_1_and_2[0], image_1_and_2[1]
    recon_1, recon_2 = recon_1_and_2[0], recon_1_and_2[1]

    # original and reconstruction for images 1 and 2
    for i, (image, recon) in enumerate(zip([image_1, image_2], [recon_1, recon_2])):
        fig = plt.figure(figsize=(6, 3 * num_sub_image))
        for j, (sub_image_key, (channel_start, channel_end)) in enumerate(zip(image_keys, image_channels_start_end)):
            sub_image = image[..., channel_start:channel_end]
            sub_recon = recon[..., channel_start:channel_end]

            ax = fig.add_subplot(num_sub_image, 2, 2 * j + 1, xticks=[], yticks=[])
            ax.imshow(sub_image)
            ax.set_ylabel(sub_image_key)

            ax = fig.add_subplot(num_sub_image, 2, 2 * j + 2, xticks=[], yticks=[])
            ax.imshow(sub_recon)
            ax.set_title("recon loss: {0:.3f}".format(((sub_image - sub_recon) ** 2).sum()))
        fig.tight_layout()
        writer.add_figure("image_reconstruction_{}".format(i + 1), fig, step + 1)
        plt.close("all")

    # interpolation between images 1 and 2
    num_col = num_interpolation + 2
    for i, recon_intrpl_i in enumerate(recon_interpolation):
        fig = plt.figure(figsize=(3 * num_col, 3 * num_sub_image))
        for j, (sub_image_key, (channel_start, channel_end)) in enumerate(zip(image_keys, image_channels_start_end)):
            sub_recon_1 = recon_1[..., channel_start:channel_end]
            sub_recon_2 = recon_2[..., channel_start:channel_end]

            ax = fig.add_subplot(num_sub_image, num_col, j * num_col + 1, xticks=[], yticks=[])
            ax.imshow(sub_recon_1)
            ax.set_ylabel(sub_image_key)

            for k, recon_intrpl_i_k in enumerate(recon_intrpl_i[..., channel_start:channel_end]):
                ax = fig.add_subplot(num_sub_image, num_col, j * num_col + k + 2, xticks=[], yticks=[])
                ax.imshow(recon_intrpl_i_k)

            ax = fig.add_subplot(num_sub_image, num_col, j * num_col + num_col, xticks=[], yticks=[])
            ax.imshow(sub_recon_2)
        fig.tight_layout()
        writer.add_figure("interpolation_feature_{}".format(i + 1), fig, step + 1)
        plt.close("all")
