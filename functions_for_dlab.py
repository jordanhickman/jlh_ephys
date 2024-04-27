
## raw data stuff

def raw_heatmap(data, pre=1, post=2, dists=None, vmin=None, vmax=None, 
                save=False, save_path=None, save_type='png', title=None, ax=None):
    """
    Plots a heatmap of data with options for customization and saving.

    Args:
        data (np.array): The data to plot, shape = (trials, samples, channels).
        pre (int, optional): pre_window in ms. Defaults to 1.
        post (int, optional): post_window in ms. Defaults to 2.
        dists (np.array, optional): Distance from stimulation array, same shape as channels.
        vmin (float, optional): Minimum value for heatmap scaling. Defaults to None.
        vmax (float, optional): Maximum value for heatmap scaling. Defaults to None.
        save (bool, optional): If True, save the figure. Defaults to False.
        save_path (str, optional): Path to save the figure. Defaults to None.
        save_type (str, optional): Format of saved figure. Defaults to 'png'.
        title (str, optional): Figure title. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None (creates new figure).

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """

    # Check if an axes object was provided, if not create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        created_fig = True
    else:
        created_fig = False  # Flag to avoid creating a new figure

    channels = data.shape[2]
    data_to_plot = np.mean(data, axis=0).T  # average across trials

    time_ms = np.linspace(-pre, post, data.shape[1])

    cax = ax.imshow(data_to_plot, aspect='auto', extent=[time_ms[0], time_ms[-1], 0, channels],
                    vmax=vmax, vmin=vmin, origin='lower', cmap='viridis')

    if created_fig:
        fig.colorbar(cax, ax=ax, pad=0.20)
    ax.set_ylabel('Channel')

    if dists is not None:
        ax_dist = ax.twinx()
        ax_dist.set_ylim(ax.get_ylim())
        ax_dist.set_yticks(np.arange(0, len(dists), 25))
        ax_dist.set_yticklabels([int(d) for d in dists[::25]])
        ax_dist.set_ylabel('Distance from Stimulation (units)')
        zero_dist_channel = np.argmin(np.abs(dists))
        ax.axhline(y=zero_dist_channel, color='red', linestyle='--')

    ax.set_xlabel('Time (ms)')
    if title:
        ax.set_title(title)

    if created_fig:
        plt.tight_layout()

    if save and created_fig:
        plt.savefig(save_path, format=save_type)

    return ax

