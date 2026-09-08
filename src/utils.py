import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import mne

def plot_eeg_gif(eeg, save_path="eeg_animation.gif", fps=5):
    eeg.rename_channels(lambda x: x.replace("EEG ", "").replace("-REF", "").strip())

    # 1. Drop non-EEG channels
    non_eeg = ["EKG EKG", "SPO2", "HR", "1", "2", "MK"]
    eeg.drop_channels([ch for ch in non_eeg if ch in eeg.ch_names])

    # 2. Normalize names to montage casing
    montage = mne.channels.make_standard_montage("standard_1020")
    montage_names = {name.upper(): name for name in montage.ch_names}
    eeg.rename_channels(
        {ch: montage_names[ch.upper()] for ch in eeg.ch_names
        if ch.upper() in montage_names}
    )
    eeg.set_montage(montage, on_missing="warn")

    # 3. Pick EEG channels (define picks here — it was missing before)
    picks = mne.pick_types(eeg.info, eeg=True)
    info = mne.pick_info(eeg.info, picks)
    sfreq = eeg.info["sfreq"]

    # 4. Window + downsample
    t_start, t_stop = 60.0, 80.0
    start, stop = eeg.time_as_index([t_start, t_stop])
    seg = eeg.get_data(picks=picks)[:, start:stop]
    step = int(sfreq // 10)
    frames = range(0, seg.shape[1], step)

    vmin, vmax = seg.min(), seg.max()

    # 5. Figure with room for a colorbar
    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.suptitle("PN00-3 — Scalp EEG Topography", fontsize=13, fontweight="bold")

    # Persistent colorbar (legend), drawn once from a fixed-scale mappable
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("Voltage (µV)", rotation=270, labelpad=15)

    def update(i):
        ax.clear()
        mne.viz.plot_topomap(
            seg[:, i], info, axes=ax, show=False,
            vlim=(vmin, vmax), contours=0, cmap="RdBu_r",
        )
        ax.set_title(f"t = {t_start + i / sfreq:.2f} s", fontsize=10)

    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)