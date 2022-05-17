import matplotlib.pyplot as plt
import librosa.display

def filterchar_show(x, amp, phase, N):
    """
    Show figure of filter characteristic (amplitude and phase).

    Parameters:
        amp : np.ndarray
            amplitude
        phase : np.ndarray
            phase
    
    Returns:
        fig : matplotlib.figure.Figure
            figure of filter characteristic
    """

    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 8))
    fig.subplots_adjust(hspace = 0.6)

    ax[0].plot(x, amp[0 : N // 2 + 1])
    ax[0].set(
        title = "Filter amplitude",
        xlabel = "Frequency [Hz]",
        ylabel = "Amplitude [dB]",
        xlim = (0, 8000),
    )
    ax[0].grid(ls = "--")

    ax[1].plot(x, phase[0 : N // 2 + 1])
    ax[1].set(
        title = "Filter phase",
        xlabel = "Frequency [Hz]",
        ylabel = "Phase [rad]",
        xlim = (0, 8000),
    )
    ax[1].grid(ls = "--")

    fig.align_labels()

    plt.show()

    return fig

def double_specshow(
    db1,
    db2,
    sr,
    hop_length,
    x_axis = "time",
    y_axis = "linear",
    cmap = "rainbow",
    title1 = None,
    title2 = None,
):
    """ 
    Display two spectrograms side by side.

    Parameters:
        db1 : np.ndarray
            first db data to display
        db2 : np.ndarray
            second db data to display
        sr : int
            sample rate used to determine time scale in x-axis
        x_axis, y_axis : None or str
            range for the x- and y-axes
        cmap : None or str
            the sequential colormap name (Default: "rainbow")
        title1, title2 : None or str
            title of spectrograms
    """

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
    plt.subplots_adjust(wspace = 0.4)

    img = librosa.display.specshow(
        db1,
        sr = sr,
        hop_length = hop_length,
        x_axis = x_axis,
        y_axis = y_axis,
        ax = ax[0],
        cmap = cmap,
    )
    ax[0].set(
        title = title1,
        xlabel = "time [s]",
        ylabel = "Frequency [Hz]",
        ylim = (0, 8000),
    )

    fig.colorbar(img, aspect = 40, pad = 0.03, ax = ax[0], format = "%+2.f dB")

    img = librosa.display.specshow(
        db2,
        sr = sr,
        hop_length = hop_length,
        x_axis = x_axis,
        y_axis = y_axis,
        ax = ax[1],
        cmap = cmap,
    )
    ax[1].set(
        title = title2,
        xlabel = "time [s]",
        ylabel = "Frequency [Hz]",
        ylim = (0, 8000),
    )
    fig.colorbar(img, aspect = 40, pad = 0.03, ax = ax[1], format = "%+2.f dB")

    plt.show()
    return fig