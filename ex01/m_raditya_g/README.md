# ## First B4 Circular Assignment
## Summary of the assignment
This assignment is to extract and draw a spectrogram of audio.

## Assignment

1. record audio and display the spectrogram!
2. convert the spectrogram back to a time signal!

## How to proceed with the assignment

1. audio recording
   - Any recording equipment and audio can be used.
   - 1ch, 16 kHz wav format is recommended
   - If you want to use the recording equipment in the lab, please consult with us. 2.
2) Draw a spectrogram of the recorded audio
   - Leave appropriate comments in the script.
   - Each axis of the graph should be clearly marked. 3.
3. presentation (next week)
   - Explain what you have done so that others can understand what you have done.
   - You may use the Github page as is to explain your code.
      You may use the Github page to explain your code, but make it easy to read!
   - It would be interesting to see the difference in spectrograms by phoneme.
   - The presenter will be chosen at random on the day of the presentation **All presenters should prepare slides** 4.
Upload your slides to `Procyon/Presentation/B4Round Lecture/2022
## Example results

! [Example result](figs/result.png)

## Tip

- If you use MATLAB|functions
  - Audio loading: `audioread
  - (fast) Fourier transform: `fft
  - Window functions: `hann`, `hamming
  - Displaying color data: `imagesc`.

- Using python｜Modules, Libraries, Functions
  - Linear algebra: `scipy`, `numpy
  - Reading audio: `soundfile`, `librosa.core.load`, `scipy.io.read`, `wave
    - SoundFile and LibROSA can be installed by the following methods
      - pip: `pip install librosa` and `pip install SoundFile`.
      - Anaconda: `conda install -c conda-forge librosa`, `conda install -c conda-forge pysoundfile`.
  - (fast) Fourier transform: `scipy.fftpack.fft`, `numpy.fft`
  - Window functions: `scipy.signal
  - Drawing graphs: `matplotlib.pyplot`, `seaborn`, `plotly` (for gurus)
    - It might be useful someday if you know how to use them in various ways!
  - Argument parsing: `argparse`.

## Caution

- If you use Python, you must use `virtualenv` or `plotly`:
   - It is useful to use `virtualenv` or `pyenv` to manage libraries. but you don't need it at first.
   - **If you use Python, be sure to use Series 3: ！！！！(Series 2 will be unsupported soon.) **
   - For Takeda Lab, use the `virtualenv` environment created in the setup (I suspect you haven't done it?).
      - Activation example: `source ~/workspace3/myvenv/bin/activate`.  
      - After activation, `pip install ... Install libraries with `pip install ...
   - Python has a code style guide called `PEP8`. Follow this guide when coding.
      - [Reference](https://blog-ja.sideci.com/entry/python-lint-pickup-5tools)
- Do not use `spectrogram` function (MATLAB) or other functions to obtain spectrogram directly from waveform data.  
- Do not use `matplotlib.pyplot.specgram` function (Python) to draw spectrogram directly from waveform data  
- Do not use `librosa.core.stft` function (Python) etc. to obtain spectrogram directly from waveform data.  

Translated with www.DeepL.com/Translator (free version)