import argparse, numpy, pyaudio, struct, sys, wave
from scipy import fft

# Sample rate in frames per second.
SAMPLE_RATE = 48_000

ap = argparse.ArgumentParser()
ap.add_argument(
    "-o", "--outfile",
    help="wav output file",
)
ap.add_argument(
    "-a", "--ampls",
    help="comma-separated list of relative band amplitudes in dB",
    default = "1.0"
)
ap.add_argument(
    "-f", "--freqs",
    help="comma-separated list of band frequency splitpoints in Hz",
    default = ""
)
ap.add_argument(
    "-b", "--blocksize",
    help="block size in frames",
    type=int,
    default = 4096,
)
ap.add_argument(
    "-l", "--lap",
    help="window overlap in frames",
    type=int,
)
ap.add_argument(
    "wavfile",
    help="wav input file",
)
args = ap.parse_args()

# Read a wave file.
def read(filename):
    with wave.open(filename, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == SAMPLE_RATE
        nframes = w.getnframes()
        frames = w.readframes(nframes)
        framedata = struct.unpack(f"<{nframes}h", frames)
        samples = [s / (1 << 15) for s in framedata]
        return w.getparams(), samples

# Write a wave file.
def write(f, samples, params):
    nframes = len(samples)
    framedata = [int(s * (1 << 15)) for s in samples]
    frames = struct.pack(f"<{nframes}h", *framedata)
    with wave.open(f, "wb") as w:
        w.setparams(params)
        w.writeframes(frames)

blocksize = args.blocksize

# Play a tone on the computer.
def play(samples):
    # Set up and start the stream.
    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate = SAMPLE_RATE,
        channels = 1,
        format = pyaudio.paFloat32,
        output = True,
        frames_per_buffer = blocksize,
    )

    # Write the samples.
    wav = iter(samples)
    done = False
    while not done:
        buffer = list()
        for _ in range(blocksize):
            try:
                sample = next(wav)
            except StopIteration:
                done = True
                break
            buffer.append(sample)
        pbuffer = struct.pack(f"{len(buffer)}f", *buffer)
        stream.write(pbuffer)

    # Tear down the stream.
    stream.close()

# Collect the samples.
params, samples = read(args.wavfile)

# Calculate band boundaries and amplitudes.  Frequencies
# should be log-scaled, but currently are not.
bandampls = [10**(float(b)/20) for b in args.ampls.replace("+", "").split(",")]
nbands = len(bandampls)
# The real FFT will return positive frequencies only,
# which is fine for our purposes. This means that
# the returned block will be half-sized. It will
# also return the DC component at position 0, which
# we will remove as irrelevant.
bands = [0]
for b in bandampls:
    bands += [b] * (blocksize // 2 // nbands)
# Pad out the last band because rounding error.
bands += [0] * (blocksize // 2 + 1 - len(bands))
bandampls = numpy.array(bands)
assert len(bandampls) == blocksize // 2 + 1

# Build the window.
if args.lap is None:
    lap = blocksize // 32
else:
    lap = args.lap
trap1 = numpy.linspace(0, 1, lap, endpoint=False)
trap2 = numpy.ones(blocksize - 2 * lap)
trap3 = 1 - trap1
window = numpy.append(trap1, trap2)
window = numpy.append(window, trap3)

# Run the filter.
nsamples = len(samples)
insamples = numpy.array(samples)
outsamples = numpy.zeros(nsamples)
start = 0
while start + blocksize <= nsamples:
    samples_in = samples[start:start + blocksize] * window
    freqs = fft.rfft(samples_in)
    freqs *= bandampls
    samples_out = fft.irfft(freqs)
    outsamples[start:start + blocksize] += samples_out * window
    start += blocksize - lap

# Play the result.
if args.outfile:
    write(args.outfile, outsamples, params)
play(outsamples)
