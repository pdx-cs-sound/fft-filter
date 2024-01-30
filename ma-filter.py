import argparse, numpy as np, sounddevice, struct, sys, wave
from scipy import fft

ap = argparse.ArgumentParser()
ap.add_argument(
    "-o", "--outfile",
    help="wav output file",
)
ap.add_argument(
    "-b", "--blocksize",
    help="block size in frames",
    type=int,
    default = 8,
)
ap.add_argument(
    "wavfile",
    help="wav input file",
)
args = ap.parse_args()

blocksize = args.blocksize

# Read a wave file.
def read(filename):
    with wave.open(filename, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        nframes = w.getnframes()
        frames = w.readframes(nframes)
        framedata = struct.unpack(f"<{nframes}h", frames)
        samples = [s / (1 << 15) for s in framedata]
        return w.getparams(), samples, w.getframerate()

# Collect the samples.
params, samples, sample_rate = read(args.wavfile)

# Write a wave file.
def write(f, samples, params):
    nframes = len(samples)
    framedata = [int(s * (1 << 15)) for s in samples]
    frames = struct.pack(f"<{nframes}h", *framedata)
    with wave.open(f, "wb") as w:
        w.setparams(params)
        w.writeframes(frames)

# Play a tone on the computer.
def play(samples):
    # Set up and start the stream.
    stream = sounddevice.RawOutputStream(
        samplerate = sample_rate,
        blocksize = blocksize,
        channels = 1,
        dtype = 'float32',
    )
    stream.start()

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
        assert not stream.write(pbuffer), "overrun"

    # Tear down the stream.
    stream.stop()
    stream.close()

outsamples = np.array([0.0] * (len(samples) + blocksize - 1))
for i in range(blocksize):
    outsamples += np.append(np.array([0.0] * (i - 1)), samples[:-i])
outsamples /= blocksize

outsamples = np.clip(outsamples, -0.95, 0.95)

# Play the result.
if args.outfile:
    write(args.outfile, outsamples, params)
play(outsamples)
