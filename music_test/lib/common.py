# Audio stream settings
CHUNK = 2048 * 2  # Number of audio samples per frame
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate in Hz

# Overlapping window settings
OVERLAP_RATIO = 0.75  # 75% overlap
HOP_SIZE = int(CHUNK * (1 - OVERLAP_RATIO))
BUFFER_SIZE = CHUNK * 2  # Double chunk for circular buffer

def set_overlap_ratio(overlap_ratio):
    global OVERLAP_RATIO
    OVERLAP_RATIO = overlap_ratio
def get_overlap_ratio():
    return OVERLAP_RATIO

def get_buffer_size():
    return CHUNK * 2
def get_hop_size():
    return int(CHUNK * (1 - OVERLAP_RATIO))

def get_chunk():
    return CHUNK
def set_chunk(chunk):
    global CHUNK
    CHUNK = chunk

def get_rate():
    return RATE
def set_rate(rate):
    global RATE
    RATE = rate

def get_channels():
    return CHANNELS
def set_channels(channels):
    global CHANNELS
    CHANNELS = channels


def clear_line():
    """Clear the current line by printing spaces and returning to start"""
    print(" " * 80, end='\r')
