import textgrids


def ctm_to_textgrid(filename, tier_name = "ctm_words", textgrid = None,
    output_dir = ""):
    """converts a ctm file to a textgrid which can be viewed with praat.
    a ctm file a standard ASR output
    filename nchannels start_time duration word confidence
    """
    intervals = load_ctm(filename)
    tg = intervals_to_textgrid(intervals,filename,tier_name,textgrid,output_dir)
    return tg

def table_to_textgrid(filename, tier_name = "table_words", textgrid = None,
    output_dir = ""):
    """converts a table file to a textgrid which can be viewed with praat.
    a table file is created by the wav2vec2 audio server code
    word start_time end_time
    """
    intervals = load_table(filename)
    tg = intervals_to_textgrid(intervals,filename,tier_name,textgrid,output_dir)
    return tg

def wav2vec_to_textgrid(output, tier_name = "words", textgrid = None,
    output_dir = ""):
    """converts the output from wav2vec to a textgrid.
    output is a dict with a key "chunks" which contains a list words
    with start and end times
    """
    intervals = wav2vec_output_to_intervals(output)
    tg = intervals_to_textgrid(intervals,filename,tier_name,textgrid,output_dir)
    return tg

def intervals_to_textgrid(intervals,filename, tier_name,textgrid=None,
    output_dir=""):
    """creates a textgrid based on a list of intervals."""
    output_filename = input_filename_to_textgrid_filename(filename, output_dir)
    textgrid = add_tier(intervals, tier_name, textgrid)
    textgrid = add_xmin_xmin_to_textgrid(textgrid)
    print("saving textgrid:",output_filename)
    textgrid.write(output_filename)
    return textgrid

def wav2vec_output_to_intervals(output):
    """converts the wav2vec output (see wav2vec_to_textgrid) to a list of
    intervals.
    """
    table = pipeline_output2table(output)
    intervals = []
    for line in table:
        word, start, end = line
        intervals.append( make_interval(word, start, end) )
    return intervals

def load(filename):
    """loads a tsv file into a list of lists."""
    return [line.split("\t") for line in open(filename).read().split("\n")]


def load_ctm(filename):
    """loads a ctm file and stores each line in an interval object."""
    t = load(filename)
    intervals = []
    for line in t:
        word = line[4]
        start = round( float(line[2]), 2)
        duration = round( float(line[3]), 2)
        end = round( start + duration, 2)
        intervals.append( make_interval(word, start, end) )
    return intervals 


def load_table(filename):
    """loads a table file and stores each line in an interval object."""
    t = load(filename)
    intervals = []
    for line in t:
        word = line[0]
        start = round( float(line[1]), 2)
        end = round( float(line[2]), 2)
        intervals.append(  make_interval(word, start, end) )
    return intervals

def pipeline_output2table(output):
    """convert pipeline output to table (word,start,end)."""
    table = []
    for d in output["chunks"]:
        start, end = d["timestamp"]
        table.append([d["text"], start, end])
    return table

def input_filename_to_textgrid_filename(filename, output_dir = ""):
    """creates textgrid filename based on the input filename and optionally
    an output dir.
    """
    stem = ".".join(filename.split(".")[:-1])
    filename = stem + ".textgrid"
    if output_dir: 
        if not output_dir.endswith("/"): output_dir += "/"
        filename = output_dir + filename.split("/")[-1]
    return filename

def make_interval(text,start,end):
    """creates an interval, an object that stores text and start and end time.
    """
    return textgrids.Interval(text, start, end)

def add_tier(intervals, tier_name, textgrid = None):
    """creates a new tier based on a list of intervals
    returns a textgrid with the new tier.
    """
    if not textgrid: textgrid = textgrids.TextGrid()
    textgrid[tier_name] = textgrids.Tier()
    textgrid[tier_name].extend(intervals)
    return textgrid

def add_xmin_xmin_to_textgrid(textgrid):
    """a textgrid needs a xmin and xmax.
    this function finds the earliest xmin and latest xmax in all tiers
    and stores them on the textgrid
    """
    xmin = 10**27
    xmax = 0
    for tier in textgrid.values():
        start = sorted(tier, key = lambda x:x.xmin)[0].xmin
        if start < xmin: xmin = start
        end = sorted(tier, key = lambda x:x.xmax)[-1].xmax
        if end < xmax: xmax = end
    textgrid.xmin = xmin
    textgrid.xmax = xmax
    return textgrid
        

    
    
