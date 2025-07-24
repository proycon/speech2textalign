import argparse
import argument_handler 
import audio
import glob
import gpu_selector as gpus
from logger import log
from numba import cuda
import os 
import socket
import time
import torch
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ProcessorWithLM

""" 
More info about pipelines for ASR see:
https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline
"""
from transformers import AutomaticSpeechRecognitionPipeline as ap

default_recognizer_dir = "/vol/bigdata2/datasets2/SSHOC-T44-LISpanel-2021/"
default_recognizer_dir += "TEXT_ANALYSIS/homed_lm_recognizers/cgn/"

status_directory = "/vol/tensusers/mbentum/AUDIOSERVER/STATUS/"

def load_model(recognizer_dir = default_recognizer_dir):
    model = Wav2Vec2ForCTC.from_pretrained(recognizer_dir)
    return model

def load_processor_with_lm(recognizer_dir = default_recognizer_dir):
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(recognizer_dir)
    return processor
    

def load_processor(recognizer_dir = default_recognizer_dir):
    processor = Wav2Vec2Processor.from_pretrained(recognizer_dir)
    return processor

def load_pipeline(recognizer_dir=None, model = None,processor = None, 
    chunk_length_s = 10, 
    device = -1):
    """
    loads a pipeline object that can transcribe audio files
    recognizer_dir      directory that stores the wav2vec2 model
    model               preloaded wav2vec model (speeds up loading pipeline)
    chunk_length_s      chunking duration of long audio files
                        wav2vec2 is memory hungry, the pipeline employs
                        a sliding window to handle long audio files
                        and the edge effects from chunking
    """
    print("using device:",device)
    log("using device: "+ str(device),device)
    if not recognizer_dir: recognizer_dir = default_recognizer_dir
    if not model:
        print("loading model:",recognizer_dir)
        log("loading model: "+ str(recognizer_dir), device)
        model = load_model(recognizer_dir)
    if not processor:
        print("loading processor")
        log("loading processor",device)
        print(recognizer_dir)
        p= load_processor(recognizer_dir)
    else: p = processor 
    print("loading pipeline")
    log("loading pipeline",device)
    pipeline = ap(
        feature_extractor =p.feature_extractor,
        model = model,
        tokenizer = p.tokenizer,
        chunk_length_s = chunk_length_s,
        device = device
    )
    return pipeline


def decode_audiofile(filename, pipeline, start=0.0, end=None, 
    timestamp_type = "word"):
    """
    transcribe an audio file with pipeline object
    loads the audio with librosa
    """
    a = audio.load_audio(filename,start,end)
    output = pipeline(a, return_timestamps = timestamp_type)
    return output 


def _make_decoding_output_filename(audio_filename, output_dir, extension):
    """ 
    make a filename for transcription based on audio_filename 
    and output_dir.
    """
    filename = audio_filename.replace(".wav",extension)
    filename = output_dir + filename.split("/")[-1]
    return filename

def save_pipeline_output_to_files(output,audio_filename,output_dir=""):
    table = pipeline_output2table(output)
    ctm = pipeline_output2ctm(output,audio_filename)
    save(_table2str(table),audio_filename,".table",output_dir)
    save(_table2str(ctm,sep = " "),audio_filename,".ctm",output_dir)
    save(output["text"],audio_filename,".txt",output_dir)

def save(t, audio_filename, extension, output_dir = ""):
    """save pipeline output to a file."""
    filename= _make_decoding_output_filename(audio_filename, output_dir,
        extension)
    print("saving to:",filename)
    try:
        with open(filename,"w") as fout:
            fout.write(t)
    except PermissionError:
        print("could not write file to", filename, "due to a permission error")

def pipeline_output2table(output):
    """convert pipeline output to table (word\tstart\tend)."""
    table = []
    for d in output["chunks"]:
        start, end = d["timestamp"]
        table.append([d["text"], start, end])
    return table

def stem_filename(filename):
    if "/" in filename: filename = filename.split("/")[-1]
    if "." in filename: 
        filename= filename.split(".")[:-1]
        if len(filename) > 1: filename = ".".join(filename)
        else: filename = filename[0]
    return filename

def pipeline_output2ctm(output, filename):
    filename = stem_filename(filename)
    table = pipeline_output2table(output)
    ctm = []
    for line in table:
        word, start, end = line
        duration = round(end - start,2)
        line = [filename,1,start,duration,word,"1.00"]
        ctm.append(line)
    return ctm
    

def _table2str(table, sep = "\t"):
    """convert output table to string."""
    output = []
    for line in table:
        output.append(sep.join(list(map(str,line))))
    return "\n".join(output)

class Transcriber:
    """transcribe audio files in input_dir or the audio file filename."""
    def __init__(self, model_dir = None, input_dir = None, output_dir = None,
        model = None, pipeline = None, device = -1, filename = "",
        timestamp_type = "word"):
        """transcribe audio files in input_dir
        model_dir       directory of the wav2vec2 model
        input_dir       directory for audio files that need to be transcribed
        output_dir      directory for output files
        """
        self.model_dir = model_dir if model_dir else default_recognizer_dir
        self.input_dir = input_dir 
        self.output_dir = output_dir 
        self.device = device
        self.filename = filename
        self.timestamp_type = timestamp_type
        if pipeline: self.pipeline = pipeline
        elif model: 
            self.model = model
            self.pipeline = load_pipeline(model = model,device = device)
        else: self.pipeline = load_pipeline(recognizer_dir = self.model_dir,
            device = device)
        self.transcribed_audio_files = {}
        self.did_transcription= False
    
    def load_audio_filenames(self):
        self.ok = True
        if self.input_dir:
            self.audio_filenames = glob.glob(self.input_dir + "*.wav")
        elif self.filename:
            self.audio_filenames = [self.filename]
        else: self.ok = False
        m ="transcribed audio files" 
        m += " ".join(self.transcribed_audio_files.keys())

    def transcribe(self):
        self.load_audio_filenames()
        self.did_transcription= False
        for filename in self.audio_filenames:
            if filename not in self.transcribed_audio_files.keys():
                print("transcribing: ",filename)
                log("transcribing: "+filename,self.device)
                try: o = decode_audiofile(filename, self.pipeline,
                    timestamp_type = self.timestamp_type)
                except ValueError:
                    log("failed to transcribe : "+filename,self.device)
                save_pipeline_output_to_files(o, filename,self.output_dir)
                self.transcribed_audio_files[filename] = o
                self.did_transcription = True
                log("transcribed: "+filename,self.device)
                

def check_device(device):
    """
    Check whether the gpu device is valid and available, otherwise tries
    to find an available device.
    if device equals -1, cpu is used for decoding
    """
    print("checking gpu device")
    log("checking gpu device",device)
    s = gpus.Selector()
    if device == None:
        print("no device specified, trying to find an available gpu")
        log("no device specified, trying to find an available gpu",device)
        output = s.get_available_divice_on_this_server()
        log("found: "+str(output),device)
    elif device == -1:
        print("running on cpu, this will be slow")
        output=device
    elif type(args.device) == int:
        if type(s.device_available(device)): output = device
        else: output = None 
    if output == None:
        print("could not find a device, doing nothing. Device:",output)
        log("could not find a device, doing nothing. Device: "+str(output),
            device)
    return output


def remove_status(device):
    """ removes the old status of the transcribe function in the status_directory. """
    print("removing current status:")
    server_name = socket.gethostname()
    fn = glob.glob(status_directory + server_name + "_" + device + "_*")
    print("\n".join(fn))
    log("removing current status: " + "\t".join(fn),device)
    for f in fn:
        os.system("rm " + f)

def set_status(device,status):
    """ sets the status of transcribe function in the status_directory. """
    if device == "None": return
    remove_status(device)
    print("setting status:", status)
    log("setting status: " + status,device)
    server_name= socket.gethostname()
    f =open(status_directory + server_name + "_" + device + "_" + status,"w")
    f.close()

def pre_checks(args):
    """check device and input / output dir"""
    set_status(str(args.device),"started")
    device = check_device(args.device)
    input_dir, output_dir = args.input_dir, args.output_dir
    if device == None: 
        raise ValueError("could not start device:",device)
    if type(input_dir) == str and not input_dir.endswith("/"):
        input_dir += "/"
    if not output_dir: output_dir = os.getcwd() 
    if not output_dir.endswith("/"):
        output_dir += "/"
    if not args.filename or type(args.filename) == str: filename = ""
    return device, input_dir, output_dir, filename
    

def _check_transcriber_ok(transcriber):
    transcriber.load_audio_filenames()
    if not transcriber.ok:
        m ="could not load audiofiles, input_dir: "+str(transcriber.input_dir)
        m += ", filename: " + transcriber.filename
        log(m, device)
        log("closing down transcriber", transcriber.device)
        set_status(str(transcriber.device),"closed")
        return False
    return True
        

def transcribe(args):
    device, input_dir, output_dir, filename = pre_checks(args)
    log("keep_alive: "+str(args.keep_alive_minutes), device)
    if args.keep_alive_minutes == None: args.keep_alive_minutes = 0
    keep_alive_seconds = args.keep_alive_minutes * 60
    timestamp_type = "char" if args.label_timestamps else "word"
    print("using timestamp type:",timestamp_type)
    set_status(str(device),"active")
    print("loading transcriber")
    log("loading transcriber", device)
    transcriber = Transcriber(args.model_dir, input_dir, output_dir,
        device = device, filename = args.filename, 
        timestamp_type = timestamp_type)
    if not _check_transcriber_ok(transcriber): return
    print("start transcribing")
    log("start transcribing", device)
    log("keep_alive: "+str(keep_alive_seconds), device)
    last_transcription = time.time()
    log("start transcription: "+str(last_transcription), device)
    while True:
        transcriber.transcribe()
        if transcriber.did_transcription:
            last_transcription = time.time()
        time.sleep(1)
        if time.time() - last_transcription > keep_alive_seconds:
            break
    print("closing down transcriber")
    log("closing down transcriber", device)
    set_status(str(device),"closed")
    return transcriber.transcribed_audio_files
            
    
                
    
        
if __name__ == "__main__":
    args = argument_handler.transcribe_arguments(add_device_field = True)
    argument_handler.check_arguments(args)
    transcribe(args)

