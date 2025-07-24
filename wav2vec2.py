import argument_handler 
import audio
import glob
import sys
from argparse import Namespace
from typing import Optional, Union
import os 
import time
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ProcessorWithLM

""" 
More info about pipelines for ASR see:
https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline
"""
from transformers import AutomaticSpeechRecognitionPipeline


def load_model(recognizer_dir) -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC.from_pretrained(recognizer_dir)
    return model

def load_processor_with_lm(recognizer_dir) -> Wav2Vec2ProcessorWithLM:
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(recognizer_dir)
    return processor
    

def load_processor(recognizer_dir) -> Wav2Vec2Processor:
    processor = Wav2Vec2Processor.from_pretrained(recognizer_dir)
    return processor

def load_pipeline(recognizer_dir: str, model: Optional[Wav2Vec2ForCTC] = None,processor: Optional[Wav2Vec2Processor] = None, 
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
    print("using device:",device, file=sys.stderr)
    if not model:
        print("loading model:",recognizer_dir, file=sys.stderr)
        model = load_model(recognizer_dir)
    if not processor:
        print("loading processor", file=sys.stderr)
        print(recognizer_dir, file=sys.stderr)
        p = load_processor(recognizer_dir)
    else: 
        p = processor 
    print("loading pipeline", file=sys.stderr)
    pipeline = AutomaticSpeechRecognitionPipeline(
        feature_extractor =p.feature_extractor,
        model = model,
        tokenizer = p.tokenizer,
        chunk_length_s = chunk_length_s,
        device = device
    )
    return pipeline


def decode_audiofile(filename: str , pipeline: AutomaticSpeechRecognitionPipeline, start: float =0.0, end: Optional[float] =None, 
                     timestamp_type: str = "word") -> dict[str,Union[str,float,int,bool]]:
    """
    transcribe an audio file with pipeline object
    loads the audio with librosa
    """
    a = audio.load_audio(filename,start,end)
    output = pipeline(a, return_timestamps = timestamp_type)
    return output 


def _make_decoding_output_filename(audio_filename: str, output_dir: str, extension: str) -> str:
    """ 
    make a filename for transcription based on audio_filename 
    and output_dir.
    """
    filename = audio_filename.replace(".wav",extension)
    filename = output_dir + filename.split("/")[-1]
    return filename

def save_pipeline_output_to_files(output: dict,audio_filename: str,output_dir: str =""):
    table = pipeline_output2table(output)
    ctm = pipeline_output2ctm(output,audio_filename)
    save(_table2str(table),audio_filename,".table",output_dir)
    save(_table2str(ctm,sep = " "),audio_filename,".ctm",output_dir)
    save(output["text"],audio_filename,".txt",output_dir)

def save(table_str: str, audio_filename: str, extension: str, output_dir: str = ""):
    """save pipeline output to a file."""
    filename= _make_decoding_output_filename(audio_filename, output_dir,
        extension)
    print("saving to:",filename, file=sys.stderr)
    try:
        with open(filename,"w") as fout:
            fout.write(table_str)
    except PermissionError:
        print("could not write file to", filename, "due to a permission error", file=sys.stderr)

def pipeline_output2table(output: dict) -> list[tuple]:
    """convert pipeline output to table (word\tstart\tend)."""
    table = []
    for d in output["chunks"]:
        start, end = d["timestamp"]
        table.append((d["text"], start, end))
    return table

def stem_filename(filename: str) -> str:
    if "/" in filename: filename = filename.split("/")[-1]
    if "." in filename: 
        filename_components = filename.split(".")[:-1]
        if len(filename_components) > 1: 
            filename = ".".join(filename_components)
        else:
            filename = filename_components[0]
    return filename

def pipeline_output2ctm(output: dict, filename: str) -> list[tuple]:
    filename = stem_filename(filename)
    table = pipeline_output2table(output)
    ctm = []
    for line in table:
        word, start, end = line
        duration = round(end - start,2)
        line = (filename,1,start,duration,word,"1.00")
        ctm.append(line)
    return ctm
    

def _table2str(table: list[tuple], sep = "\t") -> str:
    """convert output table to string."""
    output = []
    for line in table:
        output.append(sep.join(list(map(str,line))))
    return "\n".join(output)

class Transcriber:
    """transcribe audio files in input_dir or the audio file filename."""
    def __init__(self, model_dir: str, input_dir: str, output_dir: str,
                 model: Optional[Wav2Vec2ForCTC] = None, pipeline: Optional[AutomaticSpeechRecognitionPipeline] = None, device: int = -1, filename: str = "",
                 timestamp_type: str = "word"):
        """transcribe audio files in input_dir
        model_dir       directory of the wav2vec2 model
        input_dir       directory for audio files that need to be transcribed
        output_dir      directory for output files
        """
        self.model_dir = model_dir
        self.input_dir = input_dir 
        self.output_dir = output_dir 
        self.device = device
        self.filename = filename
        self.timestamp_type = timestamp_type
        if pipeline:
            self.pipeline = pipeline
        elif model: 
            self.model = model
            self.pipeline = load_pipeline(recognizer_dir= self.model_dir, model = model,device = device)
        else:
            self.pipeline = load_pipeline(recognizer_dir = self.model_dir, device = device)
        self.transcribed_audio_files = {}
        self.did_transcription= False
    
    def load_audio_filenames(self):
        self.ok = True
        if self.input_dir:
            self.audio_filenames = glob.glob(self.input_dir + "*.wav")
        elif self.filename:
            self.audio_filenames = [self.filename]
        else: self.ok = False
        m = "transcribed audio files" 
        m += " ".join(self.transcribed_audio_files.keys())

    def transcribe(self):
        self.load_audio_filenames()
        self.did_transcription= False
        for filename in self.audio_filenames:
            if filename not in self.transcribed_audio_files.keys():
                print(f"transcribing {filename} on device {self.device}", file=sys.stderr)
                try: 
                    o = decode_audiofile(filename, self.pipeline,
                    timestamp_type = self.timestamp_type)
                except ValueError:
                    print(f"failed to transcribe {filename} on device {self.device}",file=sys.stderr)
                    return
                save_pipeline_output_to_files(o, filename,self.output_dir)
                self.transcribed_audio_files[filename] = o
                self.did_transcription = True
                print(f"transcribed {filename} on device {self.device}",file=sys.stderr)


def pre_checks(args: Namespace) -> tuple[int,str,str]:
    """check device and input / output dir"""
    device = args.device
    input_dir, output_dir = args.input_dir, args.output_dir
    if type(input_dir) == str and not input_dir.endswith("/"):
        input_dir += "/"
    if not output_dir: output_dir = os.getcwd() 
    if not output_dir.endswith("/"):
        output_dir += "/"
    return device, input_dir, output_dir
    

def _check_transcriber_ok(transcriber: Transcriber) -> bool:
    transcriber.load_audio_filenames()
    if not transcriber.ok:
        print(f"could not load audiofiles, input_dir: {transcriber.input_dir}, filename: {transcriber.filename}", file=sys.stderr)
        return False
    return True
        

def transcribe(args: Namespace):
    device, input_dir, output_dir = pre_checks(args)
    print(f"keep_alive: {args.keep_alive_minutes} on device {device}", file=sys.stderr)
    if args.keep_alive_minutes == None: 
        args.keep_alive_minutes = 0
    keep_alive_seconds = args.keep_alive_minutes * 60
    timestamp_type = "char" if args.label_timestamps else "word"
    print("using timestamp type:",timestamp_type, file=sys.stderr)
    print("loading transcriber", file=sys.stderr)
    transcriber = Transcriber(args.model_dir, input_dir, output_dir,
        device = device, filename = args.filename, 
        timestamp_type = timestamp_type)
    if not _check_transcriber_ok(transcriber): return
    print("start transcribing", file=sys.stderr)
    last_transcription = time.time()
    while True:
        transcriber.transcribe()
        if transcriber.did_transcription:
            last_transcription = time.time()
        time.sleep(1)
        if time.time() - last_transcription > keep_alive_seconds:
            break
    print("closing down transcriber", file=sys.stderr)
    return transcriber.transcribed_audio_files

if __name__ == "__main__":
    args = argument_handler.transcribe_arguments(add_device_field = True)
    argument_handler.check_arguments(args)
    transcribe(args)

