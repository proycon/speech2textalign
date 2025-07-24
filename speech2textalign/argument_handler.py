import argparse
import os

def check_arguments(args):
    """check if argument are usable"""
    if not args.filename and not args.input_dir and not args.show_available:
        raise ValueError("either --filename or --input-dir must be provided")
    if args.filename and args.input_dir:
        raise ValueError("either --filename or --input-dir must be provided")
    if args.filename:
        if not os.path.isfile(args.filename):
            raise FileNotFoundError(args.filename,"does not exists")
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            raise FileNotFoundError(args.input_dir,"does not exists")
        if not args.prepare and len(os.listdir(args.input_dir)) == 0:
            raise ValueError("no files in input directory",args.input_dir)
    

def arguments_to_command(args, verbose = False):
    if verbose: print(args)
    command = ""
    if args.filename:
        command += " --filename " + os.path.abspath(args.filename)
    if args.input_dir:
        command += " --input-dir " + os.path.abspath(args.input_dir)
    if args.output_dir:
        command += " --output-dir " + os.path.abspath(args.output_dir)
    else:
        command += " --output-dir " + os.getcwd() + "/"
    if args.model_dir:
        command += " --model-dir " + os.path.abspath(args.model_dir)
    if args.keep_alive_minutes:
        command += " --keep-alive-minutes " + str(args.keep_alive_minutes)
    if args.prepare:
        command += " --prepare " 
    if args.label_timestamps:
        command += " --label-timestamps"
    return command

def transcribe_arguments(verbose = False, add_device_field = False):
    m = "transcribe wav audio file with wav2vec2 "
    p = argparse.ArgumentParser(description=m)
    if add_device_field:
        p.add_argument("--device",type=int,
            help="gpu device number (set -1 for cpu)",default=0)
    p.add_argument("--model-dir", type=str,
        help="directory where the model is located", required = False)
    p.add_argument("--filename",type=str,
        help="audio filename to be transcribed",required = False)
    p.add_argument("--input-dir",type=str,
        help="directory with audio files to be transcribed",required = False)
    p.add_argument("--output-dir",type=str,
        help="directory to store transcription",required = False)
    p.add_argument("--keep-alive-minutes",type=int,
        help="how long the transcriber proces should linger",required = False)
    p.add_argument("--prepare",action="store_true",
        help="start transcriber without audio files in directory",
        required = False)
    p.add_argument("--show-available", action="store_true",
        help="show available and selected gpus", required = False)
    p.add_argument("--label-timestamps", action="store_true",
        help="get timestamps at the label level instead of word leve", 
        required = False)
    args = p.parse_args()
    check_arguments(args)
    return args

