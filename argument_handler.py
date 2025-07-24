import argparse
import glob
import os
import logger
import time

def check_arguments(args):
    """check if argument are usable"""
    if not args.filename and not args.input_dir and not args.show_available:
        raise ValueError("either -filename or -input_dir must be provided")
    if args.filename and args.input_dir:
        raise ValueError("either -filename or -input_dir must be provided")
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
        command += " -filename " + os.path.abspath(args.filename)
    if args.input_dir:
        command += " -input_dir " + os.path.abspath(args.input_dir)
    if args.output_dir:
        command += " -output_dir " + os.path.abspath(args.output_dir)
    else:
        command += " -output_dir " + os.getcwd() + "/"
    if args.model_dir:
        command += " -model_dir " + os.path.abspath(args.model_dir)
    if args.keep_alive_minutes:
        command += " -keep_alive_minutes " + str(args.keep_alive_minutes)
    if args.prepare:
        command += " -prepare " 
    if args.label_timestamps:
        command += " -label_timestamps"
    return command

def transcribe_arguments(verbose = False, add_device_field = False):
    m = "transcribe wav audio file with wav2vec2 "
    p = argparse.ArgumentParser(description=m)
    if add_device_field:
        p.add_argument("-device",type=int,
            help="gpu device number",required = True)
    p.add_argument("-model_dir", type=str,
        help="directory where the model is located", required = False)
    p.add_argument("-filename",type=str,
        help="audio filename to be transcribed",required = False)
    p.add_argument("-input_dir",type=str,
        help="directory with audiofilename to be transcribed",required = False)
    p.add_argument("-output_dir",type=str,
        help="directory to store transcription",required = False)
    p.add_argument("-keep_alive_minutes",type=int,
        help="how long the transcriber proces should linger",required = False)
    p.add_argument("-prepare",action="store_true",
        help="start transcriber without audio files in directory",
        required = False)
    p.add_argument("-show_available", action="store_true",
        help="show available and selected gpus", required = False)
    p.add_argument("-label_timestamps", action="store_true",
        help="get timestamps at the label level instead of word leve", 
        required = False)
    args = p.parse_args()
    check_arguments(args)
    return args

def show_log_arguments():
    m = "show log for gpu device"
    p = argparse.ArgumentParser(description=m)
    p.add_argument("-server_name", type=str,
        help="server where the gpu is located", required = True)
    p.add_argument("-device",type=int,
        help="gpu device number",required = True)
    p.add_argument("-start_time",type=int,
        help="epoch time as integer",required = False)
    args = p.parse_args()
    return args
