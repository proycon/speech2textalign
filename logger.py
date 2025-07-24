import argument_handler
import os
import socket
import time

log_directory = "/vol/tensusers/mbentum/AUDIOSERVER/LOG/"

def make_activity_log_filename(device):
    name = socket.gethostname()
    filename = log_directory + name + "_" + str(device)
    return filename

def log(line, device):
    t = time.strftime("%Y-%m-%d %H:%M:%S") + "\t" + str(int(time.time())) + "\t"
    line = t + line
    with open(make_activity_log_filename(device), "a") as fout:
        fout.write(line + "\n")

def delta_time_last_activity(gpu):
    filename = log_directory + gpu.name + "_" + str(gpu.device)
    if not os.path.isfile(filename): return None
    with open(filename) as fin:
        t = fin.read().split("\n")
    if len(t) == 0: return None
    t = t[::-1] # reverse log
    i = 0
    # find the last line with an epoch timestamp
    while True:
        if i >= len(t): return None
        items = t[i].split("\t")
        if len(items) > 1: last_time = items[1]
        else: last_time = ""
        try: last_time = int(last_time)
        except ValueError: i += 1
        else: return time.time() - last_time

def show_log(server_name,device, start = None):
    """show the log for a specific gpu / device"""
    if not start: start = time.time()
    filename = log_directory + server_name + "_" + str(device)
    print("showing log",filename)
    if not os.path.isfile(filename):
        with open(filename, "a") as fout:
            fout.close()
    lines = []
    last_change = time.time()
    while True:
        new = _check_log(filename,start,lines)
        done = False
        for line in new:
            print(line)
            lines.append(line)
            if "setting status: closed" in line: done = True
            if time.time() - last_change > 3600: done =True
        if done: 
            print("transcription is done, stop showing log file")
            break
        time.sleep(1)

def _check_log(filename, start_time,read_lines):
    """check whether the log contains new information and return it."""
    with open(filename) as fin:
        lines = fin.read().split("\n")
    selection = []
    for line in lines:
        if not line:continue
        if line in read_lines: continue
        line_time =  int(line.split("\t")[1]) 
        if line_time > start_time: selection.append(line)
    return selection


if __name__ == "__main__":
    """show log of a specific pony / gpu / device """
    args = argument_handler.show_log_arguments()
    if not args.start_time: start_time = time.time()
    else: start_time = args.start_time
    show_log(args.server_name,args.device,start_time)
   
