import argparse
import GPUtil as gpu_util
import socket
import time

output_directory = "/vol/tensusers/mbentum/AUDIOSERVER/gpu_table/"

def gpu_to_table_line(name,gpu):
    line = [name,gpu.id,round(gpu.load,2)]
    line += [round(gpu.memoryUtil,2),int(gpu.memoryTotal)]
    line += [int(gpu.memoryUsed),int(gpu.temperature),int(time.time())]
    line = map(str,line)
    return "\t".join(line)

def make_filename():
    filename = time.strftime("gpu-%Y-%m-%d")
    return output_directory + filename

def write_gpu_line(name,gpu):
    l = gpu_to_table_line(name,gpu)
    filename = make_filename()
    with open(filename,"a") as fout:
        fout.write(l +"\n")

def log_gpus(name = None, directory = None):
    if not name: name = socket.gethostname()
    if directory and type(directory) == str and os.path.isdir(directory):
        global output_directory
        output_directory = directory
    gpus = gpu_util.getGPUs()
    for gpu in gpus:
        write_gpu_line(name,gpu)


if __name__ == "__main__":
    m = "log gpu usage to: (default dir) " + output_directory
    p = argparse.ArgumentParser(description=m)
    p.add_argument("-output_dir",type=str,
        help="directory to store gpu log",required = False)
    p.add_argument("-name", type=str,
        help="server name to be stored in log", required = False)
    args = p.parse_args()
    log_gpus(args.name,args.output_dir)
    



