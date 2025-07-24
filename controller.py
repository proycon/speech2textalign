import argument_handler 
import gpu_selector as gpus
import logger
import os
import ponies
import sys
import threading
import time

sys.tracebacklimit=0
pony_dict = ponies.pony_dict
log_dir = "/vol/tensusers/mbentum/AUDIOSERVER/LOG/"


def activate_command():
    m="source /vol/tensusers/mbentum/AUDIOSERVER/audioserver_env/bin/activate;"
    return m

def make_ssh_command(gpu,args):
    argument_command = argument_handler.arguments_to_command(args)
    device = gpu.device
    server_name = gpu.name
    m = "ssh " + server_name + " "
    m += """ + activate_command()
    m += "python /vol/tensusers/mbentum/AUDIOSERVER/repo/wav2vec2.py "
    m += "-device " + str(device) + argument_command + """
    return m

def show_log_command(gpu):
    m = "python /vol/tensusers/mbentum/AUDIOSERVER/repo/logger.py "
    m += "-server_name " + gpu.name + " -device " + str(gpu.device)
    return m
    
    
def get_gpus_selector():
    """create gpu selector object, return if there are available gpus,
    otherwise wait and try again.
    """
    s = gpus.Selector()
    if not s.selected_gpu:
        print("could not find an available gpu")
        print("\n".join(map(str,s.rejected)))
        print("waiting a minute to try again")
        time.sleep(60)
        return get_gpus_selector()
    return s

def handle_show_available():
    s = gpus.Selector()
    print("not available")
    print("\n".join(map(str,s.rejected)))
    if s.selected_gpu:
        device = s.selected_gpu.device
        server_name = s.selected_gpu.name
        print("selected gpu: ",server_name,pony_dict[server_name],
            ", device:",device)

def transcribe(args = None, verbose = True):
    if args.show_available: return handle_show_available()
    s = get_gpus_selector()
    device = s.selected_gpu.device
    server_name = s.selected_gpu.name
    print("selected gpu: ",server_name,pony_dict[server_name],
        ", device:",device)
    ssh_command = make_ssh_command(s.selected_gpu,args)
    if verbose: print("ssh command:",ssh_command)
    start = time.time()
    x = threading.Thread(target = logger.show_log, 
        args = (server_name,device,start))
    x.start()
    os.system(ssh_command)






if __name__ == "__main__":
    args = argument_handler.transcribe_arguments()
    transcribe(args)
