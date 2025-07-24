import gpu_logger
import glob
import logger
import os 
import socket
import time
import ponies

pony_dict = ponies.pony_dict

status_dir = "/vol/tensusers/mbentum/AUDIOSERVER/STATUS/"

class Selector:
    """select gpu that is not in use based on log files.
    """
    def __init__(self, to_old = 600, free_memory = 11000, max_load = .1):
        """select available gpu based on restrictions
        to_old          cut of for to old log data default = 10 minutes
                        log data should be update every 5 minutes with
                        a cron job
        free_memory     minimal amount of free gpu memory
        max_load        maximum load of the gpu processing power default 10%
        """
        self.to_old = to_old
        self.free_memory = free_memory
        self.max_load = max_load
        self.filename = gpu_logger.make_filename()
        self.table = load_table(self.filename)
        self.ok = True if self.table else False
        if self.ok: self._make_gpus()
        if self.navailable_gpus == 0 and self.nbusy_gpus == 0: self.ok = False

    def __repr__(self):
        m = "gpu selector | available gpus: " + str(self.navailable_gpus) 
        m += " | busy gpus: " + str(self.nbusy_gpus)
        m += "\nselected gpu:\n"
        if self.selected_gpu: m += self.selected_gpu.__repr__()
        else: m += "none"
        return m

    def _make_gpus(self):
        """load gpu object based on info in the log file.
        gpus are sorted on available free memory
        """
        self.gpus = []
        self.no_memory = []
        self.overload = []
        self.rejected = []
        self.active = []
        for line in self.table[::-1]:
            gpu = Gpu(line)
            reject = False
            if not gpu.ok:continue
            if gpu.name == "mlp10":
                #thunderlane gives an segmentation fault if I try to laod
                # wav2vec2
                reject = True
            if gpu.free_memory < self.free_memory: 
                if gpu not in self.no_memory: self.no_memory.append(gpu)
                reject = True
            if gpu.load > self.max_load: 
                if gpu not in self.overload:self.overload.append(gpu)
                reject = True
            if gpu.delta_time > self.to_old: reject = True
            if gpu.status == "active": 
                if gpu not in self.active:self.active.append(gpu)
                reject = True
            if reject:
                if gpu not in self.rejected:self.rejected.append(gpu)
            else:
                if gpu not in self.gpus: self.gpus.append(gpu)
        self.gpus = sorted(self.gpus, reverse = True)

    @property
    def selected_gpu(self):
        """select available gpu, if available"""
        if len(self.gpus) > 0: return self.gpus[0]
        return False

    @property
    def navailable_gpus(self):
        """number of available gpus."""
        return len(self.gpus)

    @property
    def nbusy_gpus(self):
        """number of busy gpus."""
        return len(self.overload) + len(self.no_memory)

    def device_available(self, device):
        name = socket.gethostname()
        for gpu in self.gpus:
            if gpu.name == name and gpu.device == device: return True
        return False

    def get_available_divice_on_this_server(self):
        name = socket.gethostname()
        for gpu in self.gpus:
            if gpu.name == name: return gpu.device
        return None

class Gpu:
    """object that contains info about a single gpu device."""
    def __init__(self,line):
        """object that contains info about a single gpu device.
        line    one line from the gpu logger file
        """
        self.line = line.split("\t")
        if len(self.line) == 8:
            self._read_line()
        else:self.ok = False

    def __repr__(self):
        m = self.name + " " + self.pony_name.ljust(12) + " "
        m += str(self.device).ljust(3) + " " 
        m += self.readable_time.ljust(9)
        m += " " + str(self.delta_time).ljust(5) 
        m += " " + str(self.load).ljust(5)
        m += " " + str(self.memory_load).ljust(5)
        return m

    def __eq__(self,other):
        """a gpu is identical if the server name and device number are equal."""
        if type(self) != type(other): return False
        return self.name == other.name and self.device == other.device

    def __gt__(self,other):
        """a gpu is larger if it has more memory available."""
        return self.free_memory > other.free_memory

    def _read_line(self):
        """loads the info in the gpu logger line in attributes of the object."""
        names = "name,device,load,memory_load,memory_total,memory_used"
        names += ",temperature,time"
        self.names = names.split(",")
        int_names="device,memory_total,memory_used,temperature".split(",")
        float_names = "load,memory_load,time".split(",")
        for name, value in zip(self.names,self.line):
            if name in int_names: setattr(self,name,int(value))
            elif name in float_names: setattr(self,name,float(value))
            else: setattr(self,name,value)
        self.ok = True
        self.pony_name = pony_dict[self.name]
           
    @property
    def readable_time(self):
        return time.strftime("%H:%M:%S",time.localtime(self.time))

    @property
    def delta_time(self):
        return int(time.time() - self.time)

    @property
    def free_memory(self):
        if not hasattr(self,"memory_total"): return 0
        return self.memory_total - self.memory_used

    @property
    def status(self):
        status = check_status(self)
        delta_time = self.delta_time_last_activity
        if not delta_time or delta_time > 3600:
            status = "closed"
        return status

    @property
    def delta_time_last_activity(self):
        return logger.delta_time_last_activity(self)


def load_table(filename = ""):
    """load the gpu logger file
    this file is appended every 5 minutes; most recent information at the bottom
    updating of the log file is done with a crontab on pipsqueak
    """
    if not filename:
        filename = gpu_logger.make_filename()
    if os.path.isfile(filename):
        with open(filename) as fin:
            table = fin.read().split("\n")
        return table
    else: return False

def check_status(gpu):
    filename = status_dir + gpu.name + "_" + str(gpu.device)
    fn = glob.glob(filename + "*")
    if len(fn) == 0: return None
    status = fn[0].split("_")[-1]
    return status
    

