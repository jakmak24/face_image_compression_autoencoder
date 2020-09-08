from termcolor import colored

logFile = None

def open_log(path):
    global logFile
    logFile = open(path+"/log.txt","a")


def info(*msg):
    global logFile
    s = "[I] " + " ".join(list(map(str, msg)))
    print(colored(s, "green"))
    logFile.write(s+"\n")


def debug(*msg):
    global logFile
    s = "[D] " + " ".join(list(map(str, msg)))
    print(colored(s, "yellow"))
    logFile.write(s+"\n")


def error(*msg):
    global logFile
    s = "[E] " + " ".join(list(map(str, msg)))
    print(colored(s, "red"))
    logFile.write(s+"\n")

    
def close_log():
    global logFile
    logFile.close()