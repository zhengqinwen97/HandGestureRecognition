import os
from multiprocessing import Process
import time

# recognition
def run_proc1(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))
    os.system(r"D:\Anaconda\envs\tensorflow_gpu\python.exe D:\Code\Graduation_Project\Gesture_detection_and_classify\Recognition.py")

# tracking
def run_proc2(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))
    os.system(r"D:\Anaconda\envs\tensorflow_gpu\python.exe D:\Code\Graduation_Project\Gesture_detection_and_classify\Tracking.py")

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p1 = Process(target = run_proc1, args=('test',))
    p2 = Process(target = run_proc2, args=('test',))
    print('Process1 will start.')
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print('end process.....')
