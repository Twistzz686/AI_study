import time
from threading import Thread

def aa():
    print("666")
    time.sleep(2)
def bb():
    print("999")
    time.sleep(2)
if __name__ == '__main__':
    start_time = time.time()
    thread1 = Thread(target = bb)
    thread1.start()
    aa()
    thread1.join()
    s = time.time()-start_time
    print(s)