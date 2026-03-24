from multiprocessing import Process,Lock
import multiprocessing
import time
import os

def download(i,lock):
    lock.acquire()
    print(f"正在下载文件{i}")
    time.sleep(2)
    print("下载完成")
    lock.release()

if __name__ == '__main__':
    lock = multiprocessing.Lock()
    process1 = Process(target = download,args = (1,lock))
    process2 = Process(target = download,args = (2,lock))
    process3 = Process(target = download,args = (3,lock))
    process4 = Process(target = download,args = (4,lock))
    process5 = Process(target = download,args = (5,lock))
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process5.start()
    process1.join()
    process2.join()
    process3.join()
    process4.join()
    process5.join()