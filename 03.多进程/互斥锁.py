from multiprocessing import Queue,current_process
import os
import time
import multiprocessing

def task(lock,queue,amount):
    while True:
        lock.acquire()
        money =queue.get()
        if money >= amount:
            money -= amount
            print(f"{current_process().name}缺了{amount},还有{money}")
        else:
            print("余额不足，取款失败")
            queue.put(money)
            lock.release()
            break
        queue.put(money)
        time.sleep(1)
        lock.release()

if __name__ == '__main__':
    count = 1000
    queue = Queue(5)
    queue.put(count)
    lock = multiprocessing.Lock()
    t1 = multiprocessing.Process(target = task,args=(lock,queue,50),name = '张三')
    t2 = multiprocessing.Process(target = task,args=(lock,queue,100),name='李四')
    t1.start()
    t2.start()
    t1.join()
    t2.join()