from multiprocessing import Process
import os
import time
def say_hello(name):
    time.sleep(2)
    print(f"Hello {name},Nice to me you")
if __name__ == '__main__':
    start = time.time()
    process = Process(target=say_hello,args=('Alice', ))
    process.start()
    say_hello('Bob')
    process.join()
    exe_time = time.time() - start
    print(exe_time)