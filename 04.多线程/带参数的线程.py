from threading import Thread
import time

def say_hello(name):
    time.sleep(2)
    print(f"Hello,{name}")
if __name__ == '__main__':
    start = time.time()
    thread1 = Thread(target = say_hello ,args=('Alice', ))
    thread1.start()
    say_hello('bob')
    thread1.join()
    exe_time = start - time.time()
    print(exe_time)