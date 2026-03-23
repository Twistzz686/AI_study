from multiprocessing import Process
import os
def say():
    print("hello")
if __name__ == '__main__':
    process = Process(target=say)
    process.start()
    say()
    process.join()
