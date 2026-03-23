from multiprocessing import Process,Pipe

def child_process(conn):
    conn.send('Hello from child')
    data = conn.recv()
    print('Received',data)
    conn.close()

if __name__ == '__main__':
    parent_conn,child_conn = Pipe()
    p = Process(target = child_process, args = (child_conn, ))
    p.start()

    parent_conn.send('Hello from parent')
    print('Receive',parent_conn.recv())
    p.join()
