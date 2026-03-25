from concurrent.futures import ProcessPoolExecutor

def compute_square(n):
    return n * n

if __name__ == '__main__':
    numbers = [1,2,3,4,5]
    results = []

    with ProcessPoolExecutor(max_workers = 3) as executor:
        for num in numbers:
            future = executor.submit(compute_square, num)
            results.append(future)
            for res in results:
                print(res.result())