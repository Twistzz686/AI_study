from concurrent.futures import ThreadPoolExecutor

def calculate_square(number):
    return number * number

numbers = [1,2,3,4,5,6,7,8,9,10]

executor = ThreadPoolExecutor(max_workers = 5)
results = []
for num in numbers:
    result = executor.submit(calculate_square,num)
    results.append(result)
for res in results:
    print(res.result())
executor.shutdown(wait=True)