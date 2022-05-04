#%%
import time
from distributed import Client, LocalCluster, as_completed
import numpy as np

# print(client)
def task(a, b, c):
    # time.sleep(a)
    a +=10
    b +=10
    c +=10
    return a, b, c

def task2(a):
    a[0] += 1
    a[1] += 1
    return a[0], a[1]

def task3(position, energy):
    print(position)
    print(energy)
    return position[0], energy

if __name__ == '__main__':
    # cluster = LocalCluster()
    client = Client("tcp://localhost:8786")

    # future = client.submit(task, 1)
    # print(future.result())


    # e = range(5)
    # r = range(5)
    # t = range(5)
    # p = (e,r,t)
    # p2 = (5,6)

    # futures = client.map(task2, p2)
    # results = client.gather(futures)
    # np_results = np.array(results)
    # print(results)



    future = client.submit(task3, (1,1,1), 5)
    print(future.result())

