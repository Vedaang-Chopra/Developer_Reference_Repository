import time

#  Threading: - Used to speed up programs(but not necessary, with concurrency!!)
#  Concurrency : - 

start = time.perf_counter()

def do_something(seconds):
    print(f'Sleeping {seconds} second(s)...')
    time.sleep(seconds)
    return f'Done Sleeping...{seconds}'





do_something(5)
do_something(5)
do_something(5)

finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')

# threads = []
# for _ in range(10):
#     t = threading.Thread(target=do_something, args=[1.5])
#     t.start()
#     threads.append(t)

# for thread in threads:
#     thread.join()

