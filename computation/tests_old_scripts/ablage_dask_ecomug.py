#%%
t = stopwatch.stopwatch(start=True, title='generating ecomug muons', selfexecutiontime_micros=0, time_unit='s', return_results=True)
futures = []
for event in tqdm(range(STATISTICS), disable=False):
    future = client.submit(Ecomug_generate, 0, pure=False)
    futures.append(future)
results = client.gather(futures)
t.stop()
# %%
t = stopwatch.stopwatch(start=True, title='generating ecomug muons', selfexecutiontime_micros=0, time_unit='s', return_results=True)
futures = client.map(Ecomug_generate, da.zeros(STATISTICS, dtype=bool), pure=False)
# futures = client.map(Ecomug_generate, range(STATISTICS), pure=False, batch_size=int(STATISTICS/1))
results = client.gather(futures)
t.stop()
#%%
results = []
for event in tqdm(range(STATISTICS), disable=False):
    result = delayed(Ecomug_generate)(0)
    results.append(result)

results = results.compute()

# %%