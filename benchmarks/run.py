import sys,time,statistics,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qkernel.features import *
durations=[]
for _ in range(5):
    start=time.perf_counter()
    Perceptron().fit([[0],[.2],[2.8],[3]],[-1,-1,1,1])
    durations.append(time.perf_counter()-start)
print(json.dumps({'workload':'Perceptron().fit([[0],[.2],[2.8],[3]],[-1,-1,1,1])','repetitions':len(durations),'median_seconds':statistics.median(durations),'min_seconds':min(durations)}))
