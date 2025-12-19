import streamlit as st
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

st.set_page_config(page_title="HeteroSim-X", layout="wide")

st.title("⚡ HeteroSim-X: Heterogeneous Computing Simulator")

st.markdown("""
**Live simulation of CPU–GPU task offloading inspired by OpenCL architecture**
""")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
cpu_cores = st.sidebar.slider("CPU Cores", 1, 8, 4)
gpu_units = st.sidebar.slider("GPU Work Items", 32, 1024, 256, step=32)
task_size = st.sidebar.slider("Task Size (Vector Length)", 10_000, 200_000, 50_000, step=10_000)

# Task Definition
vector_a = np.random.rand(task_size)
vector_b = np.random.rand(task_size)

def cpu_task(chunk):
    time.sleep(0.00001 * len(chunk))  # simulate cache & memory latency
    return np.sum(chunk)

def gpu_task(data):
    start = time.time()
    chunk_size = len(data) // gpu_units
    results = []
    for i in range(gpu_units):
        chunk = data[i*chunk_size:(i+1)*chunk_size]
        results.append(np.sum(chunk))
    time.sleep(0.000002 * len(data))  # simulate global memory latency
    return sum(results), time.time() - start

# CPU Execution
st.subheader("🧠 CPU Execution (Multicore)")
cpu_start = time.time()
chunks = np.array_split(vector_a + vector_b, cpu_cores)

with ThreadPoolExecutor(max_workers=cpu_cores) as executor:
    cpu_results = list(executor.map(cpu_task, chunks))

cpu_time = time.time() - cpu_start
st.write(f"⏱ CPU Time: **{cpu_time:.4f} sec**")

# GPU Execution
st.subheader("🎮 GPU Execution (Simulated OpenCL)")
gpu_result, gpu_time = gpu_task(vector_a + vector_b)
st.write(f"⏱ GPU Time: **{gpu_time:.4f} sec**")

# Performance Comparison
st.subheader("📊 Performance Comparison")
speedup = cpu_time / gpu_time if gpu_time > 0 else 0
st.metric("Speedup (CPU / GPU)", f"{speedup:.2f}x")

# Visualization
fig, ax = plt.subplots()
ax.bar(["CPU", "GPU"], [cpu_time, gpu_time])
ax.set_ylabel("Execution Time (seconds)")
ax.set_title("Heterogeneous Execution Comparison")
st.pyplot(fig)

# Profiling Info
st.subheader("🔍 OpenCL-Inspired Profiling")
st.code(f"""
Platform Model   : Host + Compute Device
Execution Model  : Task + Data Parallelism
Memory Model     : Global / Local / Private (Simulated)
CPU Cores        : {cpu_cores}
GPU Work Items   : {gpu_units}
Task Size        : {task_size}
""")

st.success("Simulation Completed Successfully 🚀")
