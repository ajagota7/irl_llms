#!/usr/bin/env python
"""
Simple GPU monitoring script to track GPU utilization during training.
Run this in a separate terminal while training to monitor GPU usage.
"""

import time
import psutil
import subprocess
import os

def get_gpu_info():
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        memory_used, memory_total, utilization, temperature = parts
                        gpu_info.append({
                            'memory_used_mb': int(memory_used),
                            'memory_total_mb': int(memory_total),
                            'utilization_percent': int(utilization),
                            'temperature_c': int(temperature)
                        })
            return gpu_info
    except Exception as e:
        print(f"Error getting GPU info: {e}")
    return []

def get_system_info():
    """Get system memory and CPU info."""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    return {
        'memory_used_gb': memory.used / (1024**3),
        'memory_total_gb': memory.total / (1024**3),
        'memory_percent': memory.percent,
        'cpu_percent': cpu_percent
    }

def monitor_gpu(interval=2):
    """Monitor GPU and system resources."""
    print("GPU and System Monitor")
    print("=" * 80)
    print("Time\t\tGPU\tMem Used\tMem Total\tUtil%\tTemp°C\tSys Mem%\tCPU%")
    print("-" * 80)
    
    try:
        while True:
            gpu_info = get_gpu_info()
            system_info = get_system_info()
            
            current_time = time.strftime("%H:%M:%S")
            
            if gpu_info:
                for i, gpu in enumerate(gpu_info):
                    mem_used_gb = gpu['memory_used_mb'] / 1024
                    mem_total_gb = gpu['memory_total_mb'] / 1024
                    print(f"{current_time}\tGPU{i}\t{mem_used_gb:.1f}GB\t{mem_total_gb:.1f}GB\t\t{gpu['utilization_percent']}%\t{gpu['temperature_c']}°C\t{system_info['memory_percent']:.1f}%\t{system_info['cpu_percent']:.1f}%")
            else:
                print(f"{current_time}\tNo GPU info available")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor GPU and system resources")
    parser.add_argument("--interval", type=int, default=2, help="Update interval in seconds")
    args = parser.parse_args()
    
    monitor_gpu(args.interval) 