#!/usr/bin/env python3
"""
GPU Monitoring Script for RLHF Training
Monitors GPU usage, memory, and performance metrics during training.
"""

import time
import psutil
import GPUtil
import argparse
from datetime import datetime
import json
import os


def get_gpu_info():
    """Get detailed GPU information."""
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return None
        
        gpu_info = []
        for i, gpu in enumerate(gpus):
            info = {
                'id': i,
                'name': gpu.name,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'gpu_load': gpu.load * 100,
                'temperature': gpu.temperature,
                'power_draw': getattr(gpu, 'powerDraw', 0),
                'timestamp': datetime.now().isoformat()
            }
            gpu_info.append(info)
        
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None


def get_system_info():
    """Get system information."""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'memory_percent': memory.percent,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error getting system info: {e}")
        return None


def print_status(gpu_info, system_info, show_details=False):
    """Print current status."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] GPU & System Status:")
    print("-" * 60)
    
    if gpu_info:
        for gpu in gpu_info:
            print(f"GPU {gpu['id']} ({gpu['name']}):")
            print(f"  Memory: {gpu['memory_used']}GB/{gpu['memory_total']}GB ({gpu['memory_percent']:.1f}%)")
            print(f"  Load: {gpu['gpu_load']:.1f}%")
            print(f"  Temp: {gpu['temperature']}°C")
            if gpu['power_draw'] > 0:
                print(f"  Power: {gpu['power_draw']:.1f}W")
    
    if system_info:
        print(f"System:")
        print(f"  CPU: {system_info['cpu_percent']:.1f}%")
        print(f"  Memory: {system_info['memory_used_gb']:.1f}GB/{system_info['memory_total_gb']:.1f}GB ({system_info['memory_percent']:.1f}%)")
    
    if show_details and gpu_info:
        print("\nDetailed GPU Info:")
        for gpu in gpu_info:
            print(f"  GPU {gpu['id']}: {json.dumps(gpu, indent=2)}")


def save_metrics(gpu_info, system_info, output_file):
    """Save metrics to file."""
    if not gpu_info and not system_info:
        return
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'gpu': gpu_info,
        'system': system_info
    }
    
    try:
        # Append to file
        with open(output_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    except Exception as e:
        print(f"Error saving metrics: {e}")


def monitor_loop(interval=5, duration=None, output_file=None, show_details=False):
    """Main monitoring loop."""
    print(f"Starting GPU monitoring...")
    print(f"Interval: {interval}s")
    if duration:
        print(f"Duration: {duration}s")
    if output_file:
        print(f"Output file: {output_file}")
    
    start_time = time.time()
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            # Get current metrics
            gpu_info = get_gpu_info()
            system_info = get_system_info()
            
            # Print status
            print_status(gpu_info, system_info, show_details)
            
            # Save metrics if output file specified
            if output_file:
                save_metrics(gpu_info, system_info, output_file)
            
            # Check if duration exceeded
            if duration and (time.time() - start_time) >= duration:
                print(f"\nMonitoring completed after {duration}s")
                break
            
            # Wait for next iteration
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\nMonitoring stopped by user after {iteration} iterations")
    
    # Print summary
    if output_file and os.path.exists(output_file):
        print(f"\nMetrics saved to: {output_file}")


def analyze_log(log_file):
    """Analyze a monitoring log file."""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    print(f"Analyzing log file: {log_file}")
    
    gpu_memory_usage = []
    gpu_load_usage = []
    timestamps = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                
                if 'gpu' in data and data['gpu']:
                    for gpu in data['gpu']:
                        gpu_memory_usage.append(gpu['memory_percent'])
                        gpu_load_usage.append(gpu['gpu_load'])
                        timestamps.append(data['timestamp'])
    
        if gpu_memory_usage:
            print(f"\nGPU Memory Usage Summary:")
            print(f"  Average: {sum(gpu_memory_usage) / len(gpu_memory_usage):.1f}%")
            print(f"  Max: {max(gpu_memory_usage):.1f}%")
            print(f"  Min: {min(gpu_memory_usage):.1f}%")
            
            print(f"\nGPU Load Summary:")
            print(f"  Average: {sum(gpu_load_usage) / len(gpu_load_usage):.1f}%")
            print(f"  Max: {max(gpu_load_usage):.1f}%")
            print(f"  Min: {min(gpu_load_usage):.1f}%")
            
            print(f"\nMonitoring Duration:")
            print(f"  Start: {timestamps[0]}")
            print(f"  End: {timestamps[-1]}")
            print(f"  Total samples: {len(timestamps)}")
    
    except Exception as e:
        print(f"Error analyzing log: {e}")


def main():
    parser = argparse.ArgumentParser(description="GPU Monitoring for RLHF Training")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, help="Total monitoring duration in seconds")
    parser.add_argument("--output", type=str, help="Output file for metrics")
    parser.add_argument("--details", action="store_true", help="Show detailed information")
    parser.add_argument("--analyze", type=str, help="Analyze existing log file")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_log(args.analyze)
    else:
        monitor_loop(
            interval=args.interval,
            duration=args.duration,
            output_file=args.output,
            show_details=args.details
        )


if __name__ == "__main__":
    main() 