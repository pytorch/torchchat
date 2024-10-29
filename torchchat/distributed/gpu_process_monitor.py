import os
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import psutil
from tabulate import tabulate


def get_nvidia_pids():
    """
    Get all PIDs currently using NVIDIA GPUs.

    Returns:
        dict: Dictionary mapping PIDs to their GPU info
    """
    try:
        # Run nvidia-smi with process info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,gpu_uuid,gpu_name,gpu_bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("Error running nvidia-smi. Make sure NVIDIA drivers are installed.")
            return {}

        gpu_processes = {}
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                # Parse the CSV output
                pid, memory, gpu_uuid, gpu_name, gpu_bus = [
                    x.strip() for x in line.split(",")
                ]
                try:
                    pid = int(pid)
                    gpu_processes[pid] = {
                        "gpu_memory": float(memory),
                        "gpu_name": gpu_name,
                        "gpu_uuid": gpu_uuid,
                        "gpu_bus": gpu_bus,
                    }
                except ValueError:
                    continue

        return gpu_processes

    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA drivers are installed.")
        return {}
    except Exception as e:
        print(f"Error getting GPU processes: {str(e)}")
        return {}


def get_gpu_memory_summary():
    """
    Get total GPU memory usage and available memory.

    Returns:
        dict: Dictionary containing GPU memory information
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=gpu_name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {}

        gpu_memory = {}
        for i, line in enumerate(result.stdout.strip().split("\n")):
            if line.strip():
                name, used, total = [x.strip() for x in line.split(",")]
                gpu_memory[i] = {
                    "name": name,
                    "used": float(used),
                    "total": float(total),
                }
        return gpu_memory

    except Exception as e:
        print(f"Error getting GPU memory info: {str(e)}")
        return {}


def get_process_folder_info(pid):
    """
    Get folder information for a running process.

    Args:
        pid (int): Process ID to look up

    Returns:
        dict: Dictionary containing process folder information
    """
    try:
        process = psutil.Process(pid)

        # Basic info that should always be available
        info = {
            "pid": pid,
            "name": "Unknown",
            "exe": "Unknown",
            "cwd": "Unknown",
            "username": "Unknown",
            "create_time": "Unknown",
            "status": "Unknown",
            "open_files": None,
            "memory_info": "Unknown",
            "cpu_percent": "Unknown",
            "cmdline": "Unknown",
            "gpu_info": None,
        }

        # Try to get each piece of information separately
        try:
            info["name"] = process.name()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        try:
            info["exe"] = process.exe()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        try:
            info["cwd"] = process.cwd()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        try:
            info["username"] = process.username()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        try:
            info["create_time"] = datetime.fromtimestamp(
                process.create_time()
            ).strftime("%Y-%m-%d %H:%M:%S")
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        try:
            info["status"] = process.status()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        try:
            info["open_files"] = [f.path for f in process.open_files()]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        try:
            memory = process.memory_info()
            info["memory_info"] = f"{memory.rss / 1024 / 1024:.1f} MB"
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        try:
            info["cpu_percent"] = f"{process.cpu_percent(interval=0.1)}%"
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        try:
            info["cmdline"] = " ".join(process.cmdline())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        return info

    except psutil.NoSuchProcess:
        return None
    except Exception as e:
        print(f"Unexpected error for PID {pid}: {str(e)}")
        return None


def get_all_processes_info(filter_name=None):
    """
    Get folder information for all running processes.

    Args:
        filter_name (str): Optional filter to only show processes containing this name

    Returns:
        list: List of dictionaries containing process information
    """
    processes = []

    # Get list of all PIDs first
    for pid in psutil.pids():
        try:
            # If filter is specified, check process name first
            if filter_name:
                try:
                    proc = psutil.Process(pid)
                    if filter_name.lower() not in proc.name().lower():
                        continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Get detailed info
            info = get_process_folder_info(pid)
            if info:
                processes.append(info)

        except Exception as e:
            print(f"Error processing PID {pid}: {str(e)}")
            continue

    return processes


def print_detailed_process_info(process_info, gpu_info=None):
    """
    Print detailed information for a single process.

    Args:
        process_info (dict): Dictionary containing process information
        gpu_info (dict): Dictionary containing GPU information for the process
    """
    if not process_info:
        print("Process not found or access denied.")
        return

    # Print basic process information
    print("\nProcess Details:")
    print("=" * 50)
    print(f"PID: {process_info['pid']}")
    print(f"Name: {process_info['name']}")
    print(f"Status: {process_info['status']}")
    print(f"Username: {process_info['username']}")
    print(f"Created: {process_info['create_time']}")
    print(f"Memory Usage: {process_info['memory_info']}")
    print(f"CPU Usage: {process_info['cpu_percent']}")

    # Print GPU information if available
    if gpu_info:
        print("\nGPU Information:")
        print(f"GPU Name: {gpu_info['gpu_name']}")
        print(f"GPU Memory Usage: {gpu_info['gpu_memory']} MB")
        print(f"GPU Bus ID: {gpu_info['gpu_bus']}")

    print("\nPaths:")
    print(f"Executable: {process_info['exe']}")
    print(f"Working Directory: {process_info['cwd']}")
    print(f"Command Line: {process_info['cmdline']}")

    # Print open files if available
    if process_info["open_files"]:
        print("\nOpen Files:")
        for file in process_info["open_files"]:
            print(f"  - {file}")
    else:
        print("\nOpen Files: None or Access Denied")


def print_process_info(process_list, show_files=False):
    """
    Print process information in a formatted table.

    Args:
        process_list (list): List of process information dictionaries
        show_files (bool): Whether to show open files in the output
    """
    if not process_list:
        print("No processes found matching criteria.")
        return

    # Prepare data for tabulate
    headers = [
        "PID",
        "Name",
        "Username",
        "Status",
        "Created",
        "Memory",
        "CPU",
        "Executable",
    ]
    rows = []

    for proc in process_list:
        row = [
            proc["pid"],
            proc["name"],
            proc["username"],
            proc["status"],
            proc["create_time"],
            proc["memory_info"],
            proc["cpu_percent"],
            proc["exe"][:50] + "..." if len(proc["exe"]) > 50 else proc["exe"],
        ]
        rows.append(row)

    # Print main process table
    print("\nRunning Processes:")
    print(
        tabulate(
            rows,
            headers=headers,
            tablefmt="grid",
            maxcolwidths=[None, None, None, None, None, None, None, 50],
        )
    )

    # Print open files if requested
    if show_files:
        print("\nOpen Files by Process:")
        for proc in process_list:
            if proc["open_files"]:
                print(f"\nPID {proc['pid']} ({proc['name']}):")
                for file in proc["open_files"]:
                    print(f"  - {file}")


def print_gpu_processes():
    """
    Print information about all processes using NVIDIA GPUs with memory summaries.
    """
    # Get GPU process information
    gpu_processes = get_nvidia_pids()

    if not gpu_processes:
        print("No GPU processes found or nvidia-smi not available.")
        return

    # Get GPU memory summary
    gpu_memory = get_gpu_memory_summary()

    # Initialize counters for memory usage by username
    username_gpu_memory = defaultdict(float)
    username_system_memory = defaultdict(float)
    username_processes = defaultdict(int)

    print("\nGPU Processes:")
    print("=" * 100)

    # Get process information for each GPU process
    for pid, gpu_info in gpu_processes.items():
        process_info = get_process_folder_info(pid)
        if process_info:
            print_detailed_process_info(process_info, gpu_info)
            print("=" * 100)

            # Accumulate memory usage by username
            username = process_info["username"]
            username_gpu_memory[username] += gpu_info["gpu_memory"]

            # Convert system memory from string to float (remove "MB" and convert)
            try:
                sys_memory = float(process_info["memory_info"].replace(" MB", ""))
                username_system_memory[username] += sys_memory
            except (ValueError, AttributeError):
                pass

            username_processes[username] += 1

    # Print GPU summary
    print("\nGPU Memory Summary:")
    print("=" * 50)
    for gpu_id, mem_info in gpu_memory.items():
        print(f"GPU {gpu_id} ({mem_info['name']}):")
        print(f"  Used: {mem_info['used']:.1f} MB / {mem_info['total']:.1f} MB")
        print(f"  Utilization: {(mem_info['used'] / mem_info['total'] * 100):.1f}%")

    # Print user summary
    print("\nUser Memory Summary:")
    print("=" * 50)
    headers = ["Username", "Process Count", "GPU Memory (MB)", "System Memory (MB)"]
    rows = []

    total_gpu_mem = 0
    total_sys_mem = 0
    total_processes = 0

    for username in sorted(username_gpu_memory.keys()):
        gpu_mem = username_gpu_memory[username]
        sys_mem = username_system_memory[username]
        proc_count = username_processes[username]

        rows.append([username, proc_count, f"{gpu_mem:.1f}", f"{sys_mem:.1f}"])

        total_gpu_mem += gpu_mem
        total_sys_mem += sys_mem
        total_processes += proc_count

    # Add totals row
    rows.append(
        ["TOTAL", total_processes, f"{total_gpu_mem:.1f}", f"{total_sys_mem:.1f}"]
    )

    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Display folder information for running processes"
    )
    parser.add_argument(
        "--pid", "-p", type=int, help="Show detailed information for specific PID"
    )
    parser.add_argument(
        "--gpu", "-g", action="store_true", help="Show only GPU processes"
    )
    parser.add_argument("--filter", "-f", help="Filter processes by name")
    parser.add_argument("--files", "-o", action="store_true", help="Show open files")
    args = parser.parse_args()

    try:
        if args.gpu:
            # Show only GPU processes
            print_gpu_processes()
        elif args.pid:
            # Get and print info for specific PID
            process_info = get_process_folder_info(args.pid)
            gpu_processes = get_nvidia_pids()
            gpu_info = gpu_processes.get(args.pid)
            print_detailed_process_info(process_info, gpu_info)
        else:
            # Get all process information
            processes = get_all_processes_info(filter_name=args.filter)

            # Sort processes by PID
            processes.sort(key=lambda x: x["pid"])

            # Print the information
            print_process_info(processes, show_files=args.files)

            # Print summary
            print(f"\nTotal processes: {len(processes)}")
            if args.filter:
                print(f"Filter applied: {args.filter}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
