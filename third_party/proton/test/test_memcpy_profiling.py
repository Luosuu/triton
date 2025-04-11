#!/usr/bin/env python3
"""
Test script for memory copy profiling in Proton.

This script tests the functionality of memory copy operation tracking
in the Proton profiler by performing various memory transfer operations
and verifying that they are correctly captured in the profiling data.
"""

import os
import json
import torch
import triton.profiler as proton

# File to store profiling results
PROFILE_NAME = "memcpy_profile"

def pretty_bytes(num_bytes):
    """Convert bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"

def test_memory_copy_profiling():
    """Test memory copy operation profiling."""
    print("Testing memory copy profiling...")
    
    # Start the profiler
    session_id = proton.start(PROFILE_NAME)
    
    # Create some tensors for memory operations
    host_tensor_a = torch.ones(1024, 1024, dtype=torch.float32)
    host_tensor_b = torch.zeros(1024, 1024, dtype=torch.float32)
    
    # Perform different types of memory transfers
    with proton.scope("host_to_device_transfers"):
        # Host to Device transfer - 4MB
        device_tensor_a = host_tensor_a.cuda()
        # Another Host to Device transfer - 4MB
        device_tensor_b = torch.zeros(1024, 1024, dtype=torch.float32, device="cuda")
        device_tensor_b.copy_(host_tensor_a)
    
    with proton.scope("device_to_host_transfers"):
        # Device to Host transfer - 4MB
        host_tensor_b.copy_(device_tensor_a)
    
    with proton.scope("device_to_device_transfers"):
        # Device to Device transfer - 4MB
        device_tensor_c = torch.zeros(1024, 1024, dtype=torch.float32, device="cuda")
        device_tensor_c.copy_(device_tensor_a)
    
    # Ensure operations are completed
    torch.cuda.synchronize()
    
    # Finalize the profiling session
    proton.finalize(session_id)
    
    # Check if the profile file was created
    profile_path = f"{PROFILE_NAME}.hatchet"
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"Profile file {profile_path} not found")
    
    # Analyze the profile data
    with open(profile_path, 'r') as f:
        data = json.load(f)
    
    # Process and verify the results
    memory_ops = {}
    memory_ops_count = 0
    
    # Function to recursively search for memory operations in the profiling data
    def find_memory_ops(node):
        nonlocal memory_ops_count
        
        # Check if this node has metrics that might indicate a memory copy
        if 'metrics' in node and 'direction' in node.get('metrics', {}):
            direction = node['metrics']['direction']
            bytes_transferred = node['metrics'].get('bytes', 0)
            
            # Add to our counts
            if direction not in memory_ops:
                memory_ops[direction] = {'count': 0, 'bytes': 0}
            
            memory_ops[direction]['count'] += 1
            memory_ops[direction]['bytes'] += bytes_transferred
            memory_ops_count += 1
        
        # Recursively check children
        for child in node.get('children', []):
            find_memory_ops(child)
    
    # Start the recursive search from the root node
    find_memory_ops(data[0])
    
    # Print the results
    print(f"Memory copy operations detected: {memory_ops_count}")
    for direction, stats in memory_ops.items():
        print(f"- {direction}: {stats['count']} operations, total {pretty_bytes(stats['bytes'])}")
    
    # Verify the expected results
    expected_directions = {"HostToDevice", "DeviceToHost", "DeviceToDevice"}
    actual_directions = set(memory_ops.keys())
    
    if not actual_directions.issuperset(expected_directions):
        missing = expected_directions - actual_directions
        print(f"ERROR: Missing memory copy directions: {missing}")
        return False
    
    # Check if we found at least the expected number of operations
    if memory_ops_count < 4:  # We performed 4 memory operations in the test
        print(f"ERROR: Expected at least 4 memory operations, found {memory_ops_count}")
        return False
    
    print("Test passed!")
    return True

if __name__ == "__main__":
    success = test_memory_copy_profiling()
    if not success:
        exit(1)