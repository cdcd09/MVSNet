#!/usr/bin/env python
"""
Simple point cloud generation from depth maps
"""

import sys
import numpy as np
import cv2
import os

# Avoid tensorflow argparse conflicts
sys.argv = [sys.argv[0]]

from preprocess import load_pfm, load_cam

def depth_to_pointcloud(depth_map, cam, prob_map=None, prob_threshold=0.1):
    """Convert depth map to 3D point cloud"""
    
    # Get camera parameters
    intrinsic = cam[1][:3, :3]
    extrinsic = cam[0]
    
    height, width = depth_map.shape
    
    # Create mesh grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply probability mask if provided
    if prob_map is not None:
        mask = (depth_map > 0) & (prob_map > prob_threshold)
    else:
        mask = depth_map > 0
    
    # Get valid pixels
    u_valid = u[mask]
    v_valid = v[mask] 
    depth_valid = depth_map[mask]
    
    if len(u_valid) == 0:
        return np.empty((0, 3))
    
    # Back-project to 3D
    x = (u_valid - intrinsic[0, 2]) * depth_valid / intrinsic[0, 0]
    y = (v_valid - intrinsic[1, 2]) * depth_valid / intrinsic[1, 1]
    z = depth_valid
    
    # Points in camera coordinates
    points_cam = np.stack([x, y, z], axis=1)
    
    # Transform to world coordinates
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3:4]
    
    points_world = (R.T @ points_cam.T - R.T @ t).T
    
    return points_world

def generate_pointcloud(dense_folder, prob_threshold=0.1):
    """Generate point cloud from all depth maps"""
    
    image_folder = os.path.join(dense_folder, 'images')
    cam_folder = os.path.join(dense_folder, 'cams')
    depth_folder = os.path.join(dense_folder, 'depths_mvsnet')
    
    all_points = []
    
    # Process each image
    image_names = sorted(os.listdir(image_folder))
    for image_name in image_names:
        if not image_name.endswith(('.jpg', '.png')):
            continue
            
        print(f"Processing {image_name}")
        
        image_prefix = os.path.splitext(image_name)[0]
        
        # File paths
        depth_path = os.path.join(depth_folder, f"{image_prefix}_prob_filtered.pfm")
        prob_path = os.path.join(depth_folder, f"{image_prefix}_prob.pfm")
        cam_path = os.path.join(cam_folder, f"{image_prefix}_cam.txt")
        
        # Check if files exist
        if not (os.path.exists(depth_path) and os.path.exists(cam_path)):
            print(f"Missing files for {image_prefix}")
            continue
            
        # Load data
        try:
            depth_map = load_pfm(open(depth_path, 'rb'))
            
            # Load camera with simple file reading instead of load_cam
            with open(cam_path, 'r') as f:
                lines = f.readlines()
            
            # Parse camera file manually
            # Line 1: extrinsic matrix (4x4)
            extrinsic = []
            for i in range(1, 5):
                row = [float(x) for x in lines[i].strip().split()]
                extrinsic.append(row)
            extrinsic = np.array(extrinsic)
            
            # Line 6: intrinsic matrix (3x3) 
            intrinsic = []
            for i in range(6, 9):
                row = [float(x) for x in lines[i].strip().split()]
                intrinsic.append(row)
            intrinsic = np.array(intrinsic)
            
            # Create cam format like load_cam
            cam = [extrinsic, np.eye(4)]
            cam[1][:3, :3] = intrinsic
            
            prob_map = None
            if os.path.exists(prob_path):
                prob_map = load_pfm(open(prob_path, 'rb'))
            
            # Generate points
            points = depth_to_pointcloud(depth_map, cam, prob_map, prob_threshold)
            
            if len(points) > 0:
                all_points.append(points)
                print(f"  Generated {len(points)} points")
            else:
                print(f"  No valid points")
                
        except Exception as e:
            print(f"Error processing {image_prefix}: {e}")
            continue
    
    # Combine all points
    if all_points:
        final_points = np.vstack(all_points)
        print(f"\nTotal points: {len(final_points)}")
        return final_points
    else:
        print("No points generated!")
        return np.empty((0, 3))

def save_ply(points, filename):
    """Save points to PLY file"""
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

if __name__ == "__main__":
    # Simple hardcoded execution
    dense_folder = "/data/dtu/dtu_eval/scan1"
    prob_threshold = 0.1
    output_file = "scan1_pointcloud.ply"
    
    print(f"Processing {dense_folder}")
    
    # Generate point cloud
    points = generate_pointcloud(dense_folder, prob_threshold)
    
    if len(points) > 0:
        # Save to PLY
        output_path = os.path.join(dense_folder, output_file)
        save_ply(points, output_path)
        print(f"Saved point cloud to {output_path}")
    else:
        print("No points to save!")