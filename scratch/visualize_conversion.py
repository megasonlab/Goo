import numpy as np
import bpy
import bmesh
from mathutils import Vector
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def create_sphere(radius=1.0, resolution=32):
    """Create a sphere mesh in Blender."""
    # Remove existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Create sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=resolution, ring_count=resolution)
    sphere = bpy.context.active_object
    return sphere

def sample_mesh_points(obj, num_points=10000):
    """Sample points from a mesh object."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    
    # Calculate total area for weighted sampling
    total_area = sum(f.calc_area() for f in bm.faces)
    
    # Calculate points per face based on area
    points_per_face = []
    for face in bm.faces:
        face_area = face.calc_area()
        num_face_points = max(10, int(num_points * (face_area / total_area)))
        points_per_face.append(num_face_points)
    
    # Sample points
    points = []
    for face, num_face_points in zip(bm.faces, points_per_face):
        verts = face.verts
        if len(verts) > 3:
            # For non-triangular faces, use fan triangulation
            center = Vector(face.calc_center_median())
            for i in range(len(verts)):
                v1 = Vector(verts[i].co)
                v2 = Vector(verts[(i + 1) % len(verts)].co)
                v3 = center
                
                for _ in range(num_face_points // len(verts)):
                    r1, r2 = float(np.random.random()), float(np.random.random())
                    if r1 + r2 > 1:
                        r1, r2 = 1 - r1, 1 - r2
                    
                    point = v1 + v2 * r1 - v1 * r1 + v3 * r2 - v1 * r2
                    points.append(obj.matrix_world @ point)
        else:
            v1 = Vector(verts[0].co)
            v2 = Vector(verts[1].co)
            v3 = Vector(verts[2].co)
            
            for _ in range(num_face_points):
                r1, r2 = float(np.random.random()), float(np.random.random())
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                
                point = v1 + v2 * r1 - v1 * r1 + v3 * r2 - v1 * r2
                points.append(obj.matrix_world @ point)
    
    bm.free()
    return np.array(points)

def points_to_volume(points, resolution=(64, 64, 64), padding=0.1):
    """Convert point cloud to 3D volume array."""
    # Get bounds
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # Add padding
    padding_array = np.array([padding] * 3)
    min_coords -= padding_array
    max_coords += padding_array
    
    # Convert points to grid coordinates
    normalized = (points - min_coords) / (max_coords - min_coords)
    grid_points = normalized * (np.array(resolution) - 1)
    
    # Create volume array
    volume = np.zeros(resolution, dtype=np.float32)
    
    # Convert to integer coordinates
    grid_points = grid_points.astype(int)
    
    # Filter points within bounds
    mask = np.all((grid_points >= 0) & (grid_points < resolution), axis=1)
    grid_points = grid_points[mask]
    
    # Mark voxels containing points
    volume[grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]] = 1.0
    
    # Use dilation to create continuous surfaces
    volume = ndimage.binary_dilation(volume, structure=np.ones((2,2,2)))
    
    return volume

def visualize_pipeline():
    """Visualize the entire pipeline."""
    # Create sphere
    sphere = create_sphere(radius=1.0, resolution=32)
    
    # Sample points
    points = sample_mesh_points(sphere, num_points=50000)
    
    # Convert to volume
    volume = points_to_volume(points, resolution=(64, 64, 64))
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Original mesh
    ax1 = fig.add_subplot(131, projection='3d')
    # Get mesh vertices and faces
    mesh = sphere.data
    vertices = np.array([v.co for v in mesh.vertices])
    faces = np.array([[p.vertices[0], p.vertices[1], p.vertices[2]] for p in mesh.polygons])
    
    # Plot mesh
    for face in faces:
        x = vertices[face, 0]
        y = vertices[face, 1]
        z = vertices[face, 2]
        ax1.plot_trisurf(x, y, z)
    ax1.set_title('Original Mesh')
    
    # Plot 2: Point cloud
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.', alpha=0.1)
    ax2.set_title('Point Cloud')
    
    # Plot 3: Volume (middle slice)
    ax3 = fig.add_subplot(133)
    middle_slice = volume[volume.shape[0]//2, :, :]
    ax3.imshow(middle_slice, cmap='gray')
    ax3.set_title('Volume (Middle Slice)')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "conversion_pipeline.png"))
    plt.close()
    
    # Save point cloud and volume data
    np.save(os.path.join(output_dir, "sphere_points.npy"), points)
    np.save(os.path.join(output_dir, "sphere_volume.npy"), volume)
    
    print(f"Visualization saved to {output_dir}/conversion_pipeline.png")
    print(f"Point cloud data saved to {output_dir}/sphere_points.npy")
    print(f"Volume data saved to {output_dir}/sphere_volume.npy")

if __name__ == "__main__":
    visualize_pipeline() 