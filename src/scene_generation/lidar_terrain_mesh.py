import laspy
import numpy as np
import pyvista as pv

import pyvista as pv

from pyproj import Transformer

from plyfile import PlyData, PlyElement

def generate_terrain_mesh(lidar_laz_file_path, ply_save_path, src_crs="EPSG:3857", dest_crs="EPSG:32617", plot_figures=False, center_x = 0, center_y = 0):
    print("generate_terrain_mesh")
    pv.global_theme.trame.server_proxy_enabled = True
    print(pv.global_theme.trame.server_proxy_enabled)
    
    
    
    # Load the LAZ file
    las = laspy.read(lidar_laz_file_path)
    # Check for VLRs
    # for vlr in las.vlrs:
    #     print(vlr.description)
    #     print(vlr.record_id)
    #     print(vlr)
    # # Access the WKT (Well-Known Text) CRS if available
    # if las.header.evlrs:
    #     for evlr in las.header.evlrs:
    #         if evlr.description == "WKT":
    #             print("CRS WKT:", evlr.string)

    
    # Extract the classification field and filter out ground points (classification == 2)
    ground_mask = las.classification == 2
    
    # Extract x, y, and z coordinates for ground points
    x_ground = las.x[ground_mask]
    y_ground = las.y[ground_mask]
    z_ground = las.z[ground_mask]
    
    # Create a PyVista point cloud
    print(x_ground)
    print(y_ground)
    transformer = Transformer.from_crs(src_crs, dest_crs, always_xy=True)
    x_ground, y_ground = transformer.transform(x_ground, y_ground)
    print(x_ground)
    print(y_ground)

    x_ground = x_ground - center_x
    y_ground = y_ground - center_y
    print("centerx, y", center_x, center_y)


     # Normalize the Z-values by setting the minimum Z to 0
    z_ground = z_ground - np.min(z_ground)-10

    points = np.vstack((x_ground, y_ground, z_ground)).T
    point_cloud = pv.PolyData(points)
    # point_cloud["Height"] = points[:, 2]
    # plotter = pv.Plotter()
    # plotter.add_points(point_cloud, scalars="Height", cmap="viridis", point_size=5)
    
    # # Show the plot
    # plotter.show()
    
    surface_mesh = point_cloud.delaunay_2d()

            # Extract vertices and faces from the PyVista surface mesh
    vertices = surface_mesh.points
    faces = surface_mesh.faces.reshape(-1, 4)[:, 1:4]  # Ignore the first element which is the number of vertices per face

    # Prepare data for the PLY file
    vertex_data = [(vertex[0], vertex[1], vertex[2]) for vertex in vertices]
    face_data = [(list(face),) for face in faces]
    
    # Define PlyElement for vertices and faces
    vertex_element = PlyElement.describe(np.array(vertex_data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
    face_element = PlyElement.describe(np.array(face_data, dtype=[('vertex_indices', 'i4', (3,))]), 'face')
    
    # Write the PLY file manually using plyfile
    PlyData([vertex_element, face_element], text=False).write(ply_save_path)

    
    #surface_mesh.save(ply_save_path)
    # Plot the triangulated terrain
    if(plot_figures):
        pv.set_jupyter_backend('client')
        plotter = pv.Plotter()
        plotter.add_mesh(surface_mesh, scalars=surface_mesh.points[:, 2], cmap="terrain")
        plotter.show()

def remove_obj_info_from_ply(input_ply_path, output_ply_path):
    # Open the input PLY file in binary read mode
    with open(input_ply_path, 'rb') as infile:
        # Read the first few lines (assuming the header)
        header_lines = []
        for _ in range(5):
            line = infile.readline()
            header_lines.append(line)
        
        # Filter out lines that start with "obj_info" after decoding
        cleaned_header = [
            line for line in header_lines if not line.decode("utf-8", errors="ignore").startswith("obj_info")
        ]
        
        # Read the remaining content of the file
        remaining_data = infile.read()
    
    # Write the cleaned header and the rest of the binary content to the output file
    with open(output_ply_path, 'wb') as outfile:
        outfile.writelines(cleaned_header)
        outfile.write(remaining_data)

# # Example usage
# remove_obj_info_from_ply("input.ply", "cleaned_output.ply")


# Example usage


if __name__ == '__main__':
    data_dir = "output"
    mesh_data_dir = "output/mesh"
    os.makedirs(mesh_data_dir, exist_ok=True)
    center_x = -71.0602
    center_y = 42.3512
    projection_EPSG_code = "EPSG:32613"
    generate_terrain_mesh(
        os.path.join(data_dir, "test_hag.laz"),
        os.path.join(mesh_data_dir, f"lidar_terrain.ply"), src_crs="EPSG:3857", dest_crs=projection_EPSG_code,
        plot_figures=False, center_x=center_x, center_y=center_y)