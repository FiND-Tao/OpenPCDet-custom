#%%
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize saved OpenPCDet inference results')
    parser.add_argument('--input', type=str, default='/work/tl2lab/tl_folder/lidar_work/pcdet_tree2/OpenPCDet/tools/inference_outputs/000000_slice_0.npz', required=False, help='Path to saved .npz file')
    parser.add_argument('--score_thresh', type=float, default=0.0, help='Minimum score to draw boxes')
    return parser.parse_args(args=[])


def translate_boxes_to_open3d_instance(box):
    import open3d

    center = box[0:3]
    lwh = box[3:6]
    axis_angles = np.array([0, 0, box[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    return line_set


def resolve_input_path(input_arg):
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / 'inference_outputs'

    if input_arg:
        input_path = Path(input_arg).expanduser().resolve()
        if input_path.exists():
            return input_path
        raise SystemExit(f'Input file not found: {input_path}')

    if output_dir.exists():
        candidates = sorted(output_dir.glob('*.npz'))
        if candidates:
            return candidates[0].resolve()

    raise SystemExit(
        f'No .npz result file found. Provide one with --input or place files under {output_dir}'
    )


def main():
    args = parse_args()
    
    # Check for k3d
    try:
        import k3d
    except ImportError:
        raise SystemExit("Please run: pip install k3d")

    input_path = resolve_input_path(args.input)
    print(f'Loading: {input_path}')

    data = np.load(input_path)
    points = data['points']
    pred_boxes = data['pred_boxes']
    pred_scores = data['pred_scores']

    # Filter boxes
    keep = pred_scores >= args.score_thresh
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]

    # --- K3D Visualization ---
    plot = k3d.plot(name=f'Result: {input_path.name}')

    # 1. Add Point Cloud
    # Decrease point_size if it looks too chunky
    plot += k3d.points(points[:, :3], point_size=0.05, shader='flat', color=0xFFFFFF)

    # 2. Add Boxes (converting Open3D LineSets to K3D Lines)
    print(f"Rendering {len(pred_boxes)} boxes...")
    for box in pred_boxes:
        # Use your existing helper to get the O3D geometry
        line_set = translate_boxes_to_open3d_instance(box)
        
        # Extract data for K3D
        # K3D lines expects float32 vertices and uint32 indices
        vertices = np.asarray(line_set.points).astype(np.float32)
        indices = np.asarray(line_set.lines).astype(np.uint32)
        
        plot += k3d.lines(vertices=vertices, indices=indices, 
                          width=0.1, color=0x00FF00, shader='mesh')

    plot.display()
    print(f'Done! Displaying {points.shape[0]} points and {pred_boxes.shape[0]} boxes.')



if __name__ == '__main__':
    main()

# %%
