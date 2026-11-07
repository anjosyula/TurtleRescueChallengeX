"""Make a portable, editable scene whose F12 preview cannot overwrite training data."""
import argparse
from pathlib import Path
import sys
import bpy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists():raise FileExistsError('Choose a new blend file')
    bpy.ops.wm.open_mainfile(filepath=str(args.scene.resolve()))
    scene=bpy.context.scene
    preview=args.output.parent/'interactive-preview';preview.mkdir(exist_ok=True)
    scene.render.filepath=str((preview/'rgb.png').resolve())
    if scene.compositing_node_group:
        for node in scene.compositing_node_group.nodes:
            if node.bl_idname=='CompositorNodeOutputFile':node.directory=str(preview.resolve())
    scene['dataset_note']='Shell proxy, assumed scale. All sensor depth is simulated. See asset.json.'
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))


if __name__=='__main__':main()
