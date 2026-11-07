"""Render the candidate scan for review, without treating file units as meters.

Run with Blender --background --factory-startup --python inspect_shell_blender.py
-- --input /path/Scan.obj --output evidence/shell-preview
This is an inspection render, not labeled training data or anatomical repair.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True,exist_ok=True)
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.obj_import(filepath=str(args.input.resolve()),forward_axis='Y',up_axis='Z')
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    if len(meshes)!=1:raise ValueError('Inspect multi-object scan orientation manually')
    shell=meshes[0];bpy.context.view_layer.objects.active=shell;shell.select_set(True)
    # Keep a normalized display-only copy in memory. Original geometry is untouched.
    vertices=np.empty(len(shell.data.vertices)*3,np.float32)
    shell.data.vertices.foreach_get('co',vertices);vertices=vertices.reshape(-1,3)
    vertices=vertices[:,[0,2,1]];vertices[:,2]*=-1
    low,high=vertices.min(axis=0),vertices.max(axis=0)
    vertices-=np.array([(low[0]+high[0])/2,(low[1]+high[1])/2,low[2]])
    vertices/=float(max(high-low))
    shell.data.vertices.foreach_set('co',vertices.ravel());shell.data.update()
    modifier=shell.modifiers.new('Preview only decimation','DECIMATE');modifier.ratio=.03
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    for face in shell.data.polygons:face.use_smooth=True
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=24
    scene.render.threads_mode='FIXED';scene.render.threads=2
    scene.render.resolution_x=640;scene.render.resolution_y=480;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.world.color=(.25,.25,.25)
    bpy.ops.object.light_add(type='AREA',location=(1,-1,2));light=bpy.context.object;light.data.energy=150;light.data.shape='DISK';light.data.size=2
    bpy.ops.object.light_add(type='AREA',location=(-1,1,1));bpy.context.object.data.energy=75;bpy.context.object.data.size=2
    bpy.ops.object.camera_add(location=(1.2,-1.3,1.1));camera=bpy.context.object;scene.camera=camera;camera.data.lens=50
    target=Vector((0,0,.2))
    for lamp in [o for o in scene.objects if o.type=='LIGHT']:
        lamp.rotation_euler=(target-lamp.location).to_track_quat('-Z','Y').to_euler()
    for name,position in [('oblique',(1.2,-1.3,1.1)),('opposite',(-1.2,1.3,1.0))]:
        camera.location=position;camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
        scene.render.filepath=str((args.output/(name+'.png')).resolve());bpy.ops.render.render(write_still=True)
    neutral=bpy.data.materials.new('Inspection neutral material');neutral.use_nodes=True
    shader=neutral.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value=(.65,.65,.65,1);shader.inputs['Roughness'].default_value=.7
    shell.data.materials.clear();shell.data.materials.append(neutral)
    camera.location=(1.2,-1.3,1.1);camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str((args.output/'neutral.png').resolve());bpy.ops.render.render(write_still=True)
    (args.output/'inspection.json').write_text(json.dumps({'source':str(args.input.resolve()),
        'blender':bpy.app.version_string,'purpose':'qualitative candidate-scan inspection only',
        'display_transform':'axes x,z,-y; centered; longest dimension normalized to 1',
        'physical_scale':'unknown','decimation_ratio':.03,'preview_faces':len(shell.data.polygons),
        'training_use':False},indent=2)+'\n')


if __name__=='__main__':main()
