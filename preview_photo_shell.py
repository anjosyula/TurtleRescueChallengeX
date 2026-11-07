"""Local photo-texture previews on the existing scan; never edits training data."""
import json
from pathlib import Path
import sys
import bpy
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from blender_synthetic import aim, configure, render_data, sha256

out=ROOT/'output/photo-shell-preview'
out.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'datasets/blender-shell-v1/shell_workbench.blend'))
scene=bpy.context.scene
shell=bpy.data.objects['Shell scan proxy']
for obj in list(scene.objects):
    if obj!=shell:bpy.data.objects.remove(obj,do_unlink=True)
shell.hide_render=False;shell.rotation_euler=(0,0,0)
shell.location=(0,0,0)
outputs=configure(scene,640,32)
scene.view_settings.view_transform='AgX'
scene.world.node_tree.nodes['Background'].inputs[1].default_value=.35
bpy.ops.mesh.primitive_plane_add(size=4,location=(0,0,-.001))
ground=bpy.context.object;ground.pass_index=2
mat=bpy.data.materials.new('Neutral studio ground');mat.diffuse_color=(.45,.45,.45,1)
ground.data.materials.append(mat)
bpy.ops.object.camera_add(location=(.26,-.30,.35))
camera=bpy.context.object;scene.camera=camera;camera.data.lens=65;camera.data.sensor_width=36
aim(camera,(0,0,.04))
bpy.ops.object.light_add(type='AREA',location=(.1,-.1,.45))
bpy.context.object.data.energy=1.5;bpy.context.object.data.shape='DISK';bpy.context.object.data.size=.5
aim(bpy.context.object,(0,0,.04))

def photo_material(name,path,uv_name):
    mat=bpy.data.materials.new(name);mat.use_nodes=True
    shader=mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Roughness'].default_value=.85
    shader.inputs['Specular IOR Level'].default_value=.15
    image=mat.node_tree.nodes.new('ShaderNodeTexImage')
    image.image=bpy.data.images.load(str(path),check_existing=True)
    uv=mat.node_tree.nodes.new('ShaderNodeUVMap');uv.uv_map=uv_name
    mat.node_tree.links.new(uv.outputs['UV'],image.inputs['Vector'])
    mat.node_tree.links.new(image.outputs['Color'],shader.inputs['Base Color'])
    shell.data.materials.clear();shell.data.materials.append(mat)
    for poly in shell.data.polygons:poly.material_index=0

photo_material('Original scan photographic atlas',Path('assets/Scan.jpg'),'UVMap')
render_data(scene,outputs,out/'scan-atlas')

# A conservative inner-shell quadrilateral avoids road/head/limbs. This is a
# visibly approximate planar mapping, not camera-pose fitting or anatomical UV registration.
photo=ROOT/'images/39682/large/13206069.jpg'
corners=np.array([[330,160],[490,180],[580,290],[330,330]],dtype=float)
points=np.array([v.co[:] for v in shell.data.vertices])
low=points.min(0);high=points.max(0)
uv=shell.data.uv_layers.new(name='Photo preview projection')
for loop in shell.data.loops:
    x,y,_=(points[loop.vertex_index]-low)/(high-low)
    p=(1-x)*(1-y)*corners[0]+x*(1-y)*corners[1]+x*y*corners[2]+(1-x)*y*corners[3]
    uv.data[loop.index].uv=(p[0]/1024,1-p[1]/738)
photo_material('Collected shell photo - approximate preview',photo,uv.name)
z,mask,w2b=render_data(scene,outputs,out/'collected-photo')
f=640*camera.data.lens/camera.data.sensor_width
manifest=json.loads((ROOT/'images/39682/large/manifest.json').read_text())
source=next(r for r in manifest if r['file']==photo.name)
metadata={'source_photo':source,'photo_sha256':sha256(photo),
    'source_mesh':'datasets/blender-shell-v1/shell_workbench.blend',
    'original_scan_unchanged':True,'K':[[f,0,320],[0,f,320],[0,0,1]],
    'world_to_camera_cv':(np.diag([1,-1,-1,1])@w2b).tolist(),
    'photo_quad_pixels':corners.tolist(),'projection':'bilinear planar inner-shell patch on normalized mesh XY',
    'limitations':['Preview only; not species/anatomically registered',
    'Photo shading remains baked in; one patch repeats/stretches over the mesh',
    'Only color is transferred; no new scute geometry is created',
    'Recorded source license CC BY; current online photo page could not be retrieved'],
    'training_data_modified':False,'models_retrained':False,
    'blender_image_texture_docs':'https://docs.blender.org/manual/en/5.1/render/shader_nodes/textures/image.html'}
(out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
(out/'ATTRIBUTION.txt').write_text('Collected-photo texture: Lee Stayton, CC BY, as recorded in the local download manifest.\n'
    'https://www.inaturalist.org/photos/19308943\n'
    'https://www.inaturalist.org/observations/13206069\n'
    'Modifications: inner-shell patch mapped onto a different shell scan, relit, rendered, then Gaussian-rasterized.\n'
    'Preview only; approximate mapping, not anatomical reconstruction.\n')
scene.render.filepath=str((out/'interactive-preview.png').resolve())
for node in outputs.values():node.directory=str((out/'interactive-passes').resolve())
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(out/'photo-shell.blend'))
print('PHOTO_TEXTURE_PREVIEW_COMPLETE',out)
