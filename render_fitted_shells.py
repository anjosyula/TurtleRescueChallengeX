"""Blender renders of fitted photo-textured scan variants, with explicit unsupported faces."""
from pathlib import Path
import argparse
import json
import sys
import math
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from blender_synthetic import configure, render_data, read_pass, aim

def material(name,color=None,image=None):
    m=bpy.data.materials.new(name);m.use_nodes=True
    nodes=m.node_tree.nodes;nodes.clear();links=m.node_tree.links
    out=nodes.new('ShaderNodeOutputMaterial');emission=nodes.new('ShaderNodeEmission')
    links.new(emission.outputs[0],out.inputs['Surface'])
    if image:
        tex=nodes.new('ShaderNodeTexImage');tex.image=image;tex.extension='CLIP'
        uv=nodes.new('ShaderNodeUVMap');uv.uv_map='Fitted photo projection'
        links.new(uv.outputs['UV'],tex.inputs['Vector']);links.new(tex.outputs['Color'],emission.inputs['Color'])
    else:emission.inputs['Color'].default_value=(*color,1)
    return m

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fits',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    records=[];rng=np.random.default_rng(20261026)
    for fit in json.loads(args.fits.read_text()):
        folder=ROOT/fit['fit_folder'];data=np.load(folder/'fit.npz')
        base=data['vertices'];faces=data['faces'];uv=data['uv_pixels']
        R=np.array(fit['R']);t=np.array(fit['t']);K=np.array(fit['K'])
        bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
        mesh=bpy.data.meshes.new('Deformable shell');mesh.from_pydata(base.tolist(),[],faces.tolist());mesh.update()
        shell=bpy.data.objects.new('Photo-fitted scan proxy',mesh);bpy.context.collection.objects.link(shell);shell.pass_index=1
        for poly in mesh.polygons:poly.use_smooth=True
        layer=mesh.uv_layers.new(name='Fitted photo projection')
        w,h=fit['image_size_wh']
        for loop in mesh.loops:
            u,v=uv[loop.vertex_index];layer.data[loop.index].uv=((u+.5)/w,1-(v+.5)/h)
        photo=bpy.data.images.load(str(ROOT/fit['image']),check_existing=True)
        known=material('Observed photo color - baked illumination',image=photo);known.pass_index=1
        unknown=material('Unsupported surface - neutral gray',color=(.18,.18,.18));unknown.pass_index=2
        mesh.materials.append(known);mesh.materials.append(unknown)
        mask_image=bpy.data.images.load(str(folder/'source-mask.png'),check_existing=False)
        pixels=np.empty(w*h*4,np.float32);mask_image.pixels.foreach_get(pixels)
        mask=pixels.reshape(h,w,4)[::-1,:,0]>.5;bpy.data.images.remove(mask_image)
        camera_center=-R.T@t
        bvh=BVHTree.FromPolygons([Vector(v) for v in base],faces.tolist(),all_triangles=True)
        supported=[]
        for i,face in enumerate(faces):
            center=base[face].mean(0);direction=center-camera_center;distance=np.linalg.norm(direction)
            hit,normal,index,hitdistance=bvh.ray_cast(Vector(camera_center),Vector(direction/distance),distance+.01)
            xy=uv[face].mean(0);x,y=np.rint(xy).astype(int)
            inside=0<=x<w and 0<=y<h and mask[y,x]
            visible=hit is not None and abs(hitdistance-distance)<fit['length_m']*.005
            supported.append(bool(inside and visible))
            mesh.polygons[i].material_index=0 if supported[-1] else 1
        scene=bpy.context.scene;outputs=configure(scene,640,12)
        scene.view_settings.view_transform='Standard';scene.world.color=(.18,.18,.18)
        scene.view_layers[0].use_pass_material_index=True
        tree=scene.compositing_node_group;render=next(n for n in tree.nodes if n.bl_idname=='CompositorNodeRLayers')
        node=tree.nodes.new('CompositorNodeOutputFile');node.format.media_type='IMAGE';node.format.file_format='OPEN_EXR'
        node.format.color_depth='32';node.format.color_mode='RGB';node.save_as_render=False
        node.file_name='coverage';node.file_output_items.new('FLOAT','coverage')
        tree.links.new(render.outputs['Material Index'],node.inputs['coverage']);outputs['coverage']=node
        bpy.ops.mesh.primitive_plane_add(size=20,location=(0,0,-.001))
        ground=bpy.context.object;ground.pass_index=2
        ground.data.materials.append(material('Neutral ground',color=(.18,.18,.18)))
        bpy.ops.object.camera_add();camera=bpy.context.object;scene.camera=camera
        camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=36;camera.data.clip_start=.005;camera.data.clip_end=100
        poly=np.array(fit['shell_polygon_xy']);center=(poly.min(0)+poly.max(0))/2
        side=float(np.ptp(poly,axis=0).max()*1.65);origin=center-side/2;scale=640/side
        renderK=K.copy();renderK[0]*=scale;renderK[1]*=scale
        renderK[0,2]-=origin[0]*scale;renderK[1,2]-=origin[1]*scale
        aspect=renderK[0,0]/renderK[1,1]
        scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=aspect
        camera.data.lens=renderK[0,0]*36/640
        camera.data.shift_x=(320-renderK[0,2])/640
        camera.data.shift_y=(renderK[1,2]-320)*aspect/640
        for variant in range(6):
            factors=np.ones(3) if variant==0 else rng.uniform([.82,.9,.88],[1.12,1.1,1.12])
            # Keep every generated length within this example's declared prior.
            length=float(np.clip(fit['length_m']*factors[0],*fit['profile']['length_bounds_m']))
            factors[0]=length/fit['length_m']
            width=float(np.clip(fit['width_ratio']*factors[1],*fit['profile']['width_ratio_bounds']))
            height=float(np.clip(fit['height_ratio']*factors[2],*fit['profile']['height_ratio_bounds']))
            deformation=np.array([factors[0],factors[0]*width/fit['width_ratio'],factors[0]*height/fit['height_ratio']])
            vertices=base*deformation
            mesh.vertices.foreach_set('co',vertices.ravel());mesh.update()
            yaw=0 if variant==0 else float(rng.uniform(-5,5));a=math.radians(yaw)
            rz=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
            extrinsic=np.eye(4);extrinsic[:3,:3]=R@rz;extrinsic[:3,3]=t
            camera.matrix_world=Matrix(np.linalg.inv(np.diag([1,-1,-1,1])@extrinsic))
            name=f"{fit['id']}_{variant:02d}";target=args.output/name
            z,mask_out,w2b=render_data(scene,outputs,target)
            actual_cv=np.diag([1,-1,-1,1])@w2b
            coverage_pass=read_pass(target,'coverage',640)[:,:,0]
            coverage=(np.abs(coverage_pass-1)<.1)&(mask_out>0)
            fraction=float(coverage.sum()/max(mask_out.sum(),1));np.save(target/'texture_support.npy',coverage.astype(np.uint8))
            yy,xx=np.where(mask_out>0);box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
            rec={'sample_id':name,'source_id':fit['id'],'source_group':fit['source_group'],
                'synthetic':True,'review_only':True,'accepted_for_training':False,
                'geometry':'randomized deformed scan; not measured anatomy',
                'length_m':length,'width_ratio':width,'height_ratio':height,'view_yaw_delta_deg':yaw,
                'K':renderK.tolist(),'world_to_camera_cv':actual_cv.tolist(),'image_size_wh':[640,640],
                'texture_supported_shell_fraction':fraction,'source_camera_kind':fit['camera_kind'],
                'texture_policy':'Source-visible faces only; unsupported faces gray; captured lighting retained via emission shader',
                'boxes_xyxy':[box],'depth_kind':'synthetic camera_axis_z','depth_units':'meters',
                'crop_from_original':{'origin_xy':origin.tolist(),'side_pixels':side,'scale':scale},
                'source_fit':str(folder/'fit.json'),'source_photo':fit['image']}
            (target/'sample.json').write_text(json.dumps(rec,indent=2)+'\n');records.append(rec)
            print('VARIANT',name,fraction,flush=True)
            if variant==0:
                bpy.ops.file.pack_all();scene.render.filepath=str((target/'interactive-preview.png').resolve())
                for output in outputs.values():output.directory=str((target/'interactive-passes').resolve())
                bpy.ops.wm.save_as_mainfile(filepath=str(target/'fitted-shell.blend'))
    (args.output/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
