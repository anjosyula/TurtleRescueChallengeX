"""Local Blender 5.2 shell-proxy dataset: RGB, visible masks, metric simulated Z.

blender --background --factory-startup --python blender_synthetic.py -- --output DATASET
The default length is an explicit simulation assumption, not scan calibration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import bpy
import bmesh
import numpy as np
from mathutils import Vector


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aim(obj, target):
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()


def material(name, color, roughness=.8):
    m=bpy.data.materials.new(name);m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=roughness
    return m


def procedural(name, colors, scale, bump_distance, shell=False):
    m=material(name,colors[0]);nodes=m.node_tree.nodes;links=m.node_tree.links
    p=nodes.get('Principled BSDF');tex=nodes.new('ShaderNodeTexNoise')
    coord=nodes.new('ShaderNodeTexCoord');links.new(coord.outputs['Generated' if shell else 'Object'],tex.inputs['Vector'])
    tex.inputs['Scale'].default_value=scale;tex.inputs['Detail'].default_value=4
    ramp=nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color=(*colors[0],1);ramp.color_ramp.elements[1].color=(*colors[1],1)
    links.new(tex.outputs['Fac'],ramp.inputs['Fac']);links.new(ramp.outputs['Color'],p.inputs['Base Color'])
    height=tex.outputs['Fac']
    if shell:
        cells=nodes.new('ShaderNodeTexVoronoi');cells.feature='DISTANCE_TO_EDGE';cells.inputs['Scale'].default_value=5
        links.new(coord.outputs['Generated'],cells.inputs['Vector'])
        lines=nodes.new('ShaderNodeValToRGB')
        lines.color_ramp.elements[0].position=.015;lines.color_ramp.elements[0].color=(.08,.08,.08,1)
        lines.color_ramp.elements[1].position=.045;lines.color_ramp.elements[1].color=(1,1,1,1)
        links.new(cells.outputs['Distance'],lines.inputs['Fac'])
        mix=nodes.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.55
        links.new(ramp.outputs['Color'],mix.inputs[1]);links.new(lines.outputs['Color'],mix.inputs[2])
        links.new(mix.outputs[0],p.inputs['Base Color']);height=lines.outputs['Color']
    bump=nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.35;bump.inputs['Distance'].default_value=bump_distance
    links.new(height,bump.inputs['Height']);links.new(bump.outputs['Normal'],p.inputs['Normal'])
    return m


def configure(scene, size, samples):
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples
    scene.cycles.use_denoising=True;scene.cycles.max_bounces=3
    scene.render.threads_mode='FIXED';scene.render.threads=4
    scene.render.resolution_x=size;scene.render.resolution_y=size;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.render.film_transparent=False
    scene.view_settings.view_transform='AgX'
    scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.65,.75,1,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.35
    layer=scene.view_layers[0];layer.use_pass_z=True;layer.use_pass_position=True;layer.use_pass_object_index=True
    tree=bpy.data.node_groups.new('Dataset compositor','CompositorNodeTree');scene.compositing_node_group=tree
    tree.interface.new_socket(name='Image',in_out='OUTPUT',socket_type='NodeSocketColor')
    render=tree.nodes.new('CompositorNodeRLayers');out=tree.nodes.new('NodeGroupOutput')
    tree.links.new(render.outputs['Image'],out.inputs['Image'])
    outputs={}
    for name,socket,kind in [('position','Position','VECTOR'),('index','Object Index','FLOAT')]:
        node=tree.nodes.new('CompositorNodeOutputFile');node.format.media_type='IMAGE';node.format.file_format='OPEN_EXR'
        node.format.color_depth='32';node.format.color_mode='RGB';node.save_as_render=False
        node.file_name=name;node.file_output_items.new(kind,name)
        tree.links.new(render.outputs[socket],node.inputs[name]);outputs[name]=node
    return outputs


def read_pass(folder, name, size):
    paths=list(Path(folder).glob(name+'*.exr'))
    if len(paths)!=1:raise ValueError(f'Expected one {name} EXR, found {paths}')
    im=bpy.data.images.load(str(paths[0]),check_existing=False)
    pixels=np.empty(size*size*4,dtype=np.float32);im.pixels.foreach_get(pixels)
    bpy.data.images.remove(im)
    return pixels.reshape(size,size,4)[::-1,:,:3].copy()


def render_data(scene, outputs, folder):
    folder.mkdir(parents=True,exist_ok=False)
    for node in outputs.values():node.directory=str(folder.resolve())
    scene.render.filepath=str((folder/'rgb.png').resolve());bpy.ops.render.render(write_still=True)
    size=scene.render.resolution_x
    position=read_pass(folder,'position',size);index=read_pass(folder,'index',size)[:,:,0]
    # Position is world XYZ. Transform it to camera coordinates; Blender camera looks down -Z.
    w2b=np.asarray(scene.camera.matrix_world.inverted(),dtype=np.float64)
    z=-(position@w2b[2,:3]+w2b[2,3]).astype(np.float32)
    valid=np.isfinite(z)&(z>0)&(z<100)&(index>.5)
    z[~valid]=0
    mask=(np.abs(index-1)<.1).astype(np.uint8)
    np.save(folder/'depth_z_m.npy',z);np.save(folder/'shell_mask.npy',mask)
    np.save(folder/'validity.npy',valid.astype(np.uint8))
    # Keep the raw passes for reproducible checking of geometry and boundaries.
    return z,mask,w2b


def geometry_fixture(output):
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;outputs=configure(scene,64,2)
    bpy.ops.mesh.primitive_plane_add(size=10,location=(0,0,0));bpy.context.object.pass_index=2
    bpy.context.object.data.materials.append(material('fixture',(.4,.4,.4)))
    bpy.ops.object.camera_add(location=(0,0,2));camera=bpy.context.object;scene.camera=camera
    camera.rotation_euler=(0,0,0);camera.data.lens=36;camera.data.sensor_width=36
    z,mask,w2b=render_data(scene,outputs,output)
    assert np.all(z>0) and np.max(np.abs(z-2))<1e-4
    assert not mask.any()
    positions=read_pass(output,'position',64)
    yy,xx=np.mgrid[:64,:64]
    u=64*positions[:,:,0]/z+32;v=-64*positions[:,:,1]/z+32
    error=np.sqrt((u-(xx+.5))**2+(v-(yy+.5))**2)
    assert float(error.max())<1.5, 'Position pass disagrees with image orientation/intrinsics'
    write_json(output/'fixture.json',{'passed':True,'plane_camera_axis_z_m':2.,
        'max_error_m':float(np.max(np.abs(z-2))),'max_projection_error_pixels':float(error.max()),
        'note':'Plane extends across image; includes off-axis pixels. Position pass conversion, not radial range. Projection tolerance permits subpixel render sampling.'})


def import_shell(source, length):
    bpy.ops.wm.obj_import(filepath=str(source.resolve()),forward_axis='Y',up_axis='Z')
    objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
    if len(objects)!=1:raise ValueError('Expected one scan mesh')
    obj=objects[0];obj.name='Shell scan proxy';bpy.context.view_layer.objects.active=obj
    points=np.empty(len(obj.data.vertices)*3,np.float32);obj.data.vertices.foreach_get('co',points)
    points=points.reshape(-1,3)[:,[0,2,1]];points[:,2]*=-1
    low,high=points.min(axis=0),points.max(axis=0)
    points-=np.array([(low[0]+high[0])/2,(low[1]+high[1])/2,low[2]])
    gain=length/float(max(high-low));points*=gain
    obj.data.vertices.foreach_set('co',points.ravel());obj.data.update()
    mod=obj.modifiers.new('Working-copy decimation','DECIMATE');mod.ratio=.015
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bm=bmesh.new();bm.from_mesh(obj.data)
    before=sum(e.is_boundary for e in bm.edges)
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=length*1e-5)
    bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=length*1e-6)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    after=sum(e.is_boundary for e in bm.edges);bm.to_mesh(obj.data);bm.free()
    for face in obj.data.polygons:face.use_smooth=True
    obj.pass_index=1
    return obj,{'input_vertices':len(points),'working_vertices':len(obj.data.vertices),
        'working_faces':len(obj.data.polygons),'boundary_edges_before_cleanup':before,
        'boundary_edges_after_cleanup':after,'file_units_to_simulated_meters':gain,
        'assumed_longest_dimension_m':length,'scale_measured':False,
        'transform':'x,z,-y, centered XY, bottom at Z=0',
        'repair':'merge near-coincident vertices, remove degenerate edges, recalculate normals; large openings retained',
        'anatomical_reconstruction':False,'limitation':'Single incomplete shell proxy, no limbs/head or species-validated scute anatomy'}


def mesh_object(name,vertices,faces,mat):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(vertices,[],faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat);obj.pass_index=2
    return obj


def environment(rng, radius, style, materials, shell_length, occluded):
    objects=[];vertices=[];faces=[]
    # Grass ribbons are explicit geometry, so occlusion, RGB, and depth remain aligned.
    for i in range(2400 if style=='grass' else 100):
        x,y=rng.uniform(-radius,radius,2);height=rng.uniform(.025,.14);width=rng.uniform(.002,.008)
        direction=rng.uniform(0,2*math.pi);dx,dy=math.cos(direction)*width,math.sin(direction)*width
        base=len(vertices)
        vertices += [(x-dx,y-dy,.001),(x+dx,y+dy,.001),
                     (x+.006-dx*.6,y+.003-dy*.6,height*.6),
                     (x+.006+dx*.6,y+.003+dy*.6,height*.6),(x+.02,y+.01,height)]
        faces.extend([(base,base+1,base+3,base+2),(base+2,base+3,base+4)])
    objects.append(mesh_object('Grass',vertices,faces,materials['grass']))
    for i in range(22):
        x,y=rng.uniform(-radius,radius,2)
        if abs(x)<shell_length*.65 and abs(y)<shell_length*.65:continue
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=1,location=(x,y,.008))
        obj=bpy.context.object;obj.name='Rock/debris';obj.scale=tuple(rng.uniform(.008,.05,3));obj.pass_index=2
        for face in obj.data.polygons:face.use_smooth=True
        obj.data.materials.append(materials['rock']);objects.append(obj)
    if occluded:
        # Leaf-sized cover above part of the shell; actual visible mask determines labels.
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=6,radius=1,
            location=(rng.uniform(-.04,.04),rng.uniform(-.05,.05),shell_length*.44))
        obj=bpy.context.object;obj.name='Partial leaf cover';obj.scale=(shell_length*.4,shell_length*.12,.002)
        obj.rotation_euler[2]=rng.uniform(-math.pi,math.pi);obj.pass_index=2
        obj.data.materials.append(materials['leaf']);objects.append(obj)
    return objects


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('assets/Scan.obj'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=96);p.add_argument('--size',type=int,default=640)
    p.add_argument('--samples',type=int,default=8);p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--assumed-length-m',type=float,default=.2)
    p.add_argument('--fixture',action='store_true')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.fixture:geometry_fixture(args.output);return
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    if args.count<1 or args.size<32 or args.assumed_length_m<=0:raise ValueError('Invalid rendering parameters')
    args.output.mkdir(parents=True)
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    shell,inspection=import_shell(args.input,args.assumed_length_m)
    materials={'shell':[
        procedural('Olive polygon shell',[(.035,.045,.012),(.23,.24,.065)],8,.0007,True),
        procedural('Brown polygon shell',[(.025,.013,.005),(.20,.105,.025)],10,.0007,True),
        procedural('Dark mottled shell',[(.012,.016,.008),(.14,.12,.035)],12,.0007,True)],
        'soil':procedural('Soil',[(.025,.012,.006),(.18,.10,.035)],18,.004),
        'gravel':procedural('Gravel',[(.07,.065,.05),(.3,.28,.22)],35,.007),
        'grassground':procedural('Ground green',[(.025,.045,.008),(.12,.16,.03)],20,.004),
        'grass':procedural('Grass blades',[(.025,.06,.006),(.12,.21,.025)],6,.0002),
        'rock':procedural('Rock',[(.05,.045,.03),(.25,.22,.15)],25,.001),
        'leaf':procedural('Leaf cover',[(.035,.018,.004),(.18,.11,.022)],20,.0003)}
    bpy.ops.mesh.primitive_plane_add(size=60,location=(0,0,-.001));ground=bpy.context.object;ground.pass_index=2
    ground.data.materials.append(materials['soil'])
    bpy.ops.object.camera_add();camera=bpy.context.object;camera.data.lens=36;camera.data.sensor_width=36
    camera.data.sensor_fit='HORIZONTAL';camera.data.clip_end=100
    scene=bpy.context.scene;scene.camera=camera;outputs=configure(scene,args.size,args.samples)
    bpy.ops.object.light_add(type='SUN');sun=bpy.context.object;sun.data.angle=.15
    rng=np.random.default_rng(args.seed)
    write_json(args.output/'asset.json',{'source':str(args.input.resolve()),'source_sha256':sha256(args.input),
        'source_rights':'User-provided local scan; no external distribution license inferred',
        'material_source':'Procedural Blender nodes created for this experiment; no third-party photographs',
        'material_limitations':'Stylized polygon relief and mottling, not species-specific scute restoration',
        'group_id':'shell-scan-'+sha256(args.input)[:16],'partition':'train_only',
        'blender':bpy.app.version_string,'settings':vars(args)|{'input':str(args.input),'output':str(args.output)},
        'inspection':inspection})
    manifest=args.output/'manifest.jsonl'
    for index in range(args.count):
        negative=index%8==7;shell.hide_render=negative
        shell.rotation_euler[2]=rng.uniform(-math.pi,math.pi)
        shell.data.materials.clear();shell.data.materials.append(materials['shell'][index%3])
        target_pixels=float(rng.choice([20,28,40,64,96,144,220]))
        distance=args.size*args.assumed_length_m/target_pixels
        elevation=rng.uniform(math.radians(35),math.radians(78));azimuth=rng.uniform(-math.pi,math.pi)
        camera.location=(distance*math.cos(elevation)*math.cos(azimuth),distance*math.cos(elevation)*math.sin(azimuth),distance*math.sin(elevation))
        target=(rng.uniform(-.2,.2)*distance,rng.uniform(-.2,.2)*distance,args.assumed_length_m*.15)
        aim(camera,target)
        style=['soil','gravel','grass'][index%3];ground.data.materials[0]=materials['grassground' if style=='grass' else style]
        occluded=not negative and index%4==2
        objects=environment(rng,min(5,distance*.65),style,materials,args.assumed_length_m,occluded)
        sun.rotation_euler=(rng.uniform(.1,1.0),rng.uniform(-.5,.5),rng.uniform(-math.pi,math.pi));sun.data.energy=rng.uniform(1,4)
        scene.cycles.seed=args.seed+index
        folder=args.output/'samples'/f'{index:06d}'
        z,mask,w2b=render_data(scene,outputs,folder)
        y,x=np.where(mask>0)
        box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x)>=8 else None
        # Do not label invisible/tiny fragments as ordinary visually detectable positives.
        accepted=negative or box is not None
        f=args.size*camera.data.lens/camera.data.sensor_width
        k=[[f,0,args.size/2],[0,f,args.size/2],[0,0,1]]
        cv_extrinsic=np.diag([1,-1,-1,1])@w2b
        record={'sample_id':f'{index:06d}','split':'train','synthetic':True,'sensor':'blender_simulation',
            'group_id':'shell-scan-'+sha256(args.input)[:16], 'accepted':accepted,
            'label_semantics':'visible shell-proxy pixels, exclusive xyxy; not a complete animal model',
            'boxes_xyxy':[box] if box else [],'class_id':0,'visible_mask_pixels':int(mask.sum()),
            'negative_scene':negative,'scene_style':style,'partial_cover_requested':occluded,
            'target_width_pixels_requested':target_pixels,'image_size_wh':[args.size,args.size],
            'depth_kind':'simulated camera_axis_z','depth_units':'meters','assumed_shell_length_m':args.assumed_length_m,
            'scale_measured':False,'K':k,'world_to_camera_cv':cv_extrinsic.tolist(),
            'files':{name:str((folder/file).relative_to(args.output)) for name,file in
                     [('rgb','rgb.png'),('depth','depth_z_m.npy'),('mask','shell_mask.npy'),('validity','validity.npy')]}}
        write_json(folder/'sample.json',record)
        with manifest.open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        write_json(args.output/'progress.json',{'rendered':index+1,'requested':args.count})
        if index==0:
            bpy.ops.wm.save_as_mainfile(filepath=str((args.output/'scene.blend').resolve()))
        for obj in objects:
            mesh=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
            if mesh.users==0:bpy.data.meshes.remove(mesh)
        print(f'DATASET_SAMPLE {index+1}/{args.count} visible={len(x)} accepted={accepted}',flush=True)
    write_json(args.output/'completed.json',{'samples':args.count,'partition':'train_only','original_scan_modified':False})


if __name__=='__main__':main()
