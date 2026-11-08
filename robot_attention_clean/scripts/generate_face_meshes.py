#!/usr/bin/env python3
"""Generate original humanoid and eyelid meshes using only the standard library.

Coordinates: head faces +Y with Z up; eye-local forward is -Z and up is +Y.
This is a procedural approximation, not a scan of the reference image.
"""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent.parent / 'gazebo/models/eye/meshes'


def write_mesh(name, points, faces):
    normals = [[0.0, 0.0, 0.0] for _ in points]
    for a, b, c in faces:
        u = [points[b][i] - points[a][i] for i in range(3)]
        v = [points[c][i] - points[a][i] for i in range(3)]
        n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        for idx in (a, b, c):
            for k in range(3): normals[idx][k] += n[k]
    for n in normals:
        length = math.sqrt(sum(x*x for x in n)) or 1.0
        for k in range(3): n[k] /= length
    ns = 'http://www.collada.org/2005/11/COLLADASchema'
    ET.register_namespace('', ns)
    def elem(p, tag, text=None, **attrs):
        e = ET.SubElement(p, '{'+ns+'}'+tag, attrs)
        if text is not None: e.text = text
        return e
    doc = ET.Element('{'+ns+'}COLLADA', version='1.4.1')
    asset = elem(doc, 'asset'); elem(asset, 'unit', name='meter', meter='1'); elem(asset, 'up_axis', 'Z_UP')
    lib = elem(doc, 'library_geometries'); geo = elem(lib, 'geometry', id='surface'); mesh = elem(geo, 'mesh')
    for name_, array in [('positions', points), ('normals', normals)]:
        source = elem(mesh, 'source', id=name_)
        elem(source, 'float_array', ' '.join(f'{v:.7f}' for p in array for v in p), id=name_+'-array', count=str(len(array)*3))
        technique = elem(source, 'technique_common')
        acc = elem(technique, 'accessor', source='#'+name_+'-array', count=str(len(array)), stride='3')
        for axis in 'XYZ': elem(acc, 'param', name=axis, type='float')
    verts = elem(mesh, 'vertices', id='vertices'); elem(verts, 'input', semantic='POSITION', source='#positions')
    tris = elem(mesh, 'triangles', count=str(len(faces)))
    elem(tris, 'input', semantic='VERTEX', source='#vertices', offset='0')
    elem(tris, 'input', semantic='NORMAL', source='#normals', offset='1')
    elem(tris, 'p', ' '.join(f'{v} {v}' for tri in faces for v in tri))
    scenes = elem(doc, 'library_visual_scenes'); scene = elem(scenes, 'visual_scene', id='scene')
    node = elem(scene, 'node', id='surface-node'); elem(node, 'instance_geometry', url='#surface')
    scene = elem(doc, 'scene'); elem(scene, 'instance_visual_scene', url='#scene')
    OUT.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(doc).write(OUT/name, encoding='utf-8', xml_declaration=True)


def grid(rows, cols, sample, keep=lambda p: True):
    points = [sample(i/rows, j/cols) for i in range(rows+1) for j in range(cols+1)]
    faces = []
    for i in range(rows):
        for j in range(cols):
            a = i*(cols+1)+j; b = a+cols+1
            for tri in [(a,b,a+1), (a+1,b,b+1)]:
                mid = tuple(sum(points[v][k] for v in tri)/3 for k in range(3))
                if keep(mid): faces.append(tri)
    return points, faces


def gaussian(x,z,cx,cz,sx,sz):
    return math.exp(-0.5*((x-cx)/sx)**2 - 0.5*((z-cz)/sz)**2)


def head():
    # Anatomical silhouette: chin, jaw, cheeks, temples, forehead, cranium.
    rings = [(1.35,.12,.32,.24),(1.52,.60,.63,.47),(1.82,.88,.69,.66),
             (2.18,1.04,.65,.83),(2.53,1.15,.72,.99),(2.9,1.13,.63,1.05),
             (3.18,1.12,.57,1.08),(3.43,1.12,.74,1.09),(3.8,1.14,.78,1.07),
             (4.1,1.04,.72,.96),(4.35,.78,.57,.74),(4.5,.35,.28,.34),(4.54,.01,.01,.01)]
    def sample(u,v):
        z=1.35+u*(4.54-1.35)
        for index,(a,b) in enumerate(zip(rings,rings[1:])):
            if a[0] <= z <= b[0]+1e-9:
                t=(z-a[0])/(b[0]-a[0]); span=b[0]-a[0]
                before=rings[max(0,index-1)]; after=rings[min(len(rings)-1,index+2)]
                values=[]
                for k in (1,2,3):
                    m0=(b[k]-before[k])/(b[0]-before[0])
                    m1=(after[k]-a[k])/(after[0]-a[0])
                    values.append(max(.005,(2*t**3-3*t*t+1)*a[k]+(t**3-2*t*t+t)*span*m0
                                      +(-2*t**3+3*t*t)*b[k]+(t**3-t*t)*span*m1))
                w,front,back=values
                w *= 1.16  # Broader face encloses the eyeballs instead of exposing their sides.
                break
        theta=2*math.pi*v; x=w*math.sin(theta); f=math.cos(theta)
        # Extend the cranium along Y while retaining the eye socket positions.
        y=(front*f**.6 if f>=0 else 1.55*back*f)-.12
        if f>0:
            # Nose bridge/tip, alae, cheekbones, brow ridges, lips and chin.
            relief=(.27*gaussian(x,z,0,2.95,.15,.40)+.40*gaussian(x,z,0,2.62,.18,.15)
                    +.16*gaussian(x,z,0,1.65,.38,.14)
                    +.13*gaussian(x,z,0,2.12,.40,.065)+.12*gaussian(x,z,0,2.01,.38,.06))
            for sign in (-1,1):
                relief += .13*gaussian(x,z,sign*.25,2.52,.12,.11)
                relief += .15*gaussian(x,z,sign*.73,2.65,.24,.20)
                relief += .14*gaussian(x,z,sign*.60,3.38,.35,.10)
                relief -= .08*gaussian(x,z,sign*.56,3.1,.38,.22)
            y+=relief*f**2
        return (x,y,z)
    def keep(p):
        x,y,z=p
        # Open eye sockets; the independently controlled eyes fill these holes.
        return not (y>0 and any(((x-s*.59)/.345)**2+((z-3.14)/.195)**2<1 for s in (-1,1)))
    points,faces=grid(140,160,sample,keep)
    write_mesh('head.dae',points,faces)


def lids():
    for name,lo,hi in [('upper_lid.dae',28,90),('lower_lid.dae',-90,-38)]:
        def sample(u,v):
            lat=math.radians(lo+(hi-lo)*u); lon=2*math.pi*v
            return (.412*math.cos(lat)*math.cos(lon),.412*math.sin(lat),.412*math.cos(lat)*math.sin(lon))
        write_mesh(name,*grid(22,64,sample))
    # Ears are thin ellipsoids with a recessed center, mounted on the skull sides.
    def ear(u,v):
        lat=-math.pi/2+math.pi*u; lon=2*math.pi*v
        return (.14*math.cos(lat)*math.cos(lon),.25*math.cos(lat)*math.sin(lon),.43*math.sin(lat))
    points,faces=grid(28,48,ear)
    write_mesh('ear.dae',points,[(a,c,b) for a,b,c in faces])


if __name__ == '__main__':
    head(); lids()
