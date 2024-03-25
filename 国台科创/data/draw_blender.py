import bpy
import pandas as pd

df = pd.read_csv('data.csv', nrows=100)

scene = bpy.context.scene

obj = bpy.context.active_object

psys = obj.modifiers.new("Particles", 'PARTICLE_SYSTEM').particle_system
psys.settings.count = len(df)
psys.settings.frame_start = 1
psys.settings.frame_end = 1
psys.settings.lifetime = 100

for i, row in df.iterrows():
    particle = psys.particles[i]
    particle.location = (row.iloc[0], row.iloc[1], row.iloc[2])

scene.update()