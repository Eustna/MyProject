import krpc
import time
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import multiprocessing
from matplotlib.animation import FuncAnimation
from krpcLibrary import set_vessel,get_part_list_tag,get_tag_liquid_fuel,node_circular_orbit_pe
from krpcLibrary import burning_time,reference,node_Homan_earth_system,tranfer

def show(ref_name):
    conn = krpc.connect(name='绘制图片')
    vessel = set_vessel(conn)
    time.sleep(1)
    ref = reference(conn,ref_name)

    fig = plt.figure(figsize=(5, 2.5))
    ax = fig.add_subplot(111, projection='3d')
    xdata, ydata, zdata = [], [], []
    scat = ax.scatter(xdata, ydata, zdata,s=3)
    line, = ax.plot(xdata, ydata,zdata,linewidth=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([]) 
    ax.set_xlabel('X')  
    ax.set_ylabel('Y')  
    ax.set_zlabel('Z')  
    ax.set_title(f"Track under {ref_name} reference frame")
    def update(frame):
        for text in ax.texts:
            text.remove()
        rate = conn.space_center.warp_rate
        x = vessel.position(ref)[0]
        y = vessel.position(ref)[2]
        z = vessel.position(ref)[1]
        xdata.append(x)
        ydata.append(y)
        zdata.append(z)
        ax.set_xlabel(f'X={round(vessel.position(ref)[0],2)}')  
        ax.set_ylabel(f'Y={round(vessel.position(ref)[2],2)}')  
        ax.set_zlabel(f'Z={round(vessel.position(ref)[1],2)}')
        scat._offsets3d = (xdata, ydata, zdata)
        line.set_data(xdata, ydata)
        line.set_3d_properties(zdata)
        ax.view_init(elev=20, azim=frame)
        ax.set_xlim([min(xdata)-10, max(xdata)+10])  
        ax.set_ylim([min(ydata)-10, max(ydata)+10])  
        ax.set_zlim([min(zdata)-10, max(zdata)+10])  
        ax.text(x, y, z, f"Altitude:{round(vessel.flight().mean_altitude,2)} \n Latitude:{round(vessel.flight().latitude,4)}\n Longitude:{round(vessel.flight().longitude,4)}")
        plt.pause(1/rate)
        return line, scat,
    ani = FuncAnimation(fig, update, frames=np.linspace(0,360, 256), blit=False, repeat=True)
    plt.show()

def show_2dim():
    conn = krpc.connect(name='绘制图片')
    vessel = set_vessel(conn)
    time.sleep(1)
    fig = plt.figure(figsize=(5, 2.5))
    ax = fig.add_subplot(111)
    xdata, ydata = [], []
    scat = ax.scatter(xdata, ydata,s=1.5)
    line, = ax.plot(xdata, ydata,linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Longitude')  
    ax.set_ylabel('Latitude')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_title("Ground track")
    ax.grid(True, which='both')
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    def update(frame):
        for text in ax.texts:
            text.remove()
        rate = conn.space_center.warp_rate
        x = vessel.flight().longitude
        y = vessel.flight().latitude
        xdata.append(x)
        ydata.append(y)
        ax.set_xlabel(f'Longitude:{round(vessel.flight().longitude,4)}')
        ax.set_ylabel(f'Latitude:{round(vessel.flight().latitude,4)}')  
        scat.set_offsets(np.column_stack((xdata, ydata)))
        line.set_data(xdata, ydata)
        ax.text(x, y,  f"Altitude:{round(vessel.flight().mean_altitude,2)}")
        plt.pause(1/rate)
        return line,scat,
    ani = FuncAnimation(fig, update, frames=np.linspace(0,360, 256), blit=False, repeat=True)
    plt.show()



if __name__ == '__main__':
    t1 = multiprocessing.Process(target=show, args=("Celestial",))
    t2 = multiprocessing.Process(target=show, args=("Body",))
    t3 = multiprocessing.Process(target=show_2dim)
    t1.start()
    t2.start()
    t3.start()


    conn = krpc.connect(name='长征五号发射')
    print("任务简述：发射探测器抵达Mun轨道")
    print("发射前检查开始")
    vessel = set_vessel(conn)
    ref = reference(conn)

    Time= conn.add_stream(getattr, conn.space_center, 'ut')
    Altitude=conn.add_stream(getattr, vessel.flight(), 'mean_altitude')
    Apoapsis=conn.add_stream(getattr, vessel.orbit, 'apoapsis_altitude')
    engine_list = get_part_list_tag(conn,"engine")
    booster_tank_list = get_part_list_tag(conn,"booster_fuel")
    first_detach_list = get_part_list_tag(conn,"first_stage_detach")
    first_detach = first_detach_list[0]
    second_engine_list = get_part_list_tag(conn,"second_engine")
    second_engine = second_engine_list[0]
    plantform_list = get_part_list_tag(conn,"plantform")
    palens_list = get_part_list_tag(conn,"palen")
    mu=conn.space_center.bodies["Kerbin"].gravitational_parameter

    engine_activate = False
    fly_stage_1 = True
    fly_stage_2 = False
    orbit_arrange_1 = False
    booster_detach = False
    first_stage_detach = False
    fairing_detach = False
    wait_to_node1 = False
    print_change = False
    goal_apoapsis = 120000
    fuel_amount_booster = get_tag_liquid_fuel(conn,"booster_fuel") + 10
    fuel_amount_first_stage = get_tag_liquid_fuel(conn,"first_stage_fuel") + 10

    angle = 90
    burn_time = 0
    vessel.auto_pilot.engage()
    vessel.auto_pilot.target_pitch_and_heading(90, 90)#初始航向
    vessel.control.toggle_action_group(1)
    print("检查正常，可以发射")
    for i in range(10):
        time.sleep(1)
        print(f"倒计时：{10-i}")

    while True:
        time.sleep(0.1)
        if not print_change:
            if angle <= 70:
                print("开始重力转弯")
                print_change = True

        if not engine_activate:
            vessel.control.throttle = 1
            time.sleep(0.9)
            vessel.control.toggle_action_group(3)
            for part in engine_list:
                part.engine.active = True
            engine_activate = True
            print("点火")

        if not booster_detach:
            if fuel_amount_booster<=0.1:
                vessel.control.toggle_action_group(2)
                booster_detach = True
                print("助推器分离")
            elif abs(fuel_amount_booster-get_tag_liquid_fuel(conn,"booster_fuel")) <= 0.1 and vessel.control.throttle != 0:
                vessel.control.toggle_action_group(2)
                booster_detach = True
                print("助推器分离")
            fuel_amount_booster = get_tag_liquid_fuel(conn,"booster_fuel")

        if booster_detach:
            if not first_stage_detach:
                if fuel_amount_first_stage<=0.1:
                    vessel.control.throttle = 0
                    time.sleep(0.1)
                    vessel.control.toggle_action_group(5)
                    first_stage_detach = True
                    print("一级火箭分离")
                elif abs(fuel_amount_first_stage-get_tag_liquid_fuel(conn,"first_stage_fuel")) <= 0.1 and vessel.control.throttle >= 0.5 :
                    vessel.control.throttle = 0
                    time.sleep(0.1)
                    vessel.control.toggle_action_group(5)
                    first_stage_detach = True
                    print("一级火箭分离")
                fuel_amount_first_stage = get_tag_liquid_fuel(conn,"first_stage_fuel")

        if not fairing_detach:
            if Altitude()>= vessel.orbit.body.atmosphere_depth:
                vessel.control.toggle_action_group(4)
                fairing_detach = True
                print("整流罩分离")


        if fly_stage_1:
            if Apoapsis()<goal_apoapsis*0.95:
                angle=90*(goal_apoapsis-Apoapsis())/(goal_apoapsis)
            else:
                vessel.control.throttle = 0
                fly_stage_1 = True
                fly_stage_2 = True
            vessel.auto_pilot.target_pitch_and_heading(angle, 90)

        if fly_stage_2:
            finish = False
            vessel.auto_pilot.target_pitch_and_heading(0, 90)
            if Apoapsis() <= goal_apoapsis-100:
                vessel.control.throttle = 1
            elif goal_apoapsis-100 <Apoapsis() <=goal_apoapsis-10:
                vessel.control.throttle = 0.1
            elif goal_apoapsis-10 <Apoapsis() <=goal_apoapsis:
                vessel.control.throttle = 0.05
            else:
                vessel.control.throttle = 0
                finish = True
            if Altitude()>= vessel.orbit.body.atmosphere_depth and finish:
                fly_stage_2 = False
                orbit_arrange_1 = True

            if orbit_arrange_1:
                R = vessel.orbit.apoapsis
                a = vessel.orbit.semi_major_axis
                now_speed = math.sqrt( mu*( (2/R) - (1/a) ) )
                need_speed = math.sqrt( mu*( (1/R) ) )
                node1 = vessel.control.add_node(vessel.orbit.time_to_apoapsis+Time(),
                                                prograde=need_speed-now_speed
                )
                vessel.auto_pilot.disengage()
                time.sleep(0.1)
                vessel.control.sas = True
                time.sleep(0.1)
                vessel.control.sas_mode = conn.space_center.SASMode.maneuver
                burn_time = burning_time(conn,need_speed-now_speed)
                print(f"规划入轨轨道，预计燃烧时间为：{round(burn_time,2)}s     ")
                orbit_arrange_1 = False
                wait_to_node1 = True
                break

    while True:
        time.sleep(0.001)
        
        if wait_to_node1:
            next_node = vessel.control.nodes[0]
            if burn_time <=5:
                if next_node.time_to < burn_time/2:
                    vessel.control.throttle = 1
                    time.sleep(burn_time)
                    vessel.control.throttle = 0
                    next_node.remove()
                    wait_to_node1 = False
                    tranfer_stage = True
            else:
                if next_node.time_to > burn_time/2:
                    vessel.control.throttle = 0
                elif -burn_time/2<= next_node.time_to <= burn_time/2:
                    vessel.control.throttle = 1
                else:
                    vessel.control.throttle = 0
                    next_node.remove()
                    print("加速结束，进入预定轨道")
                    wait_to_node1 = False
                    break
        if next_node.time_to-burn_time/2 >=0:
            print(f"还有{round(next_node.time_to-burn_time/2,2)}s开始加速     ", end='\r')
        elif next_node.time_to-burn_time/2 <0:
            print(f"剩余加速时间：{round(burn_time-(burn_time/2-next_node.time_to),2)}s     ", end='\r')

    if not first_stage_detach:
        vessel.control.throttle = 0
        time.sleep(0.1)
        vessel.control.toggle_action_group(5)
        first_stage_detach = True
        vessel.control.toggle_action_group(6)
        print("一级火箭分离")
        

    obdy_name = "Minmus"

    nodes = node_Homan_earth_system(conn,obdy_name)
    while True:
        time.sleep(0.1)
        if nodes[1].time_to - nodes[0].time_to <= burning_time(conn,nodes[0].delta_v)+120:
            sys.stdout.write("\033[5A")
            sys.stdout.write("\033[J")
            conn.space_center.warp_to(nodes[1].ut+1)
            nodes = node_Homan_earth_system(conn,obdy_name)
        else: 
            break
    
    tranfer(conn,nodes)

    vessel.control.throttle = 0
    time.sleep(0.1)
    vessel.control.toggle_action_group(7)
    vessel.control.throttle = 1
    print("二级火箭分离")
    time.sleep(1)
    vessel.control.throttle = 0
    nodes = node_circular_orbit_pe(conn,1000000)
    tranfer(conn,nodes)
    print("任务圆满成功")