import krpc
import time
import math
from krpcLibrary import set_vessel,IaI,node_transfer_fuelless,point_line_distance
from krpcLibrary import get_tag_liquid_fuel,get_tag_solid_fuel,symmetry_point
from krpcLibrary import get_part_list_tag,burning_time,set_target_vessel,launch_orbit_control
from krpcLibrary import bfminus,bftime,bfplus,Ae_n,IaI,reference,ang_of_AB,A_B

conn = krpc.connect(name='GDLV3_launch')
vessel = set_vessel(conn)

target_vessel_name = "station"
target_vessel = set_target_vessel(conn,target_vessel_name)

ref = reference(conn)
Time= conn.add_stream(getattr, conn.space_center, 'ut')
Altitude=conn.add_stream(getattr, vessel.flight(), 'mean_altitude')
Apoapsis=conn.add_stream(getattr, vessel.orbit, 'apoapsis_altitude')
engine_list = get_part_list_tag(conn,"engine")
solid_engine_list = get_part_list_tag(conn,"solid_fuel")
separate_list = get_part_list_tag(conn,"separate")
support_list = get_part_list_tag(conn,"support")
palens_list = get_part_list_tag(conn,"palen")
mu=conn.space_center.bodies["Kerbin"].gravitational_parameter

goal_apoapsis=80000

engine_statment =  False
stage_1_fly = False
stage_1_fly_adjust = False
open_fairing = False
orbit_arrange_1 = False
wait_to_node1 = False
judgment = [True,
            False,
            False,
            False,
            False,
            0,
            0]

angle = 0
vessel.auto_pilot.target_pitch_and_heading(90, 90)#初始航向
vessel.auto_pilot.engage()

for part in support_list:
    part.launch_clamp.release()

while True:
    if not engine_statment:
        vessel.control.throttle = 1
        for part in engine_list:
            part.engine.active = True
        for part in solid_engine_list:
            part.engine.active = True
        engine_statment = True

    if get_tag_solid_fuel(conn,"solid_fuel")<=0.1:
        vessel.control.toggle_action_group(1)
        # for part in separate_list:
        #     part.decoupler.decouple()
    
    judgment = launch_orbit_control(conn,90,120000,judgment[0],judgment[1],judgment[2],judgment[3],judgment[5],judgment[6])
    # print(judgment)
    if judgment[4]:
        break
    # if not stage_1_fly:
    #     if Apoapsis()<goal_apoapsis*0.95:
    #         angle=90*(goal_apoapsis-Apoapsis())/goal_apoapsis
    #     else:
    #         vessel.control.throttle = 0
    #         stage_1_fly = True
    #         stage_1_fly_adjust = True
    #     vessel.auto_pilot.target_pitch_and_heading(angle, 90)

    # if stage_1_fly_adjust:
    #     finsih = False
    #     vessel.auto_pilot.target_pitch_and_heading(0, 90)
    #     if Apoapsis() <= goal_apoapsis-100:
    #         vessel.control.throttle = 1
    #     elif goal_apoapsis-100 <Apoapsis() <=goal_apoapsis-10:
    #          vessel.control.throttle = 0.1
    #     elif goal_apoapsis-10 <Apoapsis() <=goal_apoapsis:
    #          vessel.control.throttle = 0.05
    #     else:
    #         vessel.control.throttle = 0
    #         finish = True
    #     if Altitude()>= vessel.orbit.body.atmosphere_depth and finish:
    #         stage_1_fly_adjust = False
    #         orbit_arrange_1 = True

    if not open_fairing:
        if Altitude()>= vessel.orbit.body.atmosphere_depth:
            vessel.control.toggle_action_group(2)
            time.sleep(1)
            for part in palens_list:
                part.solar_panel.deployed = True
            open_fairing = True
            
    # if orbit_arrange_1:
    #     R = vessel.orbit.apoapsis
    #     a = vessel.orbit.semi_major_axis
    #     now_speed = math.sqrt( mu*( (2/R) - (1/a) ) )
    #     need_speed = math.sqrt( mu*( (1/R) ) )
    #     node1 = vessel.control.add_node(vessel.orbit.time_to_apoapsis+Time(),
    #                                     prograde=need_speed-now_speed
    #     )

    #     if True:#姿态调整
    #         vessel.auto_pilot.disengage()
    #         time.sleep(0.1)
    #         vessel.control.sas = True
    #         time.sleep(0.1)
    #         vessel.control.sas_mode = conn.space_center.SASMode.maneuver
    #         while True:
    #             pitch_0=vessel.flight().pitch
    #             heading_0=vessel.flight().heading
    #             time.sleep(1)
    #             pitch_1=vessel.flight().pitch
    #             heading_1=vessel.flight().heading
    #             if abs(pitch_1-pitch_0)<0.1 and abs(heading_1-heading_0)<0.1:
    #                 break
    #     burn_time = burning_time(conn,need_speed-now_speed)
    #     print(f"燃烧时间为：{round(burn_time,2)}秒")
    #     orbit_arrange_1 = False
    #     wait_to_node1 = True

    # if wait_to_node1:
    #     next_node = vessel.control.nodes[0]
    #     if burn_time <=5:
    #         if next_node.time_to < burn_time/2:
    #             vessel.control.throttle = 1
    #             time.sleep(burn_time)
    #             vessel.control.throttle = 0

    #             next_node.remove()
    #             wait_to_node1 = False
    #             tranfer_stage = True
    #     else:
    #         if next_node.time_to > burn_time/2:
    #             vessel.control.throttle = 0
    #         elif -burn_time/2<= next_node.time_to <= burn_time/2:
    #             vessel.control.throttle = 1
    #         else:
    #             vessel.control.throttle = 0
    #             next_node.remove()
    #             wait_to_node1 = False
    #             break

    time.sleep(0.1)


time.sleep(5)
for j in range(2):
    info = node_transfer_fuelless(conn,target_vessel_name)
    for i in range(4):
        next_node = vessel.control.nodes[0]
        burn_time = burning_time(conn,abs(info[2][i]))
        if 1 < burn_time <=3:
            rate = 0.2
            burn_time = burning_time(conn,abs(info[2][i]),None,rate)
        elif 0.5 < burn_time <=1:
            rate = 0.1
            burn_time = burning_time(conn,abs(info[2][i]),None,rate)
        elif 0.25< burn_time <= 0.5:
            rate = 0.05
            burn_time = burning_time(conn,abs(info[2][i]),None,rate)
        elif 0.1<= burn_time <= 0.25:
            rate = 0.02
            burn_time = burning_time(conn,abs(info[2][i]),None,rate)
        elif burn_time <=0.1:
            rate = 0.005
            burn_time = burning_time(conn,abs(info[2][i]),None,rate)
        else:
            rate = 1
        print(f"需要燃烧的时间为：{round(burn_time,2)}")
        time.sleep(0.1)
        vessel.control.sas_mode = conn.space_center.SASMode.maneuver
        time.sleep(1)
        conn.space_center.warp_to(next_node.time_to+conn.space_center.ut-burn_time-20)
        while True:
            if next_node.time_to > burn_time/2:
                    vessel.control.throttle = 0
            elif -burn_time/2<= next_node.time_to <= burn_time/2:
                vessel.control.throttle = 1*rate
            else:
                vessel.control.throttle = 0
                next_node.remove()
                break
            time.sleep(0.001)


print("开始交汇修正")
vessel.auto_pilot.engage()
vessel.auto_pilot.reference_frame = ref
vessel.control.sas = False
vessel.control.rcs = True

while True:
    time.sleep(0.1)
    r_A = vessel.position(ref)
    r_B = bfplus(target_vessel.position(ref),bftime(200,Ae_n(target_vessel.position(ref))))
    dr = bfminus(r_B,r_A)
    if IaI(dr)<= 200:
        k = 0
    else:
        k = 50/(1+math.exp(500/IaI(dr)))
    vc = bftime(k,Ae_n(dr))
    v_A = vessel.velocity(ref)
    v_B = target_vessel.velocity(ref)
    dv = bfminus(v_B,v_A)
    vessel.auto_pilot.target_direction = Ae_n(bfplus(vc,dv))

    while True:
        vessel.auto_pilot.target_direction = Ae_n(bfplus(vc,dv))

        if ang_of_AB(vessel.auto_pilot.target_direction,vessel.flight(ref).direction) <= 1:
            F = vessel.available_thrust
            m0 = vessel.mass
            rate=IaI(dv)*m0/0.1/F
            if rate > 1:
                rate = 1
            vessel.control.throttle = 1*rate
            time.sleep(0.1)
            vessel.control.throttle = 0
            break
        time.sleep(0.1)

    r_A = vessel.position(ref)
    r_B = bfplus(target_vessel.position(ref),bftime(200,Ae_n(target_vessel.position(ref))))
    dr = bfminus(r_B,r_A)
    v_A = vessel.velocity(ref)
    v_B = target_vessel.velocity(ref)
    dv = bfminus(v_B,v_A)
    if IaI(dr) <=1000 and IaI(dv) <= 10:
        break


print("进入外泊点")
record =0
while True:
    r_A = vessel.position(ref)
    r_B = bfplus(target_vessel.position(ref),bftime(200,Ae_n(target_vessel.position(ref))))
    dr = bfminus(r_B,r_A)
    v_A = vessel.velocity(ref)
    v_B = target_vessel.velocity(ref)
    dv = bfminus(v_B,v_A)
    if IaI(dr) <=50:
        v_goal = 0
    else:
        v_goal = 40/(1+math.exp(200/IaI(bfminus(vessel.position(ref),target_vessel.position(ref)))))
    dr = bfminus(r_B,r_A)
    vc = bftime(v_goal,Ae_n(dr))
    rcs_forward = conn.space_center.transform_direction(
        (0, 1, 0), 
        vessel.reference_frame, 
        ref)
    rcs_up = conn.space_center.transform_direction(
        (0, 0, -1), 
        vessel.reference_frame, 
        ref)
    rcs_right = conn.space_center.transform_direction(
        (1, 0, 0), 
        vessel.reference_frame, 
        ref)
    v=Ae_n(bfplus(vc,dv))
    vessel.control.up = A_B(v,rcs_up)
    vessel.control.forward = A_B(v,rcs_forward)
    vessel.control.right = A_B(v,rcs_right)
    if IaI(bfminus(target_vessel.velocity(ref),vessel.velocity(ref))) <= 0.5 and IaI(dr)<=60:
        record +=1
    if record >= 10:
        print("外泊点进入")
        break



######需要修改对接口
# target_port = target_vessel.parts.docking_ports[0]

docking_ports = target_vessel.parts.with_module('ModuleDockingNode')
for part in docking_ports:
    if part.docking_port.state == conn.space_center.DockingPortState.ready:
        target_port = part.docking_port

vessel_port = vessel.parts.docking_ports[0]

vessel.auto_pilot.reference_frame = ref
target_port_forward = conn.space_center.transform_direction((0, -1, 0), target_port.reference_frame, ref)
if True:
    print("检测航线安全")
    distance=point_line_distance(vessel_port.position(ref),
                                 bfplus(target_port.position(ref),bftime(-60,Ae_n(target_port_forward))),
                                 target_vessel.position(ref)
                                 )
    if distance >40:
        print(f"安全，目标距离航线：{round(distance,2)}米，处于安全距离,可以进入下一泊点")
        Safe = True
    else :
        print(f"危险，目标距离航线：{round(distance,2)}米,有撞击风险")
        Safe = False

if not Safe:
    print("修正航线")
    target_port_new = symmetry_point(vessel_port.position(ref),
                                     bfplus(target_port.position(ref),bftime(-60,Ae_n(target_port_forward))),
                                     target_vessel.position(ref),
                                     50,
                                     False
                                     )
    target_port_new=bfminus(target_port_new,target_port.position(ref))

    record =0
    vessel.control.forward = 1
    time.sleep(5)
    while True:
        target_port_forward = conn.space_center.transform_direction((0, -1, 0), target_port.reference_frame, ref)
        vessel.auto_pilot.target_direction = target_port_forward
        r_A = vessel_port.position(ref)
        r_B = bfplus(target_port_new,target_port.position(ref))
        dr = bfminus(r_B,r_A)
        v_A = vessel.velocity(ref)
        v_B = target_vessel.velocity(ref)
        dv = bfminus(v_B,v_A)
        if IaI(dr) <=10:
            v_goal = 0
        else:
            v_goal = 10/(1+math.exp(40/IaI(bfminus(vessel.position(ref),target_vessel.position(ref)))))
        dr = bfminus(r_B,r_A)
        vc = bftime(v_goal,Ae_n(dr))
        rcs_forward = conn.space_center.transform_direction(
            (0, 1, 0), 
            vessel.reference_frame, 
            ref)
        rcs_up = conn.space_center.transform_direction(
            (0, 0, -1), 
            vessel.reference_frame, 
            ref)
        rcs_right = conn.space_center.transform_direction(
            (1, 0, 0), 
            vessel.reference_frame, 
            ref)
        v=Ae_n(bfplus(vc,dv))
        vessel.control.up = A_B(v,rcs_up)
        vessel.control.forward = A_B(v,rcs_forward)
        vessel.control.right = A_B(v,rcs_right)
        if IaI(bfminus(target_vessel.velocity(ref),vessel.velocity(ref))) <= 0.5 and IaI(dr)<=20:
            record +=1
        if record >= 20:
            print("航线修正完毕，进入下一泊点")
            break
    
record =0
vessel.control.forward = 1
time.sleep(5)
while True:
    target_port_forward = conn.space_center.transform_direction((0, -1, 0), target_port.reference_frame, ref)
    vessel.auto_pilot.target_direction = target_port_forward
    r_A = vessel_port.position(ref)
    r_B = bfplus(target_port.position(ref),bftime(-60,Ae_n(target_port_forward)))
    dr = bfminus(r_B,r_A)
    v_A = vessel.velocity(ref)
    v_B = target_vessel.velocity(ref)
    dv = bfminus(v_B,v_A)
    if IaI(dr) <=10:
        v_goal = 0
    else:
        v_goal = 10/(1+math.exp(40/IaI(bfminus(vessel.position(ref),target_vessel.position(ref)))))
    dr = bfminus(r_B,r_A)
    vc = bftime(v_goal,Ae_n(dr))
    rcs_forward = conn.space_center.transform_direction(
        (0, 1, 0), 
        vessel.reference_frame, 
        ref)
    rcs_up = conn.space_center.transform_direction(
        (0, 0, -1), 
        vessel.reference_frame, 
        ref)
    rcs_right = conn.space_center.transform_direction(
        (1, 0, 0), 
        vessel.reference_frame, 
        ref)
    v=Ae_n(bfplus(vc,dv))
    vessel.control.up = A_B(v,rcs_up)
    vessel.control.forward = A_B(v,rcs_forward)
    vessel.control.right = A_B(v,rcs_right)
    if IaI(bfminus(target_vessel.velocity(ref),vessel.velocity(ref))) <= 0.2 and IaI(dr)<=15:
        record +=1
    if record >= 50:
        print("泊点已到达，准备对接")
        break


distance = 50
target_vessel.control.sas = True
while True:
    if distance <=1:
        distance -= 0.1
    elif distance<=0:
        distance = 0
    elif distance>15:
        distance -= 5
    else:
        distance -= 1
    record = 0
    if distance<=1:
        time.sleep(0.01)
    else:
        time.sleep(0.1)
    while True:
        try:
            time.sleep(0.01)
            if True:#将对接口对准对接口
                port_forward = conn.space_center.transform_direction((0, 1, 0), vessel_port.reference_frame, ref)
                vessel_forward = conn.space_center.transform_direction((0, 1, 0), vessel.reference_frame, ref)
                delta_direction = bfminus(vessel_forward,port_forward)
                target_port_forward = conn.space_center.transform_direction((0, -1, 0), target_port.reference_frame, ref)
                target_port_forward_f = bfplus(target_port_forward,delta_direction)
                vessel.auto_pilot.target_direction = target_port_forward_f
            x = IaI(bfminus(vessel_port.position(ref),bfplus(target_port.position(ref),bftime(-distance,Ae_n(target_port_forward)))))
            if x <=0:
                v_goal = 0
            else:
                if distance>=5:
                    v_goal = 10/(1+math.exp(10/x))
                elif 0.1<=distance<5:
                    v_goal = 0.5/(1+math.exp(1/x))
                else:
                    v_goal = 4/(1+math.exp(-10/x))-2

            r_A = vessel_port.position(ref)
            r_B = bfplus(target_port.position(ref),bftime(-distance,Ae_n(target_port_forward)))
            dr = bfminus(r_B,r_A)
            vc = bftime(v_goal,Ae_n(dr))
            v_A = vessel.velocity(ref)
            v_B = target_vessel.velocity(ref)
            dv = bfminus(v_B,v_A)
            rcs_forward = conn.space_center.transform_direction(
                (0, 1, 0), 
                vessel.reference_frame, 
                ref)

            rcs_up = conn.space_center.transform_direction(
                (0, 0, -1), 
                vessel.reference_frame, 
                ref)

            rcs_right = conn.space_center.transform_direction(
                (1, 0, 0), 
                vessel.reference_frame, 
                ref)
            v=Ae_n(bfplus(vc,dv))
            vessel.control.up = A_B(v,rcs_up)
            vessel.control.forward = A_B(v,rcs_forward)
            vessel.control.right = A_B(v,rcs_right)
            if IaI(bfminus(target_vessel.velocity(ref),vessel.velocity(ref))) <= 0.5 and x<=3:
                if distance >= 5:
                    record +=1  
                else:
                    if IaI(bfminus(target_vessel.velocity(ref),vessel.velocity(ref))) <= 0.2 and x<=0.5:
                        record +=1 
            if distance<0.5:
                if record >= 1:
                    print(f"下一调整距离为：{round(distance,2)}米")
                    break
            else:
                if record >= 1:
                    print(f"下一调整距离为：{round(distance,2)}米")
                    break
        except Exception as e:
            print("对接完成")
            break
    if record <=0:
        break