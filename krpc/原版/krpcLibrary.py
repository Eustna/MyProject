import math
import numpy as np
import krpc
from sympy import symbols, Eq, solve, pi, sqrt
from scipy.optimize import fsolve
import sys
import time
#########################################数学工具
def bf(name):
    name=np.array(name)
    return name

def bfplus(value1,value2):
    value1=bf(value1)
    value2=bf(value2)
    if len(value1) == len(value2):
        value=value1+value2
    else:
        raise ValueError(f"数据输入为{value1}和{value2}，请检查数据")
    return tuple(value)

def bfminus(value1,value2):
    value1=bf(value1)
    value2=bf(value2)
    if len(value1) == len(value2):
        value=value1-value2
    else:
        raise ValueError(f"数据输入为{value1}和{value2}，请检查数据")
    return tuple(value)

def bftime(value1=float,value2=tuple):
    value2=bf(value2)
    value=value1*value2
    return tuple(value)

def bfdivid(value1=tuple,value2=float):
    value1=bf(value1)
    value=value1/value2
    return tuple(value)

def rad_to_ang(value):#弧度转换角度
    value=value*180/math.pi
    return float(value)

def ang_to_rad(value):#角度转换弧度
    value=value*math.pi/180
    return float(value)

def AxB(value1,value2):#计算叉乘
    value = bf(([0.0, 0.0, 0.0]))
    value1=bf(value1)
    value1 = (value1[0], value1[2], value1[1])
    value2=bf(value2)
    value2 = (value2[0], value2[2], value2[1])
    levi_civita = bf([[[int((i - j) * (j - k) * (k - i) / 2) 
                    for k in range(3)] 
                    for j in range(3)] 
                    for i in range(3)])
    if len(value1) == 3 and len(value2)==3:
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    value[i] += levi_civita[i][j][k] * value1[j] * value2[k]
    else:
        raise ValueError(f"数据输入为{value1}和{value2}，请检查数据")
    
    value = (value[0], value[2], value[1])
    return tuple(value)

def A_B(value1,value2):#计算点乘
    value1=bf(value1)
    value2=bf(value2)
    if len(value1) == len(value2):
        value = np.dot(value1,value2)
    else:
        raise ValueError(f"数据输入为{value1}和{value2}，请检查数据")
    return float(value)

def IaI(value):#计算模长
    value=bf(value)
    if len(value) == 3:
        value=np.linalg.norm(value)
    else:
        raise ValueError(f"数据输入为{value}，请检查数据")
    return float(value)

def Ae_n(value):#A矢量方向的单位矢量
    value=bf(value)
    if len(value) == 3:
        value=value/np.linalg.norm(value)
    else:
        raise ValueError(f"数据输入为{value}，请检查数据")
    return tuple(value)

def rad_of_AB(value1,value2):#计算AB弧度夹角
    value1=bf(value1)
    value2=bf(value2)
    if len(value1) == 3 and len(value2)==3:
        value = math.acos(A_B(value1,value2)/IaI(value1)/IaI(value2))
    else:
        raise ValueError(f"数据输入为{value1}和{value2}，请检查数据")
    return float(value)

def ang_of_AB(value1,value2):#计算AB夹角
    value1=bf(value1)
    value2=bf(value2)
    if len(value1) == 3 and len(value2)==3:
        value = math.acos(A_B(value1,value2)/IaI(value1)/IaI(value2))*180/math.pi
    else:
        raise ValueError(f"数据输入为{value1}和{value2}，请检查数据")
    return float(value)

def A_pro_CD(value1,value2,value3):#计算A在面CD的投影
    value1=bf(value1)
    value2=bf(value2)
    value3=bf(value3)
    if  len(value1) == 3 and len(value2) ==3  and len(value3) ==3:
        n=Ae_n(AxB(value2,value3))
        A_n=A_B(value1,n)*bf(n)
        value=value1-A_n
    else:
        raise ValueError(f"数据输入为{value1}、{value2}和{value3}，请检查数据")
    return tuple(value)




def f_to_time(f,e,a,mu):
    E=2*math.atan(math.sqrt((1-e)/(1+e))*math.tan(f/2))
    M=E-e*math.sin(E)
    if M<=0:
        M +=2*math.pi
    time = (2*math.pi-M)*math.sqrt(a**3/mu)
    return time

def f1_to_f2_time(f1,f2,e,a,mu):
    E1=2*math.atan(math.sqrt((1-e)/(1+e))*math.tan(f1/2))
    M1=E1-e*math.sin(E1)
    if M1<=0:
        M1 +=2*math.pi
    time1 = (2*math.pi-M1)*math.sqrt(a**3/mu)

    E2=2*math.atan(math.sqrt((1-e)/(1+e))*math.tan(f2/2))
    M2=E2-e*math.sin(E2)
    if M2<=0:
        M2 +=2*math.pi
    time2 = (2*math.pi-M2)*math.sqrt(a**3/mu)

    round_time = (2*math.pi)*math.sqrt(a**3/mu)

    Time =round_time-(time2-time1) 
    if Time >=round_time:
        Time -=round_time
    return Time

def symmetry_point(point_1,point_2,point,num=0,form=True):
    if form:
        index = 2
    else:
        index = 1
    line = Ae_n(bfminus(point_2,point_1))
    point_on_line = bftime(A_B(bfminus(point,point_1),line),line)
    vertical = bfminus(bfminus(point,point_1),point_on_line)
    result = bfplus(bfplus(point,bftime(-index,vertical)),bftime(-num,Ae_n(vertical)))
    return tuple(result)

def point_line_distance(point_1,point_2,point):
    line = Ae_n(bfminus(point_2,point_1))
    point_on_line = bftime(A_B(bfminus(point,point_1),line),line)
    vertical = bfminus(bfminus(point,point_1),point_on_line)
    return IaI(vertical)
#######################################ksp检测
def set_vessel(conn,vessel_name=None):
    try :
        vessel = conn.space_center.bodies[vessel_name]
    except Exception as e:
        if vessel_name == None:
            vessel = conn.space_center.active_vessel
        else:
            Found=False
            all_vessels = conn.space_center.vessels
            for target_vessel in all_vessels:
                if target_vessel.name == vessel_name:
                    vessel = target_vessel
                    Found = True
                    break
            if not Found :
                print("未找到目标，请检查输入名称是否在以下列表")
                for vesl in all_vessels:
                    print(vesl.name)
    return vessel

def reference(conn,ref_name=None,vessel_name=None):
    vessel=set_vessel(conn,vessel_name)
    ref_frame_dict = {  "Celestial": vessel.orbit.body.non_rotating_reference_frame, 
                        "Orbital": vessel.orbital_reference_frame,
                        "Body" : vessel.orbit.body.reference_frame,
                        "Surface" :vessel.surface_reference_frame,
                        "Velocity":vessel.surface_velocity_reference_frame,
                        "Center_orbital": vessel.orbit.body.orbital_reference_frame,
                        "Center": vessel.reference_frame
                      }
    if ref_name in ref_frame_dict:
        ref_frame = ref_frame_dict[ref_name]
    elif ref_name == None:
        ref_frame= vessel.orbit.body.non_rotating_reference_frame
    else:
        print(f"没有找到参考系{ref_name},可支持的参考系名字有：\nCelestial\nOrbital\nBody\nSurface\nVelocity\nCenter_orbital")
        ref_frame = None
    return ref_frame

def get_f(conn,r = None,name = None):
    vessel = set_vessel(conn,name)
    ref = reference(conn)
    pe_position = vessel.orbit.position_at(conn.space_center.ut+vessel.orbit.time_to_periapsis,ref)
    current_position = vessel.position(ref)
    current_velocity = vessel.velocity(ref)
    if r == None:
        r = current_position
    H = AxB(current_position,current_velocity)
    if A_B(AxB(pe_position,r),H) >=0:
        f = rad_of_AB(pe_position,r)
    else:
        f = 2*math.pi - rad_of_AB(pe_position,r)
    return f

def set_target_vessel(conn,vessel_name):
    all_vessels = conn.space_center.vessels
    Found = False
    for target_vessel in all_vessels:
        if target_vessel.name == vessel_name:
            conn.space_center.target_vessel = target_vessel
            Found = True
            break
    if Found :
        pass
    else:
        print("未找到目标，请检查输入名称是否在以下列表")
        for vesl in all_vessels:
            print(vesl.name)
    return target_vessel

def set_target_planet(conn,planet_name):
    try:
        target_body = conn.space_center.bodies[planet_name]
        conn.space_center.active_vessel.target = target_body
    except Exception as e:
        target_body = None
        print("输入有误")
    return target_body

def round_time(conn,vessel_name=None):
    vessel=set_vessel(conn,vessel_name)
    Time = vessel.orbit.period
    return Time

def round_area(conn,vessel_name=None):
    vessel = set_vessel(conn,vessel_name)
    a = vessel.orbit.semi_major_axis
    e = vessel.orbit.eccentricity
    S=math.pi*a*math.sqrt(1-e**2)*a
    return S





def time_to_trano(conn,Time=None,name=None):
    vessel=set_vessel(conn,name)
    if Time == None:
        true_anomaly=vessel.orbit.true_anomaly_at_ut(conn.space_center.ut)
    else:
        true_anomaly=vessel.orbit.true_anomaly_at_ut(conn.space_center.ut+Time)
    return true_anomaly

def trano_to_time(conn,true_anomaly,vessel_name=None):
    vessel=set_vessel(conn,vessel_name)
    Time=vessel.orbit.ut_at_true_anomaly(true_anomaly)-conn.space_center.ut
    return Time
    
def time_to_An(conn,target_vessel_name,name=None):
    vessel=set_vessel(conn,name)
    target=set_vessel(conn,target_vessel_name)
    set_target_vessel(conn,f"{target_vessel_name}")
    An_time=vessel.orbit.ut_at_true_anomaly(vessel.orbit.true_anomaly_at_an(target.orbit))-conn.space_center.ut
    return An_time

def time_to_Dn(conn,target_vessel_name,name=None):
    vessel=set_vessel(conn,name)
    target=set_vessel(conn,target_vessel_name)
    set_target_vessel(conn,f"{target_vessel_name}")
    Dn_time=vessel.orbit.ut_at_true_anomaly(vessel.orbit.true_anomaly_at_dn(target.orbit))-conn.space_center.ut
    return Dn_time

def time_to(conn,name=None,vessel_name=None,detective=False):
    if detective == False:
        vessle=set_vessel(conn,vessel_name)
        name_dict ={"Pe": vessle.orbit.time_to_periapsis,
                    "Ap": vessle.orbit.time_to_apoapsis
                    }
        if name == None:
            Time=min(name_dict.get("Pe"), name_dict.get("Ap"))
        elif name == "Pe" or "Ap":
            Time = name_dict.get(name)
        else:
            raise ValueError(f"数据输入为{name}，可使用的有：\n Pe\n Ap")
    else:
        vessle=set_vessel()
        name_dict ={"An": time_to_An(conn,vessel_name),
                    "Dn": time_to_Dn(conn,vessel_name)
                    }
        if name == None:
            Time=min(name_dict.get("An"), name_dict.get("Dn"))
        elif name == "An" or "Dn":
            Time = name_dict.get(name)
        else:
            raise ValueError(f"数据输入为{name}，可使用的有：\n An\n Dn")
    return Time

def get_vector(conn,Name=None,Time=None,vessel_name=None,ref_name=None,body_name=None):
    if body_name ==None:
        body_name="Kerbin"
    vessel=set_vessel(conn,vessel_name)
    ref_frame=reference(conn,ref_name)
    r_now=bf(vessel.position(ref_frame))
    v_now=bf(vessel.velocity(ref_frame))
    if Time is not None:
        r=bf(vessel.orbit.position_at(conn.space_center.ut+Time,ref_frame))
        h=bf(AxB(r_now,v_now))
        mu=vessel.orbit.body.gravitational_parameter
        e=vessel.orbit.eccentricity
        f=time_to_trano(conn,Time)
        er=bf(Ae_n(r))
        et=bf(Ae_n(AxB(h,r)))
        v =bf((mu/IaI(h)*e*math.sin(f))*er+(mu/IaI(h)*(1+e*math.cos(f)))*et)
    else:
        r=r_now
        v=v_now
    vector_dict={
                   "r": r,
                   "v": v 
                }
    if Name is None:
        vector = r
    elif Name == "r" or "v":
        vector = vector_dict.get(Name)
    else:
        raise ValueError(f"数据输入为{Name}，可使用的为：\n r\n v")
    return tuple(vector)

def decouple(part):
        try:
            part.decoupler.decouple()
        except krpc.error.RPCError:
            pass 

def get_part_list_tag(conn,tag,vessel_name=None):
    if vessel_name == None :
        vessel = conn.space_center.active_vessel
    else:
        vessel = set_vessel(conn,vessel_name)
    part_list = vessel.parts.with_tag(tag)
    return part_list

def get_part_list_name(conn,part_name,vessel_name=None):
    if vessel_name == None :
        vessel = conn.space_center.active_vessel
    else:
        vessel = set_vessel(conn,vessel_name)
    part_list = vessel.parts.with_name(part_name)
    return part_list

def get_tag_liquid_fuel(conn,tag,vessel_name=None):
    liquid_fuel_tag_list = get_part_list_tag(conn,tag,vessel_name)
    total_amount = 0
    for tank in liquid_fuel_tag_list:
        for resource in tank.resources.with_resource('LiquidFuel'):
            total_amount += resource.amount
    return total_amount

def get_tag_solid_fuel(conn,tag,vessel_name=None):
    solid_fuel_tag_list = get_part_list_tag(conn,tag,vessel_name)
    total_amount = 0
    for tank in solid_fuel_tag_list:
        for resource in tank.resources.with_resource('SolidFuel'):
            total_amount += resource.amount
    return total_amount

def burning_time(conn,delta_v,vessel_name = None,rate=None):
    if rate == None:
        rate = 1
    vessel = set_vessel(conn,vessel_name)
    F = vessel.available_thrust
    Isp = vessel.specific_impulse * 9.82
    m0 = vessel.mass
    m1 = m0 / math.exp((delta_v)/Isp)
    flow_rate = F / Isp *rate
    burn_time = (m0 - m1) / flow_rate
    return burn_time



def node_transfer_An_ap(conn,target_name,vessel_name = None):
    if True:#基础设置
        # if body_name ==None:
        #     body_name="Kerbin"
        vessel=set_vessel(conn,vessel_name)
        ref_frame=reference(conn)
        target_vessel=set_vessel(conn,target_name)
        mu=vessel.orbit.body.gravitational_parameter
        time_to_transfer=time_to_An(conn,target_name)

    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()

    if True:#计算在转移点处原始的矢量
        r_0=get_vector(conn,"r",time_to_transfer)
        v_0=get_vector(conn,"v",time_to_transfer)
        h=AxB(r_0,v_0)
        H=AxB(set_vessel(conn,target_name).position(ref_frame),set_vessel(conn,target_name).velocity(ref_frame))

    if True:#计算需要变换的矢量
        v_goal=bftime(math.sqrt(mu/IaI(r_0)),Ae_n(AxB(H,r_0)))
        delta_v=bfminus(v_goal,v_0)

    if True:#设置原始轨道的基矢
        e_p=Ae_n(v_0)
        e_n=Ae_n(h)
        e_r=Ae_n(AxB(v_0,h))

    if True:#计算delta v在三个方向的投影
        delta_v_p=A_B(delta_v,e_p)
        delta_v_n=A_B(delta_v,e_n)
        delta_v_r=A_B(delta_v,e_r)

    if True:
        time_1 = conn.space_center.ut + time_to_transfer
        node1 = vessel.control.add_node(time_1, 
                                    prograde=delta_v_p, 
                                    normal=delta_v_n,
                                    radial=delta_v_r
                                    )
    
    if True:#获取进入第二个节点的时间
        r_pe = target_vessel.orbit.position_at(target_vessel.orbit.time_to_apoapsis+conn.space_center.ut,ref_frame)
        if A_B(AxB(r_0,r_pe),H) >= 0:#计算进入矢量和开始转移的夹角
            rad_r0_to_rA = rad_of_AB(r_0,r_pe)
        else:
            rad_r0_to_rA = 2*math.pi-rad_of_AB(r_0,r_pe)
        time_to_rA = (rad_r0_to_rA*node1.orbit.period)/2/math.pi
        time_2 = time_1+time_to_rA

    if True:#计算整个转移需要的时间t
        t = target_vessel.orbit.time_to_apoapsis-time_to_transfer-time_to_rA
        while True:
                if t<=0:
                    t += target_vessel.orbit.period
                else:
                    break

    if True:#求解方程
        R=IaI(node1.orbit.position_at(time_2, ref_frame))
        d=R-IaI(r_pe)
        def f(x):
            return math.pi*math.sqrt(x**3/mu) + math.pi*math.sqrt((x-d/2)**3/mu) - t
        try:
            a = fsolve(f, x0=R)
        except Exception as e:
            t += target_vessel.orbit.period
            a = fsolve(f, x0=R)
        while True:
            if 2*a <=R+target_vessel.orbit.body.atmosphere_depth+target_vessel.orbit.body.equatorial_radius:
                t += target_vessel.orbit.period
                a = fsolve(f, x0=R)
            else:
                break

    if True:#活力公式计算速度
        delta_v_1 = IaI(delta_v)
        v_2 = math.sqrt( mu*( (2/R)-(1/a) ) )
        delta_v_2 = v_2-math.sqrt( mu*( 1/R ) )
        node2 = vessel.control.add_node(time_2,
                    prograde=delta_v_2)
        
        time_3=time_2+node2.orbit.period/2
        v_3 = math.sqrt( mu*( ( 2/(2*a-R) )-( 2/(2*a-d) ) ) )
        delta_v_3 = v_3-math.sqrt( mu*( ( 2/(2*a-R) )-( 1/(a) ) ) )
        node3 = vessel.control.add_node(time_3,
                        prograde=delta_v_3)
        
        time_4=time_3+node3.orbit.period/2
        v_4 = math.sqrt( mu*( (2/IaI(r_pe))-(1/target_vessel.orbit.semi_major_axis) ))
        delta_v_4 = v_4-math.sqrt( mu*( (2/IaI(r_pe))-(1/node3.orbit.semi_major_axis) ))
        node4 = vessel.control.add_node(time_4,
                            prograde=delta_v_4,
                            radial = 0)
        
        all_delta_v = abs(delta_v_1)+abs(delta_v_2)+abs(delta_v_3)+abs(delta_v_4)
        all_time = time_4-conn.space_center.ut
        delta_v = (delta_v_1,delta_v_2,delta_v_3,delta_v_4)
        times = (time_1-conn.space_center.ut,time_2-time_1,time_3-time_2,time_4-time_3)
        return_information=(all_delta_v,all_time,delta_v,times)
    return return_information

def node_transfer_Dn_ap(conn,target_name,vessel_name = None):
    if True:#基础设置
        # if body_name ==None:
        #     body_name="Kerbin"
        vessel=set_vessel(conn,vessel_name)
        ref_frame=reference(conn)
        target_vessel=set_vessel(conn,target_name)
        mu=vessel.orbit.body.gravitational_parameter
        time_to_transfer=time_to_Dn(conn,target_name)

    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()
    
    if True:#计算在转移点处原始的矢量
        r_0=get_vector(conn,"r",time_to_transfer)
        v_0=get_vector(conn,"v",time_to_transfer)
        h=AxB(r_0,v_0)
        H=AxB(set_vessel(conn,target_name).position(ref_frame),set_vessel(conn,target_name).velocity(ref_frame))

    if True:#计算需要变换的矢量
        v_goal=bftime(math.sqrt(mu/IaI(r_0)),Ae_n(AxB(H,r_0)))
        delta_v=bfminus(v_goal,v_0)

    if True:#设置原始轨道的基矢
        e_p=Ae_n(v_0)
        e_n=Ae_n(h)
        e_r=Ae_n(AxB(v_0,h))

    if True:#计算delta v在三个方向的投影
        delta_v_p=A_B(delta_v,e_p)
        delta_v_n=A_B(delta_v,e_n)
        delta_v_r=A_B(delta_v,e_r)

    if True:
        time_1 = conn.space_center.ut + time_to_transfer
        node1 = vessel.control.add_node(time_1, 
                                    prograde=delta_v_p, 
                                    normal=delta_v_n,
                                    radial=delta_v_r
                                    )
    
    if True:#获取进入第二个节点的时间
        r_pe = target_vessel.orbit.position_at(target_vessel.orbit.time_to_apoapsis+conn.space_center.ut,ref_frame)
        if A_B(AxB(r_0,r_pe),H) >= 0:#计算进入矢量和开始转移的夹角
            rad_r0_to_rA = rad_of_AB(r_0,r_pe)
        else:
            rad_r0_to_rA = 2*math.pi-rad_of_AB(r_0,r_pe)
        time_to_rA = (rad_r0_to_rA*node1.orbit.period)/2/math.pi
        time_2 = time_1+time_to_rA

    if True:#计算整个转移需要的时间t
        t = target_vessel.orbit.time_to_apoapsis-time_to_transfer-time_to_rA
        while True:
                if t<=0:
                    t += target_vessel.orbit.period
                else:
                    break

    if True:#求解方程
        R=IaI(node1.orbit.position_at(time_2, ref_frame))
        d=R-IaI(r_pe)
        def f(x):
            return math.pi*math.sqrt(x**3/mu) + math.pi*math.sqrt((x-d/2)**3/mu) - t
        try:
            a = fsolve(f, x0=R)
        except Exception as e:
            t += target_vessel.orbit.period
            a = fsolve(f, x0=R)
        while True:
            if 2*a <=R+target_vessel.orbit.body.atmosphere_depth+target_vessel.orbit.body.equatorial_radius:
                t += target_vessel.orbit.period
                a = fsolve(f, x0=R)
            else:
                break

    if True:#活力公式计算速度
        delta_v_1 = IaI(delta_v)
        v_2 = math.sqrt( mu*( (2/R)-(1/a) ) )
        delta_v_2 = v_2-math.sqrt( mu*( 1/R ) )
        node2 = vessel.control.add_node(time_2,
                    prograde=delta_v_2)
        
        time_3=time_2+node2.orbit.period/2
        v_3 = math.sqrt( mu*( ( 2/(2*a-R) )-( 2/(2*a-d) ) ) )
        delta_v_3 = v_3-math.sqrt( mu*( ( 2/(2*a-R) )-( 1/(a) ) ) )
        node3 = vessel.control.add_node(time_3,
                        prograde=delta_v_3)
        
        time_4=time_3+node3.orbit.period/2
        v_4 = math.sqrt( mu*( (2/IaI(r_pe))-(1/target_vessel.orbit.semi_major_axis) ))
        delta_v_4 = v_4-math.sqrt( mu*( (2/IaI(r_pe))-(1/node3.orbit.semi_major_axis) ))
        node4 = vessel.control.add_node(time_4,
                            prograde=delta_v_4,
                            radial = 0)

        all_delta_v = abs(delta_v_1)+abs(delta_v_2)+abs(delta_v_3)+abs(delta_v_4)
        all_time = time_4-conn.space_center.ut
        delta_v = (delta_v_1,delta_v_2,delta_v_3,delta_v_4)
        times = (time_1-conn.space_center.ut,time_2-time_1,time_3-time_2,time_4-time_3)
        return_information=(all_delta_v,all_time,delta_v,times)
    return return_information

def node_transfer_An_pe(conn,target_name,vessel_name = None):
    if True:#基础设置
        # if body_name ==None:
        #     body_name="Kerbin"
        vessel=set_vessel(conn,vessel_name)
        ref_frame=reference(conn)
        target_vessel=set_vessel(conn,target_name)
        mu=vessel.orbit.body.gravitational_parameter
        time_to_transfer=time_to_An(conn,target_name)

    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()

    if True:#计算在转移点处原始的矢量
        r_0=get_vector(conn,"r",time_to_transfer)
        v_0=get_vector(conn,"v",time_to_transfer)
        h=AxB(r_0,v_0)
        H=AxB(set_vessel(conn,target_name).position(ref_frame),set_vessel(conn,target_name).velocity(ref_frame))

    if True:#计算需要变换的矢量
        v_goal=bftime(math.sqrt(mu/IaI(r_0)),Ae_n(AxB(H,r_0)))
        delta_v=bfminus(v_goal,v_0)

    if True:#设置原始轨道的基矢
        e_p=Ae_n(v_0)
        e_n=Ae_n(h)
        e_r=Ae_n(AxB(v_0,h))

    if True:#计算delta v在三个方向的投影
        delta_v_p=A_B(delta_v,e_p)
        delta_v_n=A_B(delta_v,e_n)
        delta_v_r=A_B(delta_v,e_r)

    if True:
        time_1 = conn.space_center.ut + time_to_transfer
        node1 = vessel.control.add_node(time_1, 
                                    prograde=delta_v_p, 
                                    normal=delta_v_n,
                                    radial=delta_v_r
                                    )
    
    if True:#获取进入第二个节点的时间
        r_pe = target_vessel.orbit.position_at(target_vessel.orbit.time_to_periapsis+conn.space_center.ut,ref_frame)
        if A_B(AxB(r_0,r_pe),H) >= 0:#计算进入矢量和开始转移的夹角
            rad_r0_to_rA = rad_of_AB(r_0,r_pe)
        else:
            rad_r0_to_rA = 2*math.pi-rad_of_AB(r_0,r_pe)
        time_to_rA = (rad_r0_to_rA*node1.orbit.period)/2/math.pi
        time_2 = time_1+time_to_rA

    if True:#计算整个转移需要的时间t
        t = target_vessel.orbit.time_to_periapsis-time_to_transfer-time_to_rA
        while True:
                if t<=0:
                    t += target_vessel.orbit.period
                else:
                    break

    if True:#求解方程
        R=IaI(node1.orbit.position_at(time_2, ref_frame))
        d=R-IaI(r_pe)
        def f(x):
            return math.pi*math.sqrt(x**3/mu) + math.pi*math.sqrt((x-d/2)**3/mu) - t
        try:
            a = fsolve(f, x0=R)
        except Exception as e:
            t += target_vessel.orbit.period
            a = fsolve(f, x0=R)
        while True:
            if 2*a <=R+target_vessel.orbit.body.atmosphere_depth+target_vessel.orbit.body.equatorial_radius:
                t += target_vessel.orbit.period
                a = fsolve(f, x0=R)
            else:
                break

    if True:#活力公式计算速度
        delta_v_1 = IaI(delta_v)
        v_2 = math.sqrt( mu*( (2/R)-(1/a) ) )
        delta_v_2 = v_2-math.sqrt( mu*( 1/R ) )
        node2 = vessel.control.add_node(time_2,
                    prograde=delta_v_2)
        
        time_3=time_2+node2.orbit.period/2
        v_3 = math.sqrt( mu*( ( 2/(2*a-R) )-( 2/(2*a-d) ) ) )
        delta_v_3 = v_3-math.sqrt( mu*( ( 2/(2*a-R) )-( 1/(a) ) ) )
        node3 = vessel.control.add_node(time_3,
                        prograde=delta_v_3)
        
        time_4=time_3+node3.orbit.period/2
        v_4 = math.sqrt( mu*( (2/IaI(r_pe))-(1/target_vessel.orbit.semi_major_axis) ))
        delta_v_4 = v_4-math.sqrt( mu*( (2/IaI(r_pe))-(1/node3.orbit.semi_major_axis) ))
        node4 = vessel.control.add_node(time_4,
                            prograde=delta_v_4,
                            radial = 0)
        
        all_delta_v = abs(delta_v_1)+abs(delta_v_2)+abs(delta_v_3)+abs(delta_v_4)
        all_time = time_4-conn.space_center.ut
        delta_v = (delta_v_1,delta_v_2,delta_v_3,delta_v_4)
        times = (time_1-conn.space_center.ut,time_2-time_1,time_3-time_2,time_4-time_3)
        return_information=(all_delta_v,all_time,delta_v,times)
    return return_information

def node_transfer_Dn_pe(conn,target_name,vessel_name = None):
    if True:#基础设置
        # if body_name ==None:
        #     body_name="Kerbin"
        vessel=set_vessel(conn,vessel_name)
        ref_frame=reference(conn)
        target_vessel=set_vessel(conn,target_name)
        mu=vessel.orbit.body.gravitational_parameter

        time_to_transfer=time_to_Dn(conn,target_name)

    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()

    if True:#计算在转移点处原始的矢量
        r_0=get_vector(conn,"r",time_to_transfer)
        v_0=get_vector(conn,"v",time_to_transfer)
        h=AxB(r_0,v_0)
        H=AxB(set_vessel(conn,target_name).position(ref_frame),set_vessel(conn,target_name).velocity(ref_frame))

    if True:#计算需要变换的矢量
        v_goal=bftime(math.sqrt(mu/IaI(r_0)),Ae_n(AxB(H,r_0)))
        delta_v=bfminus(v_goal,v_0)

    if True:#设置原始轨道的基矢
        e_p=Ae_n(v_0)
        e_n=Ae_n(h)
        e_r=Ae_n(AxB(v_0,h))

    if True:#计算delta v在三个方向的投影
        delta_v_p=A_B(delta_v,e_p)
        delta_v_n=A_B(delta_v,e_n)
        delta_v_r=A_B(delta_v,e_r)

    if True:
        time_1 = conn.space_center.ut + time_to_transfer
        node1 = vessel.control.add_node(time_1, 
                                    prograde=delta_v_p, 
                                    normal=delta_v_n,
                                    radial=delta_v_r
                                    )
    
    if True:#获取进入第二个节点的时间
        r_pe = target_vessel.orbit.position_at(target_vessel.orbit.time_to_periapsis+conn.space_center.ut,ref_frame)
        if A_B(AxB(r_0,r_pe),H) >= 0:#计算进入矢量和开始转移的夹角
            rad_r0_to_rA = rad_of_AB(r_0,r_pe)
        else:
            rad_r0_to_rA = 2*math.pi-rad_of_AB(r_0,r_pe)
        time_to_rA = (rad_r0_to_rA*node1.orbit.period)/2/math.pi
        time_2 = time_1+time_to_rA

    if True:#计算整个转移需要的时间t
        t = target_vessel.orbit.time_to_periapsis-time_to_transfer-time_to_rA
        while True:
                if t<=0:
                    t += target_vessel.orbit.period
                else:
                    break

    if True:#求解方程
        R=IaI(node1.orbit.position_at(time_2, ref_frame))
        d=R-IaI(r_pe)
        def f(x):
            return math.pi*math.sqrt(x**3/mu) + math.pi*math.sqrt((x-d/2)**3/mu) - t
        try:
            a = fsolve(f, x0=R)
        except Exception as e:
            t += target_vessel.orbit.period
            a = fsolve(f, x0=R)
        while True:
            if 2*a <=R+target_vessel.orbit.body.atmosphere_depth+target_vessel.orbit.body.equatorial_radius:
                t += target_vessel.orbit.period
                a = fsolve(f, x0=R)
            else:
                break

    if True:#活力公式计算速度
        delta_v_1 = IaI(delta_v)
        v_2 = math.sqrt( mu*( (2/R)-(1/a) ) )
        delta_v_2 = v_2-math.sqrt( mu*( 1/R ) )
        node2 = vessel.control.add_node(time_2,
                    prograde=delta_v_2)
        
        time_3=time_2+node2.orbit.period/2
        v_3 = math.sqrt( mu*( ( 2/(2*a-R) )-( 2/(2*a-d) ) ) )
        delta_v_3 = v_3-math.sqrt( mu*( ( 2/(2*a-R) )-( 1/(a) ) ) )
        node3 = vessel.control.add_node(time_3,
                        prograde=delta_v_3)
        
        time_4=time_3+node3.orbit.period/2
        v_4 = math.sqrt( mu*( (2/IaI(r_pe))-(1/target_vessel.orbit.semi_major_axis) ))
        delta_v_4 = v_4-math.sqrt( mu*( (2/IaI(r_pe))-(1/node3.orbit.semi_major_axis) ))
        node4 = vessel.control.add_node(time_4,
                            prograde=delta_v_4,
                            radial = 0)
        
        all_delta_v = abs(delta_v_1)+abs(delta_v_2)+abs(delta_v_3)+abs(delta_v_4)
        all_time = time_4-conn.space_center.ut
        delta_v = (delta_v_1,delta_v_2,delta_v_3,delta_v_4)
        times = (time_1-conn.space_center.ut,time_2-time_1,time_3-time_2,time_4-time_3)
        return_information=(all_delta_v,all_time,delta_v,times)
    return return_information

def node_transfer_fuelless(conn,target_name,vessel_name = None):
    
    info_1=node_transfer_An_ap(conn,target_name,vessel_name)

    info_2=node_transfer_Dn_ap(conn,target_name,vessel_name)

    info_3=node_transfer_An_pe(conn,target_name,vessel_name)

    info_4=node_transfer_Dn_pe(conn,target_name,vessel_name)

    list = (info_1[0],
            info_2[0],
            info_3[0],
            info_4[0])
    
    if info_1[0] == min(list):
        info = node_transfer_Dn_ap(conn,target_name,vessel_name)
    elif info_2[0] == min(list):
        info = node_transfer_Dn_ap(conn,target_name,vessel_name)
    elif info_3[0] == min(list):
        info = node_transfer_An_pe(conn,target_name,vessel_name)
    else:
        info = node_transfer_Dn_pe(conn,target_name,vessel_name)
    print(f"合计最少需要消耗的dv为：{round(info[0],2)}")
    print(f"合计需要消耗的时间为：{round(info[1],2)}秒，为：{round((info[1])/60,2)}分，为：{round((info[1])/3600,2)}时，为：{round((info[1])/60/60/24,2)}天")
    return info

def node_transfer_timeless(conn,target_name,vessel_name = None):

    info_1=node_transfer_An_ap(conn,target_name,vessel_name)

    info_2=node_transfer_Dn_ap(conn,target_name,vessel_name)

    info_3=node_transfer_An_pe(conn,target_name,vessel_name)

    info_4=node_transfer_Dn_pe(conn,target_name,vessel_name)

    list = (info_1[1],
            info_2[1],
            info_3[1],
            info_4[1])
    
    if info_1[1] == min(list):
        info = node_transfer_Dn_ap(conn,target_name,vessel_name)
    elif info_2[1] == min(list):
        info = node_transfer_Dn_ap(conn,target_name,vessel_name)
    elif info_3[1] == min(list):
        info = node_transfer_An_pe(conn,target_name,vessel_name)
    else:
        info = node_transfer_Dn_pe(conn,target_name,vessel_name)
    print(f"合计需要消耗的dv为：{round(info[0],2)}")
    print(f"合计最少需要消耗的时间为：{round(info[1],2)}秒，为：{round((info[1])/60,2)}分，为：{round((info[1])/3600,2)}时，为：{round((info[1])/60/60/24,2)}天")
    return info









########################轨道机动
def node_intersect_An_pe(conn,target_name,vessel_name = None):
    if True:#基础设置
        vessel=set_vessel(conn,vessel_name)
        ref_frame=reference(conn)
        target_vessel=set_vessel(conn,target_name)
        mu=vessel.orbit.body.gravitational_parameter
        time_to_transfer=time_to_An(conn,target_name)

    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()

    if True:#计算在转移点处原始的矢量
        r_0=get_vector(conn,"r",time_to_transfer)
        v_0=get_vector(conn,"v",time_to_transfer)
        h=AxB(r_0,v_0)
        H=AxB(set_vessel(conn,target_name).position(ref_frame),set_vessel(conn,target_name).velocity(ref_frame))

    if True:#计算需要变换的矢量
        v_goal=bftime(math.sqrt(mu/IaI(r_0)),Ae_n(AxB(H,r_0)))
        delta_v=bfminus(v_goal,v_0)

    if True:#设置原始轨道的基矢
        e_p=Ae_n(v_0)
        e_n=Ae_n(h)
        e_r=Ae_n(AxB(v_0,h))

    if True:#计算delta v在三个方向的投影
        delta_v_p=A_B(delta_v,e_p)
        delta_v_n=A_B(delta_v,e_n)
        delta_v_r=A_B(delta_v,e_r)

    if True:
        time_1 = conn.space_center.ut + time_to_transfer
        node1 = vessel.control.add_node(time_1, 
                                    prograde=delta_v_p, 
                                    normal=delta_v_n,
                                    radial=delta_v_r
                                    )
    
    if True:#获取进入第二个节点的时间
        r_pe = target_vessel.orbit.position_at(target_vessel.orbit.time_to_periapsis+conn.space_center.ut,ref_frame)
        if A_B(AxB(r_0,r_pe),H) >= 0:#计算进入矢量和开始转移的夹角
            rad_r0_to_rA = rad_of_AB(r_0,r_pe)
        else:
            rad_r0_to_rA = 2*math.pi-rad_of_AB(r_0,r_pe)
        time_to_rA = (rad_r0_to_rA*node1.orbit.period)/2/math.pi
        time_2 = time_1+time_to_rA

    if True:#计算整个转移需要的时间t
        t = target_vessel.orbit.time_to_periapsis-time_to_transfer-time_to_rA
        while True:
                if t<=0:
                    t += target_vessel.orbit.period
                else:
                    break

    if True:#求解方程
        R=IaI(node1.orbit.position_at(time_2, ref_frame))
        d=R-IaI(r_pe)
        def f(x):
            return math.pi*math.sqrt(x**3/mu) + math.pi*math.sqrt((x-d/2)**3/mu) - t
        try:
            a = fsolve(f, x0=R)
        except Exception as e:
            t += target_vessel.orbit.period
            a = fsolve(f, x0=R)
        while True:
            if 2*a <=R+target_vessel.orbit.body.atmosphere_depth+target_vessel.orbit.body.equatorial_radius:
                t += target_vessel.orbit.period
                a = fsolve(f, x0=R)
            else:
                break

    if True:#活力公式计算速度
        v_2 = math.sqrt( mu*( (2/R)-(1/a) ) )
        delta_v_2 = v_2-math.sqrt( mu*( 1/R ) )
        node2 = vessel.control.add_node(time_2,
                    prograde=delta_v_2)
        
        time_3=time_2+node2.orbit.period/2
        v_3 = math.sqrt( mu*( ( 2/(2*a-R) )-( 2/(2*a-d) ) ) )
        delta_v_3 = v_3-math.sqrt( mu*( ( 2/(2*a-R) )-( 1/(a) ) ) )
        node3 = vessel.control.add_node(time_3,
                        prograde=delta_v_3)
        
        time_4=time_3+node3.orbit.period/2
        v_4 = math.sqrt( mu*( (2/IaI(r_pe))-(1/target_vessel.orbit.semi_major_axis) ))
        delta_v_4 = v_4-math.sqrt( mu*( (2/IaI(r_pe))-(1/node3.orbit.semi_major_axis) ))
        node4 = vessel.control.add_node(time_4,
                            prograde=delta_v_4,
                            radial = 0)
        nodes = [node1,node2,node3,node4]
    return nodes

def node_intersect_Dn_pe(conn,target_name,vessel_name = None):
    if True:#基础设置
        vessel=set_vessel(conn,vessel_name)
        ref_frame=reference(conn)
        target_vessel=set_vessel(conn,target_name)
        mu=vessel.orbit.body.gravitational_parameter

        time_to_transfer=time_to_Dn(conn,target_name)

    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()

    if True:#计算在转移点处原始的矢量
        r_0=get_vector(conn,"r",time_to_transfer)
        v_0=get_vector(conn,"v",time_to_transfer)
        h=AxB(r_0,v_0)
        H=AxB(set_vessel(conn,target_name).position(ref_frame),set_vessel(conn,target_name).velocity(ref_frame))

    if True:#计算需要变换的矢量
        v_goal=bftime(math.sqrt(mu/IaI(r_0)),Ae_n(AxB(H,r_0)))
        delta_v=bfminus(v_goal,v_0)

    if True:#设置原始轨道的基矢
        e_p=Ae_n(v_0)
        e_n=Ae_n(h)
        e_r=Ae_n(AxB(v_0,h))

    if True:#计算delta v在三个方向的投影
        delta_v_p=A_B(delta_v,e_p)
        delta_v_n=A_B(delta_v,e_n)
        delta_v_r=A_B(delta_v,e_r)

    if True:
        time_1 = conn.space_center.ut + time_to_transfer
        node1 = vessel.control.add_node(time_1, 
                                    prograde=delta_v_p, 
                                    normal=delta_v_n,
                                    radial=delta_v_r
                                    )
    
    if True:#获取进入第二个节点的时间
        r_pe = target_vessel.orbit.position_at(target_vessel.orbit.time_to_periapsis+conn.space_center.ut,ref_frame)
        if A_B(AxB(r_0,r_pe),H) >= 0:#计算进入矢量和开始转移的夹角
            rad_r0_to_rA = rad_of_AB(r_0,r_pe)
        else:
            rad_r0_to_rA = 2*math.pi-rad_of_AB(r_0,r_pe)
        time_to_rA = (rad_r0_to_rA*node1.orbit.period)/2/math.pi
        time_2 = time_1+time_to_rA

    if True:#计算整个转移需要的时间t
        t = target_vessel.orbit.time_to_periapsis-time_to_transfer-time_to_rA
        while True:
                if t<=0:
                    t += target_vessel.orbit.period
                else:
                    break

    if True:#求解方程
        R=IaI(node1.orbit.position_at(time_2, ref_frame))
        d=R-IaI(r_pe)
        def f(x):
            return math.pi*math.sqrt(x**3/mu) + math.pi*math.sqrt((x-d/2)**3/mu) - t
        try:
            a = fsolve(f, x0=R)
        except Exception as e:
            t += target_vessel.orbit.period
            a = fsolve(f, x0=R)
        while True:
            if 2*a <=R+target_vessel.orbit.body.atmosphere_depth+target_vessel.orbit.body.equatorial_radius:
                t += target_vessel.orbit.period
                a = fsolve(f, x0=R)
            else:
                break

    if True:#活力公式计算速度
        v_2 = math.sqrt( mu*( (2/R)-(1/a) ) )
        delta_v_2 = v_2-math.sqrt( mu*( 1/R ) )
        node2 = vessel.control.add_node(time_2,
                    prograde=delta_v_2)
        
        time_3=time_2+node2.orbit.period/2
        v_3 = math.sqrt( mu*( ( 2/(2*a-R) )-( 2/(2*a-d) ) ) )
        delta_v_3 = v_3-math.sqrt( mu*( ( 2/(2*a-R) )-( 1/(a) ) ) )
        node3 = vessel.control.add_node(time_3,
                        prograde=delta_v_3)
        
        time_4=time_3+node3.orbit.period/2
        v_4 = math.sqrt( mu*( (2/IaI(r_pe))-(1/target_vessel.orbit.semi_major_axis) ))
        delta_v_4 = v_4-math.sqrt( mu*( (2/IaI(r_pe))-(1/node3.orbit.semi_major_axis) ))
        node4 = vessel.control.add_node(time_4,
                            prograde=delta_v_4,
                            radial = 0)
        nodes = [node1,node2,node3,node4]
    return nodes

def node_intersect_An_ap(conn,target_name,vessel_name = None):
    if True:#基础设置
        vessel=set_vessel(conn,vessel_name)
        ref_frame=reference(conn)
        target_vessel=set_vessel(conn,target_name)
        mu=vessel.orbit.body.gravitational_parameter
        time_to_transfer=time_to_An(conn,target_name)

    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()

    if True:#计算在转移点处原始的矢量
        r_0=get_vector(conn,"r",time_to_transfer)
        v_0=get_vector(conn,"v",time_to_transfer)
        h=AxB(r_0,v_0)
        H=AxB(set_vessel(conn,target_name).position(ref_frame),set_vessel(conn,target_name).velocity(ref_frame))

    if True:#计算需要变换的矢量
        v_goal=bftime(math.sqrt(mu/IaI(r_0)),Ae_n(AxB(H,r_0)))
        delta_v=bfminus(v_goal,v_0)

    if True:#设置原始轨道的基矢
        e_p=Ae_n(v_0)
        e_n=Ae_n(h)
        e_r=Ae_n(AxB(v_0,h))

    if True:#计算delta v在三个方向的投影
        delta_v_p=A_B(delta_v,e_p)
        delta_v_n=A_B(delta_v,e_n)
        delta_v_r=A_B(delta_v,e_r)

    if True:
        time_1 = conn.space_center.ut + time_to_transfer
        node1 = vessel.control.add_node(time_1, 
                                    prograde=delta_v_p, 
                                    normal=delta_v_n,
                                    radial=delta_v_r
                                    )
    
    if True:#获取进入第二个节点的时间
        r_pe = target_vessel.orbit.position_at(target_vessel.orbit.time_to_apoapsis+conn.space_center.ut,ref_frame)
        if A_B(AxB(r_0,r_pe),H) >= 0:#计算进入矢量和开始转移的夹角
            rad_r0_to_rA = rad_of_AB(r_0,r_pe)
        else:
            rad_r0_to_rA = 2*math.pi-rad_of_AB(r_0,r_pe)
        time_to_rA = (rad_r0_to_rA*node1.orbit.period)/2/math.pi
        time_2 = time_1+time_to_rA

    if True:#计算整个转移需要的时间t
        t = target_vessel.orbit.time_to_apoapsis-time_to_transfer-time_to_rA
        while True:
                if t<=0:
                    t += target_vessel.orbit.period
                else:
                    break

    if True:#求解方程
        R=IaI(node1.orbit.position_at(time_2, ref_frame))
        d=R-IaI(r_pe)
        def f(x):
            return math.pi*math.sqrt(x**3/mu) + math.pi*math.sqrt((x-d/2)**3/mu) - t
        try:
            a = fsolve(f, x0=R)
        except Exception as e:
            t += target_vessel.orbit.period
            a = fsolve(f, x0=R)
        while True:
            if 2*a <=R+target_vessel.orbit.body.atmosphere_depth+target_vessel.orbit.body.equatorial_radius:
                t += target_vessel.orbit.period
                a = fsolve(f, x0=R)
            else:
                break

    if True:#活力公式计算速度
        v_2 = math.sqrt( mu*( (2/R)-(1/a) ) )
        delta_v_2 = v_2-math.sqrt( mu*( 1/R ) )
        node2 = vessel.control.add_node(time_2,
                    prograde=delta_v_2)
        
        time_3=time_2+node2.orbit.period/2
        v_3 = math.sqrt( mu*( ( 2/(2*a-R) )-( 2/(2*a-d) ) ) )
        delta_v_3 = v_3-math.sqrt( mu*( ( 2/(2*a-R) )-( 1/(a) ) ) )
        node3 = vessel.control.add_node(time_3,
                        prograde=delta_v_3)
        
        time_4=time_3+node3.orbit.period/2
        v_4 = math.sqrt( mu*( (2/IaI(r_pe))-(1/target_vessel.orbit.semi_major_axis) ))
        delta_v_4 = v_4-math.sqrt( mu*( (2/IaI(r_pe))-(1/node3.orbit.semi_major_axis) ))
        node4 = vessel.control.add_node(time_4,
                            prograde=delta_v_4,
                            radial = 0)
        nodes = [node1,node2,node3,node4]
    return nodes

def node_intersect_Dn_ap(conn,target_name,vessel_name = None):
    if True:#基础设置
        vessel=set_vessel(conn,vessel_name)
        ref_frame=reference(conn)
        target_vessel=set_vessel(conn,target_name)
        mu=vessel.orbit.body.gravitational_parameter
        time_to_transfer=time_to_Dn(conn,target_name)

    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()
    
    if True:#计算在转移点处原始的矢量
        r_0=get_vector(conn,"r",time_to_transfer)
        v_0=get_vector(conn,"v",time_to_transfer)
        h=AxB(r_0,v_0)
        H=AxB(set_vessel(conn,target_name).position(ref_frame),set_vessel(conn,target_name).velocity(ref_frame))

    if True:#计算需要变换的矢量
        v_goal=bftime(math.sqrt(mu/IaI(r_0)),Ae_n(AxB(H,r_0)))
        delta_v=bfminus(v_goal,v_0)

    if True:#设置原始轨道的基矢
        e_p=Ae_n(v_0)
        e_n=Ae_n(h)
        e_r=Ae_n(AxB(v_0,h))

    if True:#计算delta v在三个方向的投影
        delta_v_p=A_B(delta_v,e_p)
        delta_v_n=A_B(delta_v,e_n)
        delta_v_r=A_B(delta_v,e_r)

    if True:
        time_1 = conn.space_center.ut + time_to_transfer
        node1 = vessel.control.add_node(time_1, 
                                    prograde=delta_v_p, 
                                    normal=delta_v_n,
                                    radial=delta_v_r
                                    )
    
    if True:#获取进入第二个节点的时间
        r_pe = target_vessel.orbit.position_at(target_vessel.orbit.time_to_apoapsis+conn.space_center.ut,ref_frame)
        if A_B(AxB(r_0,r_pe),H) >= 0:#计算进入矢量和开始转移的夹角
            rad_r0_to_rA = rad_of_AB(r_0,r_pe)
        else:
            rad_r0_to_rA = 2*math.pi-rad_of_AB(r_0,r_pe)
        time_to_rA = (rad_r0_to_rA*node1.orbit.period)/2/math.pi
        time_2 = time_1+time_to_rA

    if True:#计算整个转移需要的时间t
        t = target_vessel.orbit.time_to_apoapsis-time_to_transfer-time_to_rA
        while True:
                if t<=0:
                    t += target_vessel.orbit.period
                else:
                    break

    if True:#求解方程
        R=IaI(node1.orbit.position_at(time_2, ref_frame))
        d=R-IaI(r_pe)
        def f(x):
            return math.pi*math.sqrt(x**3/mu) + math.pi*math.sqrt((x-d/2)**3/mu) - t
        try:
            a = fsolve(f, x0=R)
        except Exception as e:
            t += target_vessel.orbit.period
            a = fsolve(f, x0=R)
        while True:
            if 2*a <=R+target_vessel.orbit.body.atmosphere_depth+target_vessel.orbit.body.equatorial_radius:
                t += target_vessel.orbit.period
                a = fsolve(f, x0=R)
            else:
                break

    if True:#活力公式计算速度
        v_2 = math.sqrt( mu*( (2/R)-(1/a) ) )
        delta_v_2 = v_2-math.sqrt( mu*( 1/R ) )
        node2 = vessel.control.add_node(time_2,
                    prograde=delta_v_2)
        
        time_3=time_2+node2.orbit.period/2
        v_3 = math.sqrt( mu*( ( 2/(2*a-R) )-( 2/(2*a-d) ) ) )
        delta_v_3 = v_3-math.sqrt( mu*( ( 2/(2*a-R) )-( 1/(a) ) ) )
        node3 = vessel.control.add_node(time_3,
                        prograde=delta_v_3)
        
        time_4=time_3+node3.orbit.period/2
        v_4 = math.sqrt( mu*( (2/IaI(r_pe))-(1/target_vessel.orbit.semi_major_axis) ))
        delta_v_4 = v_4-math.sqrt( mu*( (2/IaI(r_pe))-(1/node3.orbit.semi_major_axis) ))
        node4 = vessel.control.add_node(time_4,
                            prograde=delta_v_4,
                            radial = 0)
        nodes = [node1,node2,node3,node4]
    return nodes

def node_intersect_fuelless(conn,target_name,vessel_name = None):

    node=node_intersect_An_pe(conn,target_name,vessel_name)
    delt_v1 = node[0].delta_v+node[1].delta_v+node[2].delta_v+node[3].delta_v
    node=node_intersect_Dn_pe(conn,target_name,vessel_name)
    delt_v2 = node[0].delta_v+node[1].delta_v+node[2].delta_v+node[3].delta_v
    node=node_intersect_An_ap(conn,target_name,vessel_name)
    delt_v3 = node[0].delta_v+node[1].delta_v+node[2].delta_v+node[3].delta_v
    node=node_intersect_Dn_ap(conn,target_name,vessel_name)
    delt_v4 = node[0].delta_v+node[1].delta_v+node[2].delta_v+node[3].delta_v

    list = (delt_v1,
            delt_v2,
            delt_v3,
            delt_v4)
    
    if delt_v1 == min(list):
        nodes = node_intersect_An_pe(conn,target_name,vessel_name)
    elif delt_v2 == min(list):
        nodes = node_intersect_Dn_pe(conn,target_name,vessel_name)
    elif delt_v3 == min(list):
        nodes = node_intersect_An_ap(conn,target_name,vessel_name)
    else:
        nodes = node_intersect_Dn_ap(conn,target_name,vessel_name)
    print(f"规划轨道方案，预计总共消耗Delta V：{round(nodes[0].delta_v+nodes[1].delta_v+nodes[2].delta_v+nodes[3].delta_v,2)}m/s，以下为详情\n第一阶段：{round(nodes[0].delta_v,2)}m/s\n第二阶段：{round(nodes[1].delta_v,2)}m/s\n第三阶段：{round(nodes[2].delta_v,2)}m/s\n第四阶段：{round(nodes[3].delta_v,2)}m/s\n ")
    print(f"预计{round(nodes[3].time_to,2)}s后交汇\n ")
    return nodes

def node_intersect_timeless(conn,target_name,vessel_name = None):

    node=node_intersect_An_pe(conn,target_name,vessel_name)
    delt_v1 = node[3].time_to
    node=node_intersect_Dn_pe(conn,target_name,vessel_name)
    delt_v2 = node[3].time_to
    node=node_intersect_An_ap(conn,target_name,vessel_name)
    delt_v3 = node[3].time_to
    node=node_intersect_Dn_ap(conn,target_name,vessel_name)
    delt_v4 = node[3].time_to

    list = (delt_v1,
            delt_v2,
            delt_v3,
            delt_v4)
    
    if delt_v1 == min(list):
        nodes = node_intersect_An_pe(conn,target_name,vessel_name)
    elif delt_v2 == min(list):
        nodes = node_intersect_Dn_pe(conn,target_name,vessel_name)
    elif delt_v3 == min(list):
        nodes = node_intersect_An_ap(conn,target_name,vessel_name)
    else:
        nodes = node_intersect_Dn_ap(conn,target_name,vessel_name)
    print(f"规划轨道方案，预计总共消耗Delta V：{round(nodes[0].delta_v+nodes[1].delta_v+nodes[2].delta_v+nodes[3].delta_v,2)}m/s，以下为详情\n第一阶段：{round(nodes[0].delta_v,2)}m/s\n第二阶段：{round(nodes[1].delta_v,2)}m/s\n第三阶段：{round(nodes[2].delta_v,2)}m/s\n第四阶段：{round(nodes[3].delta_v,2)}m/s\n ")
    print(f"预计{round(nodes[3].time_to,2)}s后交汇\n ")
    return nodes

def node_circular_orbit_ap(conn,goal_apoapsis,vessel_name = None):
    
    vessel = set_vessel(conn,vessel_name)
    mu = vessel.orbit.body.gravitational_parameter
    R = vessel.orbit.apoapsis
    a = vessel.orbit.semi_major_axis
    goal_apoapsis +=vessel.orbit.body.equatorial_radius
    v_0 = math.sqrt( mu*( (2/R) - (1/a) ) )
    v_1 = math.sqrt( mu*( (2/R) - (2/(R+goal_apoapsis)) ) )
    v_0_1 = math.sqrt( mu*( (2/goal_apoapsis) - (2/(R+goal_apoapsis)) ) )
    v_1_1 = math.sqrt( mu*( (1/goal_apoapsis) ) )
    delta_v = v_1-v_0
    delta_v_1 = v_1_1-v_0_1
    t = vessel.orbit.time_to_apoapsis+conn.space_center.ut
    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()
    node1 = vessel.control.add_node(t,
                                    prograde=delta_v)
    node2 = vessel.control.add_node(t+node1.orbit.period/2,
                                prograde=delta_v_1)
    print(f"规划轨道方案，预计总共消耗Delta V：{round(node1.delta_v+node2.delta_v,2)}m/s，以下为详情\n第一阶段：{round(node1.delta_v,2)}m/s\n第二阶段：{round(node2.delta_v,2)}m/s\n ")

    nodes = [node1,node2]
    return nodes

def node_circular_orbit_pe(conn,goal_periapsis,vessel_name = None):
    
    vessel = set_vessel(conn,vessel_name)
    mu = vessel.orbit.body.gravitational_parameter
    R = vessel.orbit.periapsis
    a = vessel.orbit.semi_major_axis
    goal_periapsis +=vessel.orbit.body.equatorial_radius
    v_0 = math.sqrt( mu*( (2/R) - (1/a) ) )
    v_1 = math.sqrt( mu*( (2/R) - (2/(R+goal_periapsis)) ) )
    v_0_1 = math.sqrt( mu*( (2/goal_periapsis) - (2/(R+goal_periapsis)) ) )
    v_1_1 = math.sqrt( mu*( (1/goal_periapsis) ) )
    delta_v = v_1-v_0
    delta_v_1 = v_1_1-v_0_1
    t = vessel.orbit.time_to_periapsis+conn.space_center.ut
    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()
    node1 = vessel.control.add_node(t,
                                    prograde=delta_v)
    node2 = vessel.control.add_node(t+node1.orbit.period/2,
                                prograde=delta_v_1)
    print(f"规划轨道方案，预计总共消耗Delta V：{round(node1.delta_v+node2.delta_v,2)}m/s，以下为详情\n第一阶段：{round(node1.delta_v,2)}m/s\n第二阶段：{round(node2.delta_v,2)}m/s\n ")

    nodes = [node1,node2]
    return nodes

def node_Homan_earth_system(conn,target_name,vessel_name = None):
    vessel=set_vessel(conn,vessel_name)
    ref = reference(conn)
    body = conn.space_center.bodies[target_name]
    if True:#基础设置
        ref_frame=reference(conn)
        planet = set_target_planet(conn,target_name)
        mu=vessel.orbit.body.gravitational_parameter
        time_to_transfer = min(vessel.orbit.ut_at_true_anomaly(vessel.orbit.true_anomaly_at_an(planet.orbit))-conn.space_center.ut,
                               vessel.orbit.ut_at_true_anomaly(vessel.orbit.true_anomaly_at_dn(planet.orbit))-conn.space_center.ut)
    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()

    if True:#计算在转移点处原始的矢量
        r_0=get_vector(conn,"r",time_to_transfer)
        v_0=get_vector(conn,"v",time_to_transfer)
        h=AxB(r_0,v_0)
        H=AxB(planet.position(ref_frame),planet.velocity(ref_frame))

    if True:#计算需要变换的矢量
        v_g=bftime(math.sqrt(mu/IaI(r_0)),Ae_n(AxB(H,r_0)))
        delta_v=bfminus(v_g,v_0)

    if True:#设置原始轨道的基矢
        e_p=Ae_n(v_0)
        e_n=Ae_n(h)
        e_r=Ae_n(AxB(v_0,h))

    if True:#计算delta v在三个方向的投影
        delta_v_p=A_B(delta_v,e_p)
        delta_v_n=A_B(delta_v,e_n)
        delta_v_r=A_B(delta_v,e_r)

    if True:#圆化共面轨道
        time_1 = conn.space_center.ut + time_to_transfer
        node1 = vessel.control.add_node(time_1, 
                                    prograde=delta_v_p, 
                                    normal=delta_v_n,
                                    radial=delta_v_r
                                    )
    rg = bftime(-1,planet.orbit.position_at(time_1,ref))
    r0 = vessel.orbit.position_at(time_1,ref)
    t = f1_to_f2_time(get_f(conn,r0),get_f(conn,rg),
                    vessel.orbit.eccentricity,
                    vessel.orbit.semi_major_axis,
                    mu)
    r = node1.orbit.position_at(t+time_1,ref)
    a = (IaI(r)+IaI(planet.position(ref)))/2
    v_goal = math.sqrt( mu*( (2/IaI(r))-(1/a) ) )
    er = Ae_n(r)
    en = Ae_n(AxB(planet.position(ref),planet.velocity(ref)))
    ep = Ae_n(AxB(en,er))
    v_goal = bftime(v_goal,ep)
    if True:
        h = AxB(r0,v_g)
        er=Ae_n(r)
        et=Ae_n(AxB(h,r))
        e = node1.orbit.eccentricity
        pe = node1.orbit.position_at(node1.orbit.time_to_periapsis+time_1,ref)
        f = rad_of_AB(pe,r)
        if A_B(AxB(pe,r),h) <0:
            f = 2*math.pi - f
        v =bfplus(bftime((mu/IaI(h)*e*math.sin(f)),er),bftime((mu/IaI(h)*(1+e*math.cos(f))),et))
    delta_v = bfminus(v_goal,v)
    Ap = Ae_n(v)
    An = en
    Ar = Ae_n(AxB(An,Ap))
    node2 = vessel.control.add_node(t+time_1, 
                                prograde=A_B(delta_v,Ap), 
                                normal=A_B(delta_v,An),
                                radial=A_B(delta_v,Ar)
                                )
    while True:
        node2.ut +=1
        if node2.ut - time_1 >vessel.orbit.period:
            node2.ut -= vessel.orbit.period
        if node2.orbit.next_orbit == None:
            pass
        elif node2.orbit.next_orbit.body.name == body.name:
            break
        time.sleep(0.001)

    time2 = node2.orbit.time_to_soi_change

    node3 = vessel.control.add_node(conn.space_center.ut+time2+1, 
                                prograde=0, 
                                normal=0,
                                radial=0
                                )
    node3.ut+=node3.orbit.time_to_periapsis
    mu = node3.orbit.body.gravitational_parameter
    r = node3.orbit.position_at(node3.ut,reference(conn))
    a = node3.orbit.semi_major_axis
    v_0 = math.sqrt( mu*( (2/IaI(r))-(1/a) ) )
    v = math.sqrt( mu*( (1/IaI(r))) )
    node3.prograde = v - v_0
    num=0
    check = 1
    rate = 10
    while True:
        error0 = abs(node3.orbit.apoapsis - node3.orbit.periapsis)
        time.sleep(0.001)
        node3.prograde -=rate
        error = abs(node3.orbit.apoapsis - node3.orbit.periapsis)
        node3.prograde +=rate
        if check*(error-error0)<=0:
            num+=1
        if num>=50:
            break
        elif 20>num>=1:
            rate = 1
        elif 50>num>=20:
            rate = 0.1
        if error<=error0:
            node3.prograde -=rate
        else:
            node3.prograde +=rate
        check = error-error0
    
    print(f"规划轨道方案，预计总共消耗Delta V：{round(node1.delta_v+node2.delta_v+node3.delta_v,2)}m/s，以下为详情\n第一阶段：{round(node1.delta_v,2)}m/s\n第二阶段：{round(node2.delta_v,2)}m/s\n第三阶段：{round(node3.delta_v,2)}m/s\n ")

    nodes = [node1,node2,node3]
    return nodes

##########################轨道控制
def tranfer(conn,nodes):
    vessel = set_vessel(conn)
    ref = reference(conn)
    delta_vs = []
    delta_rotate = []
    for i in range(len(nodes)):
        delta_vs.append(nodes[i].delta_v)

    if True:#计算飞行器最大旋转速度
        pitch_0 = vessel.flight().pitch
        heading = vessel.flight().heading
        vessel.control.sas = False
        vessel.auto_pilot.engage()
        while True:
            time.sleep(0.1)
            vessel.auto_pilot.target_pitch_and_heading(pitch_0, heading)
            if abs(vessel.flight().pitch - pitch_0) <=0.1:
                break
        for i in range(50):
            vessel.auto_pilot.target_pitch_and_heading(pitch_0-180, heading)
            delta_rotate.append(IaI(vessel.angular_velocity(ref)))
            time.sleep(0.1)
        delta_rotate = max(delta_rotate)
        t_rotate = math.pi/delta_rotate
        vessel.auto_pilot.disengage()

    for i in range(len(nodes)):
        next_node = vessel.control.nodes[0]
        burn_time = burning_time(conn,delta_vs[i])

        if burn_time>=10:
            rate = 1
        else:
            rate = burn_time/10

        burn_time = burning_time(conn,delta_vs[i],None,rate)
        print(f"规划加速方案，预计加速时间为：{round(burn_time,1)}s")

        vessel.control.sas = True
        time.sleep(1)
        vessel.control.sas_mode = conn.space_center.SASMode.maneuver
        time.sleep(1)
        conn.space_center.warp_to(next_node.time_to+conn.space_center.ut-burn_time/2-t_rotate*2)
        while True:
            if vessel.control.sas_mode == conn.space_center.SASMode.maneuver and vessel.control.sas == True:
                pass
            else:
                vessel.control.throttle = 0
                try:
                    vessel.control.sas = True
                    time.sleep(0.1)
                    vessel.control.sas_mode = conn.space_center.SASMode.maneuver
                    time.sleep(0.1)
                    vessel.control.throttle = 1
                except RuntimeError:
                    pass
            if next_node.time_to-burn_time/2 >=0:
                print(f"{round(next_node.time_to-burn_time/2,1)}s后开始加速     ", end='\r')
            elif next_node.time_to-burn_time/2 <0:
                print(f"剩余加速时间：{round(burn_time-(burn_time/2-next_node.time_to),1)}s      ", end='\r')
            if next_node.time_to > burn_time/2:
                    vessel.control.throttle = 0
            elif -burn_time/2<= next_node.time_to <= burn_time/2:
                vessel.control.throttle = 1*rate
            else:
                if IaI(next_node.remaining_burn_vector())>0.1:
                    print("速度存在误差，需要修正     ")
                    num = 0
                    error = 0
                    while True:
                        if vessel.control.sas_mode == conn.space_center.SASMode.maneuver and vessel.control.sas == True:
                            pass
                        else:
                            try:
                                vessel.control.sas = True
                                time.sleep(0.1)
                                vessel.control.sas_mode = conn.space_center.SASMode.maneuver
                                time.sleep(0.1)
                            except RuntimeError:
                                pass
                        print(f"速度误差：{round(IaI(next_node.remaining_burn_vector()),2)}m/s      ",end ="\r")
                        if IaI(next_node.remaining_burn_vector())>=10:
                            vessel.control.throttle = 1
                            sleep_rate_B = 0.1
                        elif 10>IaI(next_node.remaining_burn_vector())>=0.1:
                            vessel.control.throttle = 0.5
                            sleep_rate_B = 0.001
                        else:
                            vessel.control.throttle = 0.1
                            sleep_rate_B = 0.0001
                        if abs(error -IaI(next_node.remaining_burn_vector()))<=0.005:
                            num +=1
                        error = IaI(next_node.remaining_burn_vector())
                        if IaI(next_node.remaining_burn_vector())<=0.1:
                            print("误差修正完毕，可以结束加速     ")
                            break
                        elif num>=400:
                            if IaI(next_node.remaining_burn_vector())<=1:
                                print("误差修正完毕，可以结束加速     ")
                                break
                            else:
                                print("存在不可修正误差,请注意修正     ")
                                break
                        time.sleep(sleep_rate_B)
                
                vessel.control.throttle = 0
                next_node.remove()
                print("加速结束，进入预定轨道     ")
                time.sleep(0.1)
                vessel.control.sas = False
                break
            if round(burn_time-(burn_time/2-next_node.time_to),1) <=5:
                sleep_rate = 0.0001
            else:
                sleep_rate = 0.1
            time.sleep(sleep_rate)

def revise_intersect(conn,target_name,vessel_name = None):
    time.sleep(0.1)
    vessel = set_vessel(conn,vessel_name)
    ref = reference(conn)
    target_vessel = set_target_vessel(conn,target_name)

    if True:#计算飞行器最大旋转速度
        pitch_0 = vessel.flight().pitch
        heading = vessel.flight().heading
        vessel.control.sas = False
        vessel.auto_pilot.engage()
        while True:
            time.sleep(0.1)
            vessel.auto_pilot.target_pitch_and_heading(pitch_0, heading)
            if abs(vessel.flight().pitch - pitch_0) <=0.1:
                break
        for i in range(50):
            vessel.auto_pilot.target_pitch_and_heading(pitch_0-180, heading)
            delta_rotate = IaI(vessel.angular_velocity(ref))
            time.sleep(0.1)
        t_rotate = math.pi/delta_rotate
        vessel.auto_pilot.disengage()

    vessel.control.sas = False
    vessel.auto_pilot.engage()
    vessel.auto_pilot.reference_frame = ref
    vessel.control.rcs = True

    r_A = vessel.position(ref)
    r_B = target_vessel.position(ref)
    dr = bfminus(r_B,r_A)
    v_A = vessel.velocity(ref)
    v_B = target_vessel.velocity(ref)
    dv = bfminus(v_B,v_A)
    print(f"\r准备进行交汇修正 \n相对距离：{round(IaI(dr),2)}m\n相对速度：{round(IaI(dv),2)}m/s\n ")

    while True:
        time.sleep(0.1)
        r_A = vessel.position(ref)
        r_B = target_vessel.position(ref)
        dr = bfminus(r_B,r_A)
        if IaI(dr)<= 500:
            k = 0
        else:
            k = 100*(2/(1+math.exp(-(IaI(dr)/2500)**3))-1)
        vc = bftime(k,Ae_n(dr))
        v_A = vessel.velocity(ref)
        v_B = target_vessel.velocity(ref)
        dv = bfminus(v_B,v_A)
        while True:
            if t_rotate*IaI(dv) <= IaI(dr):
                if A_B(Ae_n(bfplus(vc,dv)),dr)<=0:
                    vessel.auto_pilot.target_direction = bftime(-1,Ae_n(bfplus(vc,dv)))
                else:
                    vessel.auto_pilot.target_direction = Ae_n(bfplus(vc,dv))
            else:
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
        r_B = target_vessel.position(ref)
        dr = bfminus(r_B,r_A)
        v_A = vessel.velocity(ref)
        v_B = target_vessel.velocity(ref)
        dv = bfminus(v_B,v_A)
        sys.stdout.write("\033[4A")
        sys.stdout.write("\033[J")
        print(f"\r正在进行交汇修正 \n相对距离：{round(IaI(dr),2)}m\n相对速度：{round(IaI(dv),2)}m/s\n ")

        if IaI(dr) <=1000 and IaI(dv) <= 5:
            sys.stdout.write("\033[4A")
            sys.stdout.write("\033[J")
            print("交汇修正完毕，可进入停泊点")
            break

#########################对接








def transfer_fuelless(conn,target_name,vessel_name = None):
    vessel = set_vessel(conn)
    info = node_transfer_fuelless(conn,target_name,vessel_name)
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
        vessel.auto_pilot.disengage()
        vessel.control.sas = True
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

def transfer_timeless(conn,target_name,vessel_name = None):
    vessel = set_vessel(conn)
    info = node_transfer_timeless(conn,target_name,vessel_name)
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
        vessel.auto_pilot.disengage()
        vessel.control.sas = True
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

def correct_interaction(conn,target_name,vessel_name = None):
    time.sleep(0.1)
    vessel = set_vessel(conn,vessel_name)
    ref = reference(conn)
    target_vessel = set_target_vessel(conn,target_name)
    print("开始交汇修正")
    vessel.control.sas = False
    vessel.auto_pilot.engage()
    vessel.auto_pilot.reference_frame = ref
    vessel.control.rcs = True

    while True:
        time.sleep(0.1)
        r_A = vessel.position(ref)
        r_B = target_vessel.position(ref)
        dr = bfminus(r_B,r_A)
        if IaI(dr)<= 500:
            k = 0
        else:
            k = 100*(2/(1+math.exp(-(IaI(dr)/2500)**3))-1)
        vc = bftime(k,Ae_n(dr))
        v_A = vessel.velocity(ref)
        v_B = target_vessel.velocity(ref)
        dv = bfminus(v_B,v_A)
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
        r_B = target_vessel.position(ref)
        dr = bfminus(r_B,r_A)
        v_A = vessel.velocity(ref)
        v_B = target_vessel.velocity(ref)
        dv = bfminus(v_B,v_A)
        if IaI(dr) <=1000 and IaI(dv) <= 5:
            break

    print("进入外泊点")
    record =0
    while True:
        r_A = vessel.position(ref)
        r_B = target_vessel.position(ref)
        dr = bfminus(r_B,r_A)
        v_A = vessel.velocity(ref)
        v_B = target_vessel.velocity(ref)
        dv = bfminus(v_B,v_A)
        if IaI(dr) <=200:
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
        if IaI(bfminus(target_vessel.velocity(ref),vessel.velocity(ref))) <= 0.5 and IaI(dr)<=205:
            record +=1
        if record >= 10:
            print("外泊点进入")
            vessel.auto_pilot.disengage()
            vessel.control.rcs = False
            break

def docking(conn,target_name,tag1=None,tag2=None,vessel_name=None,num1=50,num2=60,num3=50,num4=40):
    #num1为最终对接开始的距离
    # num2为准备对接泊点的距离
    # num3为如果航线危险，沿中点向外拓展的距离
    # num4为安全距离
    time.sleep(0.1)
    vessel = set_vessel(conn,vessel_name)
    ref = reference(conn)
    target_vessel = set_target_vessel(conn,target_name)
    print("进入内泊点")
    vessel.control.sas = False
    vessel.auto_pilot.engage()
    vessel.auto_pilot.reference_frame = ref
    vessel.control.rcs = True

    if tag1==None:
        docking_ports = target_vessel.parts.with_module('ModuleDockingNode')
        for part in docking_ports:
            if part.docking_port.state == conn.space_center.DockingPortState.ready:
                target_port = part.docking_port
    else:
        docking_ports = target_vessel.parts.with_tag(tag1)
        for part in docking_ports:
            if part.docking_port.state == conn.space_center.DockingPortState.ready:
                target_port = part.docking_port

    if tag2==None:
        docking_ports = vessel.parts.with_module('ModuleDockingNode')
        for part in docking_ports:
            if part.docking_port.state == conn.space_center.DockingPortState.ready:
                vessel_port = part.docking_port
                
    else:
        docking_ports = vessel.parts.with_tag(tag2)
        for part in docking_ports:
            if part.docking_port.state == conn.space_center.DockingPortState.ready:
                vessel_port = part.docking_port

    vessel.control.lights = False
    target_vessel.control.lights = False
    time.sleep(1)
    vessel.control.lights = True
    target_vessel.control.lights = True

    target_port_forward = conn.space_center.transform_direction((0, -1, 0), target_port.reference_frame, ref)
    if True:
        print("检测航线是否安全")
        distance=point_line_distance(vessel_port.position(ref),
                                    bfplus(target_port.position(ref),bftime(-num2,Ae_n(target_port_forward))),
                                    target_vessel.position(ref)
                                    )
        if distance >num4:
            print(f"安全，目标距离航线：{round(distance,2)}米，处于安全距离,可以进入下一泊点")
            Safe = True
        else :
            print(f"危险，目标距离航线：{round(distance,2)}米,有撞击风险")
            Safe = False

    if not Safe:
        print("修正航线")
        target_port_new = symmetry_point(vessel_port.position(ref),
                                        bfplus(target_port.position(ref),bftime(-num2,Ae_n(target_port_forward))),
                                        target_vessel.position(ref),
                                        num3,
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

    distance = num1
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
                vessel = set_vessel(conn)
                vessel.control.lights = True
                time.sleep(1)
                vessel.control.lights = False
                print("对接完成")
                break
        if record <=0:
            break

def launch_orbit_control(conn,basic_orientation,goal_apoapsis = 80000,
                            turn_on = True,
                            adjust_turn_on = False,
                            wait_to_node1 = False,
                            wait_to_node2 = False,
                            burn_time = 0,
                            burn_time_1 = 0
                            ):
    # 建议使用judgment = launch_orbit_control(conn,
    #                                     基础方向，
    #                                     需要的轨道高度,
    #                                     judgment[0],
    #                                     judgment[1],
    #                                     judgment[2],
    #                                     judgment[3],
    #                                     judgment[5],
    #                                     judgment[6])
    All_finsih = False
    vessel = set_vessel(conn)
    if turn_on:
        vessel.auto_pilot.engage()
        if vessel.orbit.apoapsis_altitude<80000*0.95:
            angle=90*(80000-vessel.orbit.apoapsis_altitude)/80000
        else:
            angle = 0
            vessel.control.throttle = 0
            turn_on = False
            adjust_turn_on = True
        vessel.auto_pilot.target_pitch_and_heading(angle, basic_orientation)

    if adjust_turn_on:
        Finish = False
        vessel.auto_pilot.target_pitch_and_heading(0, basic_orientation)
        if vessel.orbit.apoapsis_altitude <= 80000-100:
            vessel.control.throttle = 1
        elif 80000-100 <vessel.orbit.apoapsis_altitude <=80000-10:
             vessel.control.throttle = 0.1
        elif 80000-10 <vessel.orbit.apoapsis_altitude <=80000:
             vessel.control.throttle = 0.05
        else:
            vessel.control.throttle = 0
            Finish = True
        if vessel.flight().mean_altitude >= vessel.orbit.body.atmosphere_depth and Finish:
            adjust_turn_on = False
            mu = vessel.orbit.body.gravitational_parameter
            R = vessel.orbit.apoapsis
            a = vessel.orbit.semi_major_axis
            goal_apoapsis +=vessel.orbit.body.equatorial_radius
            v_0 = math.sqrt( mu*( (2/R) - (1/a) ) )
            v_1 = math.sqrt( mu*( (2/R) - (2/(R+goal_apoapsis)) ) )
            v_0_1 = math.sqrt( mu*( (2/goal_apoapsis) - (2/(R+goal_apoapsis)) ) )
            v_1_1 = math.sqrt( mu*( (1/goal_apoapsis) ) )
            delta_v = v_1-v_0
            delta_v_1 = v_1_1-v_0_1
            t = vessel.orbit.time_to_apoapsis+conn.space_center.ut
            node1 = vessel.control.add_node(t,
                                            prograde=delta_v
            )
            time.sleep(0.1)
            node2 = vessel.control.add_node(t+node1.orbit.period/2,
                                            prograde=delta_v_1
            )
            time.sleep(0.1)
            vessel.auto_pilot.disengage()
            vessel.control.sas = True
            time.sleep(0.1)
            

            burn_time = burning_time(conn,delta_v)
            if 1 < burn_time <=3:
                rate = 0.2
                burn_time = burning_time(conn,delta_v,None,rate)
            elif 0.5 < burn_time <=1:
                rate = 0.1
                burn_time = burning_time(conn,delta_v,None,rate)
            elif 0.25< burn_time <= 0.5:
                rate = 0.05
                burn_time = burning_time(conn,delta_v,None,rate)
            elif 0.1<= burn_time <= 0.25:
                rate = 0.02
                burn_time = burning_time(conn,delta_v,None,rate)
            elif burn_time <=0.1:
                rate = 0.005
                burn_time = burning_time(conn,delta_v,None,rate)
            else:
                rate = 1
            print(f"转移轨道燃烧时间为：{round(burn_time,2)}秒")
            burn_time = [burn_time,rate,v_1]

            burn_time_1 = burning_time(conn,delta_v_1)
            if 1 < burn_time_1 <=3:
                rate = 0.2
                burn_time_1 = burning_time(conn,delta_v_1,None,rate)
            elif 0.5 < burn_time_1 <=1:
                rate = 0.1
                burn_time_1 = burning_time(conn,delta_v_1,None,rate)
            elif 0.25< burn_time_1 <= 0.5:
                rate = 0.05
                burn_time_1 = burning_time(conn,delta_v_1,None,rate)
            elif 0.1<= burn_time_1 <= 0.25:
                rate = 0.02
                burn_time_1 = burning_time(conn,delta_v_1,None,rate)
            elif burn_time <=0.1:
                rate = 0.005
                burn_time_1 = burning_time(conn,delta_v_1,None,rate)
            else:
                rate = 1
            print(f"圆化轨道燃烧时间为：{round(burn_time_1,2)}秒")
            burn_time_1 = [burn_time_1,rate,v_1_1]

            next_node = vessel.control.nodes[0]
            conn.space_center.warp_to(next_node.time_to+conn.space_center.ut-burn_time[0]-20)
            vessel.control.sas_mode = conn.space_center.SASMode.maneuver
            wait_to_node1 = True

    if wait_to_node1:
        next_node = vessel.control.nodes[0]
        if burn_time[0] <=5:
            if next_node.time_to < burn_time[0]/2:
                vessel.control.throttle = burn_time[1]
                time.sleep(burn_time[0])
                vessel.control.throttle = 0
                next_node.remove()
                wait_to_node1 = False
                wait_to_node2 = True
                next_node = vessel.control.nodes[0]
                conn.space_center.warp_to(next_node.time_to+conn.space_center.ut-burn_time_1[0]-20)
                vessel.control.sas_mode = conn.space_center.SASMode.maneuver
        else:
            if next_node.time_to > burn_time[0]/2:
                vessel.control.throttle = 0
            elif -burn_time[0]/2<= next_node.time_to <= burn_time[0]/2:
                vessel.control.throttle = burn_time[1]
            else:
                vessel.control.throttle = 0
                next_node.remove()
                wait_to_node1 = False
                wait_to_node2 = True
                next_node = vessel.control.nodes[0]
                conn.space_center.warp_to(next_node.time_to+conn.space_center.ut-burn_time_1[0]-20)
                vessel.control.sas_mode = conn.space_center.SASMode.maneuver
        if abs(IaI(vessel.velocity(reference(conn))) - burn_time[2])<=0.1:
                vessel.control.throttle = 0
                next_node.remove()
                wait_to_node1 = False
                wait_to_node2 = True
                next_node = vessel.control.nodes[0]
                conn.space_center.warp_to(next_node.time_to+conn.space_center.ut-burn_time_1[0]-20)
                vessel.control.sas_mode = conn.space_center.SASMode.maneuver

    if wait_to_node2:
        next_node = vessel.control.nodes[0]
        if burn_time_1[0] <=5:
            if next_node.time_to < burn_time_1[0]/2:
                vessel.control.throttle = burn_time_1[1]
                time.sleep(burn_time_1[0])
                vessel.control.throttle = 0
                next_node.remove()
                wait_to_node2 = False
                All_finsih = True
                vessel.control.sas = False
        else:
            if next_node.time_to > burn_time_1[0]/2:
                vessel.control.throttle = 0
            elif -burn_time_1[0]/2<= next_node.time_to <= burn_time_1[0]/2:
                vessel.control.throttle = burn_time_1[1]
            else:
                vessel.control.throttle = 0
                next_node.remove()
                wait_to_node2 = False
                All_finsih = True
                vessel.control.sas = False
        if abs(IaI(vessel.velocity(reference(conn))) - burn_time_1[2])<=0.1:
            vessel.control.throttle = 0
            next_node.remove()
            wait_to_node2 = False
            All_finsih = True
            vessel.control.sas = False
    
    return turn_on,adjust_turn_on,wait_to_node1,wait_to_node2,All_finsih,burn_time,burn_time_1
    
def into_round_orbit(conn,goal_apoapsis,vessel_name = None):

    vessel = set_vessel(conn,vessel_name)
    mu = vessel.orbit.body.gravitational_parameter
    R = vessel.orbit.apoapsis
    a = vessel.orbit.semi_major_axis
    goal_apoapsis +=vessel.orbit.body.equatorial_radius
    v_0 = math.sqrt( mu*( (2/R) - (1/a) ) )
    v_1 = math.sqrt( mu*( (2/R) - (2/(R+goal_apoapsis)) ) )
    v_0_1 = math.sqrt( mu*( (2/goal_apoapsis) - (2/(R+goal_apoapsis)) ) )
    v_1_1 = math.sqrt( mu*( (1/goal_apoapsis) ) )
    delta_v = v_1-v_0
    delta_v_1 = v_1_1-v_0_1
    t = vessel.orbit.time_to_apoapsis+conn.space_center.ut
    if True:#清理节点
        nodes = vessel.control.nodes
        for node in nodes:
            node.remove()
    node1 = vessel.control.add_node(t,
                                    prograde=delta_v)
    node2 = vessel.control.add_node(t+node1.orbit.period/2,
                                prograde=delta_v_1)
    burn_time = burning_time(conn,delta_v)
    if 1 < burn_time <=3:
        rate = 0.2
        burn_time = burning_time(conn,delta_v,None,rate)
    elif 0.5 < burn_time <=1:
        rate = 0.1
        burn_time = burning_time(conn,delta_v,None,rate)
    elif 0.25< burn_time <= 0.5:
        rate = 0.05
        burn_time = burning_time(conn,delta_v,None,rate)
    elif 0.1<= burn_time <= 0.25:
        rate = 0.02
        burn_time = burning_time(conn,delta_v,None,rate)
    elif burn_time <=0.1:
        rate = 0.005
        burn_time = burning_time(conn,delta_v,None,rate)
    else:
        rate = 1
    print(f"转移轨道燃烧时间为：{round(burn_time,2)}秒")

    burn_time_1 = burning_time(conn,delta_v_1)
    if 1 < burn_time_1 <=3:
        rate_1 = 0.2
        burn_time_1 = burning_time(conn,delta_v_1,None,rate)
    elif 0.5 < burn_time_1 <=1:
        rate_1 = 0.1
        burn_time_1 = burning_time(conn,delta_v_1,None,rate)
    elif 0.25< burn_time_1 <= 0.5:
        rate_1 = 0.05
        burn_time_1 = burning_time(conn,delta_v_1,None,rate)
    elif 0.1<= burn_time_1 <= 0.25:
        rate_1 = 0.02
        burn_time_1 = burning_time(conn,delta_v_1,None,rate)
    elif burn_time_1 <=0.1:
        rate_1 = 0.005
        burn_time_1 = burning_time(conn,delta_v_1,None,rate)
    else:
        rate_1 = 1
    print(f"圆化轨道燃烧时间为：{round(burn_time_1,2)}秒")
    time.sleep(0.1)
    vessel.auto_pilot.disengage()
    vessel.control.sas = True
    time.sleep(0.1)

    next_node = vessel.control.nodes[0]
    conn.space_center.warp_to(next_node.time_to+conn.space_center.ut-burn_time-20)
    vessel.control.sas_mode = conn.space_center.SASMode.maneuver
    while True:
        time.sleep(0.01)
        if burn_time <=5:
            if next_node.time_to < burn_time/2:
                vessel.control.throttle = rate
                time.sleep(burn_time)
                vessel.control.throttle = 0
                next_node.remove()
                next_node = vessel.control.nodes[0]
                conn.space_center.warp_to(next_node.time_to+conn.space_center.ut-burn_time_1-20)
                vessel.control.sas_mode = conn.space_center.SASMode.maneuver
                break
        else:
            if next_node.time_to > burn_time/2:
                vessel.control.throttle = 0
            elif -burn_time/2<= next_node.time_to <= burn_time/2:
                vessel.control.throttle = rate
            else:
                vessel.control.throttle = 0
                next_node.remove()
                next_node = vessel.control.nodes[0]
                conn.space_center.warp_to(next_node.time_to+conn.space_center.ut-burn_time_1-20)
                vessel.control.sas_mode = conn.space_center.SASMode.maneuver
                break
    while True:
        time.sleep(0.01)
        if burn_time_1 <=5:
            if next_node.time_to < burn_time_1/2:
                vessel.control.throttle = rate_1
                time.sleep(burn_time_1)
                vessel.control.throttle = 0
                next_node.remove()
                vessel.control.sas = False
                break
        else:
            if next_node.time_to > burn_time_1/2:
                vessel.control.throttle = 0
            elif -burn_time_1/2<= next_node.time_to <= burn_time_1/2:
                vessel.control.throttle = rate_1
            else:
                vessel.control.throttle = 0
                next_node.remove()
                vessel.control.sas = False
                break

def separate(conn,rename=None):
    vessels_before = conn.space_center.vessels
    while True:
        time.sleep(0.1)
        vessels_after = conn.space_center.vessels
        new_vessels = [v for v in vessels_after if v not in vessels_before]
        if new_vessels == []:
            pass
        else:
            if rename == None:
                pass
            else:
                new_vessels[0].name = f"{rename}"
            break
    return new_vessels[0]

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    conn = krpc.connect(name="测试")
    vessel = set_vessel(conn)
    node = node_intersect_Dn_ap(conn,"Tianzhou")
    # tranfer(conn,node)
    # correct_interaction(conn,"station")
    # docking(conn,"station",None,None,"GDLV3_4")
    # vessel=separate(conn,"depart1")
    # # print(vessel.name)
    # # vessel = set_vessel(conn)
    # node1 = vessel.control.add_node(vessel.orbit.time_to_apoapsis+conn.space_center.ut,
    #                                 prograde=v_1-v_0
    # )
    # node1.delta_v
    # # node1.orbit.period
    # # vessel.auto_pilot.disengage()
    # # into_round_orbit(conn,300000)
    # # vessel=set
    # # vessel.orbit.body.gravitational_parameter
    

    # vessel = set_vessel(conn,name)
    # pe_position = vessel.orbit.position_at(conn.space_center.ut+vessel.orbit.time_to_periapsis,ref)