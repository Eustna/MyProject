import krpc
import math

class Vector:
    def __init__(self, B_x):
        if len(B_x) != 3:
            raise ValueError("输入数组维度错误")
        self.x,self.z,self.y = tuple(B_x)
        self.tuple = B_x

    def __add__(self, other):#矢量加法
        if len(other) != 3:
            raise ValueError("变量类型有误")
        value1 = self.tuple
        value2 = other
        return Vector(
            (value1[0]+value2[0],
             value1[1]+value2[1],
             value1[2]+value2[2])
            )
    
    def __sub__(self, other):#矢量减法
        if len(other) != 3:
            raise ValueError("变量类型有误")
        value1 = self.tuple
        value2 = other
        return Vector(
            (value1[0]-value2[0],
             value1[1]-value2[1],
             value1[2]-value2[2])
             )

    def __mul__(self, other):#矢量数乘和点乘
        value1 = self.tuple
        if isinstance(other, Vector):
            value2 = other.tuple
        elif isinstance(other, (int,float)):
            value2 = (other,0)
        else:
            value2 = other
        if len(value2) == 2:
            return Vector(
                (value1[0] * other,
                 value1[1] * other,
                 value1[2] * other)
                 )
        elif len(value2) == 3:
            return value1[0]*value2[0]+value1[1]*value2[1]+value1[2]*value2[2]
        else:
            raise ValueError("乘法类型有误")

    def __truediv__(self, other):#矢量数除
        if not isinstance(other,(int,float)):
            raise ValueError("变量类型有误")
        value1 = self.tuple
        value = other
        return Vector(
            (value1[0]/value,
             value1[1]/value,
             value1[2]/value)
            )

    def dot(self,other):#矢量点乘
        if len(other) != 3:
            raise ValueError("变量类型有误")
        value1 = self.tuple
        value2 = other
        return value1[0]*value2[0]+value1[1]*value2[1]+value1[2]*value2[2]
            
    def cross(self, other):#右手系矢量叉乘
        value = [0.0, 0.0, 0.0]
        value1 = [self[0], self[1], self[2]]
        value2 = [other[0], other[1], other[2]]
        levi_civita = [[[int((i - j) * (j - k) * (k - i) / 2) 
                        for k in range(3)] 
                        for j in range(3)] 
                        for i in range(3)]
        if len(value1) == 3 and len(value2)==3:
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        value[i] += levi_civita[i][j][k] * value1[j] * value2[k]
        else:
            raise ValueError(f"数据输入为{value1}和{value2}，请检查数据")
        return Vector(
            (value[0],
             value[1],
             value[2])
             )

    def X(self, other):#左手系矢量叉乘
        value = [0.0, 0.0, 0.0]
        value1 = [self[0], self[2], self[1]]
        value2 = [other[0], other[2], other[1]]
        levi_civita = [[[int((i - j) * (j - k) * (k - i) / 2) 
                        for k in range(3)] 
                        for j in range(3)] 
                        for i in range(3)]
        if len(value1) == 3 and len(value2)==3:
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        value[i] += levi_civita[i][j][k] * value1[j] * value2[k]
        else:
            raise ValueError(f"数据输入为{value1}和{value2}，请检查数据")
        return Vector(
            (value[0],
             value[2],
             value[1])
            )
    
    def __abs__(self):#矢量模长
        value = self.tuple
        return (value[0]**2+value[1]**2+value[2]**2)**0.5

    def e_n(self):
        value = self.tuple
        l_value_l = (value[0]**2+value[1]**2+value[2]**2)**0.5
        if l_value_l ==0:
            return Vector((
                0,
                0,
                0
                ))
        else:
            return Vector((
                value[0]/l_value_l,
                value[1]/l_value_l,
                value[2]/l_value_l
                ))

    def rad(self,other):
        if len(other) != 3:
            raise ValueError("变量类型有误")
        value1 = self.tuple
        value2 = other
        rad = math.acos((value1[0]*value2[0]+value1[1]*value2[1]+value1[2]*value2[2])/(value1[0]**2+value1[1]**2+value1[2]**2)**0.5/(value2[0]**2+value2[1]**2+value2[2]**2)**0.5)
        return rad

    def __rmul__(self, other):
        return self.__mul__(other)
    def __str__(self):
        return f"({self.x},{self.z},{self.y})"
    def __getitem__(self, index):
        return self.tuple[index]
    def __len__(self):
        return len(self.tuple)

class Orbit_inf():
    def __init__(self,r_in,v_in,gravitational_parameter):
        ref = (
            (1,0,0),
            (0,0,1),
            (0,1,0)
        )
        B_x = Vector(ref[0])
        B_y = Vector(ref[1])
        B_z = Vector(ref[2])

        mu = gravitational_parameter
        self.mu = mu

        B_r = Vector(r_in)
        r = abs(B_r)
        self.B_r = B_r
        self.r = r

        B_v = Vector(v_in)
        v = abs(B_v)
        self.B_v = B_v
        self.r = r

        a = 1/(2/r-v**2/mu)
        T = 2*math.pi*(a**3/mu)**0.5
        self.a = a
        self.T = T

        B_e = 1/mu*((v**2-mu/r)*B_r-(B_r.dot(B_v))*B_v)
        e = abs(B_e)
        self.e = e

        B_h = B_r.X(B_v)
        h = abs(B_h)
        self.B_h = B_h
        self.h = h

        i = math.acos(B_z.dot(B_h)/h)
        self.i = i

        B_n = B_z.X(B_h)
        if B_n[0] == 0 and B_n[1] == 0 and B_n[2] == 0:
            B_n = B_x
        n = abs(B_n)

        if e == 1:
            p = h**2/mu
            self.p = p
        if e == 0:
            omega = 0
            if B_y.dot(B_n) < 0:
                Omega = 2*math.pi-math.acos(B_n.dot(B_x)/n)
            else:
                Omega = math.acos(B_n.dot(B_x)/n)
            self.Omega = Omega
            self.omega = omega
            self.B_ap = r*B_x
            self.B_pe = (-r)*B_x
            self.f = self.true_anomaly()
            
        else:
            if B_z.dot(B_e) < 0:
                omega = 2*math.pi-math.acos(B_n.dot(B_e)/n/e)
            else:
                omega = math.acos(B_n.dot(B_e)/n/e)
            
            if B_y.dot(B_n) < 0:
                Omega = 2*math.pi-math.acos(B_n.dot(B_x)/n)
            else:
                Omega = math.acos(B_n.dot(B_x)/n)

            self.Omega = Omega
            self.omega = omega
            self.B_ap = self.ap()
            self.B_pe = self.pe()
            self.f = self.true_anomaly()
 
    def ap(self):
        a =self.a
        e = self.e
        i = self.i
        omega = self.omega
        Omega = self.Omega
        r = a * (1 + e)
        x = -r * (math.cos(Omega) * math.cos(omega) - math.sin(Omega) * math.sin(omega) * math.cos(i))
        y = -r * (math.sin(Omega) * math.cos(omega) + math.cos(Omega) * math.sin(omega) * math.cos(i))
        z = -r * (math.sin(i) * math.sin(omega))
        return Vector((x,z,y))
        
    def pe(self):
        a =self.a
        e = self.e
        i = self.i
        omega = self.omega
        Omega = self.Omega
        r = a * (1 - e)
        x = r * (math.cos(Omega) * math.cos(omega) - math.sin(Omega) * math.sin(omega) * math.cos(i))
        y = r * (math.sin(Omega) * math.cos(omega) + math.cos(Omega) * math.sin(omega) * math.cos(i))
        z = r * (math.sin(i) * math.sin(omega))
        return Vector((x,z,y))
    
    def true_anomaly(self,B_r = None):
        if B_r ==None:
            B_r = self.B_r
        else:
            if len(B_r) != 3:
                raise ValueError("输入数组维度错误")
            B_r = Vector(B_r)
        if self.B_pe.X(self.B_r)*self.B_h >= 0:
            rad = self.B_pe.rad(B_r)
        else :
            rad = 2*math.pi - self.B_pe.rad(B_r)
        return rad
    
    def eccentricity_anomaly(self,f = None):
        if f ==None:
            f = self.f
        E=2*math.atan((((1-self.e)/(1+self.e))**0.5)*math.tan(f/2))
        return E

    def mean_anomaly(self,E = None):
        if E == None:
            E = self.eccentricity_anomaly()
        M=E-self.e*math.sin(E)
        if M<=0:
            M +=2*math.pi
        return M

    def TimeTo_pe(self,B_r = None):
        if B_r ==None:
            B_r = self.B_r
            M = self.mean_anomaly(self.eccentricity_anomaly(self.true_anomaly()))
        else:
            if len(B_r) != 3:
                raise ValueError("输入数组维度错误")
            B_r = Vector(B_r)
            M = self.mean_anomaly(self.eccentricity_anomaly(self.true_anomaly(B_r)))
        t = (2*math.pi-M)*((self.a**3/mu)**0.5)
        return t

    def TimeTo_ap(self,B_r = None):
        if B_r ==None:
            B_r = self.B_r
            t = self.TimeTo_pe()
        else:
            if len(B_r) != 3:
                raise ValueError("输入数组维度错误")
            B_r = Vector(B_r)
            t = self.TimeTo_pe(B_r)
        if t >= self.T/2:
            t -= self.T/2
        else:
            t += self.T/2
        return t
    
    def f1_TimeTo_f2(self,f1,f2):
        mu = self.mu
        a = self.a
        M1 = self.mean_anomaly(self.eccentricity_anomaly(f1))
        if M1<=0:
            M1 +=2*math.pi
        t1 = (2*math.pi-M1)*((a**3/mu)**0.5)
        M2 = self.mean_anomaly(self.eccentricity_anomaly(f2))
        if M2<=0:
            M2 +=2*math.pi
        t2 = (2*math.pi-M2)*((a**3/mu)**0.5)
        T = self.T
        t =T-(t2-t1) 
        if t >=T:
            t -=T
        return t
    
    def f_to_r(self,f):
        if not isinstance(f,(int,float)):
            raise ValueError("输入数据类型错误")
        a =self.a
        e = self.e
        i = self.i
        omega = self.omega
        Omega = self.Omega
        r_f = a * (1 - e**2) / (1 + e * math.cos(f))
        x_rf = r_f * (math.cos(Omega) * math.cos(omega + f) - math.sin(Omega) * math.sin(omega + f) * math.cos(i))
        y_rf = r_f * (math.sin(Omega) * math.cos(omega + f) + math.cos(Omega) * math.sin(omega + f) * math.cos(i))
        z_rf = r_f * math.sin(omega + f) * math.sin(i)
        B_r_f = (x_rf,z_rf,y_rf)
        return Vector(B_r_f)
    
    def f_to_v(self,f):
        if not isinstance(f,(int,float)):
            raise ValueError("输入数据类型错误")
        B_r_f = self.f_to_r(f)
        ref = (
            B_r_f.e_n(),#轴项
            self.B_h.e_n(),#法向
            self.B_h.e_n().X(B_r_f.e_n()).e_n()#顺向
        )
        v_r = self.mu/self.h*self.e*math.sin(f)
        v_f = self.mu/self.h*(1+self.e*math.cos(f))
        B_v_r = v_r*ref[0]
        B_v_f = v_f*ref[2]
        B_v = B_v_r + B_v_f
        return B_v

class Ref():
    def __init__(self,conn):
        self.conn = conn

    def Vessel(self,ref_name = None,name = None):
        conn = self.conn
        vessel = set(conn,name)
        if ref_name == None:
            ref_name = "base"
        ref_frame_dict = {
            "base" : vessel.reference_frame,
            "orb"  : vessel.orbital_reference_frame,
            "sur"  : vessel.surface_reference_frame,
            "sv"   : vessel.surface_velocity_reference_frame
        }
        if ref_name in ref_frame_dict:
            ref_frame = ref_frame_dict[ref_name]
        else:
            print(f"没有找到参考系{ref_name}")
            exit()
        self.B_x = conn.space_center.transform_direction((1,0,0),ref_frame,vessel.orbit.body.non_rotating_reference_frame)
        self.B_y = conn.space_center.transform_direction((0,1,0),ref_frame,vessel.orbit.body.non_rotating_reference_frame)
        self.B_z = conn.space_center.transform_direction((0,0,1),ref_frame,vessel.orbit.body.non_rotating_reference_frame)
        return self
    
    def Body(self,ref_name = None,name = None):
        conn = self.conn
        if name == None:
            body = set(conn).orbit.body
        else:
            body = self.conn.space_center.bodies[name]
        if ref_name == None:
            ref_name = "base"
        ref_frame_dict = {
            "base" : body.non_rotating_reference_frame,
            "body"  : body.reference_frame,
            "orb"  : body.orbital_reference_frame
        }
        if ref_name in ref_frame_dict:
            ref_frame = ref_frame_dict[ref_name]
        else:
            print(f"没有找到参考系{ref_name}")
            exit()
        self.B_x = conn.space_center.transform_direction((1,0,0),ref_frame,vessel.orbit.body.non_rotating_reference_frame)
        self.B_y = conn.space_center.transform_direction((0,1,0),ref_frame,vessel.orbit.body.non_rotating_reference_frame)
        self.B_z = conn.space_center.transform_direction((0,0,1),ref_frame,vessel.orbit.body.non_rotating_reference_frame)
        return self

    def Node(self,node,ref_name = None):
        conn = self.conn
        body = node.orbit.body
        if ref_name == None:
            ref_name = "base"
        ref_frame_dict = {
            "base" : node.reference_frame,
            "orb"  : node.orbital_reference_frame
        }
        if ref_name in ref_frame_dict:
            ref_frame = ref_frame_dict[ref_name]
        else:
            print(f"没有找到参考系{ref_name}")
            exit()
        self.B_x = conn.space_center.transform_direction((1,0,0),ref_frame,body.non_rotating_reference_frame)
        self.B_y = conn.space_center.transform_direction((0,1,0),ref_frame,body.non_rotating_reference_frame)
        self.B_z = conn.space_center.transform_direction((0,0,1),ref_frame,body.non_rotating_reference_frame)
        return self

class AutoPilot():
    def __init__(self,conn):
        self.conn = conn
        self.e_0_roll = 0
        self.int_roll = 0
        self.e_0_ph = 0
        self.int_ph = 0
        self.e_0_p = 0
        self.int_p = 0
        self.e_0_h = 0
        self.int_h = 0
        self.Heading = None
        self.Pitch = None
        self.Num1 = 0
    

    def renew(self):
        self.conn = conn
        self.e_0_roll = 0
        self.int_roll = 0
        self.Heading = None
        self.Pitch = None
        self.Num1 = 0

    def roll(self,target_roll,k=(1,0.00001,10)):
        vessel = set(self.conn)
        roll = vessel.flight().roll
        kp = k[0]
        ki = k[1]
        kd = k[2]
        if roll == 90 or roll == -90:
            output = 0
        else:
            if roll < 0 :
                roll += 360
            roll = roll*math.pi/180
            target_roll = target_roll*math.pi/180
            if target_roll -  roll > math.pi:
                target_roll -= 2*math.pi
            elif target_roll - roll < -math.pi:
                target_roll += 2*math.pi
            error = target_roll - roll
            self.int_roll += error
            de = error - self.e_0_roll
            if ki*self.int_roll <= 0.1:
                kii = ki*self.int_roll
            else:
                kii = ki*self.int_roll/abs(ki*self.int_roll)*0.1
                self.int_roll -= error
            output = kp*error + kii + kd*de
            self.e_0_roll = error
        vessel.control.roll = output
        return output
    
    def lock_roll(self,rad=0,k=(1,0.00001,10)):#里面rad用来设置每周期旋转角度
        vessel = set(self.conn)
        self.Num1 += rad
        if self.Num1 >=360:
            self.Num1 -=360
        elif self.Num1 <=-360:
            self.Num1 +=360
        if self.Heading == None:
            self.Heading = vessel.flight().heading
        num = vessel.flight().heading-self.Heading+self.Num1
        while num>=360 or num <=-360:
            if num >=360:
                num -=360
            elif num <=-360:
                num +=360
        self.roll(num,k)

    def pitch_heading(self,g_pitch,g_heading,k=(1,0.00001,10)):
        vessel = set(self.conn)
        kp = k[0]
        ki = k[1]
        kd = k[2]
        pitch = vessel.flight().pitch
        heading = vessel.flight().heading
        A = (math.sin(pitch*math.pi/180),math.cos(heading*math.pi/180),math.sin(heading*math.pi/180))
        B = (math.sin(g_pitch*math.pi/180),math.cos(g_heading*math.pi/180),math.sin(g_heading*math.pi/180))
        error = abs(Vector(A).rad(B))
        self.int_ph += error
        de = error - self.e_0_ph
        if ki*self.int_ph <= 0.1:
            kii = ki*self.int_ph
        else:
            kii = ki*self.int_ph/abs(ki*self.int_ph)*0.1
            self.int_ph -= error
        output = kp*error + kii + kd*de
        self.e_0_ph = error

        if pitch >= 0:
            R = 1-pitch/90
            position = (R*math.cos(heading*math.pi/180),R*math.sin(heading*math.pi/180),0)
            g_R = 1-g_pitch/90
            g_position = (g_R*math.cos(g_heading*math.pi/180),g_R*math.sin(g_heading*math.pi/180),0)
            depos = Vector(Vector(g_position)-Vector(position)).e_n()
            theta_h = heading + 90 -vessel.flight().roll
            theta_p = heading -vessel.flight().roll
            e_h = (math.cos(theta_h*math.pi/180),math.sin(theta_h*math.pi/180),0)
            e_p = (math.cos(theta_p*math.pi/180),math.sin(theta_p*math.pi/180),0)
            pitchout = -output*Vector(depos).dot(e_p)
            headingout = output*Vector(depos).dot(e_h)
        else:
            pitch = -pitch
            g_pitch = -g_pitch
            heading = 360 - heading
            g_heading = 360 - g_heading
            R = 1-pitch/90
            position = (R*math.cos(heading*math.pi/180),R*math.sin(heading*math.pi/180),0)
            g_R = 1-g_pitch/90
            g_position = (g_R*math.cos(g_heading*math.pi/180),g_R*math.sin(g_heading*math.pi/180),0)
            depos = Vector(Vector(g_position)-Vector(position)).e_n()
            theta_h = heading + 90 -vessel.flight().roll
            theta_p = heading -vessel.flight().roll
            e_h = (math.cos(theta_h*math.pi/180),math.sin(theta_h*math.pi/180),0)
            e_p = (math.cos(theta_p*math.pi/180),math.sin(theta_p*math.pi/180),0)
            pitchout = output*Vector(depos).dot(e_p)
            headingout = -output*Vector(depos).dot(e_h)

        vessel.control.pitch = pitchout
        vessel.control.yaw = headingout

        # print(error,pitchout,headingout)
        print(error,pitchout,headingout)
        # d = math.acos(math.sin((-g_pitch+90)*math.pi/180)*math.sin((-pitch+90)*math.pi/180)*math.cos(g_heading-heading)+math.cos((-g_pitch+90)*math.pi/180)*math.cos((-pitch+90)*math.pi/180))





        # if not pitch >= 0:
        #     pitch = -pitch
        #     g_pitch = -g_pitch
        # R = 1-pitch/90
        # position = (R*math.cos(heading*math.pi/180),R*math.sin(heading*math.pi/180))
        # # if g_pitch >=0:
        # g_R = 1-g_pitch/90
        # g_position = (g_R*math.cos(g_heading*math.pi/180),g_R*math.sin(g_heading*math.pi/180))
        # # if g_pitch <=0:
        # #      g_R = 1-g_pitch/90
        # dposition = (g_position[0]-position[0],g_position[1]-position[1])

        # # if pitch <0:
        # #     R = 1+pitch/90
        # #     position = (R*math.cos(heading*math.pi/180),R*math.sin(heading*math.pi/180))
        # # print(position)
        # theta_h = vessel.flight().heading + 90 -vessel.flight().roll
        # theta_p = vessel.flight().heading -vessel.flight().roll
        # e_h = (math.cos(theta_h*math.pi/180),math.sin(theta_h*math.pi/180))
        # e_p = (math.cos(theta_p*math.pi/180),math.sin(theta_p*math.pi/180))


        # e_pitch = dposition[0]*e_p[0]+dposition[1]*e_p[1]
        # e_heading = dposition[0]*e_h[0]+dposition[1]*e_h[1]

        # # e_pitch = dposition[0]*e_p[0]+dposition[1]*e_p[1]
        # # e_heading = dposition[0]*e_h[0]+dposition[1]*e_h[1]

        # # self.int_p += e_pitch
        # # de_p = e_heading - self.e_0_p
        # # if ki*self.int_p <= 0.1:
        # #     kii_p = ki*self.int_p
        # # else:
        # #     kii_p = ki*self.int_p/abs(ki*self.int_p)*0.1
        # #     self.int_p -= e_pitch
        # # output_p = kp*e_heading + kii_p + kd*de_p

        # # vessel.control.pitch = -output_p
        # # 2/(1+math.exp(output_p))-1
        # # 
        # # print(e_pitch,e_heading)

        # # self.int_h += e_heading
        # # de_h = e_heading - self.e_0_h
        # # if ki*self.int_h <= 0.1:
        # #     kii_h = ki*self.int_h
        # # else:
        # #     kii_h = ki*self.int_h/abs(ki*self.int_h)*0.1
        # #     self.int_h -= e_heading
        # # output_h = kp*e_heading + kii_h + kd*de_h

        # # vessel.control.pitch = -output_p
        # # vessel.control.yaw = output_h

        # # print(e_pitch,-output_p)

        # error = (dposition[0]**2+dposition[1]**2)**(1/2)
        # self.int_ph += error
        # de = error - self.e_0_ph
        # if ki*self.int_ph <= 0.1:
        #     kii = ki*self.int_ph
        # else:
        #     kii = ki*self.int_ph/abs(ki*self.int_ph)*0.1
        #     self.int_ph -= error
        # output = kp*error + kii + kd*de
        # self.e_0_ph = error
        # # vessel.control.pitch = -output*e_pitch/error
        # # vessel.control.yaw = output*e_heading/error
        # print(error)


    # def other_roll(self,target_roll):
    #     conn = self.conn
    #     vessel = set(conn)
    #     roll = Flight_inf(conn).roll_other()-math.pi
    #     if roll < 0 :
    #         roll += 2*math.pi
    #     target_roll = target_roll*math.pi/180
    #     if target_roll -  roll > math.pi:
    #         target_roll -= 2*math.pi
    #     elif target_roll - roll < -math.pi:
    #         target_roll += 2*math.pi
    #     error = target_roll - roll
    #     self.int_roll += error
    #     de = error - self.e_0_roll
    #     output = self.kp*error + self.ki*self.int_roll + self.kd*de
    #     self.e_0_roll = error
    #     vessel.control.roll = -output
    #     return output





class Flight_inf():
    def __init__(self,conn,name = None):
        self.conn = conn
        self.name = name
    
    # def roll_other(self):#滚转角定义有问题
    #     conn = self.conn
    #     name = self.name
    #     rf = Ref(conn).Vessel("base",name)
    #     srf = Ref(conn).Vessel("sur",name)
    #     dir_in_rf = Vector(rf.B_y)
    #     dir_in_srf= Vector(srf.B_x)
    #     delta_vec = dir_in_srf - dir_in_rf
    #     tg_in_re = Vector(rf.B_z) + delta_vec
    #     tg_in_srf= Vector(srf.B_y)
    #     rad = tg_in_srf.rad(tg_in_re) 
    #     if tg_in_srf.X(tg_in_re).dot(dir_in_srf) > 0:
    #         rad = 2*math.pi - rad
    #     return rad 
    
    def roll(self):
        conn = self.conn
        name = self.name
        rf = Ref(conn).Vessel("base",name)
        orf = Ref(conn).Vessel("orb",name)
        dir_in_rf = Vector(rf.B_y)
        dir_in_srf= Vector(orf.B_x)
        if dir_in_rf.rad(dir_in_srf) == 0 or dir_in_rf.rad(dir_in_srf) == math.pi:
            rad = Vector(rf.B_z).rad(Vector(orf.B_z))
            if Vector(orf.B_z).X(Vector(rf.B_z)).dot(dir_in_rf) < 0:
                rad = 2*math.pi - rad
        else:
            vessel = set(conn,name)

            A = conn.space_center.transform_direction((1,0,0),vessel.reference_frame,vessel.orbital_reference_frame)
            A = conn.space_center.transform_direction((A[0],0,A[2]),vessel.orbital_reference_frame,vessel.orbit.body.non_rotating_reference_frame)
            B = conn.space_center.transform_direction((1,0,0),vessel.orbital_reference_frame,vessel.orbit.body.non_rotating_reference_frame)
            rad = Vector(A).dot(Vector(B))
            if Vector(A).X(Vector(B)).dot(Vector(orf.B_y)) < 0 :
                rad = -rad
        return rad
    
    def pitch(self):
        conn = self.conn
        name = self.name
        rf = Ref(conn).Vessel("base",name)
        srf = Ref(conn).Vessel("sur",name)
        dir_in_rf = Vector(rf.B_y)
        dir_in_srf= Vector(srf.B_x)
        rad = math.pi/2-dir_in_rf.rad(dir_in_srf)
        return rad

    def heading(self):
        conn = self.conn
        name = self.name
        vessel = set(conn,name)
        dir_in_rf = conn.space_center.transform_direction((0,1,0),vessel.reference_frame,vessel.surface_reference_frame)
        dir_in_srf= conn.space_center.transform_direction((0,1,0),vessel.surface_reference_frame,vessel.orbit.body.non_rotating_reference_frame)
        shadow = (0,dir_in_rf[1],dir_in_rf[2])
        dir_in_rf = Vector(conn.space_center.transform_direction(shadow,vessel.surface_reference_frame,vessel.orbit.body.non_rotating_reference_frame))
        support = conn.space_center.transform_direction((1,0,0),vessel.surface_reference_frame,vessel.orbit.body.non_rotating_reference_frame)
        rad = dir_in_rf.rad(dir_in_srf)
        if dir_in_rf.X(dir_in_srf).dot(support) < 0:
            rad = 2*math.pi - rad
        return rad



def set(conn,name=None):
    try :
        body = conn.space_center.bodies[name]
        return body
    except Exception as e:
        if name == None:
            vessel = conn.space_center.active_vessel
        else:
            Found=False
            all_vessels = conn.space_center.vessels
            for target_vessel in all_vessels:
                if target_vessel.name == name:
                    vessel = target_vessel
                    Found = True
                    break
            if not Found :
                print("未找到目标，请检查输入名称是否在以下列表")
                for vesl in all_vessels:
                    print(vesl.name)
        return vessel

def set_target(conn,name):
    try :
        target_body = conn.space_center.bodies[name]
        conn.space_center.active_vessel.target = target_body
        return target_body
    except Exception as e:
        all_vessels = conn.space_center.vessels
        Found = False
        for target_vessel in all_vessels:
            if target_vessel.name == name:
                conn.space_center.target_vessel = target_vessel
                Found = True
                break
            if not Found :
                print("未找到目标，请检查输入名称是否在以下列表")
                for vesl in all_vessels:
                    print(vesl.name)
        return target_vessel

def get_PartList_tag(conn,tag,vessel_name=None):
    if vessel_name == None :
        vessel = conn.space_center.active_vessel
    else:
        vessel = set(conn,vessel_name)
    part_list = vessel.parts.with_tag(tag)
    if part_list == []:
        print("未找到部件，请检查标签")
    return part_list

def get_FuelAmount(part):
    Fuel_list = []
    for i in range(len(part.resources.names)):
        Fuel_list.append((part.resources.names[i],part.resources.amount(part.resources.names[i])))
    return tuple(Fuel_list)

def reference(conn,ref_name=None,name=None):
    try :
        body = conn.space_center.bodies[name]
        vessel = set(conn)
    except Exception as e:
        vessel = set(conn,name)
        body = vessel.orbit.body
    ref_frame_dict = {  "Celestial": body.non_rotating_reference_frame, 
                        "Orbital": vessel.orbital_reference_frame,
                        "Body" : vessel.orbit.body.reference_frame,
                        "Surface" :vessel.surface_reference_frame,
                        "Velocity":vessel.surface_velocity_reference_frame,
                        "Center_orbital": body.orbital_reference_frame,
                        "Center": vessel.reference_frame
                      }
    if ref_name in ref_frame_dict:
        ref_frame = ref_frame_dict[ref_name]
    elif ref_name == None:
        ref_frame= body.non_rotating_reference_frame
    else:
        print(f"没有找到参考系{ref_name},可支持的参考系名字有：\nCelestial\nOrbital\nBody\nSurface\nVelocity\nCenter_orbital")
        ref_frame = None
    return ref_frame


# def ddzd(a_f,e_f,i_f,Omega_f,omega_f,f_f,mu):
#     def orbit_r_v(a,e,i,Omega,omega,f):
#         r = a * (1 - e**2) / (1 + e * math.cos(f))
#         x = r * (math.cos(Omega) * math.cos(omega + f) - math.sin(Omega) * math.sin(omega + f) * math.cos(i))
#         y = r * (math.sin(Omega) * math.cos(omega + f) + math.cos(Omega) * math.sin(omega + f) * math.cos(i))
#         z = r * math.sin(omega + f) * math.sin(i)
#         b_r = Vector(x,y,z)
#         r = abs(b_r)
#         h = (r*(1+e*math.cos(f))*mu)**0.5
#         b_x = Vector((math.sin(i)*math.sin(Omega),
#                       (-1)*math.sin(i)*math.cos(Omega),
#                       math.cos(i)))
#         b_f = mu/h*e*math.sin(f)*b_r.e_n() + mu/h*(1+e*math.cos(f))*b_x.e_n().cross(b_r.e_n()).e_n()
#         ref = (
#             b_x,
#             b_r.e_n(),
#             b_r.e_n().cross(b_x).e_n()
#         )
#         return (b_r,b_f,ref)
#     inf = orbit_r_v(a_f,e_f,i_f,Omega_f,omega_f,f_f)
#     b_N = inf[2][0].cross((0,0,1)).e_n()
#     b_e3 = inf[2][2]
#     b_rf = inf[0]
#     b_vf = inf[1]
#     b_temp = (2*b_vf*(b_rf*b_N)-b_rf*(b_vf*b_N)-(b_rf*b_vf)*b_N)
#     yita = (
#         (-b_vf[0],)
#     )


if __name__ == "__main__":
    # from krpcLibrary import set_vessel
    import time
    conn = krpc.connect(name="测试")
    vessel = conn.space_center.active_vessel
    # vessel.auto_pilot.engage()
    # pilot = AutoPilot(conn,1,0.000001,10)
    pilot = AutoPilot(conn)
    ref = reference(conn)
    mu = vessel.orbit.body.gravitational_parameter
    Time= conn.add_stream(getattr, conn.space_center, 'ut')
    Altitude=conn.add_stream(getattr, vessel.flight(), 'mean_altitude')
    Apoapsis=conn.add_stream(getattr, vessel.orbit, 'apoapsis_altitude')
    goal_apoapsis = 160000

    vessel.control.throttle = 1
    vessel.control.toggle_action_group(2)
    time.sleep(3)
    vessel.control.toggle_action_group(1)

    
    while True:
        time.sleep(0.01)
        if Altitude() <10000:
            angle_error = 1
        else:
            angle_error = 0.2
        if Apoapsis()<goal_apoapsis*0.95 or Apoapsis() > goal_apoapsis:
            angle=90*(goal_apoapsis-Apoapsis())/(goal_apoapsis)
        else:
            angle = 0
        if abs(vessel.flight().pitch - angle) >= angle_error:
            pilot.pitch_heading(angle,vessel.flight().heading)
        # pilot.roll(0)
        # pilot.lock_roll()
        # print(vessel.flight().roll)
        # pilot.pitch_heading(-20,180)
        # print(vessel.flight().heading)
        # print((Flight_inf(conn).roll_other()-math.pi)*180/math.pi)
        # print(vessel.flight().roll)
    # while True:
    #     pass
    # roll = (0,0)
    # while True:
    #     vessel.control.pitch = 0.5
    # while True:
    #     time.sleep(0.001)
        # # error_set = pilot_target_roll(conn,vessel,0,error_set[0],error_set[1])
        # print(vessel.flight().roll)
        # print(Flight_inf(conn).heading()*180/math.pi)
        # print(Flight_inf(conn).roll()*180/math.pi,vessel.flight().roll)
        # roll = Pilot_roll(conn,45,roll[0],roll[1])
        # print(Flight_inf(conn).pitch()*180/math.pi-vessel.flight().pitch,Flight_inf(conn).heading()*180/math.pi-vessel.flight().heading)
        # print(Flight_inf(conn).roll()*180/math.pi,vessel.flight().roll)
        # A = Vector(conn.space_center.transform_direction((0,1,0),vessel.reference_frame,ref))
        # B = Vector(conn.space_center.transform_direction((1,0,0),vessel.surface_reference_frame,ref))
        # # print(A.rad(B)*180/math.pi)
        # d = B-A
        # C = Vector(conn.space_center.transform_direction((0,0,1),vessel.reference_frame,ref))
        # C = C+d
        # D = Vector(conn.space_center.transform_direction((0,1,0),vessel.surface_reference_frame,ref))
        # print(D.rad(C)*180/math.pi)
    # error_set = pilot_target_pitch_head_roll(conn,vessel,(0,0,0),(0,0,0),90,90)
    # while True:
    #     time.sleep(0.1)
    #     error_set = pilot_target_pitch_head_roll(conn,vessel,error_set[0],error_set[1],90,90)
    # vessel.auto_pilot.engage()
    # vessel.auto_pilot.target_direction = (0,1,0)
    # print(vessel.flight().roll)

    # while True:
    #     vessel.control.pitch = 1

    # if True:#计算阶段
    #     vessel.auto_pilot.engage()
    #     vessel.auto_pilot.target_pitch
    #     vessel.auto_pilot.target_heading
    #     vessel.auto_pilot.target_roll
    #     vessel.auto_pilot.stopping_time
    # pass

    # print(vessel.auto_pilot.stopping_time[2])
    # while True:
    #     time.sleep(0.1)
    #     vessel.auto_pilot.engage()
    #     vessel.auto_pilot.target_direction = (0,1,0)
    #     print(vessel.auto_pilot.pitch_error)
    #     vessel.auto_pilot.disengage()
    # vessel.auto_pilot.engage()
    # while True:
    #     time.sleep(0.1)

    #     vessel.auto_pilot.target_roll = 0
    #     vessel.auto_pilot.target_heading = 0
    #     vessel.auto_pilot.target_pitch = 0
