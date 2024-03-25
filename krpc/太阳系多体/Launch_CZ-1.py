from Library_real import set,reference,get_PartList_tag,get_FuelAmount
import krpc
import time
import pyautogui

def reset(text):
    with open(r"D:\SteamLibrary\steamapps\common\Kerbal Space Program\Ships\Script\pilot.ks", 'w') as file:
        file.write(text)

def vessel_pilot(pitch = 90,heading = 90):
    text = f"LOCK STEERING TO HEADING({heading},{pitch}).\nWAIT UNTIL FALSE."
    reset(text)
    vessel.control.toggle_action_group(open_terminal)
    pyautogui.hotkey('ctrl', 'c')
    pyautogui.write('RUN pilot.')
    pyautogui.press('enter')
    vessel.control.toggle_action_group(close_terminal)

def pilot_engage():
    vessel.control.toggle_action_group(open_terminal)
    pyautogui.write('SWITCH TO 0.')
    pyautogui.press('enter')
    vessel.control.toggle_action_group(close_terminal)

def pilot_disengage():
    vessel.control.toggle_action_group(open_terminal)
    pyautogui.hotkey('ctrl', 'c')
    vessel.control.toggle_action_group(close_terminal)

conn = krpc.connect(name='长征一号发射')
vessel = set(conn)
ref = reference(conn)
mu = vessel.orbit.body.gravitational_parameter 
Time= conn.add_stream(getattr, conn.space_center, 'ut')
Altitude=conn.add_stream(getattr, vessel.flight(), 'mean_altitude')
Apoapsis=conn.add_stream(getattr, vessel.orbit, 'apoapsis_altitude')

stage1_fuel = get_PartList_tag(conn,"一级燃料")[0]
stage2_fuel = get_PartList_tag(conn,"二级燃料")[0]
stage3_fuel = get_PartList_tag(conn,"三级")[0]

pilot_1 = True
engine_1 = False
engine_2 = False
engine_3 = False
depart_1 = False
depart_2 = False
depart_3 = False
fairing_depart = False

heading = 60
angle = 90
goal_apoapsis = 160000

launch_action = 1
engine_1_open = 2
engine_1_close = 3
depart_1_action = 4
engine_2_open = 5
engine_2_close = 6
depart_2_action = 7
engine_3_open = 8
engine_3_close = 9
depart_3_action = 10
depart_fairing = 11
open_terminal = 13
close_terminal = 14

if True:
    vessel.control.throttle = 1
    for i in range(10):
        print(f"倒计时：{10-i}")
        time.sleep(1)
    print("点火")
    vessel.control.toggle_action_group(engine_1_open)

if True:
    pilot_engage()
    # vessel.control.toggle_action_group(open_terminal)
    # pyautogui.write('SWITCH TO 0.')
    # pyautogui.press('enter')
    # vessel.control.toggle_action_group(close_terminal)
time.sleep(1)

if True:
    vessel_pilot(angle,heading)
    # text = f"LOCK STEERING TO HEADING({heading},{angle}).\nWAIT UNTIL FALSE."
    # reset(text)
    # vessel.control.toggle_action_group(open_terminal)
    # pyautogui.write('RUN pilot.')
    # pyautogui.press('enter')
    # vessel.control.toggle_action_group(close_terminal)
time.sleep(2)
vessel.control.toggle_action_group(launch_action)

while True:
    time.sleep(0.1)
    if pilot_1 :
        if Altitude() <10000:
            angle_error = 1
        else:
            angle_error = 0.2
        if Apoapsis()<goal_apoapsis*0.95 or Apoapsis() > goal_apoapsis:
            angle=90*(goal_apoapsis-Apoapsis())/(goal_apoapsis)
        else:
            angle = 0
        if abs(vessel.flight().pitch - angle) >= angle_error:
            vessel_pilot(angle,heading)
            # text = f"LOCK STEERING TO HEADING({heading},{angle}).\nWAIT UNTIL FALSE."
            # reset(text)
            # vessel.control.toggle_action_group(open_terminal)
            # pyautogui.hotkey('ctrl', 'c')
            # pyautogui.write('RUN pilot.')
            # pyautogui.press('enter')
            # vessel.control.toggle_action_group(close_terminal)

    # if not fairing_depart:
    #     # if Altitude() > vessel.orbit.body.atmosphere_depth:
    #     if Altitude() > 70000:
            
    #         fairing_depart = True

    if not depart_1:
        if get_FuelAmount(stage1_fuel)[0][1] <=400 or get_FuelAmount(stage1_fuel)[1][1] <=400:
            vessel.control.toggle_action_group(engine_2_open)
            vessel.control.toggle_action_group(engine_1_close)
            vessel.control.toggle_action_group(depart_fairing)
            time.sleep(3)
            vessel.control.toggle_action_group(depart_1_action)
            vessel.control.toggle_action_group(depart_fairing)
            depart_1 = True
            
    if not depart_2:
        if get_FuelAmount(stage2_fuel)[0][1] <=400 or get_FuelAmount(stage2_fuel)[1][1] <=400 or get_FuelAmount(stage2_fuel)[2][1] <=400:
            if True:
                vessel_pilot(angle,heading)
                # text = f"LOCK STEERING TO HEADING({heading},0).\nWAIT UNTIL FALSE."
                # reset(text)
                # vessel.control.toggle_action_group(open_terminal)
                # pyautogui.hotkey('ctrl', 'c')
                # pyautogui.write('RUN pilot.')
                # pyautogui.press('enter')
                # vessel.control.toggle_action_group(close_terminal)

            time.sleep(3)
            vessel.control.toggle_action_group(engine_2_close)
            if True:
                pilot_1 = False
                pilot_disengage()
                # vessel.control.toggle_action_group(open_terminal)
                # pyautogui.hotkey('ctrl', 'c')
                # vessel.control.toggle_action_group(close_terminal)

            time.sleep(1)
            vessel.control.sas = True
            vessel.control.toggle_action_group(depart_2_action)
            time.sleep(0.2)
            vessel.control.toggle_action_group(engine_3_open)
            depart_2 = True

    if not depart_3:
        if get_FuelAmount(stage3_fuel)[0][1] <=200 :
            time.sleep(20)
            vessel.control.toggle_action_group(12)
            time.sleep(0.1)
            vessel.control.toggle_action_group(depart_3_action)
            break


