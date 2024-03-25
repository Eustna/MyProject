import threading
import subprocess

# 定义一个函数，用于运行Python脚本
def run_script(script_name):
    subprocess.call(['python', script_name])

# 创建新线程
thread1 = threading.Thread(target=run_script, args=(r'E:\work\python_code\code\watch_pro\start.py',))
thread2 = threading.Thread(target=run_script, args=(r'E:\work\python_code\code\watch_pro\start2.py',))

# 开启线程
thread1.start()
thread2.start()

# 等待所有线程完成
thread1.join()
thread2.join()
