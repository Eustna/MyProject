# 把pth模型转onnx类型
import os
from nets.siamese import Siamese
import torch

root_folder = os.path.dirname(
        os.path.abspath(__file__)
        )

def pth_to_onnx(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = path
    model = Siamese((105,105,3))
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()
    dummy_input = [torch.randn(1,3,105,105),torch.randn(1,3,105,105)]
    torch.onnx.export(
        model,dummy_input,os.path.join(root_folder,"logs","classify.onnx"),verbose=True,input_names=["x1","x2"],output_names=["output"],
    )
    print("转换成功")

pth_to_onnx(r"E:\work\python_code\verification_code_detection\Siamese-pytorch\logs\ep043-loss0.008-val_loss0.017.pth")