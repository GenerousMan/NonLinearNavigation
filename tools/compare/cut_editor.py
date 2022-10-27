import time
import progressbar
from model.cut_editor.train_model import load_model, ConcatDropoutNet

def load_train_model():
    net = ConcatDropoutNet()
    net = load_model(net, 2000)
    return net

def calc_accpet_of_vshots(vshots):
    pass

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # train()
    pass