import torch
import numpy as np
from torch.autograd import Variable
from modules.network import UNet
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load Pytorch UNet model
NUM_CHANNELS = 1
NUM_CLASSES = 2
state = "checkpoint/unet-2000-ex-none.pth"
model_pytorch = UNet(NUM_CLASSES, NUM_CHANNELS)
model_pytorch.load_state_dict(torch.load(state, map_location=torch.device('cpu')))

dummy_input = torch.from_numpy(np.random.uniform(0, 1, (1, 1, 240, 320))).float()
dummy_output = model_pytorch(dummy_input)

# Export to ONNX format
torch.onnx.export(model_pytorch, dummy_input, './models/model_simple.onnx')

# Load ONNX model and convert to TensorFlow format
model_onnx = onnx.load('./models/model_simple.onnx')
tf_rep = prepare(model_onnx)

# Export model as .pb file
tf_rep.export_graph('./models/model_simple.pb')


graph = tf.Graph()
sess = tf.InteractiveSession(graph = graph)

with tf.gfile.GFile('./models/model_simple.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

print('Check out the input placeholders:')
#nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
nodes = [n.name + '=>' + n.op for n in graph_def.node]
for node in nodes:
    print(node)
