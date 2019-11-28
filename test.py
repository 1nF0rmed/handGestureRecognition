from flask import Flask, flash, redirect, url_for, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import uuid
import os


UPLOAD_FOLDER = './tmp_files/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

state = "unknown"

_graph = None
_labels = None
model = None

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())

    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()

    for l in proto_as_ascii_lines:
        label.append(l.rstrip())

    return label

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=128, input_std=128):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                            name="png_reader")

    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name="jpeg_reader")

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    sess = tf.Session()
    result = sess.run(normalized)

    return result

def loadModel(frozen_graph_filename, label_filename):
    global model

    # Load the tensorflow frozen graph and label files
    graph = load_graph(frozen_graph_filename)
    labels = load_labels(label_filename)

    return graph,labels

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getPrediction(file_path):

    # Get the loaded graph and labels
    global _graph
    global _labels

    # Read tensor from image file
    t = read_tensor_from_image_file(file_path)

    input_name = "import/Mul"
    output_name = "import/final_result"
    input_operation = _graph.get_operation_by_name(input_name)
    output_operation = _graph.get_operation_by_name(output_name)

    # Start the tensorflow session to compute the graph
    with tf.Session(graph=_graph) as sess:
        results = sess.run(output_operation.outputs[0],
                            {input_operation.outputs[0]: t})

    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    print("TOP K: ")
    print(results.argsort())

    return str("{}".format(_labels[top_k[0]])), np.max(results)

@app.route('/api/v1/<name>')
def success(name):
    return "welcome %s" % name

def gen_random_filename(filename):
    # split extension and name
    b = filename.split(".")

    # Set name to a unique id
    b[0] = str(uuid.uuid4())

    # return the new filename
    return '.'.join(b)

@app.route('/api/v1/predictGesture', methods=['POST'])
def setState():
    global state
    filename = ""

    # Get the file from the POST Request
    if 'file' not in request.files:
        print 'No file part'
        return redirect(request.url)

    _file = request.files['file']
    if _file and allowed_file(_file.filename):
        filename = gen_random_filename(_file.filename)
        _file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # TODO Classify the image
    state,acc =  getPrediction(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # TODO Set the state to the label

    return "State:"+str(state)+":"+str(acc)

@app.route('/api/v1/getState', methods=['GET'])
def getState():
    global state

    # TODO get the current state of the plant

    # TODO return the state of the plant
    return state

if __name__ == "__main__":

    # Load the graph and labels
    global _graph
    global _labels

    graph_filename = "retrained_model.pb"
    graph_labels = "labels.txt"

    _graph,_labels = loadModel(graph_filename, graph_labels)

    # print getPrediction("./test4.jpg")

    # Load the flask api
    app.run(host="0.0.0.0", port=80)
