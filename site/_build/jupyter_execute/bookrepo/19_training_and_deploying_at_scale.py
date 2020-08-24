**Chapter 19 – Training and Deploying TensorFlow Models at Scale**

_This notebook contains all the sample code in chapter 19._

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/19_training_and_deploying_at_scale.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

# Setup
First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20 and TensorFlow ≥2.0.


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
    !echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" > /etc/apt/sources.list.d/tensorflow-serving.list
    !curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
    !apt update && apt-get install -y tensorflow-model-server
    !pip install -q -U tensorflow-serving-api
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deploy"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Deploying TensorFlow models to TensorFlow Serving (TFS)
We will use the REST API or the gRPC API.

## Save/Load a `SavedModel`

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., np.newaxis].astype(np.float32) / 255.
X_test = X_test[..., np.newaxis].astype(np.float32) / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_new = X_test[:3]

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28, 1]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-2),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

np.round(model.predict(X_new), 2)

model_version = "0001"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
model_path

!rm -rf {model_name}

tf.saved_model.save(model, model_path)

for root, dirs, files in os.walk(model_name):
    indent = '    ' * root.count(os.sep)
    print('{}{}/'.format(indent, os.path.basename(root)))
    for filename in files:
        print('{}{}'.format(indent + '    ', filename))

!saved_model_cli show --dir {model_path}

!saved_model_cli show --dir {model_path} --tag_set serve

!saved_model_cli show --dir {model_path} --tag_set serve \
                      --signature_def serving_default

!saved_model_cli show --dir {model_path} --all

Let's write the new instances to a `npy` file so we can pass them easily to our model:

np.save("my_mnist_tests.npy", X_new)

input_name = model.input_names[0]
input_name

And now let's use `saved_model_cli` to make predictions for the instances we just saved:

!saved_model_cli run --dir {model_path} --tag_set serve \
                     --signature_def serving_default    \
                     --inputs {input_name}=my_mnist_tests.npy

np.round([[1.1739199e-04, 1.1239604e-07, 6.0210604e-04, 2.0804715e-03, 2.5779348e-06,
           6.4079795e-05, 2.7411186e-08, 9.9669880e-01, 3.9654213e-05, 3.9471846e-04],
          [1.2294615e-03, 2.9207937e-05, 9.8599273e-01, 9.6755642e-03, 8.8930705e-08,
           2.9156188e-04, 1.5831805e-03, 1.1311053e-09, 1.1980456e-03, 1.1113169e-07],
          [6.4066830e-05, 9.6359509e-01, 9.0598064e-03, 2.9872139e-03, 5.9552520e-04,
           3.7478798e-03, 2.5074568e-03, 1.1462728e-02, 5.5553433e-03, 4.2495009e-04]], 2)

## TensorFlow Serving

Install [Docker](https://docs.docker.com/install/) if you don't have it already. Then run:

```bash
docker pull tensorflow/serving

export ML_PATH=$HOME/ml # or wherever this project is
docker run -it --rm -p 8500:8500 -p 8501:8501 \
   -v "$ML_PATH/my_mnist_model:/models/my_mnist_model" \
   -e MODEL_NAME=my_mnist_model \
   tensorflow/serving
```
Once you are finished using it, press Ctrl-C to shut down the server.

Alternatively, if `tensorflow_model_server` is installed (e.g., if you are running this notebook in Colab), then the following 3 cells will start the server:

os.environ["MODEL_DIR"] = os.path.split(os.path.abspath(model_path))[0]

%%bash --bg
nohup tensorflow_model_server \
     --rest_api_port=8501 \
     --model_name=my_mnist_model \
     --model_base_path="${MODEL_DIR}" >server.log 2>&1

!tail server.log

import json

input_data_json = json.dumps({
    "signature_name": "serving_default",
    "instances": X_new.tolist(),
})

repr(input_data_json)[:1500] + "..."

Now let's use TensorFlow Serving's REST API to make predictions:

import requests

SERVER_URL = 'http://localhost:8501/v1/models/my_mnist_model:predict'
response = requests.post(SERVER_URL, data=input_data_json)
response.raise_for_status() # raise an exception in case of error
response = response.json()

response.keys()

y_proba = np.array(response["predictions"])
y_proba.round(2)

### Using the gRPC API

from tensorflow_serving.apis.predict_pb2 import PredictRequest

request = PredictRequest()
request.model_spec.name = model_name
request.model_spec.signature_name = "serving_default"
input_name = model.input_names[0]
request.inputs[input_name].CopyFrom(tf.make_tensor_proto(X_new))

import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel('localhost:8500')
predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
response = predict_service.Predict(request, timeout=10.0)

response

Convert the response to a tensor:

output_name = model.output_names[0]
outputs_proto = response.outputs[output_name]
y_proba = tf.make_ndarray(outputs_proto)
y_proba.round(2)

Or to a NumPy array if your client does not include the TensorFlow library:

output_name = model.output_names[0]
outputs_proto = response.outputs[output_name]
shape = [dim.size for dim in outputs_proto.tensor_shape.dim]
y_proba = np.array(outputs_proto.float_val).reshape(shape)
y_proba.round(2)

## Deploying a new model version

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28, 1]),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-2),
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

model_version = "0002"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
model_path

tf.saved_model.save(model, model_path)

for root, dirs, files in os.walk(model_name):
    indent = '    ' * root.count(os.sep)
    print('{}{}/'.format(indent, os.path.basename(root)))
    for filename in files:
        print('{}{}'.format(indent + '    ', filename))

**Warning**: You may need to wait a minute before the new model is loaded by TensorFlow Serving.

import requests

SERVER_URL = 'http://localhost:8501/v1/models/my_mnist_model:predict'
            
response = requests.post(SERVER_URL, data=input_data_json)
response.raise_for_status()
response = response.json()

response.keys()

y_proba = np.array(response["predictions"])
y_proba.round(2)

# Deploy the model to Google Cloud AI Platform

Follow the instructions in the book to deploy the model to Google Cloud AI Platform, download the service account's private key and save it to the `my_service_account_private_key.json` in the project directory. Also, update the `project_id`:

project_id = "onyx-smoke-242003"

import googleapiclient.discovery

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "my_service_account_private_key.json"
model_id = "my_mnist_model"
model_path = "projects/{}/models/{}".format(project_id, model_id)
model_path += "/versions/v0001/" # if you want to run a specific version
ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

def predict(X):
    input_data_json = {"signature_name": "serving_default",
                       "instances": X.tolist()}
    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    if "error" in response:
        raise RuntimeError(response["error"])
    return np.array([pred[output_name] for pred in response["predictions"]])

Y_probas = predict(X_new)
np.round(Y_probas, 2)

# Using GPUs

tf.test.is_gpu_available()

tf.test.gpu_device_name()

tf.test.is_built_with_cuda()

from tensorflow.python.client.device_lib import list_local_devices

devices = list_local_devices()
devices

# Distributed Training

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

def create_model():
    return keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=7, activation="relu",
                            padding="same", input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                            padding="same"), 
        keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                            padding="same"),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=10, activation='softmax'),
    ])

batch_size = 100
model = create_model()
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-2),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10,
          validation_data=(X_valid, y_valid), batch_size=batch_size)

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

distribution = tf.distribute.MirroredStrategy()

# Change the default all-reduce algorithm:
#distribution = tf.distribute.MirroredStrategy(
#    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

# Specify the list of GPUs to use:
#distribution = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

# Use the central storage strategy instead:
#distribution = tf.distribute.experimental.CentralStorageStrategy()

#resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
#tf.tpu.experimental.initialize_tpu_system(resolver)
#distribution = tf.distribute.experimental.TPUStrategy(resolver)

with distribution.scope():
    model = create_model()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-2),
                  metrics=["accuracy"])

batch_size = 100 # must be divisible by the number of workers
model.fit(X_train, y_train, epochs=10,
          validation_data=(X_valid, y_valid), batch_size=batch_size)

model.predict(X_new)

Custom training loop:

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

K = keras.backend

distribution = tf.distribute.MirroredStrategy()

with distribution.scope():
    model = create_model()
    optimizer = keras.optimizers.SGD()

with distribution.scope():
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().batch(batch_size)
    input_iterator = distribution.make_dataset_iterator(dataset)
    
@tf.function
def train_step():
    def step_fn(inputs):
        X, y = inputs
        with tf.GradientTape() as tape:
            Y_proba = model(X)
            loss = K.sum(keras.losses.sparse_categorical_crossentropy(y, Y_proba)) / batch_size

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    per_replica_losses = distribution.experimental_run(step_fn, input_iterator)
    mean_loss = distribution.reduce(tf.distribute.ReduceOp.SUM,
                                    per_replica_losses, axis=None)
    return mean_loss

n_epochs = 10
with distribution.scope():
    input_iterator.initialize()
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        for iteration in range(len(X_train) // batch_size):
            print("\rLoss: {:.3f}".format(train_step().numpy()), end="")
        print()

batch_size = 100 # must be divisible by the number of workers
model.fit(X_train, y_train, epochs=10,
          validation_data=(X_valid, y_valid), batch_size=batch_size)

## Training across multiple servers

A TensorFlow cluster is a group of TensorFlow processes running in parallel, usually on different machines, and talking to each other to complete some work, for example training or executing a neural network. Each TF process in the cluster is called a "task" (or a "TF server"). It has an IP address, a port, and a type (also called its role or its job). The type can be `"worker"`, `"chief"`, `"ps"` (parameter server) or `"evaluator"`:
* Each **worker** performs computations, usually on a machine with one or more GPUs.
* The **chief** performs computations as well, but it also handles extra work such as writing TensorBoard logs or saving checkpoints. There is a single chief in a cluster. If no chief is specified, then the first worker is the chief.
* A **parameter server** (ps) only keeps track of variable values, it is usually on a CPU-only machine.
* The **evaluator** obviously takes care of evaluation. There is usually a single evaluator in a cluster.

The set of tasks that share the same type is often called a "job". For example, the "worker" job is the set of all workers.

To start a TensorFlow cluster, you must first specify it. This means defining all the tasks (IP address, TCP port, and type). For example, the following cluster specification defines a cluster with 3 tasks (2 workers and 1 parameter server). It's a dictionary with one key per job, and the values are lists of task addresses:

```
{
    "worker": ["my-worker0.example.com:9876", "my-worker1.example.com:9876"],
    "ps": ["my-ps0.example.com:9876"]
}
```

Every task in the cluster may communicate with every other task in the server, so make sure to configure your firewall to authorize all communications between these machines on these ports (it's usually simpler if you use the same port on every machine).

When a task is started, it needs to be told which one it is: its type and index (the task index is also called the task id). A common way to specify everything at once (both the cluster spec and the current task's type and id) is to set the `TF_CONFIG` environment variable before starting the program. It must be a JSON-encoded dictionary containing a cluster specification (under the `"cluster"` key), and the type and index of the task to start (under the `"task"` key). For example, the following `TF_CONFIG` environment variable defines a simple cluster with 2 workers and 1 parameter server, and specifies that the task to start is the first worker:

import os
import json

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["my-work0.example.com:9876", "my-work1.example.com:9876"],
        "ps": ["my-ps0.example.com:9876"]
    },
    "task": {"type": "worker", "index": 0}
})
print("TF_CONFIG='{}'".format(os.environ["TF_CONFIG"]))

Some platforms (e.g., Google Cloud ML Engine) automatically set this environment variable for you.

Then you would write a short Python script to start a task. The same script can be used on every machine, since it will load the `TF_CONFIG` variable, which will tell it which task to start:

import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
worker0 = tf.distribute.Server(resolver.cluster_spec(),
                               job_name=resolver.task_type,
                               task_index=resolver.task_id)

Another way to specify the cluster specification is directly in Python, rather than through an environment variable:

cluster_spec = tf.train.ClusterSpec({
    "worker": ["127.0.0.1:9901", "127.0.0.1:9902"],
    "ps": ["127.0.0.1:9903"]
})

You can then start a server simply by passing it the cluster spec and indicating its type and index. Let's start the two remaining tasks (remember that in general you would only start a single task per machine; we are starting 3 tasks on the localhost just for the purpose of this code example):

#worker1 = tf.distribute.Server(cluster_spec, job_name="worker", task_index=1)
ps0 = tf.distribute.Server(cluster_spec, job_name="ps", task_index=0)

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["127.0.0.1:9901", "127.0.0.1:9902"],
        "ps": ["127.0.0.1:9903"]
    },
    "task": {"type": "worker", "index": 1}
})
print(repr(os.environ["TF_CONFIG"]))

distribution = tf.distribute.experimental.MultiWorkerMirroredStrategy()

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["127.0.0.1:9901", "127.0.0.1:9902"],
        "ps": ["127.0.0.1:9903"]
    },
    "task": {"type": "worker", "index": 1}
})
#CUDA_VISIBLE_DEVICES=0 

with distribution.scope():
    model = create_model()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-2),
                  metrics=["accuracy"])

import tensorflow as tf
from tensorflow import keras
import numpy as np

# At the beginning of the program (restart the kernel before running this cell)
distribution = tf.distribute.experimental.MultiWorkerMirroredStrategy()

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., np.newaxis] / 255.
X_test = X_test[..., np.newaxis] / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_new = X_test[:3]

n_workers = 2
batch_size = 32 * n_workers
dataset = tf.data.Dataset.from_tensor_slices((X_train[..., np.newaxis], y_train)).repeat().batch(batch_size)
    
def create_model():
    return keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=7, activation="relu",
                            padding="same", input_shape=[28, 28, 1]),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                            padding="same"), 
        keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                            padding="same"),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=10, activation='softmax'),
    ])

with distribution.scope():
    model = create_model()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-2),
                  metrics=["accuracy"])

model.fit(dataset, steps_per_epoch=len(X_train)//batch_size, epochs=10)

# Hyperparameter tuning

# Only talk to ps server
config_proto = tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:%d' % tf_config['task']['index']])
config = tf.estimator.RunConfig(session_config=config_proto)
# default since 1.10

strategy.num_replicas_in_sync