<center>[![AnalyticsDojo](https://raw.githubusercontent.com/rpi-techfundamentals/fall2018-materials/master/fig/final-logo.png)](http://rpi.analyticsdojo.com)
<h1>Ludwig</h1></center>
<center><h3><a href = 'http://rpi.analyticsdojo.com'>rpi.analyticsdojo.com</a></h3></center>



# Ludwig

Ludwig, a project of Uber, provides a new data type-based approach to deep learning model design that makes the tool suited for many different applications. Rather than building out the architecture, you just need to specify the data.

First let's install ludwig and grab some data. 

!pip install ludwig

!wget https://raw.githubusercontent.com/rpi-techfundamentals/fall2018-materials/master/input/train.csv && wget https://raw.githubusercontent.com/rpi-techfundamentals/fall2018-materials/master/input/test.csv
  



### Model Definition File

Here in order to describe the model, we need to create/download a model definition file. This is a simple file that describes the data.  


```
input_features:
    -
        name: text
        type: text
        level: word
        encoder: parallel_cnn

output_features:
    -
        name: class
        type: category
        
```

!wget https://raw.githubusercontent.com/rpi-techfundamentals/spring2019-materials/master/13-deep-learning3/model_definition.yaml

!cat model_definition.yaml

### Training the Model
We are good to now train the model. 

While previously we have always done splits and done all of our training in core python.  Here we are just going to call the ludwig command line tool. 

ludwig experiment \
  --data_csv reuters-allcats.csv \
  --model_definition_file model_definition.yaml

!ludwig experiment  --data_csv train.csv \
--model_definition_file model_definition.yaml


!ludwig visualize --visualization learning_curves --training_statistics results/experiment_run_0/training_statistics.json

!cd results/experiment_run_0/ &&ls

!ludwig predict  --data_csv train.csv --model_path results/experiment_run_0/model/

!ls results/experiment_run_0/model/
