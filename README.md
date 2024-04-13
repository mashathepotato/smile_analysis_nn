#### Dental Smile Analysis using Neural Networks ####
Internet of Mirrors project for a digital smile analysis using neural networks.

### Running Instructions: ###
1. Run $pip install -r requirements.txt in your environment

2. To train the CNN, run cnn/sequentialmodel.py. This is the baseline for the model. The checkpoints will be saved in models/sequential

3. To run the smile capture code that captures a camera frame, crops it, and runs it through the saved model, run main.py.
   
5. To run everything else with the User Interface and smile capture, run interface.py.
