# Dream-Prediction
This is a semester project from Machine Learning and Optimization Laboratory (MLO) @ EPFL conducted jointly with Center for Sleep Research and Inverstigation @ CHUV Hospital and L2F @ EPFL Innovation Park in fall 2018.

## Dream prediction by neural networks on EEG time series collected from sleeping humans.
Usage of neural networks in neuroimaging area is becoming increasingly common due to their representation power. Especially ConvNets are increasingly popular in the classification of Electroencephalogram (EEG) and Magnetic Resonance Imaging (fMRI). Herein, we worked on modelling and classification of the patterns thought exist in EEG recordings of dreaming brain. We reduced the problem into image and video classification while preserving multimodel information of EEG. Our best spatiotemporal network display 85\% accuracy in distinguishing signals of dreaming brain from not dreaming. We further studied the distinguishing of the recalling ability of dreams. The same networks achieved 55\% accuracy. We showed that although more subtle, dreaming as other cognitive events can be detected via EEG classification with neural networks. 

### File Structure
#### Notebooks
Under this folder, enumerated Jupyter notebooks show the development process of the project. They are left to give an idea of the progress.
#### Scripts
All models, helper functions and scripts for preprocessing are under this folder. To run the models one needs to run _tf\_modelRun.py_ script with required parameters. 
#### Data 
Because of privacy concerns, real data is not uploaded. However, the pseudo version is provided under this folder. Please note that this is to show the data format. Pseudo data is generated with random numbers.
#### Results
Trained and saved models are under the results file. Summary of the models and their architectures can be found in sample training outputs.
