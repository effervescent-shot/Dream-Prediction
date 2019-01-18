# Dream-Prediction
## Dream prediction by using neural networks on EEG time series collected from sleeping humans.
Usage of neural networks in neuroimaging area is becoming increasingly common due to their representation power. Especially ConvNets area particularly popular in the classification of Electroencephalogram (EEG) and Magnetic Resonance Imaging (fMRI). Herein, we worked on modelling and classification of patterns exist in EEG recordings of dreaming brain. We reduced it into image and video classification problem while preserving multimodel information of EEG. Our best spatiotemporal networks display 85\% accuracy in distinguishing signals of dreaming brain from not dreaming. We further studied to distinguish recalling ability of dreams. The same networks achieved 55\% accuracy. We showed that although more subtle, dreaming as other cognitive events can be detected via EEG classification with neural networks. 

## File Structure
### Notebooks
Under this folder, enumerated jupyter notebooks shows the development process of the project. They are left to give an idea about teh progress.
### Scripts
All models, helper functions and scripts for preprocessing are under this folder. To run the models one needs to run _tf|_modelRun.py_ script with required parameters. 
### Data 
Because of privacy concerns, real data is not uploaded but pseudo verisons can be found under this folder. Please note that this is just to show data format. Pseudo data is generated with random numbers.
### Results
Trained and saved models are under results file. Summary of model architectures can be found sample training outputs.
