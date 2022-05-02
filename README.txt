Multi Modal Emotion Classification: 

1) About the Dataset:
  - The dataset consists of 24 Actors and 120 videos per Actor
  - Out of the 2880 videos, only half of them have audio
  - The actors in the videos depict 8 emotions across them:
    • neutral
    • calm
    • happy
    • sad
    • angry
    • fearful
    • disgust
    • surprised
  - Each emotion except neutral has videos in two intensity, normal and strong
  - Each video has approximately 100 frames
  
2) Preprocessing Videos for Frames:
  - We extract 25 frames per video at equal interval
  - We use face detection from opencv to focus only on the face rather than the whole frame 

3) Preprocessing Video for Audios:
  - We extract audio from 1440 videos that contain audio and store them in wav format
  - We choose wav over mp3, as mp3 is compressed format and mp3 causes conflict with gpu while loading with torchaudio.load()
    
 4) Emotion Classification using Videos:
  - Resnet18 with RNN:
    • We use a Resnet18 backbone and pass the 25 frames through an LSTM for preserving spatiotemporal information
    • We use Cross Entropy Loss loss function and Adam optimizer
    • With batch size 4, we train for 15 epochs
  - 3D Resnet18:
    • We use 3D Resnet18 and pass the 25 frames stacked through this network, it has 2D spatial convolution and 1D temporal convolution, therefore preserving spatiotemporal information
    • We use Cross Entropy Loss loss function and Adam optimizer
    • With batch size 4, we train for 15 epochs

5) Emotion Classification using Audios:
  - 1D Convolution:
    • We load the waveform from the wav file and then resample it for a new frequency
    • We pass the dataset comprising of the padded waveforms through our custom 1D convolution network
    • We use Cross Entropy Loss loss function and Adam optimizer
    • With batch size 4, we train for 20 epochs
    
6) Inference for Emotion Classification using Videos:
  - Compared to the baseline technique from the reference paper, where the accuracy on validation set is 36.08%:
    • We achieved a validation accuracy of 47.7% for Resnet18 with RNN
    • We achieved a validation accuracy of 62.7% for 3D Resnet18

7) Inference for Emotion Classification using Audios:
  - Compared to the baseline technique from the reference paper for videos, where the accuracy on validation set is 36.08%, for our audio based network, we achieved a validation accuracy of 44.2% for 1D CNN
  
8) Miscellaneous:
  - The Confusion Matrix, and Precision, Recall and F1-Score are mentioned in the notebooks
    
