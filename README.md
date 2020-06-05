# NLP_Project
An attempt to train a model to understand the language of animals

The basic idea of the project was as follows:
  Record a video of a particular animal for a substantial period of time from its own point of view.
  Isolate the barks and the respective timestamps of the barks when they occurred.
  Use ML algorithms to understand patterns within these barks and find categories.
  Cross verify using the timestamps in the video.
  
Unfortunately though, I could not collect my desired dataset since it required the use of a camera like a GoPro which could be attached via a harness to the animal being observed.
Nevertheless, we were lucky enough to find a dataset of Youtube Videos with barks of various dogs here: https://research.google.com/audioset/ontology/dog_1.html

Hence, I continued to build the process so that for the least part, I can atleast understand how to process the audio files and pre process the data when I get my desired Dataset.

Therefore, this project showcases a model trained using barks from the source mentioned above (dirty dataset for our usecase) which can be used in the future with better equipements.

Reference to understand audio file visulaizations: https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
