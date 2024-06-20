# drowsiness_detection
this code can detect your eyes and mouth opening to detect yawning and drowsiness ,this method is not based on an AI model to make decision but rather it uses AI to just detect face landmarks :].


process:

  -retrieve frames from the video stream .
  -for each frame ,make some image preprocessing : convert the image to a graysacale image to boost the process after , equalization of the histogramme to enhance the contraste and apply a gaussian filter .
  -use Dlib utilities to get face landmarks .
  -calculte eyes aspect ratio (EAR) and mouth opening ratio(MOR) and compare them to the thresholds.
  -make a decision based on the comparison.
