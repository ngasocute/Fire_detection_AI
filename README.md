# FireDetection
FireNet is an artificial intelligence project for real-time fire detection.
<br><br>
<img src="images/fire_net.jpg" />
<hr>
<b>FireNet</b> is a real-time fire detection project containing an annotated dataset, pre-trained models and inference codes, all created to ensure that machine learning systems can be trained
 to detect fires instantly and eliminate false alerts. This is part of <a href="https://deepquestai.com" >DeepQuest AI</a>'s to train machine learning systems to 
  perceive, understand and act accordingly in solving problems in any environment they are deployed. <br><br> 
  
  <br>

 We have also provided a [ImageAI](https://github.com/OlafenwaMoses/ImageAI) codebase to train a <b>YOLOv3</b> detection model on the images
  and perform detection in mages and videos using a pre-trained model (also using <b>YOLOv3</b>) provided in the release section of this repository.
  The python codebase is contained in the <b><a href="fire_net.py" >fire_net.py</a></b> file and the detection configuration JSON file for detection is also provided the 
  <b><a href="detection_config.json" >detection_config.json</a></b>.
<br>
Running the experiment or detection requires that you have **Tensorflow**, and **Keras**, **OpenCV** and **ImageAI** installed. You can install this dependencies via the commands below.

<br><span><b>- Tensorflow 1.4.0 (and later versions)  </b>      <a href="https://www.tensorflow.org/install/install_windows" style="text-decoration: none;" > Install</a></span> or install via pip <pre> pip3 install --upgrade tensorflow </pre> 
       
  <span><b>- OpenCV  </b>        <a href="https://pypi.python.org/pypi/opencv-python" style="text-decoration: none;" >Install</a></span> or install via pip <pre> pip3 install opencv-python </pre> 
       
   <span><b>- Keras 2.x  </b>     <a href="https://keras.io/#installation" style="text-decoration: none;" >Install</a></span> or install via pip <pre> pip3 install keras </pre> 
  
   <span><b>- ImageAI 2.0.3  </b>  
   <span>      <pre>pip3 install imageai --upgrade </pre></span> <br><br> <br>
  <span><b>- Run project:open file fire_net.py then type in terminal</b></span> <br>
 <span>   <pre> python fire_net.py </pre> </span><br> <br>

<br><br><br><br>
  <img src="images/1-detected.jpg" />


