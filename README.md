# Open World Object Detection: A Review

Welcome to the code archive for our review paper: Open World Object Detection: A Review

## Abstract:

Exploring new knowledge is a fundamental human ability that can be mirrored in the development of deep neural networks, 
especially in the field of object detection. Open world object detection (OWOD) is an emerging area of research that 
adapts this principle to explore new knowledge. It focuses on recognizing and learning from objects absent from 
initial training sets, thereby incrementally expanding its knowledge base when new class labels are introduced.

<div align="center">
  <img src="images/OWOD_framework.png"/>
</div>

We conclude most existing Open World Object Detection (OWOD) methods in literature and archive their
codes in this repository covering essential aspects, including, benchmark datasets, source codes, 
evaluation results, and a taxonomy of existing methods.

# Taxonomy of OWOD methods

## Pseudo-labeling-based methods

Pseudo-labeling-based methods adopt the pseudo-labeling technique to select unknown objects during the training process.
They usually use a self-defined objectness score to measure whether the selected region contains an object or not. 
Object proposals with the top-k objectness scores and that do not match with known categories will be pseudo-labeled 
as unknown objects.

<div align="center">
    <img src="images/PL.png"/>
</div>

<a name="ORE"></a>

**Towards Open World Object Detection**

- Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Joseph_Towards_Open_World_Object_Detection_CVPR_2021_paper.html
- Venue: CVPR 2021
- Code: https://github.com/JosephKJ/OWOD

<a name="OW-DETR"></a>

**OW-DETR: Open-World Detection Transformer**

- Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Gupta_OW-DETR_Open-World_Detection_Transformer_CVPR_2022_paper.html
- Venue: CVPR 2022
- Code: https://github.com/akshitac8/OW-DETR

<a name="Fast-OWDETR"></a>

**Fast OWDETR: transformer for open world object detection**

- Paper: https://hdl.handle.net/10356/162462
- Code: https://github.com/luckychay/Fast-OWDETR

<a name="OpenWorldDETR"></a>

**Open World DETR: Transformer based Open World Object Detection**

- Paper: https://arxiv.org/abs/2212.02969

<a name="CAT"></a>

**CAT: LoCalization and IdentificAtion Cascade Detection Transformer for Open-World Object Detection**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Ma_CAT_LoCalization_and_IdentificAtion_Cascade_Detection_Transformer_for_Open-World_Object_CVPR_2023_paper.html
- Venue: CVPR 2023
- Code: https://github.com/xiaomabufei/CAT

## Class-agnostic methods

Class-agnostic methods consider known and unknown objects as the same foreground objects. By separating the detection
of objects and the identification of each instance, these methods use a class-agnostic object proposer to measure
the objectness of proposed regions. As the class-agnostic object proposer is trained to learn the objectness 
rather than the classifier, no bias from known categories is introduced.

<div align="center">
    <img src="images/CA.png"/>
</div>

<a name="2B-OCD"></a>

**Two-branch Objectness-centric Open World Detection**

- Paper: https://dl.acm.org/doi/abs/10.1145/3552458.3556453
- Venue: HCMA 2022

<a name="PROB"></a>

**PROB: Probabilistic Objectness for Open World Object Detection**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Zohar_PROB_Probabilistic_Objectness_for_Open_World_Object_Detection_CVPR_2023_paper.html
- Venue: CVPR 2023
- Code: https://github.com/orrzohar/PROB

<a name="OW-RCNN"></a>

**Addressing the Challenges of Open-World Object Detection**

- Paper: https://arxiv.org/abs/2303.14930

<a name="OLN"></a>

**Learning Open-World Object Proposals Without Learning to Classify**

- Paper: https://ieeexplore.ieee.org/abstract/document/9697381
- Venue: RA-L & ICRA 2022
- Code: https://github.com/mcahny/object_localization_network

<a name="RandBox"></a>

**Random Boxes Are Open-world Object Detectors**

- Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Random_Boxes_Are_Open-world_Object_Detectors_ICCV_2023_paper.html
- Venue: ICCV 2023
- Code: https://github.com/scuwyh2000/RandBox

## Metric-learning methods

Metric-learning OWOD methods generally treat the classification of unknown instances as a metic-learning process. 
By projecting the features of instances on an embedding feature space, a bunch of metric-learning techniques 
can be utilized to classify between known classes, unknown classes, and backgrounds. Most metric-learning methods
use a common strategy to extract potential unknown instances and focus on distinguishing between known, unknown, 
and backgrounds. Some methods even extend to separate different unknown classes without ground truth labels, 
which is closer to real open-world settings.

<div align="center">
    <img src="images/ML.png"/>
</div>

<a name="RE-OWOD"></a>

**Revisiting Open World Object Detection**

- Paper: https://ieeexplore.ieee.org/abstract/document/10288518
- Venue: TCSVT
- Code: https://github.com/RE-OWOD/RE-OWOD

<a name="OCPL"></a>

**Open-World Object Detection via Discriminative Class Prototype Learning**

- Paper: https://arxiv.org/abs/2302.11757
- Venue: ICIP 2022

<a name="UC-OWOD"></a>

**UC-OWOD: Unknown-Classified Open World Object Detection**

- Paper: https://link.springer.com/chapter/10.1007/978-3-031-20080-9_12
- Venue: ECCV 2022
- Code: https://github.com/JohnWuzh/UC-OWOD

## Other methods

Apart from what has been included, there are also other OWOD methods that cannot be classified into any of the 
categories above.

<a name="MAVL"></a>

**Class-agnostic Object Detection with Multi-modal Transformer**

- Paper: https://link.springer.com/chapter/10.1007/978-3-031-20080-9_30
- Venue: ECCV 2022
- Code: https://github.com/mmaaz60/mvits_for_class_agnostic_od

<a name="STUD"></a>

**Unknown-Aware Object Detection: Learning What You Don't Know from Videos in the Wild**

- Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Du_Unknown-Aware_Object_Detection_Learning_What_You_Dont_Know_From_Videos_CVPR_2022_paper.html
- Venue: CVPR 2022
- Code: https://github.com/deeplearning-wisc/stud

<a name="SKDF"></a>

**Detecting the open-world objects with the help of the Brain**

- Paper: https://arxiv.org/abs/2303.11623
- Code: https://github.com/xiaomabufei/DOWB

<a name="ORTH"></a>

**Exploring Orthogonality in Open World Object Detection (OrthogonalDet)**

- Paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_Exploring_Orthogonality_in_Open_World_Object_Detection_CVPR_2024_paper.pdf  
- Venue: CVPR 2024  
- Code: https://github.com/feifeiobama/OrthogonalDet  

<a name="ALLOW"></a>

**Annealing-based Label-Transfer Learning for Open World Object Detection (ALLOW)**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Ma_Annealing-Based_Label-Transfer_Learning_for_Open_World_Object_Detection_CVPR_2023_paper.pdf  
- Venue: CVPR 2023  
- Code: https://github.com/DIG-Beihang/ALLOW  

<a name="Hyp-OW"></a>

**Exploiting Hierarchical Structure Learning with Hyperbolic Distance Enhances Open World Object Detection (Hyp-OW)**

- Paper: https://arxiv.org/pdf/2306.14291  
- Venue: AAAI 2024  
- Code: https://github.com/boschresearch/Hyp-OW  

<a name="SGROD"></a>

**Recalling Unknowns Without Losing Precision: An Effective Solution to Large Model-Guided Open World Object Detection (SGROD)**

- Paper: https://ieeexplore.ieee.org/document/10684083  
- Venue: IEEE Transactions on Image Processing (TIP)  
- Code: https://github.com/harrylin-hyl/SGROD  

<a name="KTCN"></a>

**Enhancing Open-World Object Detection with Knowledge Transfer and Class-Awareness Neutralization (KTCN)**

- Paper: https://www.ijcai.org/proceedings/2024/0162.pdf  
- Venue: IJCAI 2024  
- Code: https://github.com/xxyzll/KTCN  

<a name="MEPU"></a>

**Unsupervised Recognition of Unknown Objects for Open-World Object Detection (MEPU)**

- Paper: https://ieeexplore.ieee.org/abstract/document/10978049  
- Venue: IEEE Transactions on Neural Networks and Learning Systems (TNNLS)  
- Code: https://github.com/frh23333/mepu-owod  

<a name="FMDL"></a>

**Enhancing Open-World Object Detection with Foundation Models and Dynamic Learning (FMDL)**

- Paper: https://www.sciencedirect.com/science/article/abs/pii/S0957417425006724  
- Venue: Expert Systems with Applications (ESA)  
- Code: -  

<a name="OW-OVD"></a>

**Unified Open World and Open Vocabulary Object Detection (OW-OVD)**

- Venue: CVPR 2025  
- Code: https://github.com/xxyzll/OW_OVD  

<a name="YOLO-UniOW"></a>

**Efficient Universal Open-World Object Detection (YOLO-UniOW)**

- Paper: https://arxiv.org/abs/2412.20645  
- Venue: arXiv (preprint)  
- Code: https://github.com/THU-MIG/YOLO-UniOW 

# Dataset splits & Results

In the task of open-world object detection, two datasets are commonly used in most existing methods, MS-COCO dataset 
and PASCAL VOC dataset. These datasets are divided into several splits based on two strategies.

First, in the original OWOD task, [ORE](#ORE) integrates the MS-COCO dataset with the PASCAL VOC dataset to provide more
samples called OWOD split. Specifically, all the classes and the corresponding samples are grouped into a set of 
non-overlapping tasks $\{T_1, \cdots, T_t\}$. Classes from the PASCAL VOC dataset are treated as task $T_1$. 
The other classes are grouped into tasks by semantic drifts.

<div align="center">
    <img src="images/OWOD_split.png"/>
</div>

Most existing state-ot-the-art methods use OWOD split as their evaluation protocol, the results are concluded below:

<table border="1" cellspacing="0" cellpadding="4"
       style="border-collapse: collapse; margin: auto; text-align: center;">
  <thead>
    <tr>
      <th style="text-align: center">Task IDs</th>
      <th colspan="2" style="text-align: center">Task 1</th>
      <th colspan="4" style="text-align: center">Task 2</th>
      <th colspan="4" style="text-align: center">Task 3</th>
      <th colspan="3" style="text-align: center">Task 4</th>
    </tr>
    <tr>
      <th rowspan="2" style="text-align: center">Methods</th>
      <th rowspan="2" style="text-align: center">U-Recall</th>
      <th style="text-align: center">mAP</th>
      <th rowspan="2" style="text-align: center">U-Recall</th>
      <th colspan="3" style="text-align: center">mAP</th>
      <th rowspan="2" style="text-align: center">U-Recall</th>
      <th colspan="3" style="text-align: center">mAP</th>
      <th colspan="3" style="text-align: center">mAP</th>
    </tr>
    <tr>
      <th style="text-align: center">Current<br/>Known</th>
      <th style="text-align: center">Previously<br/>Known</th>
      <th style="text-align: center">Current<br/>Known</th>
      <th style="text-align: center">Both</th>
      <th style="text-align: center">Previously<br/>Known</th>
      <th style="text-align: center">Current<br/>Known</th>
      <th style="text-align: center">Both</th>
      <th style="text-align: center">Previously<br/>Known</th>
      <th style="text-align: center">Current<br/>Known</th>
      <th style="text-align: center">Both</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="#ORE">ORE</a></td>
      <td>4.9</td>
      <td>56.0</td>
      <td>2.9</td>
      <td>52.7</td>
      <td>26.0</td>
      <td>39.4</td>
      <td>3.9</td>
      <td>38.2</td>
      <td>12.7</td>
      <td>29.7</td>
      <td>29.6</td>
      <td>12.4</td>
      <td>25.3</td>
    </tr>
    <tr>
      <td><a href="#UC-OWOD">UC-OWOD</a></td>
      <td>–</td>
      <td>50.7</td>
      <td>–</td>
      <td>33.1</td>
      <td>30.5</td>
      <td>31.8</td>
      <td>–</td>
      <td>28.8</td>
      <td>16.3</td>
      <td>24.6</td>
      <td>25.6</td>
      <td>12.9</td>
      <td>23.2</td>
    </tr>
    <tr>
      <td><a href="#OW-DETR">OW-DETR</a></td>
      <td>7.5</td>
      <td>59.2</td>
      <td>6.2</td>
      <td>53.6</td>
      <td>33.5</td>
      <td>42.9</td>
      <td>5.7</td>
      <td>38.3</td>
      <td>15.8</td>
      <td>30.8</td>
      <td>31.4</td>
      <td>17.1</td>
      <td>27.8</td>
    </tr>
    <tr>
      <td><a href="#Fast-OWDETR">Fast-OWDETR</a></td>
      <td>9.2</td>
      <td>56.6</td>
      <td>8.8</td>
      <td>51.3</td>
      <td>28.6</td>
      <td>39.4</td>
      <td>7.8</td>
      <td>39.2</td>
      <td>15.7</td>
      <td>32.2</td>
      <td>28.2</td>
      <td>11.4</td>
      <td>25.0</td>
    </tr>
    <tr>
      <td><a href="#OCPL">OCPL</a></td>
      <td>8.3</td>
      <td>56.6</td>
      <td>7.7</td>
      <td>50.7</td>
      <td>27.5</td>
      <td>39.1</td>
      <td>11.9</td>
      <td>38.6</td>
      <td>14.7</td>
      <td>30.7</td>
      <td>30.8</td>
      <td>14.4</td>
      <td>26.7</td>
    </tr>
    <tr>
      <td><a href="#RE-OWOD">RE-OWOD</a></td>
      <td>9.1</td>
      <td>59.7</td>
      <td>9.9</td>
      <td>54.1</td>
      <td>37.3</td>
      <td>45.6</td>
      <td>11.4</td>
      <td>43.1</td>
      <td>24.6</td>
      <td>37.6</td>
      <td>38.0</td>
      <td>28.7</td>
      <td>35.7</td>
    </tr>
    <tr>
      <td><a href="#RandBox">RandBox</a></td>
      <td>10.6</td>
      <td>61.8</td>
      <td>6.3</td>
      <td>–</td>
      <td>–</td>
      <td>45.3</td>
      <td>7.8</td>
      <td>–</td>
      <td>–</td>
      <td>39.4</td>
      <td>–</td>
      <td>–</td>
      <td>35.4</td>
    </tr>
    <tr>
      <td><a href="#2B-OCD">2B-OCD</a></td>
      <td>12.1</td>
      <td>56.4</td>
      <td>9.4</td>
      <td>51.6</td>
      <td>25.3</td>
      <td>38.5</td>
      <td>11.7</td>
      <td>37.2</td>
      <td>13.2</td>
      <td>29.2</td>
      <td>30.0</td>
      <td>13.3</td>
      <td>25.8</td>
    </tr>
    <tr>
      <td><a href="#ALLOW">ALLOW</a></td>
      <td>13.6</td>
      <td>59.3</td>
      <td>10.0</td>
      <td>53.2</td>
      <td>34.0</td>
      <td>45.6</td>
      <td>14.3</td>
      <td>42.6</td>
      <td>26.7</td>
      <td>38.0</td>
      <td>33.5</td>
      <td>21.8</td>
      <td>30.6</td>
    </tr>
    <tr>
      <td><a href="#PROB">PROB</a></td>
      <td>19.4</td>
      <td>59.5</td>
      <td>17.4</td>
      <td>55.7</td>
      <td>32.2</td>
      <td>44.0</td>
      <td>19.6</td>
      <td>43.0</td>
      <td>22.2</td>
      <td>36.0</td>
      <td>35.7</td>
      <td>18.9</td>
      <td>31.5</td>
    </tr>
    <tr>
      <td><a href="#OpenWorldDETR">Open World DETR</a></td>
      <td>21.0</td>
      <td>59.9</td>
      <td>15.7</td>
      <td>51.8</td>
      <td>36.4</td>
      <td>44.1</td>
      <td>17.4</td>
      <td>38.9</td>
      <td>24.7</td>
      <td>34.2</td>
      <td>32.0</td>
      <td>19.7</td>
      <td>29.0</td>
    </tr>
    <tr>
      <td><a href="#Hyp-OW">Hyp-OW</a></td>
      <td>23.5</td>
      <td>59.4</td>
      <td>20.6</td>
      <td>–</td>
      <td>–</td>
      <td>44.0</td>
      <td>26.3</td>
      <td>–</td>
      <td>–</td>
      <td>36.8</td>
      <td>–</td>
      <td>–</td>
      <td>33.6</td>
    </tr>
    <tr>
      <td><a href="#CAT">CAT</a></td>
      <td>23.7</td>
      <td>60.0</td>
      <td>19.1</td>
      <td>55.5</td>
      <td>32.7</td>
      <td>44.1</td>
      <td>24.4</td>
      <td>42.8</td>
      <td>18.7</td>
      <td>34.8</td>
      <td>34.4</td>
      <td>16.6</td>
      <td>29.9</td>
    </tr>
    <tr>
      <td><a href="#ORTH">ORTH</a></td>
      <td>24.6</td>
      <td>61.3</td>
      <td>26.3</td>
      <td>55.5</td>
      <td>38.5</td>
      <td>47.0</td>
      <td>29.1</td>
      <td>46.7</td>
      <td>30.6</td>
      <td>41.3</td>
      <td>42.4</td>
      <td>24.3</td>
      <td>37.9</td>
    </tr>
    <tr>
      <td><a href="#MEPU">MEPU-FS</a></td>
      <td>31.6</td>
      <td>60.2</td>
      <td>30.9</td>
      <td>57.3</td>
      <td>33.3</td>
      <td>44.8</td>
      <td>30.1</td>
      <td>42.6</td>
      <td>21.0</td>
      <td>35.4</td>
      <td>34.8</td>
      <td>19.1</td>
      <td>30.9</td>
    </tr>
    <tr>
      <td><a href="#SGROD">SGROD</a></td>
      <td>34.3</td>
      <td>59.8</td>
      <td>32.6</td>
      <td>56.0</td>
      <td>32.3</td>
      <td>44.9</td>
      <td>32.7</td>
      <td>42.8</td>
      <td>22.4</td>
      <td>36.0</td>
      <td>35.5</td>
      <td>18.5</td>
      <td>31.2</td>
    </tr>
    <tr>
      <td><a href="#OW-RCNN">OW-RCNN</a></td>
      <td>37.7</td>
      <td>63.0</td>
      <td>39.9</td>
      <td>48.8</td>
      <td>41.7</td>
      <td>45.2</td>
      <td>43.0</td>
      <td>45.2</td>
      <td>31.7</td>
      <td>40.7</td>
      <td>40.3</td>
      <td>28.8</td>
      <td>37.4</td>
    </tr>
    <tr>
      <td><a href="#SKDF">SKDF</a></td>
      <td>39.0</td>
      <td>56.8</td>
      <td>36.7</td>
      <td>52.3</td>
      <td>28.3</td>
      <td>40.3</td>
      <td>36.1</td>
      <td>36.9</td>
      <td>16.4</td>
      <td>30.1</td>
      <td>31.0</td>
      <td>14.7</td>
      <td>26.9</td>
    </tr>
    <tr>
      <td><a href="#KTCN">KTCN</a></td>
      <td>41.5</td>
      <td>60.2</td>
      <td>38.6</td>
      <td>55.8</td>
      <td>36.3</td>
      <td>46.0</td>
      <td>39.7</td>
      <td>43.5</td>
      <td>22.1</td>
      <td>36.4</td>
      <td>35.1</td>
      <td>16.2</td>
      <td>30.4</td>
    </tr>
    <tr>
      <td><a href="#FMDL">FMDL</a></td>
      <td>41.6</td>
      <td>62.3</td>
      <td>38.7</td>
      <td>59.2</td>
      <td>38.6</td>
      <td>47.3</td>
      <td>35.6</td>
      <td>48.1</td>
      <td>32.2</td>
      <td>43.2</td>
      <td>44.5</td>
      <td>26.7</td>
      <td>38.3</td>
    </tr>
    <tr>
      <td><a href="#MAVL">MAVL</a></td>
      <td>50.1</td>
      <td>64.0</td>
      <td>49.5</td>
      <td>61.6</td>
      <td>30.8</td>
      <td>46.2</td>
      <td>50.9</td>
      <td>43.8</td>
      <td>22.7</td>
      <td>36.8</td>
      <td>36.2</td>
      <td>20.6</td>
      <td>32.3</td>
    </tr>
    <tr>
      <td><a href="#OW-OVD">OW-OVD</a></td>
      <td>50.0</td>
      <td>69.4</td>
      <td>51.7</td>
      <td>69.5</td>
      <td>41.7</td>
      <td>55.6</td>
      <td>50.6</td>
      <td>55.5</td>
      <td>29.8</td>
      <td>47.0</td>
      <td>47.0</td>
      <td>25.2</td>
      <td>41.6</td>
    </tr>
    <tr>
      <td><a href="#YOLO-UniOW">YOLO-UniOW</a></td>
      <td>82.6</td>
      <td>73.6</td>
      <td>82.6</td>
      <td>73.4</td>
      <td>73.4</td>
      <td>48.4</td>
      <td>60.9</td>
      <td>81.5</td>
      <td>60.9</td>
      <td>39.0</td>
      <td>53.6</td>
      <td>32.0</td>
      <td>48.2</td>
    </tr>
  </tbody>
</table>

In the latest OWOD task, [OW-DETR](#OW-DETR) proposed a new strategy by splitting the categories across super-classes, 
called MS-COCO split. Specifically, object classes are grouped into the same tasks by semantic meanings. For example,
*trucks* and *vehicles* that belong to different tasks in the combined dataset are grouped into the same super-class 
task: *Animals, Person, Vehicles*.

<div align="center">
    <img src="images/MS-COCO_split.png"/>
</div>

Several methods also reported their evaluation results based on MS-COCO split. The results are shown below:

<table border="1" cellspacing="0" cellpadding="4"
       style="border-collapse: collapse; margin: auto; text-align: center;">
  <thead>
    <tr>
      <th style="text-align: center">Task IDs</th>
      <th colspan="2" style="text-align: center">Task 1</th>
      <th colspan="4" style="text-align: center">Task 2</th>
      <th colspan="4" style="text-align: center">Task 3</th>
      <th colspan="3" style="text-align: center">Task 4</th>
    </tr>
    <tr>
      <th rowspan="2" style="text-align: center">Methods</th>
      <th rowspan="2" style="text-align: center">U-Recall</th>
      <th style="text-align: center">mAP</th>
      <th rowspan="2" style="text-align: center">U-Recall</th>
      <th colspan="3" style="text-align: center">mAP</th>
      <th rowspan="2" style="text-align: center">U-Recall</th>
      <th colspan="3" style="text-align: center">mAP</th>
      <th colspan="3" style="text-align: center">mAP</th>
    </tr>
    <tr>
      <th style="text-align: center">Current<br/>Known</th>
      <th style="text-align: center">Previously<br/>Known</th>
      <th style="text-align: center">Current<br/>Known</th>
      <th style="text-align: center">Both</th>
      <th style="text-align: center">Previously<br/>Known</th>
      <th style="text-align: center">Current<br/>Known</th>
      <th style="text-align: center">Both</th>
      <th style="text-align: center">Previously<br/>Known</th>
      <th style="text-align: center">Current<br/>Known</th>
      <th style="text-align: center">Both</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="#ORE">ORE</a></td>
      <td>1.5</td>
      <td>61.4</td>
      <td>3.9</td>
      <td>56.5</td>
      <td>26.1</td>
      <td>40.6</td>
      <td>3.6</td>
      <td>38.7</td>
      <td>23.7</td>
      <td>33.7</td>
      <td>33.6</td>
      <td>26.3</td>
      <td>31.8</td>
    </tr>
    <tr>
      <td><a href="#OW-DETR">OW-DETR</a></td>
      <td>5.7</td>
      <td>71.5</td>
      <td>6.2</td>
      <td>62.8</td>
      <td>27.5</td>
      <td>43.8</td>
      <td>6.9</td>
      <td>45.2</td>
      <td>24.9</td>
      <td>38.5</td>
      <td>38.2</td>
      <td>28.1</td>
      <td>33.1</td>
    </tr>
    <tr>
      <td><a href="#PROB">PROB</a></td>
      <td>19.4</td>
      <td>59.5</td>
      <td>17.4</td>
      <td>55.7</td>
      <td>32.2</td>
      <td>44.0</td>
      <td>19.6</td>
      <td>43.0</td>
      <td>22.2</td>
      <td>36.0</td>
      <td>35.7</td>
      <td>18.9</td>
      <td>31.5</td>
    </tr>
    <tr>
      <td><a href="#CAT">CAT</a></td>
      <td>24.0</td>
      <td>74.2</td>
      <td>23.0</td>
      <td>67.6</td>
      <td>35.5</td>
      <td>50.7</td>
      <td>24.6</td>
      <td>51.2</td>
      <td>32.6</td>
      <td>45.0</td>
      <td>45.4</td>
      <td>35.1</td>
      <td>42.8</td>
    </tr>
    <tr>
      <td><a href="#ORTH">ORTH</a></td>
      <td>24.6</td>
      <td>71.6</td>
      <td>27.9</td>
      <td>64.0</td>
      <td>39.9</td>
      <td>51.3</td>
      <td>31.9</td>
      <td>52.1</td>
      <td>42.2</td>
      <td>48.8</td>
      <td>48.7</td>
      <td>38.8</td>
      <td>46.2</td>
    </tr>
    <tr>
      <td><a href="#Hyp-OW">Hyp-OW</a></td>
      <td>23.9</td>
      <td>72.7</td>
      <td>23.3</td>
      <td>–</td>
      <td>–</td>
      <td>50.6</td>
      <td>25.4</td>
      <td>–</td>
      <td>–</td>
      <td>46.2</td>
      <td>–</td>
      <td>–</td>
      <td>44.8</td>
    </tr>
    <tr>
      <td><a href="#OW-RCNN">OW-RCNN</a></td>
      <td>23.9</td>
      <td>68.9</td>
      <td>33.3</td>
      <td>49.6</td>
      <td>36.7</td>
      <td>41.9</td>
      <td>40.8</td>
      <td>42.3</td>
      <td>30.8</td>
      <td>38.5</td>
      <td>39.4</td>
      <td>32.2</td>
      <td>37.7</td>
    </tr>
    <tr>
      <td><a href="#MEPU">MEPU-FS</a></td>
      <td>37.9</td>
      <td>74.3</td>
      <td>35.8</td>
      <td>68.0</td>
      <td>41.9</td>
      <td>54.3</td>
      <td>35.7</td>
      <td>50.2</td>
      <td>38.3</td>
      <td>46.2</td>
      <td>43.7</td>
      <td>33.7</td>
      <td>41.2</td>
    </tr>
    <tr>
      <td><a href="#SGROD">SGROD</a></td>
      <td>48.0</td>
      <td>73.2</td>
      <td>48.9</td>
      <td>64.7</td>
      <td>36.7</td>
      <td>50.0</td>
      <td>47.7</td>
      <td>47.4</td>
      <td>32.4</td>
      <td>42.4</td>
      <td>42.5</td>
      <td>32.6</td>
      <td>40.0</td>
    </tr>
    <tr>
      <td><a href="#SKDF">SKDF</a></td>
      <td>60.9</td>
      <td>69.4</td>
      <td>60.0</td>
      <td>63.8</td>
      <td>26.9</td>
      <td>44.4</td>
      <td>58.6</td>
      <td>46.2</td>
      <td>28.0</td>
      <td>40.1</td>
      <td>41.8</td>
      <td>29.6</td>
      <td>38.7</td>
    </tr>
    <tr>
      <td><a href="#OW-OVD">OW-OVD</a></td>
      <td>76.2</td>
      <td>78.6</td>
      <td>79.8</td>
      <td>78.5</td>
      <td>61.5</td>
      <td>69.6</td>
      <td>78.4</td>
      <td>69.6</td>
      <td>55.1</td>
      <td>64.7</td>
      <td>64.8</td>
      <td>56.3</td>
      <td>62.7</td>
    </tr>
    <tr>
      <td><a href="#YOLO-UniOW">YOLO-UniOW</a></td>
      <td>84.5</td>
      <td>74.4</td>
      <td>83.4</td>
      <td>74.4</td>
      <td>56.9</td>
      <td>65.2</td>
      <td>83.0</td>
      <td>65.2</td>
      <td>52.2</td>
      <td>61.0</td>
      <td>61.0</td>
      <td>52.7</td>
      <td>58.9</td>
    </tr>
  </tbody>
</table>
