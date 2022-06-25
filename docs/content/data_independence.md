
# Data Independence 
foReal's declarative approach abstracts the physical data acquisition (vibration registered by a sensor and digitized) into a logical data access pattern (request signal from time $t_0$ to time $t_1$). Similarly, we want to abstract the physical data model (for example a file stored in a filesystem on hard disk) into an easy to understand logical data model (for example a table). This abstraction is usually referred to as data independence.
 
Focusing on data models is crucial in long-term monitoring projects since data from such projects is ever evolving. Over the course of a long-term measurement, changes in file formats or storage engines are likely. Moreover, changes in data types can occur when for example new sensors are installed. 

The foReal framework is designed to work with specific logical data models for data and annotations which will be described in the following.


## Data Cubes

Environmental data is inherently multi-dimensional. Examples include 

 - time-series (1-dimension: time)
 - seismic spectrograms (2-dimensions: time,frequency)
 - images (3-dimensions: x, y, color)
 - timelapse images (4-dimensions: time,x,y,color)
 - seismic spectrograms in global context (7-dimensions: latitude,longitude,elevation,station,channel,time,frequency)

The logical data model for these kind of data are typically multi-dimensional arrays or so-called data cubes. In environmental monitoring, often the data along a dimension is meaningless without additional information, such as timestamps for the time dimension, frequency steps for the frequency dimension, or coordinates for spatial data. Therefore, data cubes for environmental data require the option of additional one dimensional arrays (from here on called coordinates) describing the content along each dimensions. These so-called labeled multi-dimensional arrays are implemented efficiently in the [xarray](https://xarray.dev/) framework, which is used internally by foReal.

foReal's request-driven data access is grounded in data cubes. A request in  foReal slices out a segment of a multi-dimensional data cube using coordinates to select the requested data segment.

## Annotations

We want to achieve data independence for annotations, meaning instead of defining annotations by their representation in the physical data storage (name of the folder containing images of respective label) we want to abstract it into a logical data model (directive to connect data with a label). This requirement comes from the fact, that the we need to be able to restructure or resample the data without loosing the annotation information. For examples if the annotations are stored as folder names (as done for certain image datasets) they are attached to the storage structure (filesystem). When we change the storage system or just restructure the data by sorting it differently the annotations information might be lost. Moreover, annotation should be reusable and independent of data type. If we, for example, annotate wind in a time-series stream, we should in principle be able to apply this annotation to a co-located seismic sensor.

In foReal, annotation can be connected to the respective data by using a similar approach as used for *requests} in the previous section.  Like a request, annotation contain an *indexers} key defining the concrete slice of data to be annotated. In addition, it contains a *tags} key containing the annotation. We do not put harsh constraints on what a target is but usually it would be the class name of the annotated event. The annotation might additionally contain more information such as how or by whom the annotation was created or an event id to trace and track multiple annotations of the same event, for example to annotate the same mountaineer on multiple, consecutive timelapse images.
The annotation format is exemplified in the following figure using images and seismic data. The figure shows graphically which segment of the image or seismic stream is annotated. In addition, three annotations are given in JSON format.

![annotation example](assets/images/annotationexample.png)
```
{ "indexers": {
    "time": { 
      "start": "2016-08-04T11:44:17", 
      "stop": "2016-08-04T12:09:32"}, 
    "sensor": ["MH36", "MH38"]}, 
  "tags": {"mountaineer": true}}
...
{ "indexers": {  
    "time": ["2016-08-12:04:12"], 
    "x": { 
      "start": 95, 
      "stop": 145},
    "y":{ 
      "start": 20, 
      "stop": 70}, 
    "sensor": ["MHDSLR"]},
  "tags": {"mountaineer": true}}
...
{ "indexers": {
    "time": {
      "start": "2016-05-02", 
      "stop": "2016-09-15"}, 
    }, 
  "tags": {
    "clear of snow": {"annotator":"mountain-lodge keeper"}}}
...
```

---


<b>Sensor-specific label or sensor-independent annotation can be described using the same syntax. *indexers* describes the actual data slice to be annotated. *tags* contains the annotation.</b>


 foReal annotations can be used for two purposes. First, a given annotation can be used to load the corresponding data segment. Second, a given data segment can be annotated with the corresponding annotation(s). 

![annotation config example](assets/images/annotationconfig.svg)

---

<b>Usage of annotations in the context of foReal. The request's start and end time are highlighted as the red line below the signals. In **a)** an annotation is used as a request to load the corresponding seismic segment. In **b)** a request from a dataset is used to load a seismic segment and all corresponding annotations. The upper path in **b)**} demonstrates how a request (one red line), which can be regarded as a one-dimensional "bounding box", overlaps with the annotation's one-dimensional "bounding boxes" (three black lines).</b>

Part a) in the above figure depicts how an annotation can be used to load a data segment. Since each annotation contains the same information as a request, it can be used as-is to request the annotated data segment. Moreover, it can be used to retrieve correlated sensor values. For example, we can use the seismic annotation to request all correlated images by replacing the annotation's *sensor* field with the sensor name of the camera.

Annotating a given data segment with annotations is a common scenario, for example when multiple mountaineers are annotated on an image and we want to crop the image while retaining the annotations within the cropped are and discard the annotations which are outside. This scenario is more complex because we need to be able to find the annotations describing a data segment of arbitrary size. 

In our framework, the data segment can be defined by a request. A request can be regarded as bounding box. For the case of a one-dimensional time series, the "bounding box" would consist of start and end time. For the case of a cropped image, the bounding box would consist of the cropping area (start and end values on both, horizontal and vertical axis). In general, a request can be regarded as a multi-dimensional axis-aligned bounding box, which means it consist of a start and end value for each dimension of the array. Similarly, an annotation can be seen as such a bounding box. It describes what (event, object, ...) is inside the bounding box area. 

The task is to find all annotations of the annotation set which are within the request's boundaries, meaning we need to find all annotation bounding boxes that overlap with the request bounding box (illustrated in the top path of the above fiugure part b) ). We implement the multi-dimensional overlap detection iteratively by first checking for overlap on the first dimension between request and annotation set. The other dimensions will only be checked for the pairs with an overlap in the first dimension.
Conceptually, the way a request slices out an annotation segment out of an annotation set is similar to the way a request slices a data segment out of the data cube as illustrated at the bottom of part b) in the above figure.

