# FDA  Submission

**Your Name:**: Kofi Ofuafor

**Name of your Device:**: Pneumonia detection algorithm

## Algorithm Description 

### 1. General Information

**Intended Use Statement:** 
More than 1 million adults are hospitalized with pneumonia and around 50,000 die from the disease every year in the US alone.

This algorithm is intended for use in _assisting the clinician_ in acute care setting with a speedy detection of pneumonia within a chest x-ray. In otherwords it is to be used a a screening test for Pneumonia. The predicate device for this algorithm is a CADx device.


**Indications for Use:**
This algorithm is used as a Pneumonia classifier in a chest x-ray taken in acute medical or surgical admission units, accident and emergency and acute care wards. It is intended for use in people aged 20-70 years with _no prior history_ of presence of edema, pleural effusion and pulmonary infiltrations


**Device Limitations:**
The device study was based on images from people of age 20- 70 and was significantly less accurate in the presence of concurrent disease and therefore not recommended for use in the catgory of patients were there is prior presence of disease in a coloumn num_prior_positive in the demographic data. The algorithm was not traing in patients over 80 years old. 

**Clinical Impact of Performance:**
* Performance statistics
    * Accuracy: 81%
    * Precision: 0.16666666666666666
    * Sensitivity: 1
    * Specificity: 0

For this algorithm, missing a detection of Pneumonia is not acceptable as human life and depends on it. False negative may slow down clinicians urgency to look at the patient xray for a physical examination. Therefore False Negative results have more impact than a False positive. Accordingly, the threshold for the algorithm has been weighed in favour of recall(sensitivity). 

When a test with high recall returns a negative result, you can be pretty confident that the result is truely negative. Recall does not take into account FP though, so you may still be labelling alot of negative cases as positive. So recall are good for screening tests.

### 2. Algorithm Design and Function

**Algorithm Fow chart**
The check_dicom function reads in a .dcm file, checks the important fields for our device, and returns a numpy array of just the imaging data.


_Dicom flow Chart_


![Dicom flow chart](/Figure/dicom-flow-chart.jpeg)



_ Architecture flow chart_


![Architecture flow chart](/Figure/architecture-diagram.jpeg)










The function, check_dicom(), is required to check the image type(MODALITY), Body Part Examined (Chest), and Image position (PA or AP ) for each DICOM image and check if the input to the algorithm is valid to be predicted by our algorithm or not.

**DICOM Checking Steps:**
There are three steps:
1. Extract the desired attributes
*Modality: DX
The image type DX is a digital radiography
*Body Part Examined: Must be Chest
*Patient Position: AP or PA
The image position could be AP or PA
*Study Description: label
This is the label for the chest x-ray.

2. Next the check_dicom function checks each image if the input to the algorithm is valid to be predicted upon by our device.

3. Finally, valid images are included as input and a numpy array of the image data is returned.

**Preprocessing Steps:**
The preprocess_image function used here takes in the image value returned from check_dicom, and the image mean, standard deviation as well as image size corresponding with that of VGG16 input image size of (1,224,224, 3). These inputs are used to standardize and resize the image which is returned as a processed image.

Image augmentation techniques were also applied prior to training the model especially to training data.

**CNN Architecture:**
The pretrained model was loaded and the first 17 layers were rendered untrainable
Here is the pretrained model



   ![Loaded Model](/Figure/pre_trained_model.jpeg)


Seven fully connected layers were added to the pretrained model as classifiers. Of the 7 layers, three were Dropout layers to prevent overfitting.


   ![My model](/Figure/my_model.jpeg)


### 3. Algorithm Training

**Parameters:**
* Types of augmentation used during training
    * horizontal_flip = True, 
    * vertical_flip = False, 
    * height_shift_range= 0.1, 
    * width_shift_range=0.1, 
    * rotation_range=20, 
    * shear_range = 0.1,
    * zoom_range=0.1)
* Batch size
    * Training: 16
    * Validation: 32
    * Performance evaluation: 100
* Optimizer learning rate
    * Adam
    * Learning rate: 5e-6. For 5 trainable layers. 
        Decaying rate per epoch: 0.5

**Layers of pre-existing architecture that were frozen**
17 Layers of pretrained model were frozen and therefore not trainable

   ![Frozen Arhitecture](/Figure/pre-existing-frozen-arch.jpeg )


**Layers of pre-existing architecture that were fine-tuned**
    * Fully connected layers were fine-tuned.  
    
  
**Layers added to pre-existing architecture**
    
   _One one convulated was layers were fine tune and four layres were added for training_
    
   ![Fine-Tuned Layers](/Figure/fine-tuned-layers.jpeg)


**Algorithm training performance visualization**

   _Train and validation loss/accuracy versus epochs._

  ![Training performance](/Figure/training-performance-plot.jpeg)

  
  _The plot of how the model distinguih between the classes._
  
  ![Roc curve](/Figure/plot-auc-curve.jpeg)

  
  _This is plot of two important parameters for the calculation of F1 Score._
  
  ![Precision recall curve](/Figure/pr.jpeg)
  
  _This is a plot of F1 Scores v Thresholds._
  
  ![F1 Scores plot](/Figure/f1-scores.jpeg)
  
  Highest f1 score is 0.34 and this corresponds to a threshold value of 0.38



**Final Threshold and Explanation:**
* Best_threshold = 38
The decision model for this threshold was based the calculation two thresholds, from  F1 scored weighted by a precision value of 0.8 and recall value of 0.8 and the point of highest f1 score both converged on recall weighted performance parameter as the best threshold for my device.

It is important for the algorithm to pick all postive cases of pneumonia. Missing cases of pneumonia is not accepetable. Recall is more valuable under the circumstance where a negative test means you have a high chance of not having the disease. 

### 4. Databases
 (For the below, include visualizations as they are useful and relevant)

**Description of Training Dataset:** 

   _Data set was split 80:20 percent in favour of training data_
    By using the len of positive cases of pneumonia to randomly select equal number of negative cases.
    
   ![Training data after split with 4:1](/Figure/train-data-b4.jpeg)
   
   _Sizes of train and validation data after discarding excess data_
   
   ![Training data after discarding excess negative data](/Figure/train-data-after.jpeg)



**Description of Validation Dataset:** 

   _Validation after split 80:20 with 20% of the entire data in the validation dataset_
   By finding using random sample to add a negative data to the in the ration of 1:4 postive to negative cases

   ![Validation data after split](/Figure/val-data-b4.jpeg)

   
   _Validation set must contain 20:80 ratio (positive to negative)_
   
   ![Validation data after split](/Figure/val-data-after.jpeg)
   
   
   

### 5. Ground Truth
The NIH dataset radiologist reports were mined using Natural Language Processing (NLP) to create the disease labels from the associated radiology reports. It is expensive and laborous to hire a radiologist to label images often require several of them through a voting mechanism to obtain ground truth. NLP is fast inexpensive considerd to be >90% accurate. It can be associated with erroneous labelling. Is considerded to be the silver standard level of ground truth, with gold standard being lung biopsy which is impractical.

**Llimitations** 
Labels obtained by natural language are not a substitute for a radiologist acquired labels as they are not as reliable.  It requires a large amount of data to obtain a decent level of accuracy. There is the ethical consideration as to whether whoc will be liable for any inaccuracy arising from disease labels obtained by NLP. These limitations can impact negatively on the model's ability to detect Pneumonia acurately.

Inaddition, concurrent diseases with similar intensity values can mimic pneumonia. Often the diagnosis of pnueumonia is coupled with history physical signs and blood results all of which are not available to the model. 



### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**
You will simply describe how a FDA Validation Plan would be conducted for your algorithm, rather than actually performing the assessment. Describe the following:

The patient population that you would request imaging data from from your clinical partner. Make sure to include:

Age ranges
Sex
Type of imaging modality
Body part imaged
Prevalence of disease of interest
Any other diseases that should be included or excluded as comorbidities in the population
Provide a short explanation of how you would obtain an optimal ground truth

Provide a performance standard that you choose based on this



 Gender distribution

 ![gender](/Figure/gender.jpeg)
 
 Ags Distribution
 
 ![Age](/Figure/age.jpeg)
 
 
 Prevalence of Pneumonia
 
 ![prevalence of Pneumonia](/Figure/distribution-of-pneumonia.jpeg)

 Distribution of Body Part examined
 
 ![Body part examined](/Figure/body-position-examined.jpeg)

 
 Concurrent disease

 ![comorbidity](/Figure/comorbidity.jpeg)
 
 


**Ground Truth Acquisiion methodology**
The NIH dataset radiologist reports were mined using Natural Language Processing (NLP) to create the disease labels from the associated radiology reports.


Often times, the gold standard(a test with the highest sensitivity and accuracy) is unattainable for an algorithm developer. 


_Silver standard_
The silver standard involves hiring several radiologists to each make their own diagnosis of an image. The final diagnosis is then determined by a voting system across all of the radiologists’ labels for each image. Note, sometimes radiologists’ experience levels are taken into account and votes are weighted by years of experience.



The other **options** to available to establish the ground truth to compare my algorithm are:

1. _Biopsy-based labeling_. Limitations: Impracticable, difficult and expensive to obtain.

2. _NLP extraction_. Limitations: may not be accurate.

3. _Expert (radiologist) labeling_. Limitations: expensive and requires a lot of time to come up with labeling protocols.

3. _Labeling by another state-of-the-art algorithm_. Limitations: may not be accurate.


**Algorithm Performance Standard:**
Recall weighted threshold, was used to minimise the number of false negatives to the bearest minimum.

Best F1 scores of my model is 0.34 compares favorably with a radiologist average F1 score of 0.38 by [Pranav Rajpurkar and Jeremy Irvin et al]('https://arxiv.org/pdf/1711.05225.pdf')

