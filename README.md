# ResTS: Residual Deep Interpretable Architecture for Plant Disease Detection
---
This is the source code of ResTS architecture described in the paper : 
## [ResTS: Residual Deep Interpretable Architecture for Plant Disease Detection](https://www.sciencedirect.com/science/article/pii/S2214317321000482)

We propose an architecture named ResTS (Residual Teacher/Student) that can be used as visualization and a classification technique for diagnosis of the plant disease. ResTS is a tertiary adaptation of formerly suggested Teacher/Student architecture. ResTS is grounded on a Convolutional Neural Network (CNN) structure that comprises two classifiers (ResTeacher and ResStudent) and a decoder. This architecture trains both the classifiers in a reciprocal mode and the conveyed representation between ResTeacher and ResStudent is used as a proxy to envision the dominant areas in the image for categorization. The proposed structure ResTS (F1 score: 0.991) has surpassed the Teacher/Student architecture (F1 score: 0.972) because of the residual connections being introduced in all the components. ResTS can yield finer visualizations of symptoms of the disease. All test results are attained on the PlantVillage dataset comprising 54306 images of 14 crop species.

![Architecture](https://github.com/jackfrost1411/Residual_Teacher_Student/blob/master/ResTS%20architecture/ResTS400dpi.png)

# For access to the pre-trained ResTS model,
## 1. Comment your request with the proper reason on this post and I will share the personal links to access the pre-trained saved model in various formats: https://www.linkedin.com/posts/dhruvilshah28_research-innovation-datascience-activity-6809383337841434624-b4b-
## 2. Cite the paper as instructed here: https://doi.org/10.1016/j.inpa.2021.06.001
## 3. You will be given access only if the request is credible and the paper is properly referenced and credited.

# Cite this paper as:
## 1. D.  Shah,  V.  Trivedi,  V.  Sheth,  A.  Shah,  U.  Chauhan,  ResTS:  Residual  DeepInterpretable Architecture for Plant Disease Detection, Information Processing in Agriculture (2021), doi: https://doi.org/10.1016/j.inpa.2021.06.001
## (https://www.sciencedirect.com/science/article/pii/S2214317321000482)

The data set used for this paper can be downloaded from https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/segmented. In this paper, we have used the segmented version with black background of [PlantVilage dataset](https://github.com/spMohanty/PlantVillage-Dataset).

### Working video of the WebApp:
![Working of Flask+React app](https://github.com/jackfrost1411/Residual_Teacher_Student/blob/master/Flask%20%2B%20ReactApp%20source%20code/Working%20of%20React%20app.gif)

## Prerequisites
To run the code, the following packages are required :

* tensorflow==2.4.1
* Keras==2.4.3
* matplotlib==3.2.2
* OpenCV==4.x
* Pillow==7.0.0



## Usage
1. install required packages.
2. Download the model
3. Copy the model in `./model`.
4. Copy input images with "black background"  in `./images`.
5. Run the following code from the project folder:
"python visualization.py"

The visualization methode can be used also in an interactive environment using "test_visualization.ipynb".
