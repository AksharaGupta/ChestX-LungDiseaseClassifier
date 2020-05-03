# ChestX-LungDiseaseClassifier

Computer Aided Disease Detection Tool deployed on a Webpage

Implementation of Computer Aided Detection (CAD) tool to help radiologists improve the accuracy of their detection of thoracic diseases using Chest X-rays. Chest X-ray examinations are one of the most
frequent and cost-effective medical imaging examinations available.
However interpreting this information for doctors manually presents
challenges as there are complexities. Thus we build a web application that
can be used by radiologists to make use of the tool in a user friendly
interface. The tool makes use of Machine Learning techniques to exploit an
extremely large dataset to predict thoracic diseases using a single chest
x-ray as an input. The application also consists of a symptom engine and
extends some information and guidance on the disease that the tool
Predicts.

OS: Linux, Windows

CNN VGG16 Model execution:
1. Install Anaconda
2. Install NIH Chest X-ray kaggle dataset - NIH Chest X-rays and insert
all in /CNNModel/data/images
3. Open Anaconda Prompt
4. Run the following to make a new environment:
    - conda create --name myenv python=3.6
    - conda activate myenv
5. Run the following to install all dependencies required
    - conda install tensorflow==1.14.0
    - conda install keras==2.2.4
    - conda install scipy==1.0.0
    - conda install PILLOW
    - conda install flask
    - pip install jupyter notebook
6. Cd into the directory called CNNModel.
8. Run the command to open jupyter notebook using the links displayed in browser jupyter notebook
9. Run the Try-Copy-Move code
10. Open the VGGtrain notebook under “Using Keras” directory
11. Add images in alient_test folder which you would like to test to see how many are correctly or wrongly predicted
12. Open testing notebook under “Using Keras” directory

Access Web application with deployed model:

1. Cd into folder flaskdeploy (in same environment as before)
2. Run in anaconda prompt- python keras_flask.py
3. Open browser : localhost:5000
4. Access web application. Login credentials - username:admin, password:admin
5. Open DataAnalysis.ipnb to see few data analysis codes on the dataset. (Add required csv to your drive at a specified location and use that in the code throughout)

