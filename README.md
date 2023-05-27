# STEREOTYPES IN MULTILINGUAL LANGUAGE MODELS

## Description
This repository contains all the code used for the realisation of the ATCS Project : Stereotypes In MultiLingual Language Models.  this project, we study stereotypes that emerge within pre-trained Multilingual Language Models(MLMs). These models are typically trained on large-scale multilingual text corpora, learning torepresent the different languages in a shared embedding space. This shared representation allows the model to generalize knowledge learned in one language to other languages, a phenomenon known as cross-lingual transfer (Conneau et al., 2019). Our objective is to build upon the work of Choenni et al., 2021 by initially contrasting emotion profilesof identical social groups across diverse languages within these MLMs. 

## Installing the Project Environment


This project uses an Anaconda environment for managing dependencies. Follow the instructions below to set up the environment:

First, ensure that you have Anaconda or Miniconda installed on your system. If not, you can download Anaconda from https://www.anaconda.com/products/distribution.

Clone the repository and navigate to the project directory:

<pre>
'''python
git clone https://github.com/username/project.git
cd project
'''
</pre>

The environment.yaml file in the project root contains the specifications for the project's conda environment. Create the environment using the following command:

<pre>
'''python
conda env create -f environment.yaml
'''
</pre>

Once the environment is created, you can activate it using:
<pre>
'''python
conda activate env_name
'''
</pre>

Replace env_name with the name of the environment specified in the environment.yaml file.

To check that the environment was installed correctly, you can list the environments:
<pre>
'''python
conda env list
'''
</pre>

You should see your new environment in the list.

Now you are ready to start working on the project!

Remember to replace username, project, and env_name with the appropriate values for your project.

## Creating the emotion profiles 

The first thing to do is to generate emotion profiles for given social groups for given languages. You can run the file run_test_normalization.py, it will generate emotion profiles for each social group and each language for a given model. Modify the arguments to modify either the used model or the top_k which corresponds to the number of predictions taken into account to generate the emotion profiles per social group per prompt. 

You can run the emotion profiles for specific fine-tuned models by changing the --fientuned_model flag to either 'french', 'english', 'greek', 'spanish' or 'base'. 

## Correlations

Ideally, you want to compare the baseline emotion profiles with fine-tuned emotion profiles, to see how the outputs of the model changed with fine-tuning. For this purpose, first run emotion profiles with the pretrained xlm-R model and with a fine-tuned model. Then run the run_correlations.py file with the right fine_tuned model. 

## Fine-tuning of the models

Fine-tuning is possible through the use of all train_*.py file. Running them directly is possible. You can also change the flags for parameters such as batch_size, number of epochs, output_directory ...
