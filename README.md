# NLU_Assignments
## Description
This repositiory is meant to contain the data and python code requested for the assignments given during the Natural Language Understanding course
***
## Requirements
In order for the modules' functions to be executed correctly, on the device running it, the library **spacy** must be installed.
You can check whether it is installed or not by running the following command in your console:
`spacy info`
Which will return spacy's version, location of installation, pipelines installed and other useful informations.

If **spacy** is not installed in your machine you can do it by running the command:
`pip install spacy`

Moreover the english pipeline of spacy is required to execute the model. Pipelines already installed can be viewed by the info command prevuiously showed. Anyway, if the english pipeline, _en_core_web_sm_, is not installed, you can do it by running:
`python -m spacy download en_core_web_sm`
***
## Content of repository
* **1st_Assignment.py**: Is the python module implementing the compulsory part of the first assignment.
* **1st_Assignment_Report.pdf**: Is the report describing what has been done inside the python code and the purposes it aims to fulfill.
***
