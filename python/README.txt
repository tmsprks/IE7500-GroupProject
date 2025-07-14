README for python directory containing requirements.txt

README.txt          		- this README file
requirements.txt    		- minimum requirements.txt to run the demo application, SA-NLP-demo.py, and all the test scripts under the 
                                  test directory
requirements-EDA-VSCode.txt	- requirements.txt to run SA-NLP.ipynb (EDA Jupyter notebook), SA-NLP-demo.py (demo application) and all the 
                                  test scripts under the test directory. requirements-EDA-VSCode.txt will pull in some Windows VS Code 
                                  related packages such as pywin32, etc.

====================================================================================================================

If your environment is Ubuntu or macOS, then using requirements.txt and manually installing required packages once you
run SA-NLP.ipyn maybe the cleanest approach.  This approach will only pull in EDA packages like matplotlib, seaborn, spacy,
wordcloud, scikit-learn, spacy and spacy's medium size vocabulary, en_core_web_md.