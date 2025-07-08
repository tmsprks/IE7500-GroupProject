README for Sentiment Analysis of 3.6M Amazon Reviews, IE 7500, Summer 2025, Group 3

README.md      - markdown README
README.txt     - this README file
SA-NLP.ipynb   - Jupyter notebook containing EDAs
benchmark/     - benchmark information
checkpoint/    - model checkpoint files
config/        - application, model configuration files
data/          - CSV files generated from the original Amazon review Kaggle dataset.
db/            - SQLite DB related files for data checks, preliminary EDA
doc/           - project docs, class and package UML diagrams
framework/     - model pipeline framework
model/         - various models including but not limited to LSTM, RNN, Self Attention
python/        - Requirements.txt, Python version
test/          - test drivers
utils/         - utility and helper classes

====================================================================

* For EDA of the Amazon Reviews data, please run the Jupyter notebook SA-NLP.ipynb

* For a complete training/validation/testing run of all the models, please run "python test/test_model_pipeline.py" from the project root directory.
  This test driver produces a summary of all the models' training/validation/testing metrics/performance numbers

* For testing the inference capability of a single model, please run "python test/test_single_model_pipeline_inference.py"
  The test driver runs the model inference pipeline of one of the model.  
  The test driver does not run the entire model pipeline from preprocessing to fit to evaluate.  
  Instead, the model's saved and trained state is loaded and then the model runs inference.
  This saves enormous compute and time.

* For testing the inference capability of a single model without using the model pipeline, please run "python test/test_single_model_inference.py"
  This test driver loads the specific model's saved state and then runs the model inference.