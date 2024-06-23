This 	Python software package is used to predict crotonylation sites in human proteins.

Usage of this Python software package:

(1) Prepare your sequence(s) file in fasta format and name it “.txt”.`<br>`(2) Run"Predictor.py" to start the forecasting programs.`<br>`(3) Click the "Select File" button, the file selection pop-up window, click to select the txt file that meets the requirements.`<br>`(4) Click the "Prediction" button, the prediction result and probability of correctness for each sample will appear on the right side of the interface.`<br>`(5) Click the "Save" button, you could save the predictions as a csv file to an address of your choice.`<br>`(6) Click the "Clear" button, the page display will be cleared.

Please notice:`<br>`(1) this software package only supports the 64-bit version of Windows operating system.`<br>`(2) The version of Python and libraries used by "Predictor.py":`<br>`- Python version used by the code: This code should be compatible with your version of Python, Python 3.x is recommended (e.g. Python 3.9 or later).`<br>`- Libraries used: torchaudio；scikit-learn；pandas；Tkinter.`<br>`(3) The models in the "Predictor.py" can be generated by "Train.py" or of your own choosing, and the training dataset of "Train.py"can be added by yourself.`<br>`(4) The training and test sets are located in the "dataset".
