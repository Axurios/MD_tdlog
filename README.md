#  Welcome to our README
## How to use it ?
right-click on the "diapo.md" file and "Open Preview" to view our presentation of this project.
(might need marp extension on vscode)

Before running main.py, please ensure that you have a virtual environment ready
if some packages needed are missing from your env, type "yes" in the terminal to download them
here's an example of what it would look like :
> Missing packages: ['numpy==1.23.5', 'pandas==1.5.3  # Adjust to latest stable version as needed']  
> Some packages are missing. Do you want to install them? (yes/no):


once this has been done, the main window of our graphical interface should open.
it will look mostly like page 9 of our presentation, i.e ![](images/main_interface.png)  


To better interact with the functionalities, you need to provide the datafiles needed.  
in the "Select MD data file" you should put "NP_1200K_desc.pkl"  
and in the "Select Theta file" : "theta.pkl" preferably or "theta_md.pkl"  
(like in main_interface.png) 

## What does it do ?

The applications was designed for researchers from the CEA Paris-Saclay working on the prediction 
of molecular dynamics. 
Their goal is to train a prediction model using the Fisher Divergence metric instead of the RMSE.
However, training the model with this metric would take too long because of the complex formulas
that are used for computing gradients. We will instead fine-tune the model by adding a linear
layer that 

## Loss Landscape button

The "Loss Landscape" button, located in the bottom left corner, allows users to switch to a 
second window where they can plot the loss landscape for a selected model and loss function.
![](images/loss_window.png)  

Initially, no model or loss function is selected. If the user attempts to plot the loss 
landscape without selecting these parameters, an error message will be displayed.

Once a model and a loss function are selected, their names will appear in the corresponding 
boxes, and the loss landscape functionality will be enabled.

The two varying weights for the loss landscape are selected randomly. As a result, if the 
user presses the "Loss Landscape" button multiple times without changing the model or loss 
function, the results will still vary.

On certain computers, a graphical bug may occur when displaying the loss landscape, causing 
a noise rectangle to appear next to the plot. This artifact often disappears if the user 
navigates back to the main window and then returns to the loss landscape window.

Returning to the main window does not erase the last generated loss landscape plot.
