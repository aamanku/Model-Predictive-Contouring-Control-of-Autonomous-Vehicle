Written By: Abhijeet Kulkarni (amkulk@udel.edu)
For: Research Project at UD
Title: Autonomous driving using Model Predictive Contouring Control.

Files:
¦   Main.py							#Main File
¦   ReadMe.txt
¦   requirements.txt				#Contains requirements of packages to run the Main.py
¦   weights.txt						#Contains weights and parameters for different behaviour. Trial and error.

+---functions
¦      alpha2data.dill				#Generated lambdified symbolic function for slip angle. Generated in Dynamics.ipynb
¦      Car_Dynamics.py				#Contains vehicle dynamics parameters.
¦      DiscDynaMatrices.dill		#Generated lambdified symbolic function for disceret dynamics's matrics . Generated in Dynamics.ipynb
¦      Dynamics.ipynb				#File to generate .dill files.	
¦      ErrorsLindata.dill			#Generated lambdified symbolic function for linearized error. Generated in Dynamics.ipynb
¦      eta_dot.dill					#Generated lambdified symbolic function for simulation of nonlinear vehicle. . Generated in Dynamics.ipynb	
¦      HelpingFunctions.py			#Contains auxilary functions. 	
¦      MPCCFunctions.py				#Contains object for the model predictive contouring control.
¦
+---results							#Contains results presented in the final report.


Procedure:
1) Install packages listed in requirements.txt
2) Run functions/Dynamics.ipynb to generate lambdified symbolic functions if .dill files not present in functions/
3) Run Main.py	
