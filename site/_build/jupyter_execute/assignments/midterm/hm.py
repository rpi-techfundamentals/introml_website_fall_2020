# Midterm - Correction

We are going to go over the Midter in class Thursday.  To prepare, please try the grading code by adding to the bottom of your notebook.  Try and understand and correct what you missed so you can pay extra attention when you go over that part.   

Feel free to post questions to Webex Teams.  You are also allowed to ask questions of others for the homeowork.  However, all the work has to be yours. 



files = "https://github.com/rpi-techfundamentals/introml_website_fall_2020/raw/master/files/midterm.zip" 
!pip install otter-grader && wget $files && unzip -o midterm.zip

#This runs all tests. 
import otter
grader = otter.Notebook()
grader.check_all()