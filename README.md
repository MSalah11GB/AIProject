# How to run the webapp

Step 0: Install Python, as well as Pickle, Scikit-learn and Streamlit library. To install any library, use this command in the terminal:

	pip install library_name

Step 1: In the web app folder, open file bengal_house_price.py. All models and datas should be included in the folder
 
Step 2: Launch the app using this command
		
	streamlit run file_path

Step 3: After closing the app, it is necessary to end both the webapp and the .py session in Task manager (this is due to a bug in streamlit that does not close the app properly)
