# A1_ML

# Please use commands to run the application 
# move the file cd ~/...
# for building and dependency resolution 
	'sudo docker build -t dash_app .'

# running the dask server 
	'docker run -p 8089:8089 dash_app'
# hit
	' localhost:8089'

# please enter the values in the range only
# Feature like engine cc is considered as ordinal variable reange more 2500 will not give good predition
# Regid linear model has no dependency any null value in feature engeering can lead to no prediction under blending column
