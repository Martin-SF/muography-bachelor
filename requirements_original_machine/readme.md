my bachelor thesis computations were made on TU Dortmunds phobos server.

I used for the computation either env_3.10 or m1. (I am not 100% sure, but they differ only marginaly)

i made two requirements.txt files on the two environments with pip freeze > requirements_ENV.txt
and pip list --format=freeze > requirements_format_freeze_ENV.txt. 
The first one gave references to local files so i made a second one with the format freeze.

