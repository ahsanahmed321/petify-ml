# SERVER PETMYPAL
This server can identify 196 breeds of dogs and will help you in finding your lost dog

**To run the server perform the following steps:**

1. Create a conda environment
**conda create -n my_pet python=3.6**
NOTE:Creating an environment with python 3.6 is mandatory to avoid any version conflict.

2. Activate conda environment
**conda activate my_pet**

3. In your Command Prompt navigate to your project
**cd your_project**

4. Install all the requirements from requirements.txt
**pip install -r requirements.txt**<br>
NOTE: Change the path in app.py file and set it according to your operating system support.

5. Make empty folders named as ID_IMAGES and GID_IMAGES in blue/api directory.
**mkdir ID_IMAGES, GID_IMAGES**

6. Start the server
gunicorn --bind 0.0.0.0:5000 --reload run:app --daemon

7. Now use the available api's:
**localhost:5000**
   a. Dog Breed List (api/breed_data)
   b. Register dog (api/register)
   c. Find guest dog (api/guest_dog)
   d. Find Dog Breed (api/dog_breeds)
   e. Guest dog matching with lost dog (api/guest_matching)
   f. List of all lost dogs (api/lost_list)
   g. List of lost dog at specific pet id (api/guest_id)

8. Shutdown the server
sudo fuser -k 5000/tcp
