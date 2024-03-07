# Auto Cell Counter (mini-project for research lab)
Automated cell counting tool using a microscope image stitcher coupled with an ORB-descriptor based brute-force feature matching algorithm. Please run the following code to install the necessary dependencies:

pip install -r requirements.txt

Also, the ORB algo assumes that the input microscope images are properly stained and a bright, white light source is used for the image captures (for better count accuracies using the algorithm). Currently, tests have only been ran using cell cultures that are stained blue. Additionally, the used criterion for defining cell colonies is the standard 50-cell threshold that is commonly used in academia (i.e. a contiguous stained mass would only be considered a "colony" if it clearly contains more than 50 individual cells).