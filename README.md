# Auto Cell Counter
Automated cell counting tool using a microscope image stitcher coupled with an ORB-descriptor based brute-force feature matching algorithm. Mini side project stemmed from research lab endeavor.

### Installation

Please run the following code to install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Also, the ORB algo assumes that the input microscope images are properly stained and a bright, white light source is used for the image captures (for better count accuracies using the algorithm). Currently, tests have only been ran using cell cultures that have been stained with crystal violet dye. Additionally, the used criterion for defining cell colonies is the standard 50-cell threshold that is commonly used in biomedical academia (i.e. a contiguous stained mass would only be considered a "colony" if it clearly contains more than 50 individual cells).
