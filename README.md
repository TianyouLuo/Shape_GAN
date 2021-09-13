# Shape_GAN
Use different GANs on shape data

## Data format
The input data is a csv file for each individual with 30,000 rows and 7 columns. The columns represent 7 different modalities, and the rows are fixed points on the common template. The values represent the value for that grid point and that modality. The first 15,000 rows correspond to left hippocampus and the last 15,000 rows correspond to right hippocampus, so we need to treat them as separate training sets and separate tasks (for now we just work with the first 15,000 rows). The rows can be arranged into a 100\times 150 matrix which shows the measurement on that extended surface. We will treat that matrix as the input.
