## Symmetry

---

### Description

A project to find patterns in terraform files which can be used to create templates of cloud resource configurations

Thus, with Symmetry templating of Terraform(IaC) will be possible

---

### How to use:

1. Ensure project script is in root infrastructure file
2. Activate python script `main.py`
3. csv will be generated in folders named:
    1. data
    2. module
    3. resource

---

### Project status

- Ended

---

### Thought process

1. Read entire project's infrastructure terraform files recursively
2. Split terraform code blocks into `data`, `resource`, `local`, `modules` code block
3. isModule ? Extract out source of module and map to the respective aws resources : Skip
4. isTagExist ? Extract tags field out from the code : Skip
5. Aggregate respective code blocks into the respective tuple lists
6. Transform list of tuple into a Pandas DataFrame
7. Pre-Process the data
    1. Term Frequency Inverse Document Frequency (TF-IDF)
    2. Similarity
8. Unsupervised Machine Learning Model
    1. Hierarchical Clustering
    2. K means
9. Output respective results

#### Unsupervised Machine Learning Model

##### Hierarchical Clustering

1. Set parameters and ingest pre-processed data
2. Utilise SciPy library to determine and visualise the clusters

##### K means

1. Set the parameters for the Kmeans Model.
2. Set the range of number of clusters to test and record the clusters.
3. Optimize the number of cluster via the elbow method
    1. Determine the formula for line A between first point of the graph and the last point of the graph
    2. Determine the formula for the perpendicular line B for the above line A
    3. For each cluster, calculate the distance of the point from line A with the use of line B formula
    4. The optimal cluster is the longest distance as that represents the turning point
4. Parse data into the optimized Kmeans

---

### Credits

- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [SciPy](https://github.com/scipy/scipy)