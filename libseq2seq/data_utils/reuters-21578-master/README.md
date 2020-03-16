# Classification of Reuters-21578 

### Requirements
<br />
To install these packages, use the following command in a <a href="http://docs.python-guide.org/en/latest/dev/virtualenvs/" target="_blank"> virtualenv</a>.

```bash
$ pip3 install -r requirements.txt
```
<br />
In alternative, install the packages listed in requirement.txt file manually
<br />
NOTE: The code works only with python >= 3.5

### Training data
<br />
Available in sgm format in the folder:

```bash
data/ 
```

### Train 

To train one of the implemented algorithms, run the following command:

```bash
$ python train_reuters.py 
```
Flags
```bash
--linearsvm            # use Linear Support Vector Machine classifier 
--knn                  # use Naive-Bayes
--logisticregression   # use Perceptron
--randomforest         # use Random Forest
```

NOTE: The presence of a flag is mandatory.


### Test 
To test one of the implemented algorithms, run the following command:

```bash
$ python test_reuters.py 
```
Flags
```bash
--linearsvm            # use Linear Support Vector Machine classifier 
--knn                  # use Naive-Bayes
--logisticregression   # use Perceptron
--randomforest         # use Random Forest
```