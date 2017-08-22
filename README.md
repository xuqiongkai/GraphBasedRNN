# GraphBasedRNN

Graph-Based Recursive Neural Network for Vertex Classification.

### Usage

* Clone the repository

```bash
git clone https://github.com/xuqiongkai/GraphBasedRNN.git --recursive
```

* Prepare the dataset (optional)

Datasets used for the evaluation in our work has been processed to the supported format. You may also prepare your own dataset according to the **Dataset format** section.

* Run the desired models by `main.lua` script. To evaluate on a dataset, simply specify the folder path with its prefix
 
	```bash
	th main/main.lua -d  NoiseGraphDataset/citeseer <arguments>
``` 

Please refer to `th main/main.lua -h` for complete list of parameters.

Please refer to 

### Dataset format

Each dataset need to be assigned with a `prefix`. Then the folder structure and file names are determined by following rules:

├── prefix  
│   ├── prefix.content   
│   ├── prefix.feature  
│   ├── prefix.cites  
│   ├── prefix.label  
│   ├── prefix.meta  
│   |     
│   ├── prefix.cites.add10  
│   ├── prefix.cites.add5  
│   ├── prefix.cites.reduce10  
│   ├── prefix.cites.reduce5  
│   |  
│   └── README  

#### Components


* **prefix.label**: parsed labels from content file, each row contains the tab separated entry ID and class label
* **prefix.feature**: serialized feature matrix in `torch.DoubleTensor`, 2-D matrix of dimension `num_examples * number_of_fetures`. 
* **prefix.cites**: links in tab-separated format with each row denoting a single edge; the two elements separated by the tab are the two vertices connected by this link
* **prefix.meta**: dictionary of all possible class labels in this dataset

#### Noisy variants

* **prefix.cites.add5**: links data with additional 5% random links
* **prefix.cites.add10**: links data with additional 10% random links
* **prefix.cites.reduce5**: links data by randomly removing 5% links from the original
* **prefix.cites.reduce10**: links data by randomly removing 10% links from the original

### Supported models

* Logistic Regression (LR)
* Iterative Classification Approach (ICA)
* Label Propagation (LP) 
* Text-Associated DeepWalk (TADW)
* **GRNN**
* **GRANN**

Please refer to our paper [cikm2017][#citation] for more details of the model.

### Acknowledgement

It would be appreciated to cite the related publications (ordered by publication date) if you decide to use these datasets in your work.

#### Citation

[TODO](http://CIKM2017)

#### Supplementary references

@article{mccallum2000automating,  
&nbsp;&nbsp;&nbsp;&nbsp;title={Automating the construction of internet portals with machine learning},  
&nbsp;&nbsp;&nbsp;&nbsp;author={McCallum, Andrew Kachites and Nigam, Kamal and Rennie, Jason and Seymore, Kristie},  
&nbsp;&nbsp;&nbsp;&nbsp;journal={Information Retrieval},  
&nbsp;&nbsp;&nbsp;&nbsp;volume={3},  
&nbsp;&nbsp;&nbsp;&nbsp;number={2},  
&nbsp;&nbsp;&nbsp;&nbsp;pages={127--163},  
&nbsp;&nbsp;&nbsp;&nbsp;year={2000},  
&nbsp;&nbsp;&nbsp;&nbsp;publisher={Springer}  
}


@inproceedings{giles1998citeseer,  
&nbsp;&nbsp;&nbsp;&nbsp;title={CiteSeer: An automatic citation indexing system},  
&nbsp;&nbsp;&nbsp;&nbsp;author={Giles, C Lee and Bollacker, Kurt D and Lawrence, Steve},  
&nbsp;&nbsp;&nbsp;&nbsp;booktitle={Proceedings of the third ACM conference on Digital libraries},  
&nbsp;&nbsp;&nbsp;&nbsp;pages={89--98},  
&nbsp;&nbsp;&nbsp;&nbsp;year={1998},  
&nbsp;&nbsp;&nbsp;&nbsp;organization={ACM}  
}  

@inproceedings{Craven:1998:LES:295240.295725,  
&nbsp;&nbsp;&nbsp;&nbsp;author = {Craven, Mark and DiPasquo, Dan and Freitag, Dayne and McCallum, Andrew and Mitchell, Tom and Nigam, Kamal and Slattery, Se\'{a}n},  
&nbsp;&nbsp;&nbsp;&nbsp;title = {Learning to Extract Symbolic Knowledge from the World Wide Web},  
&nbsp;&nbsp;&nbsp;&nbsp;booktitle = {Proceedings of the Fifteenth National/Tenth Conference on Artificial Intelligence/Innovative Applications of Artificial Intelligence},  
&nbsp;&nbsp;&nbsp;&nbsp;series = {AAAI '98/IAAI '98},  
&nbsp;&nbsp;&nbsp;&nbsp;year = {1998},  
&nbsp;&nbsp;&nbsp;&nbsp;location = {Madison, Wisconsin, USA},  
&nbsp;&nbsp;&nbsp;&nbsp;pages = {509--516},  
&nbsp;&nbsp;&nbsp;&nbsp;numpages = {8},  
&nbsp;&nbsp;&nbsp;&nbsp;publisher = {American Association for Artificial Intelligence},  
&nbsp;&nbsp;&nbsp;&nbsp;address = {Menlo Park, CA, USA},  
}   

### License

The GNU General Public License v3.0