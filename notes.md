REMEMBER, 2x3 has 2 rows and 3 columns, not 2 in length and 3 in height


I could arrange each layers activations as a matrix, where each row is a layer, 
but the only issue is that the weights cant be stored as a 3d matrix (or can they?)

wait i literally can. 

so 1 layer is represented completely by:

[a1
 a2
 a3
 ... 
 an], [a1.1 a1.2 a1.3 a1.4 a1.5 ... a1.k
       a2.1 a2.2 a2.3 ... a2.k
	   ...
	   an.1 an.2 an.3 ... an.k], and
[b1
 b2
 b3
 ...
 bn]

| A  |    |Weights
|:--:|
| a1 |
| a2 |
| ...|
| an |

where **n** is the number of neurons in the layer,
and **k** is the number of neurons in the previous layer.

this can then be turned 3-d by stacking the vectors into a 2d one, 
and stacking the matrices into a 3-d one? maybe?

okay no im doing it oop



not sure what format to use for storing the weights and biases for later.

option one:

data/
├─ weights/
│  ├─ layer 1.txt
│  ├─ ...
│  ├─ layer n.txt
├─ biases/
│  ├─ layer 1.txt
│  ├─ ...
│  ├─ layer n.txt

option two:

data/
├─ layer 1
│  ├─ weights.txt
│  ├─ biases.txt

I also don't know how to organise the numbers within the file.
could try searching up ways to store arrays in files, there might be file types that would help for all i know

JSON! OF COURSE!

i can rewrite the test data and training data using json
then i can also arrange the weights and biases from each layer as 

okay done. so now what would weights and biases look like for each layer:

one file:
{
	"layer_number": 1,
	"weights": <not sure how to change from np array to json specific object>,
	"activations": [0.0, n.m, x.y, ...],
	"biases": [0.0, 0.3, ...]
},
{
	"layer_number": 2,
	"weights": ...,
	"activations": [0.0, n.m, x.y, ...],
	"biases": [0.0, 0.3, ...]
},
...
{
	"layer_number": n,
	"weights": ...,
	"activations": [0.0, n.m, x.y, ...],
	"biases": [0.0, 0.3, ...]
}
it works out as un-zero-indexed because the input layer has no weights or biases. 

ah i can use `.tolist()` as shown here: https://stackoverflow.com/questions/48310067/making-numpy-arrays-json-serializable

what i intend to do is to say that if at any point we get a keyboard interrupt, or we've finished a training batch, we store the weights and biases so that we can keep progress