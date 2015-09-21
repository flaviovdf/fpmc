FPMC
----

Python/Cython implementation of the: "Factorizing Personalized Markov Chains
for Next-Basket Recommendation" paper.

Notes
-----

Only works for baskets of size=1. Shoulde be easy to change to other sizes. Our
datasets only have these sized baskets.

Dependencies for library
------------------------
   * Cython
   * Numpy
   * Pandas

How to install
--------------

Clone the repo

::

$ git clone https://github.com/flaviovdf/fpmc.git

Make sure you have cython and numpy. If not run as root (or use your distros package manager)

::

$ pip install numpy

::

$ pip install Cython

Install

::

$ python setup.py install

Run the main script or the cross_val script:

$ python main.py data_file num_latent_factors model.h5

This will read the data_file, decompose with num_latent_factors and save
the model under the filename model.h5

The model is a pandas HDFStore. Just read-it with:

::

>> import pandas as pd

>> pd.HDFStore('model.h5')

The keys of this store have the output matrices described in the paper.

Input Format
------------

The input file should have this format:

dt <tab> user <item> from <item> to

That is, a tab separated file where the first column is the amount of time the user 
spent on `from` before going to `to`. The second column is the user id, the third 
is the `from` object, whereas the fourth is the destination `to` object. I used 
this input on other repositores, thus the main  reason I kept it here. 
This code will ignore the first column, so you can just use any float.

References
----------
.. [1] Rendle, S. and Freudenthaler, C. and Schmidt-Thieme, L.
   "Factorizing Personalized Markov Chains
    for Next-Basket Recommendation" - WWW 2010 
