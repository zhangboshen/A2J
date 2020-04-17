
# Training code for NYU dataset.

We release our training code for NYU dataset. Other datasets can be easily reproduced with minor modifications. 

In `nyu.py` we show all of the data augmentation strategies mentioned within the paper.

In `anchor.py`, `A2J_loss` combines the **Joint position estimation loss** and **Informative anchor point surrounding loss**. Details can be found in our paper.

Simply run `nyu.py`, you can reproduce our result. (Note that different PyTorch environment can result in different performances. Actually, when I run this code with PyTorch1.2, the error is 8.58, but I can get only 8.68 with PyTorch1.3. Other things like batch size can also matters.)
