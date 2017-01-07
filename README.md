# MatchingNetworks-OSL

Coding In Progress. Will update README soon!

### Note on Omniglot Dataset
The dataset from Bredan Lake's github page gives a training split of 934 characters (30 alphabets) and
evaluation split of 659 characters (20 alphabets). Both of Matching Networks and MANN use 1200 characters for training
and 423 characters for evaluation. I was confused with this contradiction in the training/evaluation splits. Then I
checked out the Siamese Networks paper by Koch and there he mentions the splits provided by Brendan Lake is 40 alphabets
for training and 10 alphabets for evaluation. Another contradiction... Hence, to stick to 1200 characters for training
and 423 for validation, I swapped some alphabets between the `image_background` and `image_evaluation` sets from Brendan
Lake's original dataset, so that `image_background` (training split) has 1200 characters (40 alphabets) and `image_evaluation` (evaluation split) has
423 characters (10 alphabets). Phew! Providing the data here for reference.



