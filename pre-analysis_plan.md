
### Design Overview

For this study, we purchased 3 pots of basil and harvested stalks from each pot.
We then randomized the stalks to each of 4 treatments, blocking by pot so that
each treatment received at least one stalk from each pot. The four treatments 
were the 2x2 combinations of placing the basil in the fridge and placing 
the stalks in water. Each day, photos were taken of each of the plants. 
At the end of the study, we will conduct a blinded experiment where one member
of the group will present leaves from the plants to the other three participants
in a randomized order to be judged on the firmness of the leaves and the apparent
freshness of the herb based on smell/taste. Each plant will receive a score
from 1 to 5 on both qualities.

### Analysis Plan

To analyze the images, we will process each image so that the background is white,
and then count the number of green pixels in each image 
(how should we define 'green'?).
We will then compute the average number of green pixels for each treatment and
compute the following differences.

Let $\mu_{fw}$ denote the average number of green pixels for fridge treatment
level $f$ and water treatment level $w$, where a $1$ indicates the treatment
was applied and a $0$ indicates it was not. Then, the differences are:
$\mu_{00}$, (control mean)
$\mu_{01} - \mu_{00}$, (water effect)
$\mu_{10} - \mu_{00}$, (fridge effect)
$\mu_{00} - \mu_{01} - \mu_{10} + \mu_{11}$ (fridge*water interaction effect).
Note this interpretation assumes an additive structure for the treatment effects.

Alternatively, we could not suppose an additive structure and simply look for any
difference in treatments, for example by taking the range of the means...

Then, we could conduct a permutation test for each difference to assess the
probability of the outcome under the strong null hypothesis. To do so,
we would randomize each sequence of images (since the photos are collected across
multiple days) with the blocking described in the design. For each permutation,
we would compute the above statistics and draw a large number of samples to
assess the distribution and construct a p-value.

For the firmness and smell/taste test measures, we will construct the same
mean differences as described above. However, for the permutation test we will
conduct two randomizations to draw a sample. First, we will randomize the
treatments using the blocking described earlier. Then, we will randomize the
order of the samples given to the judges as well. (Does this make sense? How
do we want to conduct this randomization?)

Other considerations which should be discussed:
- Do we need a multiple testing procedure?
- How will we combine/interpret the results across outcomes?
- What exactly will be the randomization method for the smell/taste test?
- Anything else we should consider?

