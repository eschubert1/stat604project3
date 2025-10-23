
### Design Overview

For this study, we purchased 3 pots of basil and harvested stalks from each pot.
We then randomized the stalks to each of 4 treatments, blocking by pot so that
each treatment received at least one stalk from each pot. The four treatments 
were the 2x2 combinations of placing the basil in the fridge and placing 
the stalks in water. Each day, photos were taken of each of the plants. 
At the end of the study (5 days), we will conduct a blinded freshness assessment
where one member of the group will present leaves from the plants to the
other three participants in a randomized order to be judged on the firmness of
the leaves and the apparent freshness of the herb based on smell/taste. 
Ther are 3 plants, and each plant will receive a score from 1 to 5 
by each judge based on these qualities, so that each plant has 3 scores.

Additionally, soon after the start of the experiment we noticed that the stems 
placed in the fridge were wilting significantly, and we were concerned that this
may reduce the power of the comparison for the water treatment. Consequently,
we chose to purchase 3 more pots of basil and conduct another experiment with
only the water treatment (no plants were placed in the fridge). The treatment
assignment was blocked by pot, and within each pot 4 stems were assigned the
water treatment and 4 were given nothing. Both experiments were conducted for 5 days.

### Analysis Plan

Before conducting a formal test, we will do an exploratory data analysis
to understand the outcome data. We plan to make boxplots comparing
the average Likert scores for each treatment from the freshness test, as
well as for the other outcomes measured, but more visualizations may
be created if necessary.

#### Permutation Testing
Since 'freshness' is a subjective quality, our primary method of comparison
is the freshness rating from the blind judging, which we feel would most
reliably reflect the likelihood a person would be willing to cook with
or consume the basil. To assess the freshness for each plant, we will
average the scores given by the 3 judges for that plant. Then, we will
average those means across the plants within a treatment group to obtain
a freshness score for the treatment group. We will test for a treatment effect 
with both the water and fridge treatments, conditional on the presence of the 
other treatment, using the difference of the average scores for the two groups,
using a permutation test.

For testing the fridge treatment, we will permute the average scores for
the two treatments (fridge vs no fridge) conditioning on the treatment
assignment for the water treatment and blocked by pot. For a given level
of the water treatment and a given pot, the fridge treatment assignment
labels will be permuted for all the experimental units within that group,
and after a full permutation is computed, the difference in average
scores between the levels of the fridge treatment will be computed. We
will do this for 100,000 permutations to simulate the strong null
distribution and compute a p-value for the actual outcome of the experiment.

For testing the water treatment, we will do the same as the above but
swap the roles of the water and fridge treatments. We will also pool
the data for both experiments, blocking by pot as before.

Since we are conducting two tests, we will use a Bonferroni adjustment
to account for the multiplicity. The presence or absence of a
treatment effect for the water and fridge treatments will be judged
on these results. However, we also plan to conduct further
permutation tests on the other outcomes as a supplement to compare
and contrast the results from different outcomes.

#### Image Analysis
To analyze the images, we will process each image so that the background is white,
and then use a segmentation algorithm to determine the portion of the image
corresponding to plant matter. We will then compute a metric that spots high RGB values and saturation and low value of the B channel in the LAB color space of
all the pixels in this image, and subtract this from the average value
of the plant at day 0 (the start of the experiment).

We will conduct a permutation test with the same randomization scheme as
described above, but using the difference in average RGB changes as
the test statistic.

#### Weight analysis
We will conduct a permutation test on the difference in average
weight changes as well. However, this may have low power due
to difficulty in accurately measuring the weight of the stems,
which are very light.

