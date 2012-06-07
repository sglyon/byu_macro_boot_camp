"""
Created May 25, 2012

Author: Spencer Lyon
"""
import numpy as np
import scipy.stats as st

data_1 = np.array([0,1,2,0,1,0,3])
data_2 = np.array([1,3,2,2,0,1,1])
data_3 = np.hstack([data_1, data_2])

prior_a = .3
prior_b = 4.

post_a_1 = np.sum(data_1) + prior_a
post_b_1 = prior_b/(data_1.size*prior_b +1)
sample_1 = st.gamma.rvs(post_a_1, scale = post_b_1, size = 10000)
sort_sample_1 = np.sort(sample_1)
c_i_1 = [sort_sample_1[sample_1.size*.025], sort_sample_1[sample_1.size *.95]]
expected_1 = post_a_1 * post_b_1

print 'For the first week we have the following data: \n', 'Scale Parameter: ',\
    post_a_1, '\n', 'Shape parameter: ', post_b_1, '\n', 'Expected sightings: '\
    ,expected_1, '\n', "95% Confidence Interval: ", c_i_1

post_a_2 = np.sum(data_2) + post_a_1
post_b_2 = post_b_1/(data_2.size*post_b_1 +1)
sample_2 = st.gamma.rvs(post_a_2, scale = post_b_2, size = 10000)
sort_sample_2 = np.sort(sample_2)
c_i_2 = [sort_sample_2[sample_2.size*.025], sort_sample_2[sample_2.size *.95]]
expected_2 = post_a_2 * post_b_2

print '\n', '\n'
print 'For the second week we have the following data: \n', 'Scale Parameter: ',\
    post_a_2, '\n', 'Shape parameter: ', post_b_2, '\n', 'Expected sightings: '\
    ,expected_2, '\n', "95% Confidence Interval: ", c_i_2


post_a_3 = np.sum(data_3) + prior_a
post_b_3 = prior_b/(data_3.size*prior_b +1)
sample_3 = st.gamma.rvs(post_a_3, scale = post_b_3, size = 10000)
sort_sample_3 = np.sort(sample_3)
c_i_3 = [sort_sample_3[sample_3.size*.025], sort_sample_3[sample_3.size *.95]]
expected_3 = post_a_3 * post_b_3

print '\n', '\n'
print 'For both weeks we have the following data: \n', 'Scale Parameter: ',\
    post_a_3, '\n', 'Shape parameter: ', post_b_3, '\n', 'Expected sightings: '\
    ,expected_3, '\n', "95% Confidence Interval: ", c_i_3


# For problem 4 the posterior is a gamme with shape = x +.3 and scalse = 4/5

