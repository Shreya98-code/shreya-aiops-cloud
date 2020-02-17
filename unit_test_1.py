import unittest
import Preprocess
import numpy as np

class TestMyModule(unittest.TestCase):

	def setUp(self):
		return

	def test_clean_text(self):

		tweet = "Great to have a record 50 patrons"

		result = Preprocess.clean_text(tweet)

		expected_result = "great record patrons"

		self.assertEqual(result, expected_result)

	def test_tokenize_text(self):

		cleaned_text = "great record patrons"

		result = Preprocess.tokenize_text(cleaned_text)

		expected_result = ['great', 'record', 'patrons']

		self.assertEqual(result, expected_result)

	def test_replace_token_with_index(self):

		token = ['great', 'record', 'patrons']

		result = Preprocess.replace_token_with_index(token)

		expected_result = [np.array([-8.4229e-01, 3.6512e-01, -3.8841e-01, -4.6118e-01,  2.4301e-01,\
        3.2412e-01,  1.9009e+00, -2.2630e-01, -3.1335e-01, -1.0970e+00,\
        -4.1494e-03,  6.2074e-01, -5.0964e+00,  6.7418e-01,  5.0080e-01,\
        -6.2119e-01,  5.1765e-01, -4.4122e-01, -1.4364e-01,  1.9130e-01,\
        -7.4608e-01, -2.5903e-01, -7.8010e-01,  1.1030e-01, -2.7928e-01],\
        dtype='float32'), np.array([ 1.0977e+00,  3.9901e-01,  4.9718e-01, -4.6284e-01,  5.2958e-01,\
        -1.2050e-03,  3.6909e-01,  4.9868e-03,  3.8203e-01, -1.0841e+00,\
        4.6041e-01, -3.3117e-01, -3.4869e+00,  7.9319e-01,  2.6638e-01,\
        4.3072e-02, -7.6477e-01,  2.6681e-02, -5.6201e-01, -5.4023e-01,\
        -9.8459e-01,  2.1062e-01,  9.1580e-01, -6.3914e-01, -3.6684e-01],\
        dtype='float32'), np.array([ 0.31555 ,  0.41271 , -0.60845 , -0.058075, -0.28236 , -0.069237,\
        -0.70555 , -2.0864  ,  0.22163 , -0.26702 ,  0.5538  , -0.81316 ,\
        -1.2556  ,  0.24543 , -0.033796,  0.67272 ,  0.39067 , -1.003   ,\
        -0.049941,  0.85592 ,  0.37391 , -0.28982 , -1.1796  , -1.2266  ,\
        -0.11825 ], dtype='float32')]

		self.assertEqual(result, expected_result)


	def test_pad_sequence(self):

		token_ind = [np.array([-8.4229e-01,  3.6512e-01, -3.8841e-01, -4.6118e-01,  2.4301e-01,\
        3.2412e-01,  1.9009e+00, -2.2630e-01, -3.1335e-01, -1.0970e+00,\
        -4.1494e-03,  6.2074e-01, -5.0964e+00,  6.7418e-01,  5.0080e-01,\
        -6.2119e-01,  5.1765e-01, -4.4122e-01, -1.4364e-01,  1.9130e-01,\
        -7.4608e-01, -2.5903e-01, -7.8010e-01,  1.1030e-01, -2.7928e-01],\
        dtype='float32'), np.array([ 1.0977e+00,  3.9901e-01,  4.9718e-01, -4.6284e-01,  5.2958e-01,\
        -1.2050e-03,  3.6909e-01,  4.9868e-03,  3.8203e-01, -1.0841e+00,\
        4.6041e-01, -3.3117e-01, -3.4869e+00,  7.9319e-01,  2.6638e-01,\
        4.3072e-02, -7.6477e-01,  2.6681e-02, -5.6201e-01, -5.4023e-01,\
        -9.8459e-01,  2.1062e-01,  9.1580e-01, -6.3914e-01, -3.6684e-01],\
        dtype='float32'), np.array([ 0.31555 ,  0.41271 , -0.60845 , -0.058075, -0.28236 , -0.069237,\
        -0.70555 , -2.0864  ,  0.22163 , -0.26702 ,  0.5538  , -0.81316 ,\
        -1.2556  ,  0.24543 , -0.033796,  0.67272 ,  0.39067 , -1.003   ,\
        -0.049941,  0.85592 ,  0.37391 , -0.28982 , -1.1796  , -1.2266  ,\
        -0.11825 ], dtype='float32')]

		result = Preprocess.pad_sequence()

		expected_result = [np.array([-8.4229e-01,  3.6512e-01, -3.8841e-01, -4.6118e-01,  2.4301e-01,
        3.2412e-01,  1.9009e+00, -2.2630e-01, -3.1335e-01, -1.0970e+00,
       	-4.1494e-03,  6.2074e-01, -5.0964e+00,  6.7418e-01,  5.0080e-01,
       	-6.2119e-01,  5.1765e-01, -4.4122e-01, -1.4364e-01,  1.9130e-01,
       	-7.4608e-01, -2.5903e-01, -7.8010e-01,  1.1030e-01, -2.7928e-01],dtype='float32'),
       	np.array([ 1.0977e+00,  3.9901e-01,  4.9718e-01, -4.6284e-01,  5.2958e-01,
       	-1.2050e-03,  3.6909e-01,  4.9868e-03,  3.8203e-01, -1.0841e+00,
        4.6041e-01, -3.3117e-01, -3.4869e+00,  7.9319e-01,  2.6638e-01,
        4.3072e-02, -7.6477e-01,  2.6681e-02, -5.6201e-01, -5.4023e-01,
       	-9.8459e-01,  2.1062e-01,  9.1580e-01, -6.3914e-01, -3.6684e-01],
      	dtype='float32'), np.array([ 0.31555 ,  0.41271 , -0.60845 , -0.058075, -0.28236 , -0.069237,
       	-0.70555 , -2.0864  ,  0.22163 , -0.26702 ,  0.5538  , -0.81316 ,
       	-1.2556  ,  0.24543 , -0.033796,  0.67272 ,  0.39067 , -1.003   ,
       	-0.049941,  0.85592 ,  0.37391 , -0.28982 , -1.1796  , -1.2266  ,
       	-0.11825 ], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32'), np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       	0., 0., 0., 0., 0., 0., 0., 0.], dtype='float32')]

		self.assertEqual(result, expected_result)





