import pynvml as nv
nv.nvmlInit()
for i in range(nv.nvmlDeviceGetCount()):
	hndl = nv.nvmlDeviceGetHandleByIndex(i)
	if not nv.nvmlDeviceGetComputeRunningProcesses(hndl):
		visable_device = str(i)
		break
nv.nvmlShutdown()

import tensorflow as tf
import os
try:
	os.environ["CUDA_VISIBLE_DEVICES"] = visable_device
except:
	print ('No available gpu')
	exit()
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=100)
import tensorflow.contrib.layers as ly
from mnist import read_data_sets
from mnist import dense_to_one_hot

clabel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
dataset = read_data_sets('MNIST_data', one_hot=False, validation_size=0, clabel=clabel)

n_rdims = 19
n_classes = 20
n_features = 32
z_dim = 128
n_generator = 10

batch_size = 32
eigenThreshold = 0.0001
qrThreshold = 0.0001
residueThreshold = 0.0001
EVALS_UPPER_BOUND = 1
ILDA_DECAY = 0.998

learning_rate_ger = 5e-5
learning_rate_dis = 5e-5
device = '/gpu:0'

run_id = 'MNIST_CON'
log_dir = 'log/' + run_id
ckpt_dir = 'ckpt/' + run_id
if not os.path.exists(ckpt_dir):
	os.makedirs(ckpt_dir)

start_iter_step = 0
max_iter_step = 10000

def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
	if weight_decay > 0:
		regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
	else:
		regularizer = None
	return tf.get_variable(name,
												 shape=shape,
												 initializer=initializer,
												 dtype=dtype,
												 regularizer=regularizer,
												 trainable=trainable)

def giff_Sb_Sw(data, label, n_classes):
	NoS = tf.cast(tf.shape(data)[0], dtype=tf.float32)
	_label, _idx, _count = tf.unique_with_counts(label)
	classNoS_ = []
	for idx in range(n_classes):
		nos_ = tf.boolean_mask(_count, tf.equal(_label, idx))
		nos_ = tf.cond(tf.size(nos_)>0, lambda: nos_, lambda: tf.constant([0]))
		classNoS_.append(nos_)
	classNoS = tf.reshape(tf.cast(tf.stack(classNoS_), tf.float32), [n_classes])

	x_gt = tf.dynamic_partition(data, _idx, num_partitions=n_classes)
	x_g_barm = []
	x_gm =[]
	for x_g in x_gt:
		x_g_bar = tf.reduce_mean(x_g, axis=0)
		x_g_barm.append(x_g_bar)
		x_g_res = x_g - x_g_bar
		x_gm.append( (tf.matmul(x_g_res, x_g_res, transpose_a=True)) )
	_x_gm = tf.stack(x_gm)
	S_w = tf.reduce_sum(_x_gm, axis=0)

	_x_g_barm = tf.stack(x_g_barm)
	classMean_ = []
	for idx in range(n_classes):
		csm_ = tf.boolean_mask(_x_g_barm, tf.equal(_label, idx))
		csm_ = tf.cond(tf.size(csm_)>0, lambda: csm_, lambda: tf.constant(0.0, shape=(1, n_features)))
		classMean_.append(csm_)
	classMean = tf.reshape(tf.stack(classMean_), [n_classes, n_features])
	mean = tf.reduce_mean(data, axis=0)

	# S_b
	sqrt_count = tf.expand_dims(tf.sqrt(classNoS), -1)
	meanDiff_ = (classMean - mean) * sqrt_count
	S_b = tf.matmul(meanDiff_, meanDiff_, transpose_a=True)

	# # S_t
	# Xt_bar = data - tf.reduce_mean(data, axis=0)
	# m = tf.cast(tf.shape(Xt_bar)[0], dtype=tf.float32)
	# S_t = tf.matmul(Xt_bar, Xt_bar, transpose_a=True)/(m)
	# S_b = S_t - S_w + tf.eye(tf.shape(S_w)[0], tf.shape(S_w)[1])*2e-5

	return S_b, S_w, mean, classMean, NoS, classNoS

def ilda_update(data, label, saved, inputs, n_classes):
	savedSb, savedSw, savedMean, savedClassMean, savedNoS, savedClassNoS = saved
	inputSb, inputSw, inputMean, inputClassMean, inputNoS, inputClassNoS = inputs

	# UPDATE NoS
	updatedNoS = inputNoS + savedNoS
	updatedClassNoS = inputClassNoS + savedClassNoS

	# UPDATE MEAN
	updatedMean = tf.truediv((tf.multiply(tf.cast(inputNoS, tf.float32), inputMean) + \
														tf.multiply(tf.cast(savedNoS, tf.float32), savedMean)), 
														tf.cast(updatedNoS, tf.float32))

	updatedClassMean = (tf.expand_dims(inputClassNoS, -1) * inputClassMean + \
											tf.expand_dims(savedClassNoS, -1) * savedClassMean) / \
											tf.expand_dims(updatedClassNoS, -1)

	# UPDATE Sb
	sqrt_count = tf.expand_dims(tf.sqrt(updatedClassNoS), -1)
	meanDiff_ = (updatedClassMean - updatedMean) * sqrt_count
	updatedSb = tf.matmul(meanDiff_, meanDiff_, transpose_a=True)

	# UPDATE Sw
	n = tf.cast(savedClassNoS, tf.float32)
	l = tf.cast(inputClassNoS, tf.float32)
	_label, _idx, _count = tf.unique_with_counts(label)
	batch_NoS = tf.size(_label)
	inputDataClassGroup = tf.dynamic_partition(data, _idx, num_partitions=n_classes)

	term2LHS = tf.sqrt(n * l * l) / (n + l)
	term2RHS = (inputClassMean - savedClassMean) * tf.expand_dims(term2LHS, -1)
	term2 = tf.matmul(term2RHS, term2RHS, transpose_a=True)

	term3LHS = n * n / ((n + l) * (n + l))
	def _term3append(idx):
		inputDataDiff = inputDataClass - savedClassMean[_label[idx]]
		term3RHS = tf.matmul(inputDataDiff, inputDataDiff, transpose_a=True)
		return term3RHS * term3LHS[_label[idx]]

	term3Class = []
	for idx, inputDataClass in enumerate(inputDataClassGroup):
		term3append = tf.cond(idx<batch_NoS, lambda: _term3append(idx), lambda: tf.zeros([n_features, n_features], tf.float32))
		term3Class.append(term3append)
	term3 = tf.reduce_sum(tf.stack(term3Class), axis=0)

	term4LHS = l * (l + 2 * n) / ((n + l) * (n + l))
	def _term4append(idx):
		inputDataDiff = inputDataClass - inputClassMean[_label[idx]]
		term4RHS = tf.matmul(inputDataDiff, inputDataDiff, transpose_a=True)
		return term4RHS * term4LHS[_label[idx]]

	term4Class = []
	for idx, inputDataClass in enumerate(inputDataClassGroup):
		term4append = tf.cond(idx<batch_NoS, lambda: _term4append(idx), lambda: tf.zeros([n_features, n_features], tf.float32))
		term4Class.append(term4append)
	term4 = tf.reduce_sum(tf.stack(term4Class), axis=0)

	updatedSw = savedSw + term2 + term3 + term4

	return updatedSb, updatedSw, updatedMean, updatedClassMean, updatedNoS, updatedClassNoS

def giff_evals_evecs(S_b, S_w, n_rdims):
	evals_b, evecs_b = tf.self_adjoint_eig(S_b)
	evalsRest_b = tf.boolean_mask(evals_b, tf.greater(evals_b, eigenThreshold))
	evecsRest_b = evecs_b[:,-tf.shape(evalsRest_b)[0]:]
	evals_bh = tf.diag(tf.sqrt(evalsRest_b))
	S_bh = tf.matmul(tf.matmul(evecsRest_b, evals_bh), evecsRest_b, transpose_b=True)

	# S_w += tf.eye(tf.shape(S_w)[0], tf.shape(S_w)[1])*1e-4
	S_wi = tf.matrix_inverse(S_w)
	S_s = tf.matmul(tf.matmul(S_bh, S_wi), S_bh) + tf.eye(tf.shape(S_b)[0], tf.shape(S_b)[1])*1e-5
	evals, evecs = tf.self_adjoint_eig(S_s)

	evalsRest_t = tf.boolean_mask(evals, tf.greater(evals, eigenThreshold))
	evalsRest_r = evals[-n_rdims:]
	evalsRest = tf.cond(tf.greater(tf.shape(evalsRest_t)[0], n_rdims), 
											lambda: evalsRest_r, lambda: evalsRest_t)
	evecsRest = evecs[:,-tf.shape(evalsRest)[0]:]

	return evalsRest, evecsRest

def ilda_model(z_l, y_l, n_classes, update=True, decay=0.998):
	savedSw					= _get_variable('savedSw', [n_features, n_features], 
																	tf.constant_initializer(0.0), trainable=False)
	savedSb					= _get_variable('savedSb', [n_features, n_features], 
																	tf.constant_initializer(0.0), trainable=False)
	savedMean				= _get_variable('savedMean', [n_features,], 
																	tf.constant_initializer(0.0), trainable=False)
	savedClassMean	= _get_variable('savedClassMean', [n_classes, n_features], 
																	tf.constant_initializer(0.0), trainable=False)
	savedNoS				= _get_variable('savedNoS', [], 
																	tf.constant_initializer(0.0), trainable=False)
	savedClassNoS		= _get_variable('savedClassNoS', [n_classes,], 
																	tf.constant_initializer(0.0), trainable=False)

	data = z_l # [n_samples, n_features]
	label = tf.cast(y_l, tf.int32) # [n_samples, ]

	inputSb, inputSw, inputMean, inputClassMean, inputNoS, inputClassNoS = \
			giff_Sb_Sw(data, label, n_classes)

	updatedSb, updatedSw, updatedMean, updatedClassMean, updatedNoS, updatedClassNoS = \
			ilda_update(data, label,
									[savedSb, savedSw, savedMean, savedClassMean, savedNoS, savedClassNoS], 
									[inputSb, inputSw, inputMean, inputClassMean, inputNoS, inputClassNoS],
									n_classes)

	if update:
		with tf.control_dependencies(
											[savedSw.assign(updatedSw*0.9995), savedSb.assign(updatedSb),
											 savedMean.assign(updatedMean), savedClassMean.assign(updatedClassMean),
											 savedNoS.assign(updatedNoS*decay), savedClassNoS.assign(updatedClassNoS*decay)]):
			return tf.identity(updatedSb), tf.identity(updatedSw), tf.identity(updatedClassMean)
	else:
		return updatedSb, updatedSw, updatedClassMean

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def generator_conv(z, var_scope='generator', reuse=False):
	with tf.variable_scope(var_scope):
		if reuse:
			scope.reuse_variables()

		train = ly.fully_connected(
									z, 4 * 4 * 512, activation_fn=lrelu, 
									normalizer_fn=ly.batch_norm)
		train = tf.reshape(train, (-1, 4, 4, 512))

		train = ly.conv2d_transpose(train, 256, 3, stride=2, 
									activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, 
									padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
		train = ly.conv2d_transpose(train, 128, 3, stride=2,
									activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', 
									weights_initializer=tf.random_normal_initializer(0, 0.02))
		train = ly.conv2d_transpose(train, 64, 3, stride=2,
									activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', 
									weights_initializer=tf.random_normal_initializer(0, 0.02))
		train = ly.conv2d_transpose(train, 1, 3, stride=1,
									activation_fn=tf.nn.tanh, padding='SAME', 
									weights_initializer=tf.random_normal_initializer(0, 0.02))
	return train

def critic_conv(img, var_scope='critic', reuse=False):
	with tf.variable_scope(var_scope) as scope:
		if reuse:
			scope.reuse_variables()

		size = 64
		img = ly.conv2d(img, num_outputs=size, kernel_size=3,
									stride=2, activation_fn=lrelu)
		img = ly.conv2d(img, num_outputs=size * 2, kernel_size=3,
									stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
		img = ly.conv2d(img, num_outputs=size * 4, kernel_size=3,
									stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
		img = ly.conv2d(img, num_outputs=size * 8, kernel_size=3,
									stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
		u = ly.fully_connected(tf.reshape(img, [-1, size*32]), n_features, 
									activation_fn=None)

	return u

def ilda_eval(u, y, n_classes, var_scope='ilda', reuse=False, update=True, decay=0.998):
	with tf.variable_scope(var_scope) as scope:
		if reuse:
			scope.reuse_variables()

		n_rdims = n_classes-1
		Sb, Sw, mean = ilda_model(u, y, n_classes, update, decay)
		evals, evecs = giff_evals_evecs(Sb, Sw, n_rdims)

	return evals[-n_rdims:], evecs, mean

def distance_hyperplain(data, mean, evecs):
	coef_ = tf.matmul(tf.matmul(mean, evecs), evecs, transpose_b=True)
	intercept_ = -0.5 * tf.diag_part(tf.matmul(mean, coef_, transpose_b=True))
	distance = tf.matmul(data, coef_, transpose_b=True) + intercept_
	return distance

def build_graph():
	in_data = tf.placeholder(dtype=tf.float32, shape=(batch_size*n_generator, 28, 28, 1))
	real_y = tf.placeholder(dtype=tf.float32, shape=(batch_size*n_generator,))
	real_data = tf.pad(in_data, [[0,0],[2,2],[2,2],[0,0]])

	real_u = critic_conv(real_data, var_scope='critic')

	z = tf.placeholder(tf.float32, shape=(batch_size * n_generator, z_dim))
	fake_data = generator_conv(z, var_scope='generator')
	fake_y = []
	for n in range(n_generator):
		fake_y.append(tf.constant(n+n_generator, shape=[batch_size], dtype=tf.float32))

	fake_u = critic_conv(fake_data, var_scope='critic', reuse=True)
	fake_u = tf.split(fake_u, num_or_size_splits=n_generator, axis=0)

	evals, evecs, mean = ilda_eval(tf.concat(fake_u + [real_u], axis=0),
																 tf.concat(fake_y + [real_y], axis=0),
																 n_classes=n_classes, 
																 decay=ILDA_DECAY)
	
	lda_bound = evals[0]+EVALS_UPPER_BOUND
	lda_evals = tf.boolean_mask(evals, tf.greater(lda_bound, evals))
	c_loss = -tf.reduce_sum(lda_evals)

	with tf.variable_scope('evals'):
		for i in range(n_rdims):
			eval_sm = tf.summary.scalar('eval_{}'.format(i), evals[i])

	g_loss = 0

	for n in range(n_generator):
		dis_hp = distance_hyperplain(fake_u[n], mean, evecs)
		v = dis_hp - tf.expand_dims(dis_hp[:, n], -1)
		g_loss += tf.reduce_sum(v)
		g_loss -= tf.reduce_sum(dis_hp[:,n])

	g_loss_sum = tf.summary.scalar("g_loss", g_loss)
	c_loss_sum = tf.summary.scalar("c_loss", c_loss)

	img_sum = []
	for n in range(n_generator):
		img_sum.append(tf.summary.image("img"+str(n), fake_data[n*batch_size:(n+1)*batch_size], max_outputs=10))

	theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
	theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

	is_adam = True
	counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
	opt_g = ly.optimize_loss(loss=g_loss, learning_rate=learning_rate_ger,
					optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer, 
					variables=theta_g, global_step=counter_g)#, summaries = ['gradient_norm'])
	counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
	opt_c = ly.optimize_loss(loss=c_loss, learning_rate=learning_rate_dis,
					optimizer=tf.train.AdamOptimizer if is_adam is True else tf.train.RMSPropOptimizer, 
					variables=theta_c, global_step=counter_c)#, summaries = ['gradient_norm'])

	return opt_g, opt_c, z, in_data, real_y

def main():
	with tf.device(device):
		opt_g, opt_c, z, in_data, real_y = build_graph()
	merged_all = tf.summary.merge_all()
	saver = tf.train.Saver()
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.8

	def next_feed_dict():
		train_batch = dataset.train.next_batch(batch_size*n_generator)
		train_img = train_batch[0]
		train_label = train_batch[1]
		train_img = 2*train_img-1
		train_img = np.reshape(train_img, [batch_size*n_generator, 28, 28])
		train_img = np.expand_dims(train_img, -1)
		batch_v = np.random.normal(0, 1, [batch_size * n_generator, z_dim-n_generator]).astype(np.float32)
		lt = np.mgrid[0:n_generator,0:batch_size][0].flatten()
		batch_l = np.zeros((batch_size*n_generator, n_generator))
		batch_l[np.arange(batch_size*n_generator), lt] = 1
		batch_z = np.concatenate((batch_v, batch_l), axis=1)

		feed_dict = {in_data: train_img, z: batch_z, real_y: train_label}
		return feed_dict

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		# saver = tf.train.Saver()
		# saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

		run_options = tf.RunOptions(
			trace_level=tf.RunOptions.FULL_TRACE)
		run_metadata = tf.RunMetadata()

		for j in range(1500):
			sess.run([opt_c], feed_dict=next_feed_dict())

		iter_d = 0.0
		iter_g = 0.0

		i = start_iter_step
		while i<max_iter_step:
			if i%50 == 0:
				_, merged = sess.run([opt_c, merged_all], options=run_options, run_metadata=run_metadata, feed_dict=next_feed_dict())
				summary_writer.add_summary(merged, i)
				summary_writer.add_run_metadata(run_metadata, 'critic_metadata {}'.format(i), i)
			else:
				sess.run([opt_c], feed_dict=next_feed_dict())

			iter_d += 1
			while iter_d > 0:
				sess.run([opt_c], feed_dict=next_feed_dict())
				iter_d -= 1

			iter_g += 2
			while iter_g > 0:
				sess.run([opt_g], feed_dict=next_feed_dict())
				iter_g -= 1

			if i % 1000 == 999:
				saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=i)
			i += 1

main()
