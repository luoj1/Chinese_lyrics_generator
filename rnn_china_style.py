import tensorflow as tf
from vec import lrcLocal, vectorBuilderByFrequency, vectorBuilderByPinyin
import numpy as np
batch_size = 5
num_steps = 5
iteration = 50
layers = 4
learning_rate = 0.003
hidden_neuron = 1000
initializer = tf.random_uniform_initializer(-0.04,0.04)
out_address = '/tmp/model3/rnn_china_style'
charArray, _ = lrcLocal()
#print(charArray)
uniqueCharArray = list(set(charArray))
#print(uniqueCharArray)


#---------------------embed by pinyin--------------------------
'''# pinyin vector builder -> write vector to a file
charVector = list(vectorBuilderByPinyin(charArray))


with open('charVector4Pinyin.txt','w') as writer:
	for x in charVector:
		writer.write(str(x[0]))
		writer.write('\n')
		writer.write(str(x[1]))
		writer.write('\n')

'''
charVector = []
with open('charVector4Pinyin.txt','r') as read:
	r = read.readline()
	while r != None and r != '':
		temp_r =  r
		r = read.readline()
		charVector.append((int(temp_r[:-1]),int(r[:-1])))
		r = read.readline()


charVector = set(charVector)
charVector = [x for x in set(charVector)]


'''
#--------------------------embed by frequency--------------------------
charVector = list(vectorBuilderByFrequency(uniqueCharArray,charArray))
charVector.sort(key=lambda x: x[1]) #-> forfrequency
'''

dictionary = {}; dictionaryR = {}; vectionary = {}
i = 0
for (c, v) in charVector:
	#dictionary[chr(c)] = i
	vectionary[c] = v
	dictionary[c] = i
	dictionaryR[i] = chr(c)
	i = i +1



class Rnn_china_style:
	def __init__(self, isTraining = True, rep_of_generation = 10, pred_input = '在秋雪来时'):
		self.start = False
		with tf.Graph().as_default(),tf.Session() as sess:
			x = tf.placeholder(tf.int32, [batch_size,num_steps],name = 'lyricsX')
			y= tf.placeholder(tf.int32, [batch_size,num_steps], name = 'lyricsY')
			input_string = tf.placeholder(tf.int32, [batch_size], name = 'input_string')
			input_state = tf.placeholder(tf.float32,[batch_size,layers*hidden_neuron*2],name = 'input_state')

			self.lstm = tf.contrib.rnn.BasicLSTMCell(hidden_neuron,forget_bias=0.0, state_is_tuple=False)
			self.cell = tf.contrib.rnn.MultiRNNCell([self.lstm] * layers,state_is_tuple=False)
			#with tf.device("/cpu:0"):
			embedding = tf.get_variable("embed", [len(dictionary), hidden_neuron]) 
			embeddedx = tf.nn.embedding_lookup(embedding, x) 

			processed_embed = tf.placeholder(tf.float32, [batch_size,num_steps,hidden_neuron], name ='processed_embed')
			processed_embed4pred = tf.placeholder(tf.float32, [batch_size,hidden_neuron], name ='processed_embed')

			softmax_w = tf.get_variable("softmax_w", [hidden_neuron, len(dictionary)],initializer=initializer) 
			softmax_b = tf.get_variable("softmax_b", [len(dictionary)],initializer=initializer)
				
			def epoch(raw_data = charArray, dic = dictionary, batch_size = batch_size, num_steps = num_steps):
				raw_data = [dic[ord(char)] for char in raw_data]
				raw_data = np.array(raw_data, dtype=np.int32)  
				data_len = len(raw_data) 
				batch_len = data_len // batch_size 
				data = np.zeros([batch_size, batch_len], dtype=np.int32) 
				for i in range(batch_size): 
					data[i] = raw_data[batch_len * i:batch_len * (i + 1)] 
				epoch_size = (batch_len - 1) // num_steps 
				if epoch_size == 0:
					raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
				out = []
				for i in range(epoch_size):
					splitedx = data[:, i*num_steps:(i+1)*num_steps]
					splitedy = data[:, i*num_steps+1:(i+1)*num_steps+1] 
					out.append((splitedx, splitedy))
				return out

			def loss(pred, y):
				
				loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
						[pred],
						[tf.reshape(y, [-1])],
						[tf.ones([batch_size * num_steps])])
				

				print('----------loss func----------')
				print(loss)
				print(pred)
				print(y)
				print(batch_size * num_steps)
				return loss
			def cost(l):
				print('------cost------')
				rsum = tf.reduce_sum(l)/batch_size
				print(rsum)
				return rsum
			def Seq2Seq(inputs,neurons= hidden_neuron, layers = layers):
				with tf.variable_scope("RNN"):
					state = self.cell.zero_state(batch_size, tf.float32)
					outputs = [];
					for time_step in range(num_steps):
						if time_step > 0:
							tf.get_variable_scope().reuse_variables()
						
						print(str(inputs)) 
						(cell_output, state) = self.cell(inputs[:, time_step,:], state) 
						outputs.append(cell_output)

				output = tf.reshape(tf.concat(outputs, 1), [-1, neurons])

				#softmax_w = tf.get_variable("softmax_w", [neurons, len(dictionary)]) # initialized uniformly
				#softmax_b = tf.get_variable("softmax_b", [len(dictionary)]) # initialize uniformly 
				logits = tf.matmul(output, softmax_w) + softmax_b 
				print(str(logits))
				return logits
			
			#----------run prediction-------------
			with tf.variable_scope("RNN"):
				state_init = input_state
				em4prediction = tf.nn.embedding_lookup(embedding,input_string)
				em = tf.nn.dropout(processed_embed4pred, 0.5)
				(generated, state_after) = self.cell(em[:,:], state_init)
				generated = tf.reshape(tf.concat(generated, 1), [-1, hidden_neuron])	
				prediction = tf.matmul(generated, softmax_w) + softmax_b

			#-----------optimizer-------------
			optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name = 'Adam')
			logitVar = Seq2Seq(processed_embed)
			lossVar = loss(logitVar, y)
			c = cost(lossVar)
			optimizer =optimizer.minimize(c)


			self.start = True
			tf.global_variables_initializer().run()
			print('saver initing')
			model_saver = tf.train.Saver()
			if isTraining:
				
				tf.get_variable_scope().reuse_variables()


				for i in range(0,iteration):
					epoched = epoch()
					for (train,tag) in epoched:
						e = sess.run([embeddedx],{x: train})
						e = e[0]
						
						for b in range(0,batch_size):
							for n in range(0,num_steps):
								for m in range(0,hidden_neuron):
									e[b][n][m] = e[b][n][m]*vectionary[ord(dictionaryR[train[b][n]])]
						_, lossTemp,costTemp = sess.run([optimizer,lossVar,c],{x: train, processed_embed: e, y: tag})
						print(len(lossTemp))
						print(costTemp)
						print('running')
						model_saver.save(sess, out_address )
				model_saver.save(sess, out_address )
			else:
				model_saver.restore(sess,out_address )
				print('start making prediction')
				out = pred_input
				out = [ord(i) for i in out]
				out = [dictionary[char] for char in out]

				print(out)

				e = sess.run([em4prediction],{input_string:out})
				e = e[0]
				for b in range(0,batch_size):
					for m in range(0,hidden_neuron):
						e[b][m] = e[b][m]*vectionary[ord(dictionaryR[out[b]])]

				l, s = sess.run([prediction,state_after],{input_string:out, input_state: self.cell.zero_state(batch_size, tf.float32).eval(),processed_embed4pred: e})
				print(l)
				for i in range(0,20):
					visual= tf.nn.softmax(l)
					print(visual)
					visual = [p.tolist().index(max(p.tolist())) for p in visual.eval()]
					print(visual)
					out = [dictionaryR[i] for i in visual]
					print(out)
					out = [dictionary[ord(c)] for c in out]

					e = sess.run([em4prediction],{input_string:out})
					e = e[0]
					for b in range(0,batch_size):
						for m in range(0,hidden_neuron):
							selector = ord(dictionaryR[out[b]])
							e[b][m] = e[b][m]*vectionary[selector]
					l,s = sess.run([prediction,state_after],{input_string:out, input_state: s, processed_embed4pred: e})
				
Rnn_china_style(False)
