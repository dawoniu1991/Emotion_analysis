# coding: utf-8

import numpy as np
wordsList = np.load('./training_data/wordsList.npy')
wordsList = wordsList.tolist() #
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('./training_data/wordVectors.npy')



print(len(wordsList))
print(wordVectors.shape)



import tensorflow as tf
maxSeqLength = 10 #句子最大长度
numDimensions = 50 #词向量维度
firstSentence = np.zeros((maxSeqLength), dtype='int32')
firstSentence[0] = wordsList.index("i")
firstSentence[1] = wordsList.index("thought")
firstSentence[2] = wordsList.index("the")
firstSentence[3] = wordsList.index("movie")
firstSentence[4] = wordsList.index("was")
firstSentence[5] = wordsList.index("incredible")
firstSentence[6] = wordsList.index("and")
firstSentence[7] = wordsList.index("inspiring")

print(firstSentence.shape)
print(firstSentence)



with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)



from os import listdir
from os.path import isfile, join
positiveFiles = ['./training_data/positiveReviews/' + f for f in listdir('./training_data/positiveReviews/') if isfile(join('./training_data/positiveReviews/', f))]
negativeFiles = ['./training_data/negativeReviews/' + f for f in listdir('./training_data/negativeReviews/') if isfile(join('./training_data/negativeReviews/', f))]
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)       


for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)  


numFiles = len(numWords)
print('总文件数目', numFiles)
print('所有文件中单词总和', sum(numWords))
print('每个文件中的平均单词数目', sum(numWords)/len(numWords))



import matplotlib.pyplot as plt
plt.hist(numWords, 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 1200, 0, 8000])
plt.show()


# 从直方图和句子的平均单词数分布看出，可以将句子最大长度设置为 250 。



maxSeqLength = 250

#查看一条文件数据评论内容
fname = positiveFiles[3]
with open(fname) as f:
    for lines in f:
        print(lines)

# 删除标点符号、括号、问号等，只留下字母数字字符
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())



firstFile = np.zeros((maxSeqLength), dtype='int32')
with open(fname) as f:
    indexCounter = 0
    line=f.readline()
    cleanedLine = cleanSentences(line)
    split = cleanedLine.split()
    for word in split:
        try:
            firstFile[indexCounter] = wordsList.index(word)
        except ValueError:
            firstFile[indexCounter] = 399999 #
        indexCounter = indexCounter + 1


#用相同的方法来处理全部的 25000 条评论，导入电影训练集，得到一个 25000 * 250 的矩阵

# ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
# fileCounter = 0
# for pf in positiveFiles:
#    with open(pf, "r") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1 

# for nf in negativeFiles:
#    with open(nf, "r") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1 
# #Pass into embedding function and see if it evaluates. 

# np.save('idsMatrix', ids)


#上述过程耗时太长，直接导入相关矩阵
#ids维度为(25000, 250)，对于25000条数据，取每一条数据的前250个单词，多出的截取掉，不足250单词的填充0
ids = np.load('./training_data/idsMatrix.npy')



from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

#构建我们的 TensorFlow 图模型，定义一些超参数

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 50000
numDimensions=50


import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)


lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)


weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
#取最终的结果值
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)


# 接下来，我们需要定义正确的预测函数和正确率评估参数。正确的预测形式是查看最后输出的0-1向量是否和标记的0-1向量相同。
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


# 使用交叉熵损失函数来作为损失值。对于优化器，我们选择 Adam，并且采用默认的学习率。

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):

    nextBatch, nextBatchLabels = getTrainBatch();
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels}) 
    
    if (i % 1000 == 0 and i != 0):
        loss_ = sess.run(loss, {input_data: nextBatch, labels: nextBatchLabels})
        accuracy_ = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
        
        print("iteration {}/{}...".format(i+1, iterations),
              "loss {}...".format(loss_),
              "accuracy {}...".format(accuracy_))    

    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)







sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

#训练完在笔记本上要两个半小时


iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
    
    
 #训练及测试的输出结果
'''
iteration 1001/50000... loss 0.6754631996154785... accuracy 0.5416666865348816...
iteration 2001/50000... loss 0.6050778031349182... accuracy 0.7083333134651184...
iteration 3001/50000... loss 0.590325653553009... accuracy 0.625...
iteration 4001/50000... loss 0.7435455322265625... accuracy 0.4166666567325592...
iteration 5001/50000... loss 0.6158140301704407... accuracy 0.7083333134651184...
iteration 6001/50000... loss 0.7227487564086914... accuracy 0.5833333134651184...
iteration 7001/50000... loss 0.6658935546875... accuracy 0.5416666865348816...
iteration 8001/50000... loss 0.6276681423187256... accuracy 0.7083333134651184...
iteration 9001/50000... loss 0.6576926112174988... accuracy 0.6666666865348816...
iteration 10001/50000... loss 0.6462262272834778... accuracy 0.4583333432674408...
saved to models/pretrained_lstm.ckpt-10000
iteration 11001/50000... loss 0.6592984795570374... accuracy 0.625...
iteration 12001/50000... loss 0.6692590117454529... accuracy 0.5...
iteration 13001/50000... loss 0.4181993007659912... accuracy 0.75...
iteration 14001/50000... loss 0.48044511675834656... accuracy 0.75...
iteration 15001/50000... loss 0.326813280582428... accuracy 0.875...
iteration 16001/50000... loss 0.2446471005678177... accuracy 0.9583333134651184...
iteration 17001/50000... loss 0.39793500304222107... accuracy 0.75...
iteration 18001/50000... loss 0.439182847738266... accuracy 0.7916666865348816...
iteration 19001/50000... loss 0.5187193751335144... accuracy 0.7916666865348816...
iteration 20001/50000... loss 0.31241968274116516... accuracy 0.875...
saved to models/pretrained_lstm.ckpt-20000
iteration 21001/50000... loss 0.26987889409065247... accuracy 0.9166666865348816...
iteration 22001/50000... loss 0.39287447929382324... accuracy 0.9166666865348816...
iteration 23001/50000... loss 0.29798710346221924... accuracy 0.875...
iteration 24001/50000... loss 0.17066310346126556... accuracy 0.9583333134651184...
iteration 25001/50000... loss 0.2126162052154541... accuracy 0.9166666865348816...
iteration 26001/50000... loss 0.15289545059204102... accuracy 0.9583333134651184...
iteration 27001/50000... loss 0.1388673335313797... accuracy 0.875...
iteration 28001/50000... loss 0.45575758814811707... accuracy 0.8333333134651184...
iteration 29001/50000... loss 0.11559172719717026... accuracy 0.9583333134651184...
iteration 30001/50000... loss 0.3195440471172333... accuracy 0.875...
saved to models/pretrained_lstm.ckpt-30000
iteration 31001/50000... loss 0.04661552980542183... accuracy 0.9583333134651184...
iteration 32001/50000... loss 0.03961930051445961... accuracy 1.0...
iteration 33001/50000... loss 0.08754289150238037... accuracy 0.9583333134651184...
iteration 34001/50000... loss 0.29227975010871887... accuracy 0.9166666865348816...
iteration 35001/50000... loss 0.12960360944271088... accuracy 0.9583333134651184...
iteration 36001/50000... loss 0.03288501128554344... accuracy 1.0...
iteration 37001/50000... loss 0.018847020342946053... accuracy 1.0...
iteration 38001/50000... loss 0.02817232348024845... accuracy 0.9583333134651184...
iteration 39001/50000... loss 0.04794586822390556... accuracy 1.0...
iteration 40001/50000... loss 0.09287450462579727... accuracy 0.9166666865348816...
saved to models/pretrained_lstm.ckpt-40000
iteration 41001/50000... loss 0.03584953397512436... accuracy 1.0...
iteration 42001/50000... loss 0.007370330393314362... accuracy 1.0...
iteration 43001/50000... loss 0.09359293431043625... accuracy 0.9583333134651184...
iteration 44001/50000... loss 0.03562663868069649... accuracy 1.0...
iteration 45001/50000... loss 0.007080310955643654... accuracy 1.0...
iteration 46001/50000... loss 0.018065297976136208... accuracy 1.0...
iteration 47001/50000... loss 0.025063343346118927... accuracy 1.0...
iteration 48001/50000... loss 0.0035159215331077576... accuracy 1.0...
iteration 49001/50000... loss 0.007980610243976116... accuracy 1.0...
Accuracy for this batch: 91.6666686535
Accuracy for this batch: 75.0
Accuracy for this batch: 95.8333313465
Accuracy for this batch: 87.5
Accuracy for this batch: 95.8333313465
Accuracy for this batch: 75.0
Accuracy for this batch: 79.1666686535
Accuracy for this batch: 70.8333313465
Accuracy for this batch: 83.3333313465
Accuracy for this batch: 95.8333313465
'''
