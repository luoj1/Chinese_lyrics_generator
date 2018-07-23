import requests
#import tensorflow as tf 
import glob
import io
import os 
import numpy as np
import json
import re
def openSuggestion(song):
	try:
		j = requests.get('http://sug.music.baidu.com/info/suggestion?format=json&version=2&from=0&word='+song+'&_=1405404358299').text
		j = json.loads(j)
		j = j["data"]["song"][0]["songid"]
		return j
	except:
		print('no '+song)
		return None

def extractlyrics(id):
	j = getlrcbyid(id).text
	j = json.loads(j)

	try:
		if j["error_code"] != None:
			return None, None
	except:
		print("no error code return")
	#print("song: " + j["lrcContent"])
	content = io.StringIO(j["lrcContent"])
	temp = content.readline()
	out = []
	track = 0
	while temp + 'x' != 'x' :
		if track > 7:
			last = getlast(temp)
			temp = temp[last+1:len(temp)-2]
			if temp!='':
				out = out + temp.split(' ')
		temp = content.readline()
		track += 1
	separated = []
	for i in range(0, len(out)):
		separated = separated + list(out[i])
	print(separated)
	return separated, out

def getlrcbyid(id):
	try:
		return requests.get('http://tingapi.ting.baidu.com/v1/restserver/ting?method=baidu.ting.song.lry&songid='+id) #283058
	except Exception as e:
		print(e)
		return None

def lrc2lcoal():
	with open('lrcList.txt','w') as writer:
		with open('songid.txt') as sid:
			id = sid.readline()
			while id != None:
				id = id.strip()
				character,lrc = extractlyrics(id) if id != None else (None, None)
				if lrc != None and character != None:
					result = lrc
					lrcWriter(id,result)
					result = character
					oneLine = ''.join(result)
					writer.write(oneLine)
					writer.write('\n')
				id = sid.readline()

def lrcFilter(folder = './lrc/'):
	files = glob.glob(folder+'*.txt')
	for f in files:
		out = ''
		with open(f, 'r') as content:
			line = content.readline()
			#print("line: "+ str(line))
			while line != None and line != '':
				if line.find(':') != -1 or line.find('：') !=-1:
					line  = content.readline()
					continue
				line = line.strip()
				line = line.replace(' ', '\n')
				line = re.sub('\(.*?\)','',line)
				line = re.sub('（.*?）','',line)
				line = re.sub('[a-zA-z0-9:,.-_~><，。？！?!\)\(]','',line)
				line = re.sub('[^\u4e00-\u9fff]','\n',line)
				if line != '' and line != '\n' and line != None:
					out = out+line+'\n'
				#print("current out: " + str(out))
				line = content.readline()
		with open(f, 'w') as content:
			content.write(out)


def lrcWriter(name,content):
	with open('./lrc/'+name+'.txt','w') as writer:
		for i in content:
			writer.write(i)
			writer.write('\n')

def lrcLocal(folder = './lrc/'):
	files = glob.glob(folder+'*.txt')
	separate = []
	sequence = []
	for f in files:
		with open(f, 'r') as content:
			line = content.readline()
			while line!= '':
				line = re.sub('[\n]','',line)
				if line == '':
					line = content.readline()
					continue
				separate = separate + list(line)
				sequence = sequence + [line]
				line = content.readline()
	return separate, sequence
def getlast(input, index = 0,mark = ']'):
	if input.find(mark, index) == -1 :
		return index-1
	else:
		#print("out:"+str(input.find(mark,index)+1))
		out = getlast(input, index = input.find(mark,index)+1)

	return out

def vectorBuilderByFrequency(uarray,array):
	uarray = [ord(uarray[i]) for i in range(0, len(uarray)) ]
	array = [ord(array[i]) for i in range(0, len(array)) ]
	temp = []
	for char in uarray:
		temp = temp + [array.count(char)]
	temp = np.array(temp)
	temp *= 100000
	uarray = np.array(uarray)
	vec = temp+uarray
	return zip(uarray,vec.tolist())

def vectorBuilderByPinyin(array):
	carray  = [ord(array[i]) for i in range(0, len(array)) ]
	carrayout = carray
	sound = []
	pinyindict = None
	with open('pinyin.json') as content:
		pinyindict = json.loads(content.read())

	for i in range(0,len(carray)):
		try:
			pinyinjson = json.loads(requests.get('http://10.8.6.24:8080/pinyin/'+array[i]).text)
		except Exception as e:
			print('issue:' + array[i]+str(i))
			print(e)
		if len(pinyinjson["text"])==1:
			sound = sound + [pinyindict[pinyinjson["text"]]]
		else:
			last = pinyinjson["text"][-2:]
			if last == 'ng':
				#ang, ong, ing
				sound = sound + [pinyindict[pinyinjson["text"][-3:-2]]*100+ord('g')]
				continue

			if last[-2] in pinyindict:
				sound = sound + [pinyindict[last[-2]]*100+ord(last[-1])]

			else:
				sound = sound + [pinyindict[last[-1]]*100]
	sound = np.array(sound)
	carray = np.array(carray)
	carray = carray+ 100000*(sound)
	return zip(carrayout,carray.tolist())

def vectorizeByAscii(string):
	if string == None:
		return None
	carray  = [ord(string[i]) for i in range(0, len(string)) ]
	carray = np.array(carray)
	sound = []
	pinyindict = None
	with open('pinyin.json') as content:
		pinyindict = json.loads(content.read())

	for i in range(0,carray.size):
		try:
			pinyinjson = json.loads(requests.get('http://10.8.6.24:8080/pinyin/'+string[i]).text)
		except Exception as e:
			print(e)
		if len(pinyinjson["text"])==1:
			sound = sound + [pinyindict[pinyinjson["text"]]]
		else:
			last = pinyinjson["text"][-2:]
			if last == 'ng':
				#ang, ong, ing
				sound = sound + [pinyindict[pinyinjson["text"][-3:-2]]*100+ord('g')]
				continue

			if last[-2] in pinyindict:
				sound = sound + [pinyindict[last[-2]]*100+ord(last[-1])]

			else:
				sound = sound + [pinyindict[last[-1]]*100]
	sound = np.array(sound)
	carray = carray+ 100000*(sound)
	carray = sound
	return carray.tolist()
#def vectorizeByFrequency();
	#TODO: vectorize the word by its global frequency
#print(str(extractlyrics('283058')))
def vectorListGenerator(songs, verctorFunc):
	out = [];
	for name in songs:
		id = openSuggestion(name)
		if id != None:
			out = out + [id]
	songIdWriter(out)
	return out 

def songIdWriter(id):
	with open('songid.txt','w') as writer:
		for i in range(0,len(id)):
			writer.write(id[i])
			writer.write('\n')

def songslist(file):
	out = []
	with open(file) as content:
		line = content.readline()
		while line:
			if line != '\n':
				line = line.split(' ')
				line = line[1]
				temp = []
				temp.append(len(line) if line.find('?') == -1 else line.find('?') )
				temp.append(len(line) if line.find('(') == -1 else line.find('(') )
				temp.append(len(line) if line.find('-') == -1 else line.find('-') )
				if len(temp)>0:
					ind = min(temp)
					line = line[:ind]
				out.append(line)
			line = content.readline()
	return out

