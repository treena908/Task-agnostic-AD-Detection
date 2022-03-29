import pandas as pd
import sys
import os
import random
import re
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_DIR + "/../../")

from flavio_data import \
    file_tokenization,convert_list_to_string
from remove_tags import resolve_repeats,remove_tags

# clean text except the disfluency markers, and save file

def clean_text(input_file, path, filename):

    count = 1
    for line in input_file:

        for string in line.split("\n"):
            if "*PAR" in string or "*INV" in string:

                string = string.replace("\t", "").replace("\n", " ")
                string = string.replace("~", " ")

                string = re.sub(r'[\d]+', '', string)
                string = string.replace('[_]', "")
                string = string.replace('f@l', "")
                string = string.replace('|', "")
                string = string.replace('_', "")
                string = string.replace('nuk', "")
                string = string.replace('x@n', "")
                string = string.replace('+"/', "")
                string = string.replace('+"/', "")
                string = string.replace('+<', "")
                string = string.replace('+/', "")

                string = string.replace('‡', "")
                if re.findall(r"\[x[0-9 ]+\]", string):
                    string = resolve_repeats(string)
                string = re.sub(r'[\x00-\x1F]+', '', string)
                if '\x15_\x15' in string:
                    # w.remove('\x15_\x15')
                    string = string.replace('\x15_\x15', '')
                string = string.replace('@', "")


                string = string.replace('xxx', "")
                string = string.replace('[+ exc]', "")
                with open(path + filename, 'a') as f:

                    f.write(string + '\n')
                    f.close()
            print('%s file saved' % (filename))



def make_progression_data():
    print('progression')

    data=pd.read_excel("../data/Pitt-data.xlsx",sheet_name='data')
    for idx,row in data.iterrows():
        if row.mms is not None and row.mmse2 is not None:
            if abs(row.mms-row.mmse2)>=5:
                with open ('../data/progression/decline.txt','a') as f:
                    f.write(row.id+'\n')
                    f.close()



    print('done saving progression')

def select_fluency_data(fluency,result):
    folders = ['Control', 'Dementia']
    mark=False
    print('len %d\n'%len(fluency))
    while len(result)<15:
        r1 = random.randint(0, len(fluency)-1)
        result.add(fluency[r1])
    print(len(result))
    for folder in folders:

        PATH = "../data/Pitt/"+folder+"/fluency/"

        for path, dirs, files in os.walk(PATH):


            for idx,filename in enumerate(files):

                if filename not in result:
                    continue
                else:
                    fullpath = os.path.join(path, filename)
                    filename = filename.split('.')[0]
                    # result.remove(result[idx])

                    with open(fullpath, 'r', encoding="utf8",errors='ignore')as input_file:
                        # file_tokenization(input_file, "../data/Pitt/selected/",filename+'.txt')
                        clean_text(input_file, "../data/Pitt/selection/",filename+'.txt')


        print('done saving')
def found(match,file):
    for elem in file:
        if elem.split('.')[0].split('-')[0]==match:
            return True
    return False

def make_fluency_data():
    data=pd.read_pickle('../data/DA_annotation_long_all.pickle')
    file=data.file.unique()
    # print(file)
    fluency=[]
    folders = ['Control', 'Dementia']
    result = set()

    for folder in folders:

        PATH = "../data/Pitt/"+folder+"/fluency/"
        # file_to_write = open("data/segment.txt", 'a')
        for path, dirs, files in os.walk(PATH):
            for filename in files:

                if folder=='Control' and filename.split('.')[0] not in file:
                    result.add(filename)
                elif folder=='Dementia':

                    # print(filename.split('.')[0].split('-')[0])
                    # print(filename.split('.')[0].split('-')[1])
                    if filename.split('.')[0] not in file and not found(filename.split('.')[0].split('-')[0],file) and  \
                    filename.split('.')[0].split('-')[1]=='0':
                        fluency.append(filename)


    select_fluency_data(fluency,result)
def main():
    # data=pd.read_pickle('../data/all_address_DA_com.pickle')
    # print(data.columns.tolist())
    # print(data.file[0:5])
    # return
    option=[1,2]
    for opt in option:
        if opt==1:
            make_fluency_data()
        if opt==2:
            make_progression_data()
if __name__=="__main__":
    main()