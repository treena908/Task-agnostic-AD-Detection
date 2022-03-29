from __future__ import with_statement
import os
import random
import re
import pandas as pd
import numpy as np
import pickle
import sys
from remove_tags import resolve_repeats
# import nltk
# from pandas import merge
# from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, precision_score, recall_score, roc_curve
# from nltk.tokenize import sent_tokenize,word_tokenize
import xlsxwriter 
GET_INV= True
dir_result="./result/"
ft_1=[]
ft_2=[]
ft_3=[]
ft_4=[]
ft_5=[]
ft_6=[]
ft_7=[]
labels=[]
fname=[]
utterances_count=[]
word_count=[]
utterances=[]
output_list = []
PAR=[]
INV=[]
def file_tokenization(input_file,path,filename):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''

    for line in input_file:
        # print(line)
        for element in line.split("\n"):

            if "*PAR" in element or "*INV" in element :
                #remove any word after the period.
                # cleaned_string = element.split('.', 1)[0]
                # cleaned_string = element.split('.', 1)[0]
                #replace par with empty string, deleting the part of the string that starts with PAR
                if '?' in element:
                    element=element[0:element.rindex('?')+1]
                elif '.' in element:
                    element=element[0:element.rindex('.')+1]


                with open(path+filename,'a') as f:

                    f.write(element+'\n')
                    f.close()

        print('%s file saved'%filename)

                # cleaned_string=element
                # tar_PAR = 0
                # tar_INV = 0

                # if "*PAR" in element:
                #     cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*PAR',''))
                #     tar_PAR=1
                #     tar_INV = 0
                # elif "*INV" in element:
                #     cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*INV',''))
                #     tar_PAR = 0
                #     tar_INV = 1
                #substitute numerical digits, deleting underscores
                # cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
                # tokenized_list = word_tokenize(cleaned_string)
                # output_list = output_list + tokenized_list
                # output_list = output_list + ['\n'] + tokenized_list
                # output_list.append(['\n'] + tokenized_list)
                # fname.append(str_id)
                # utterances.append(cleaned_string)
                # fname.append(str_id)
                # PAR.append(tar_PAR)
                # INV.append(tar_INV)
                # labels.append(label)

    # word = sum(1 for x in output_list if x != '\n')
    # word_count.append(word)
    # print("word %d\n" % word)
    # return count
def generate_text(input_file):
    text=""
    for line in input_file:
        # print("line")
        # print(line)
        for element in line.split("\n"):
            question=False
            stop=False
            if "*PAR" in element or ["*INV","%mor","%gra"] not in element:
                if "*PAR" in element:
                    str=element
                    if "?" in element:
                        question
                        cleaned_string = element.split('?', 1)[0]
                        # adding punctuation
                        cleaned_string = cleaned_string + "?"
                    elif "." in element:
                        print("line")
                        print(element)
                        cleaned_string = element.split('.', 1)[0]
                        # adding punctuation
                        cleaned_string = cleaned_string + "."
                        print(cleaned_string)

                    # replace par with empty string, deleting the part of the string that starts with PAR
                    cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*PAR', ''))
                    print(cleaned_string)

                    # substitute numerical digits, deleting underscores
                    cleaned_string = re.sub(r'[\d]+', '', cleaned_string.replace('_', ''))
                    print(cleaned_string)

                    text = text + cleaned_string

                if "*PAR" in element:
                    par=True
                else:
                    par=False



    utterances.append(text)

def generate_feature(input_file):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    f_1=f_2=f_3=f_4=f_5=f_6=f_7=count=word=0
    file=input_file


    for line in input_file:

        for element in line.split("\n"):

            if "*PAR" in element :

                # if "?" in element:
                #     cleaned_string = element.split('?', 1)[0]
                #     # adding punctuation
                #     cleaned_string = cleaned_string + "?"
                # else:
                #     print("line")
                #     print(element)
                #     cleaned_string = element.split('.', 1)[0]
                #     # adding punctuation
                #     cleaned_string=cleaned_string+"."
                #     print(cleaned_string)



                # replace par with empty string, deleting the part of the string that starts with PAR
                # cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*PAR', ''))
                # print(cleaned_string)
                #
                # # substitute numerical digits, deleting underscores
                # cleaned_string = re.sub(r'[\d]+', '', cleaned_string.replace('_', ''))
                # print(cleaned_string)
                #
                # text=text+cleaned_string
                
                #print(element)
                if '(.)' in element:
                    f_1+=element.count('(.)')
                elif '(..)' in element:
                    f_2+=element.count('(..)')
                elif '(...)' in element:
                    f_3+=element.count('(...)')
                elif '[/]' in element:
                    f_4+=element.count('[/]')
                elif '[//]' in element:
                    f_5+=element.count('[//]')
                elif '&uh' in element:
                    f_6+=element.count('&uh')
                elif '+...' in element:
                    f_7+=element.count('+...')
                count+=1



    ft_1.append(f_1)
    ft_2.append(f_2)
    ft_3.append(f_3)
    ft_4.append(f_4)
    ft_5.append(f_5)
    ft_6.append(f_6)
    ft_7.append(f_7)
    utterances_count.append(count)
    # word_count.append(word)
    print(ft_1[0])
    print(ft_4[0])
    print(ft_2[0])
    print(ft_3[0])
    print(ft_5[0])
    print(ft_6[0])
    print(utterances_count[0])
    #return f_1,f_2,f_3,f_4,f_5,f_6,f_7
# def file_tokenization_utterances(input_file):
#     '''
#     :param input_file: single dataset file as readed by Python
#     :return: tokenized string of a single patient interview
#     '''
#     output_list = []
#     #count=0
#     length=0
#     previous="*INV"
#
#     que_list = ["else",
#                 "anything", "anymore action", "can you","tell me", "mistakes", "how about", "what's going on over here",
#                 "going on",
#
#                 "is that all", "is that", "what's happening", " happening", "action", "some more", "more"]
#     for line in input_file:
#         for element in line.split("\n"):
#             #if "*PAR" in element or ("*INV" in element and GET_INV):
#             if "*INV" in element:
#
#                 #remove any word after the period.
#                 cleaned_string = element.split('.', 1)[0]
#                 #replace par with empty string, deleting the part of the string that starts with PAR
#                 cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*INV',''))
#                 #substitute numerical digits, deleting underscores
#                 cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
#                 if previous=="*PAR":
#                     for k in range(len(que_list)):
#                         if que_list[k] in cleaned_string:
#                             count+=1
#                             break
#                 tokenized_list = word_tokenize(cleaned_string)
#                 free_tokenized_list = []
#                 for element in tokenized_list:
#                     if element is not '':
#                         free_tokenized_list.append(element)
#                 output_list.append(free_tokenized_list)
#             else:
#                 previous="*PAR"
#     return output_list, count,length
def file_tokenization_both_utterances(input_file,count,filename):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    output_list_inv = []
    output_list_par = []
    output_types=[]
    previous=None
   # print(count)
    length=0
    
    # que_list = ["else",
    #             "anything", "anymore action", "can you","tell me", "mistakes", "how about", "what's going on over here",
    #             "going on",
    #
    #             "is that all", "is that", "what's happening", " happening", "action", "some more", "more","?"]
    # neutral_list=["uhhuh","mhm","okay","alright"]
    utterance=0
    print("filename")
    print(filename)
    for line in input_file:

        for element in line.split("\n"):
            type=None
            tag=None
            if "*PAR" in element:
                return element.split(':')[1].strip()
                # with open("./data/" + str(count) + '.txt', 'a') as f:
                    # cleaned_string = element.split('.', 1)[0]

                    # substitute numerical digits,
                    #
                    #
                    # deleting underscores

                    # if len(element.split(':')[1])==1:


                    #     continue
                    # else:
                    #     cleaned_string = re.sub(r'[\d]+', '', element.replace('_', ''))
                    #     utterances.append(cleaned_string)
                    #     fname.append(filename)
                    #     utterances_count.append(utterance)
                    #     utterance+=1
                    # f.write(cleaned_string + os.linesep)


               # if "*PAR" in element or ("*INV" in element and GET_INV):
                # cleaned_string = element.split('.', 1)[0]
                    # #replace par with empty string, deleting the part of the string that starts with PAR
                # if "*INV" in cleaned_string:
                    # #cleaned_string = re.sub(r'[^\w?]+', ' ', cleaned_string.replace('*INV',''))
                    # tag="*INV"
                # elif "*PAR" in cleaned_string:
                    # #cleaned_string = re.sub(r'[^\w?]+', ' ', cleaned_string.replace('*PAR',''))
                    # tag="*PAR"
                # #substitute numerical digits, deleting underscores
                # cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
                # #tokenized_list = word_tokenize(cleaned_string)
                # #free_tokenized_list = []
                # # for element in tokenized_list:
                    # # if element is not '':
                        # # free_tokenized_list.append(element)
                # if tag=="*INV":
                    
                    # #remove any word after the period.
                    
                    # if (previous=="*INV"):
                        # with open(dir_result +str(count) +"_"+label+'.txt', 'a') as f:
                            # f.write("*PAR: "+ os.linesep)
                        

                    # #output_list_inv.append(free_tokenized_list)
                    # # for k in range(len(neutral_list)):
                        # # if que_list[k] in cleaned_string:
                       
                            # # type="question"
                            # # break
                        
                    # # if type==None:
                        # # if "laughs" in cleaned_string:
                            # # type="imitate"
                        # # else:
                            # # for k in range(len(neutral_list)):
                                # # if neutral_list[k] in cleaned_string:
                                
                                    # # type="neutral"
                                    # # break
                        
                    
                    # previous="*INV"
                    
                    # #output_types.append(type)
                # elif tag =="*PAR" :
                    # #output_list_par.append(free_tokenized_list)
                    # if previous is "*PAR":
                       # with open(dir_result +str(count) +"_"+label+'.txt', 'a') as f:
                            # f.write("*INV: "+ os.linesep)
                    # with open(dir_result +str(count) +"_"+label+'.txt', 'a') as f:
                        # f.write(cleaned_string+ os.linesep)
                    # previous="*PAR"
                    # #output_types.append(type)
                
            
    #return output_list_inv, output_list_par,output_types
    #return length

# def transcript_to_file():
#     file_list=[16,17,20,21,22,26]
#     PATH = "./data/script"
#     count=51
#     for path, dirs, files in os.walk(PATH):
#         for filename in files:
#
#             fullpath = os.path.join(path, filename)
#             filename = int(filename.split('.')[0])
#
#             with open(fullpath, 'r', encoding="utf8",errors='ignore')as input_file:
#                 file_tokenization_both_utterances(input_file,count,filename)
#                     # count+=1
#             else:
#                 continue
#                 # dementia_list.append(
#                 # {'text':tokenized_list,
#                 # 'label':label}
#                 # )
#                 # generate_feature(input_file)
#                 #
#                 # labels.append(label)
#                 # fname.append(filename)
#                 # break
#                 # dementia_list.append(
#                 # {'short_pause_count':f_1,
#                 # 'long_pause_count':f_2,
#                 # 'very_long_pause_count':f_3,
#                 # 'word_repetition_count':f_4,
#                 # 'retracing_count':f_5,
#                 # 'filled_pause_count':f_6,
#                 # 'incomplete_utterance_count':f_7,
#                 # 'label':label
#                 # }
#                 # )



def voting():
    # cc_meta = pd.read_csv("data/meta_all.txt", sep=';')
    cc_meta = pd.read_excel("data/adress_test.xlsx", sheet_name='Sheet1')
    df = pd.read_excel("data/adress_acoustic_GP.xlsx", sheet_name='Sheet1')
    dict = {"file": [], "test": [], "result": []}
    df['mmse'] = [None] * len(df)
    count = 0
    for index, row in cc_meta.iterrows():
        j = count
        rows = 0
        vals = 0
        dict["file"].append(row['file'])
        dict["test"].append(row['mmse'])
        for i in range(j, len(df)):
            name = df.iloc[i]["file"].split("-")[0]
            # name = df.iloc[i]["file"]
            # print(name)
            # print(row['file'].strip())
            if (name.strip() == row['file'].strip()):
                vals += df.iloc[i]["result"]
                count += 1
                rows += 1

            else:
                break
        dict["result"].append(vals / rows)

    df = pd.DataFrame(dict)
    writer = pd.ExcelWriter("data/adress_acoustic_GP_final.xlsx", engine='xlsxwriter')

    df.to_excel(writer, sheet_name='Sheet1', columns=list(df.columns))
    writer.save()
    # with open('data/adress_acoustic_test.pickle', 'wb') as f:
    #     pickle.dump(df, f)
def match_mmse():
    # cc_meta = pd.read_csv("data/meta_all.txt", sep=';')
    cc_meta = pd.read_excel("data/adress_test.xlsx", sheet_name='Sheet1')
    df=pd.read_excel("data/ADReSS/acoustic_feature_test.xlsx",sheet_name='Sheet2')

    df['mmse'] = [None] * len(df)
    count=0
    for index, row in cc_meta.iterrows():
        j=count
        for i in range(j,len(df)):
            name=df.iloc[i]["file"].split("-")[0]
            # name = df.iloc[i]["file"]
            print(name)
            print(row['file'].strip())
            if(name.strip()==row['file'].strip()):
                df.at[i, 'mmse']=row["mmse"]
                print(df.at[i, 'mmse'])
                count+=1

            else:
                break
    writer = pd.ExcelWriter("data/adress_acoustic_test.xlsx", engine='xlsxwriter')


    df.to_excel(writer, sheet_name='Sheet1',columns=list(df.columns))
    writer.save()
    with open('data/adress_acoustic_test.pickle', 'wb') as f:
         pickle.dump(df, f)
def convert_segment():
    # df = pd.read_pickle('data/adress_acoustic_train.pickle')
    df = pd.read_excel('data/adress_acoustic_train.xlsx', sheet_name='Sheet2')
    cc_meta = pd.read_csv("data/ADReSS/meta_all.txt", sep=';')
    # # cc_meta = pd.read_excel("data/adress_test.xlsx", sheet_name='Sheet1')
    # df=pd.read_excel("data/adress_acoustic_train.xlsx",sheet_name='Sheet1')
    # length=[]
    #
    df['mmse'] = [None] * len(df)
    rows=[]
    count=0
    for index, row in cc_meta.iterrows():
        # rows.append(row['file'].strip())
        j=count
        # list=[]
        size=0
        for i in range(j,len(df)):

            name=df.iloc[i]["file"].split("-")[0]
            # name = df.iloc[i]["file"]
            # print(name)
            # print(row['file'].strip())
            if(name.strip()==row['file'].strip()):
                df.at[i, 'mmse'] = row["mmse"]
                rows.append(name.strip())
                size+=1
                # print(df.iloc[i, 2:].values)
                # print(len(df.iloc[i, 2:].values))
                # list.extend(df.iloc[i,2:].values)

                # print(len(list))
                count+=1


            else:
                break

    print(len(rows))
    print(len(df))

    df["id"]   =rows
    writer = pd.ExcelWriter("data/adress_acoustic_train.xlsx", engine='xlsxwriter')


    df.to_excel(writer, sheet_name='Sheet1',columns=list(df.columns))
    writer.save()
    # df=pd.DataFrame(rows)
    # with open('data/adress_acoustic_train_merge.pickle', 'wb') as f:
    #      pickle.dump(df, f)
    # df=pd.read_pickle('data/adress_acoustic_train_merge.pickle')
    # df["file"]=rows

    # print(len(df.iloc[0]))
    # print(len(df.iloc[1]))
    # print(len(df))
    # df.iloc[:,:].fillna(0, inplace = True)
    # with open('data/adress_acoustic_train_merge.pickle', 'wb') as f:
    #      pickle.dump(df, f)
    # df = pd.read_pickle('data/adress_acoustic_train_merge.pickle')
    # print(df.head(1))

def match_mmse():
    # cc_meta = pd.read_csv("data/meta_all.txt", sep=';')
    cc_meta = pd.read_excel("data/adress_test.xlsx", sheet_name='Sheet1')
    df=pd.read_excel("data/ADReSS/acoustic_feature_test.xlsx",sheet_name='Sheet2')

    df['mmse'] = [None] * len(df)
    count=0
    for index, row in cc_meta.iterrows():
        j=count
        for i in range(j,len(df)):
            name=df.iloc[i]["file"].split("-")[0]
            # name = df.iloc[i]["file"]
            print(name)
            print(row['file'].strip())
            if(name.strip()==row['file'].strip()):
                df.at[i, 'mmse']=row["mmse"]
                print(df.at[i, 'mmse'])
                count+=1

            else:
                break
    writer = pd.ExcelWriter("data/adress_acoustic_test.xlsx", engine='xlsxwriter')


    df.to_excel(writer, sheet_name='Sheet1',columns=list(df.columns))
    writer.save()
    with open('data/adress_acoustic_test.pickle', 'wb') as f:
         pickle.dump(df, f)
def generate_test_df():

    # PATH = "./data/Pitt_transcripts/all"
    count = 51
    pid = []
    df_train=pd.read_excel("./data/matched_test_wid_DA.xlsx",sheet_name='Sheet1')
    df_train_exclude=df_train.iloc[:]['file']
    folders=["Control", "Dementia"]
    print("exclude")
    for item in df_train_exclude:
        # print(item)
        # print(type(item))
        pid.append(item)
    # for folder in folders:

    # for folder in folders:
    PATH = "./data/ADReSS/test/transcription/"
    for path, dirs, files in os.walk(PATH):
        for filename in files:
            fullpath = os.path.join(path, filename)
            filename = filename.split('.')[0]
            # print("file")
            # print(type(filename))
            if filename in pid:
                print("okay")
                print(filename)
                continue

            with open(fullpath, 'r', encoding="utf8", errors='ignore')as input_file:
                file_tokenization(input_file,filename,"")
                # file_tokenization_both_utterances(input_file, count, filename)
                # pid.append(file_tokenization_both_utterances(input_file, count, filename))
                # fname.append(filename)
                # if folder=="Control":
                #     labels.append(0)
                # else:
                #     labels.append(1)
    # print(len(fname))
    # print(len(PAR))
    # print(len(INV))
    df = pd.DataFrame({"file": fname, "utterance": utterances,"PAR":PAR,"INV":INV,"label":labels})
    print(df.iloc[0:2]["utterance"])
    print(df.head(5))
    with open('data/file_to_text_test.pickle', 'wb') as f:
        pickle.dump(df, f)
    # cc_meta = pd.read_excel("data/file_names.xlsx")
    # # cc_meta['label'] = [0] * len(cc_meta)
    #
    #
    #
    # # cc_meta['gender'] = np.where(cc_meta.gender == 'male', 0, 1)
    #
    # for index, row in cc_meta.iterrows():
    #     str_id = row['chat_file_name'].replace(' ', '')
    #     print(str_id)
    #     file_path = "data/Pitt_transcripts/all/" + str_id + '.cha'
    # # file_path = "data/Pitt_transcripts/all/" + "005-0" + '.cha'
    #     # if (cc_meta.at[index, 'utterances'] == None):
    #     with open(file_path, 'r', encoding="utf8")as input_file:
    # #         # print(str_id)
    # #         # output = file_pos_extraction(input_file)
    # #         # cc_meta.at[index, 'utterances'] = ''.join(output)
    # #     generate_feature(input_file)
    #
    #         # output=file_tokenization(input_file,str_id)
    #         output =convert_list_to_string(input_file,str_id)
    #         # utterances.append(output)
    #         # fname.append(str_id)
    #         # break




    # return cc_meta

def generate_full_interview_dataframe():
    """
    generates the pandas dataframe containing for each interview its label.
    :return: pandas dataframe.
    """

    dementia_list = []
    for label in ["Control", "Dementia"]:

        PATH = "data/ADReSS/train/transcription/" + label

        for path, dirs, files in os.walk(PATH):
            for filename in files:
                fullpath = os.path.join(path, filename)
                with open(fullpath, 'r',encoding="utf8")as input_file:
                    # tokenized_list = file_tokenization(input_file,label)
                    # dementia_list.append(
                        # {'text':tokenized_list,
                         # 'label':label}
                        # )
                    # generate_text(input_file)
                    name=filename.split('.')[0]
                    fname.append(name)
                    output = file_pos_extraction(input_file,name)
                    labels.append(label)
                    print(output)
                    utterances.append(output)
                    break
                break
                    # print(utterances[0:3])

                    #break
    #     dementia_list.append(
    #         {'short_pause_count':ft_1,
    #         'long_pause_count':ft_2,
    #         'very_long_pause_count':ft_3,
    #         'word_repetition_count':ft_4,
    #         'retracing_count':ft_5,
    #         'filled_pause_count':ft_6,
    #         'incomplete_utterance_count':ft_7,
    #         'label':labels,
    #          'file':fname
    #         }
    #         )
    #
    # dementia_dataframe = pd.DataFrame(dementia_list)
    # return dementia_dataframe
def generate_single_utterances_dataframe():
    
    Folder=["Control","Dementia"]
    count=0
    for folders in Folder:
        PATH = "D:/Admission/UIC/conference/BHI/annotation/source"
        for path, dirs, files in os.walk(PATH):
            for filename in files:
                fullpath = os.path.join(path, filename)
                with open(fullpath, 'r')as input_file:

                    file_tokenization_both_utterances(input_file,filename)

               
                
                
                        

    # dementia_dataframe = pd.DataFrame(dementia_list)
    # return dementia_dataframe
def generate_single_utterances_dataframe():
    
    length=[]
    
    ids=[]
    classes=[]
    files=[]
    count=0
    prev=''
    df=pd.read_csv('data/DA_annotation_data.csv')
    for label in ["Control", "Dementia"]:
        folders =  ["cookie"]
        id = 0

        for folder in folders:
            PATH = "data/Pitt_transcripts/" + label + "/" + folder
            for path, dirs, files in os.walk(PATH):
                for filename in files:
                    file_name=filename.split('.')[0]
                    if label=='Control':
                        ids.append(file_name)
                    else:
                        length.append(file_name)

                #     fullpath = os.path.join(path, filename)
                #     with open(fullpath, 'r',encoding="utf8")as input_file:
                #         file_tokenization_both_utterances(input_file)
                #
                #         # dementia_list.append(
                #         # {
                #         # 'len':length,
                #         # 'label':label,
                #         # 'id':id
                #         # }
                #         #)
                #
                #         count+=1
                #         if count>25:
                #             break
                #         id = id +1
                #         ids.append(id)
                #         type.append(label)
                #
                # if count>25:
                #     count=1
                #     break
                #         for element1,element2,element3 in zip(inv,par,type):
                #             dementia_list.append(
                #             {'par': element2,
                #             'label': label,
                #             'id':id,
                #             'inv':element1,
                #             'type':element3}
                #                 )
                #
    script=len(df['chat_file_name'].unique())
    print(script)
    print('files')
    print(ids)
    print(length)
    ids.pop()
    length.pop()
    dem=0
    con=0
    tot=0
    check=[]
    for indx,row  in df.iterrows():

        if row['chat_file_name']==prev:
            continue
        if (row['chat_file_name'] in check):
            continue
        check.append(row['chat_file_name'])

        tot = tot + 1
        # print("file")
        # print(row['chat_file_name'])
        # print("prev")
        # print(prev)
        # temp = df.loc[df['chat_file_name'] == row['chat_file_name']]
        if row['chat_file_name'].strip() in ids :
            con=con+1
            print("con")
            print(row['chat_file_name'])
            # count=count+len(temp)
            # print(len(temp))
            # type=[0] * len(temp)
            # classes.append(0)
            # cnt = cnt + 1
        elif row['chat_file_name'].strip() in length :
             dem=dem+1
             print("dem")
             print(row['chat_file_name'])
             # count = count + len(temp)
             # print(len(temp))
             # type = [1] * len(temp)
             # classes.append(1)
             # cnt=cnt+1
        else:
            # print(row['chat_file_name'].strip())



            print("none")

        prev=row['chat_file_name']
        # type.clear()
    # print(classes)
    print(tot)
    print("con")
    print(con)
    print(con/script)
    print("dem")
    print(dem)
    print(dem/script)
    # df['label']=classes
    # print(cnt)
    # dic={'class':classes}
    # df_label=pd.DataFrame(dic)
    # with open('data/DA_label_full.pickle', 'wb') as f:
    #     pickle.dump(df, f)
    # return df

    # dementia_dataframe = pd.DataFrame(dementia_list)
    # return dementia_dataframe
# import re
# from nltk.tokenize import word_tokenize

# clean pitt data
def convert_list_to_string(input_file,path,filename):
    # return_string = ""
    # print("str list")
    # print(list_strings)
    count=1
    for line in input_file:

        for string in line.split("\n"):
            if "*PAR" in string or "*INV" in string:
                # print("str before")
                # print(string)
                # string = string.replace("*PAR:","")
                # string = string.replace("%mor:","")
                # string = string.replace("%gra:","")
                string = string.replace("\t","").replace("\n"," ")
                string = string.replace("~"," ")

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


                # string = string.replace('[', "")
                # string = string.replace('(', "")
                # string = string.replace(')', "")
                # string = string.replace('<', "")
                # string = string.replace('>', "")
                # string = string.replace(')', "")
                # string = string.replace('/', "")
                # string = string.replace(':', "")
                # string = string.replace('*', "")
                # string = string.replace('+...', ".")
                # string = string.replace('+', "")

                string = string.replace('xxx', "")
                string = string.replace('[+ exc]', "")
                with open(path + filename, 'a') as f:

                    f.write(string + '\n')
                    f.close()
            print('%s file saved' % (filename))
            # string = string.replace(']', "")

                # str = string.split('.')
                # if len(str) > 1:
                #     string=str[0]+" . "
                # str1=string.split('?')
                # if len(str1) > 1:
                #     string = str1[0] + " ? "
                # print("str")
                # print(string)
                # print("return_str")

                # tokenized_list = word_tokenize(string)
                # output_list = output_list + tokenized_list
                # output_list = output_list + ['\n'] + tokenized_list
                # output_list.append( tokenized_list)
                # fname.append(str_id)
                # utterances_count.append(count)
                # count+=1

    # return return_string

def convert_mor_to_list(list_string):
    return_list = []
    element_list = list_string.split(" ")
    for element in element_list:
        if "|" in element:
            return_list.append(element.split("|")[0])
    return return_list

def convert_gra_to_list(list_string):
    return_list = []
    element_list = list_string.split(" ")
    for element in element_list:
        if "|" in element:
            return_list.append(element.split("|")[2])
    return return_list

def file_pos_extraction(input_file):
  # with open(fullpath, 'r') as input_file:
  #   id_string = input_file.name.split('/')[-1]
  #   #print(id_string)
  #   result = re.search('(.*).cha',id_string)
  #   id = result.group(1)

    par_output = []
    mor_output = []
    gra_output = []
    str=""
    i = 0
    list_files = list(input_file)
    # print(fname)


    # if (fname in ['S083', 'S114', 'S135', 'S082']):
    #     print(list_files[10])
    while i < len(list_files):
        par_list = []
        mor_list = []
        gra_list = []
        # if (fname in ['S083', 'S114', 'S135', 'S082']):
        #     print(list_files[i])
            # print(len(list_files))
        # if i == 5:
        #     if "|Control|" in list_files[i] or "|ProbableAD|" in list_files[i] or "|PossibleAD|" in list_files[i]:
        #         #age =
        #         pass
        #     else:
        #         return [],[],[], False
        if "*PAR:" in list_files[i]:

            #print(list_files[i])
            while "%mor" not in list_files[i] and "*INV:" not in list_files[i] :
                print("enter")
                # print(list_files[i])
                par_list.append(list_files[i])
                i += 1
                if "@End" in list_files[i]:
                    break
            if "@End" in list_files[i]:

                break

            if "%mor" in list_files[i]:

                #print(list_files[i])
                while "%gra" not in list_files[i]:

                    mor_list.append(list_files[i])
                    i += 1

                if "%gra" in list_files[i]:

                    #print(list_files[i])
                    while "*PAR" not in list_files[i] and "*INV" not in list_files[i] and "@End" not in list_files[i]:
                        gra_list.append(list_files[i])
                        i += 1
        else:
            i += 1
        # w = word_tokenize(convert_list_to_string(par_list))

        w = convert_list_to_string(par_list)
        if '\x15_\x15' in w:
            # w.remove('\x15_\x15')
            w_new=w.replace('\x15_\x15','')
        else:
            w_new = w

        m = convert_mor_to_list(convert_list_to_string(mor_list))
        g = convert_gra_to_list(convert_list_to_string(gra_list))
        # print(w_new)
        # print(len(w_new))

        if len(w_new) > 0:
            par_output += w_new
            str=str+w_new
        if len(m) > 0:
            mor_output.append(m)
        if len(g) > 0:
            gra_output.append(g)
    #return par_output,mor_output,gra_output,id



    return str
# def read_annotation():
#     dic_dist = [
#         'Answer:t3',
#         'Answer:t2',
#         'Answer:t1',
#         'Answer:t4',
#         'Answer:t5',
#         'Answer:t6',
#         'Answer:t7',
#         'Answer:t8',
#         'Question:General',
#         'Question:Reflexive',
#         "Answer:Yes",
#         "Answer:No",
#         "Answer:General",
#         "Instruction",
#         "Suggestion",
#         "Request",
#         "Offer",
#         "Acknowledgment",
#         "Request:Clarification",
#         "Feedback:Reflexive",
#         "Stalling",
#         "Correction",
#         "Farewell",
#         "Apology",
#
#         "Other"
#
#     ]
#     dic_dist = {
#         'Answer:t3': [],
#         'Answer:t2': [],
#         'Answer:t1': [],
#         'Answer:t4': [],
#         'Answer:t5': [],
#         'Answer:t6': [],
#         'Answer:t7': [],
#         'Answer:t8': [],
#         'Question:General': [],
#         'Question:Reflexive': [],
#         "Answer:Yes": [],
#         "Answer:No": [],
#         "Answer:General": [],
#         "Instruction": [],
#         "Suggestion": [],
#         "Request": [],
#         "Offer": [],
#         "Acknowledgment": [],
#         "Request:Clarification": [],
#         "Feedback:Reflexive": [],
#         "Stalling": [],
#         "Correction": [],
#         "Farewell": [],
#         "Apology": [],
#
#         "Other": [],
#         "file": [],
#         "text":[]
#
#     }
#     df1 = pd.read_excel('data/DA_annotattion_data.xlsx')
#     df2 = pd.read_excel('data/full_conversation__mmse.xlsx')
#     for index, row in df1.iterrows():
#         for key in dic_dist:
#             dic_dist[key].append(0)
#         text=''
#         for index2, row2 in df2.iterrows():
#             if row['chat_file_name']==row2['id']:
#                 text=row2['text']
#                 break
#         if text!='':
#             seg=segment_text(text)
#             j=0
#             for i in range(index,index+40):
#                 if df1.loc[i,'speaker']=='*PAR':
#                     for elem in df1[i,'DA_Label']:
#                         dic_dist[elem][len(dic_dist[elem])-1]=1







# match_mmse()
# convert_segment()

def plot_hist():
    import matplotlib.pyplot as plt
    df = pd.read_excel('data/ADReSS/meta_data_loso.xlsx')
    # df1 = pd.read_excel('data/ADReSS/rmse_plot.xlsx')
    fig,ax = plt.subplots()


    numBins =12
    (n1,bin1,patches1)=ax.hist(df['control'], bins=np.arange(0, 30 + 2, 2),  alpha=0.5,label='control',edgecolor='black', linewidth=1.2,align='left')
    (n2,bin2,patches2)=ax.hist(df['dementia'], bins=np.arange(0, 30 + 2, 2),  alpha=0.5,label='dementia',edgecolor='black', linewidth=1.2,align='left')
    # plt.xlim([10, 30])
    ax.set_xlabel("MMSE", fontsize=14)

    # set y-axis label
    ax.set_ylabel("count", color="blue", fontsize=14)
    plt.xticks(bin2[:-1])
    plt.legend(loc='center left')
    # twin object for two different y-axis on the sample plot
    # ax2 = ax.twinx()
    # # make a plot with different y-axis using second axis object
    # df1['value'].plot(kind='line', marker='d', ax=ax2)
    # ax2.set_ylabel("rmse", color="blue", fontsize=14)
    plt.show()
    print(n2)
    print(bin2)
    print(n1)
    print(bin1)
    dict={"dem":n2,"con":n1}
    df1=pd.DataFrame(dict)
    return df1
def plot_test():
    import matplotlib.pyplot as plt

    df1 = pd.read_excel('data/ADReSS/rmse_plot_test.xlsx')
    df2 = pd.read_excel('data/ADReSS/rmse_plot_loso.xlsx')
    fig, bar_ax = plt.subplots()
    bar1=bar_ax.bar(df1['bin'], df1['dementia'], color='blue',alpha=0.5)  # plot first y series (line)
    bar2=bar_ax.bar(df1['bin'], df1['control'], color='green', alpha=0.5)  # plot first y series (line)
    bar_ax.set_xlabel('MMSE score')  # label for x axis
    bar_ax.set_ylabel('Count',color='blue')  # label for left y axis
    bar_ax.tick_params('y', colors='blue')  # add color to left y axis
    bar_ax.set_xticklabels(df1.bin, rotation=40)
    # plt.legend(['Dementia', 'Control'], loc='center left')
    line_ax = bar_ax.twinx()
    line,=line_ax.plot(df1['bin'], df1['value'],linestyle='-', marker='o', color='red')  # plot second y series (bar)
    line_ax.set_ylabel('RMSE',color='red')  # label for right y axis
    line_ax.tick_params('y', colors='red')  # add color to right y axis
    plt.legend([bar1,bar2,line],['Dementia', 'Control','Overall RMSE'],loc=9)

    def autolabel(rects,rects2):
        """
        Attach a text label above each bar displaying its height
        """
        i=5
        # for rect in rects:
        for j in range(0,len(rects)-1):
            height = df2['total'][i]
            i+=1
            bar_ax.text(rects[j].get_x() + rects[j].get_width() / 2., 1.05 * rects[j].get_height(),\
            str(round(height,1))+'%',
                    ha='center', va='bottom',fontsize=10)
        print(len(rects2))
        for j in range(9, 10):
            height = df2['total'][i]
            i += 1
            if j==9:
                bar_ax.text(rects2[j].get_x() + rects2[j].get_width() / 2., 1.001 * rects2[j].get_height(), \
                            str(round(height, 1)) + '%',
                            ha='center', va='bottom', fontsize=8)
            else:
                bar_ax.text(rects2[j].get_x() + rects2[j].get_width() / 2., 1.05 * rects2[j].get_height(), \
                            str(round(height, 1)) + '%',
                            ha='center', va='bottom', fontsize=8)



    autolabel(bar1,bar2)
    plt.show()
    fig.savefig('test_plot_num.png')
def generate_time_frame():
    folders=['Control','Dementia']
    for folder in folders:

        PATH = "data/Pitt_time/Pitt/"+folder
        file_to_write=open("data/segment.txt",'a')
        for path, dirs, files in os.walk(PATH):
            for filename in files:
                print(filename)
                fullpath = os.path.join(path, filename)
                with open(fullpath, 'r')as input_file:
                    start=0
                    end=0
                    cur1=''
                    cur2=''
                    count=0
                    speaker=''
                    for line in input_file.readlines():
                        # print(line)

                        seg=line.split('|')
                        # if '#' in line and len(seg)>1:
                        #     cur1=seg[1]
                        #     cur2=seg[4]

                        print(seg)
                        print(seg[0])
                        if '#' not in line and len(seg)>1 and seg[0].strip()!="Totals":
                            count+=1
                            if seg[1].isspace() == False:
                                if seg[2].isspace()==False:
                                    start=int(seg[2])+end
                                    end=start+int(seg[1])
                                elif len(seg)>2 and seg[3].isspace()==False:
                                    start=int(seg[3])+end
                                    end=start+int(seg[1])
                                else:
                                    start=end
                                    end+=int(seg[1])
                                # speaker=cur1

                            elif len(seg)>5 and seg[4].isspace()==False:
                                if seg[3].isspace() == False:
                                    start = int(seg[3]) + end
                                    end = start + int(seg[4])
                                elif seg[5].isspace() == False:
                                    start = int(seg[5]) + end
                                    end = start + int(seg[4])
                                else:
                                    start=end
                                    end+=int(seg[4])
                                # speaker = cur2
                            text=filename+'_'+str(count)+" "+filename+" "+str(round(start/1000,2))+" "+str(round(end/1000,2))+"\n"

                            file_to_write.write(text)
    file_to_write.close()
# generate_time_frame()
# act=[[] for i in range(15)]
# pred=[[] for i in range(15)]
# num=[0]*10
# avg=[]*15
# # id1=[]
# # id2=[]
# df = pd.read_excel('data/adress_all_poly_loso.xlsx')
# # print(df.columns)
# for index, row in df.iterrows():
#     print(row['mmse'])
#     print(type(row['mmse']))
#     # test_rmse = np.sqrt(mean_squared_error([row['mmse']], [row['result']]))
#     if row['mmse']>=0 and row['mmse']<2:
#         print("1")
#         act[0].append(row['mmse'])
#         pred[0].append(row['result'])
#         num[0]=num[0]+1
#     elif row['mmse']>=2 and row['mmse']<4:
#         print("1")
#         act[1].append(row['mmse'])
#         pred[1].append(row['result'])
#         num[0]=num[0]+1
#     elif row['mmse']>=4 and row['mmse']<6:
#         print("1")
#         act[2].append(row['mmse'])
#         pred[2].append(row['result'])
#         num[0]=num[0]+1
#     elif row['mmse']>=6 and row['mmse']<8:
#         print("1")
#         act[3].append(row['mmse'])
#         pred[3].append(row['result'])
#         num[0]=num[0]+1
#     elif row['mmse']>=8 and row['mmse']<10:
#         print("1")
#         act[4].append(row['mmse'])
#         pred[4].append(row['result'])
#         num[0]=num[0]+1
#     elif row['mmse']>=10 and row['mmse']<12:
#         print("1")
#         act[5].append(row['mmse'])
#         pred[5].append(row['result'])
#         num[0]=num[0]+1
#
#     elif row['mmse']>=12 and row['mmse']<14:
#         print("2")
#         act[6].append(row['mmse'])
#         pred[6].append(row['result'])
#         num[1] = num[1] + 1
#     elif row['mmse'] >= 14 and row['mmse'] < 16:
#         print("3")
#         act[7].append(row['mmse'])
#         pred[7].append(row['result'])
#         num[2] = num[2] + 1
#     elif row['mmse'] >= 16 and row['mmse'] < 18:
#         print("4")
#         act[8].append(row['mmse'])
#         pred[8].append(row['result'])
#         num[3] = num[3] + 1
#     elif row['mmse'] >= 18 and row['mmse'] < 20:
#         print("5")
#         act[9].append(row['mmse'])
#         pred[9].append(row['result'])
#         num[4] = num[4] + 1
#     elif row['mmse'] >= 20 and row['mmse'] < 22:
#         print("6")
#         act[10].append(row['mmse'])
#         pred[10].append(row['result'])
#         num[5] = num[5] + 1
#     elif row['mmse'] >= 22 and row['mmse'] < 24:
#         print("7")
#         act[11].append(row['mmse'])
#         pred[11].append(row['result'])
#         num[6] = num[6] + 1
#     elif row['mmse'] >= 24 and row['mmse'] < 26:
#         print("8")
#         act[12].append(row['mmse'])
#         pred[12].append(row['result'])
#         num[7] = num[7] + 1
#     elif row['mmse'] >= 26 and row['mmse'] < 28:
#         print("9")
#         act[13].append(row['mmse'])
#         pred[13].append(row['result'])
#         num[8] = num[8] + 1
#     elif row['mmse'] >= 28 and row['mmse'] <= 30:
#         print("10")
#         act[14].append(row['mmse'])
#         pred[14].append(row['result'])
#         num[9] = num[9] + 1
#
# for i in range(0,15):
#     if len(act[i])>0:
#         print(act[i])
#         print(pred[i])
#         test_rmse = np.sqrt(mean_squared_error(act[i], pred[i]))
#         print(test_rmse)
#         avg.append(test_rmse)
#     elif len(act[i])==0:
#         avg.append(0)
# generate_test_df()
# dic={"file":fname,"count":utterances_count,"text":output_list}
# df1=pd.DataFrame(dic)

# dict={"id1":id1,"control":con,"id2":id2,"dementia":dem}
# df1=pd.DataFrame(dict)

from ast import literal_eval
#
# # generate_test_df()
# df1 = pd.read_pickle('data/adress_test.pickle')
# df1 = pd.read_excel('data/DA_vector_new.xlsx',sheet_name='Sheet1')
# # df1.to_pickle("data/DA_vector_new.pkl")
# # df1= pd.read_pickle("data/DA_vector_new.pkl")
# for index, row in df1.iterrows():
#     # print(row['text'])
#     lst=literal_eval(row['text'])
#     output_list.append(lst)
#
#     # for elem in lst:
#
# df1['utterance']=output_list
# df1.to_pickle("data/DA_vector_new.pkl")
dict={'file2':[],'DA_refined':[]}
def combine_text(text):
    size=len(text)
    count=0
    utterance=[]

    for indx,row in text.iterrows():
        result = [word for word in row['utterance'] if contains_letters(word)]
        result = [word for word in result if word not in ['gram', 'exc','xxx']]
        count+=1
        if(count==size):
            result.insert(0,'\n')
        else:
            result.insert(0,'\n')

        utterance.extend(result)
    dict['utterance'].append(utterance)
    dict['file'].append(text.iloc[0]['file'])
    dict['label'].append(text.iloc[0]['label'])
def combine_coarse_DA(da,max_length):
    coarse_tags={
        'task:question':['Question:General','Question:Reflexive',"Instruction"],
        'task:answer': ['Answer:t1', 'Answer:t6',"Answer:Yes","Answer:No","Answer:General"],
        'feedback':["Acknowledgment","Request:Clarification","Feedback:Reflexive"],
        'time_manage':["Stalling"],

        'multi':['Answer:t1', 'Answer:t2','Answer:t3', 'Answer:t4','Answer:t5', 'Answer:t6','Answer:t7', 'Answer:t8']

    }

    columns = da.columns.values.tolist()
    print(columns)
    final_list = []
    for idx, element in da.iterrows():
        j_list = [0,0,0,0,0,0]
        count=0
        # print(element)
        for j in columns:
            print(j)
            if element[j]==1:
                if j in coarse_tags['task:question'] :
                    j_list[2]=1
                elif j in coarse_tags['task:answer'] and j_list[3]==0:
                    j_list[3] = 1
                elif j in coarse_tags['feedback']:
                    j_list[4] = 1
                elif j in coarse_tags['time_manage']:
                    j_list[5] = 1
                if j in coarse_tags['multi']:
                    count+=1
                    if count>2:
                        j_list[3] = count
                        count=-12
                        break

        total=0
        print('count %d' % count)
        for ele in range(0, len(j_list)):
            total = total + j_list[ele]
        if total==0:
            j_list[1]=1

            # j_list.append(element[j])
        # print(j_list)
        final_list.append(j_list)
    size=len(final_list)
    while(size<max_length) :
        da=[0]*(len(j_list))
        da[0]=1
        final_list.append(da)
        size+=1

    dict['DA'].append(final_list)

def combine_DA(da,max_length):

    
    columns = da.columns.values.tolist()
    
    # print(columns)
    final_list = []
    for idx, element in da.iterrows():
        j_list = [0]
        # print(element)
        for j in columns:
            j_list.append(element[j])
        final_list.append(j_list)
    size=len(final_list)
    while(size<max_length) :
        da=[0]*(len(columns)+1)
        da[0]=1
        final_list.append(da)
        size+=1

    dict['DA_refined'].append(final_list)
def count_max_length(df):
    max=-1
    files=-1
    for file in df['file'].unique():
        df1=df.loc[df['file']==file]

        # print(file+" :")
        # print(len(df1))
        if(len(df1)>max):

            max=len(df1)
            files=file
            # print("maximum %d" % (max))
    return max

def wide_to_long_DA():
    df = pd.read_pickle('DA_label_wide_final.pickle')
   
    df_new = pd.read_pickle('DA_annotation_long_all.pickle')
    # df_PAR=df.loc[df['PAR']==1]
    df_PAR = df
    max=count_max_length(df_PAR)
    print('maximum %d'%max)
    print(df_new)
    # print(df_PAR.columns.tolist())
    # print(len(df_PAR.columns.tolist()))
    # print(df_PAR['Suggestion'].sum())
    # df_select=df_PAR.iloc[:,2:27]
    # print(df_select.sum(axis=0, skipna=True))
    # # max,f=count_max_length(df_PAR)
    # # print("max")
    # # print(max)
    # # print(f)
    df_drop=df_PAR.drop(['Suggestion','Offer',"Farewell","Correction","Apology","Request","Other"], axis = 1)
    print(df_PAR.columns.get_loc("Answer:t3"))
    print("columns after drop")
    print(df_drop.columns.tolist())
    start=df_drop.columns.get_loc('Answer:t3')
    stop = df_drop.columns.get_loc('file')
    # print('start stop %d %d'%(start,stop))
    # print(df_drop.columns.tolist())
    # print(len(df_drop.columns.tolist()))
    # print()
    print('start stop %d %d'%(start,stop))
    for elem in df_new['file'].unique():
        # print(elem)
        # combine_text(df.loc[df['file'].isin(da)])
        da=df_drop.loc[df_drop['file']==elem]
        
        combine_DA(da.iloc[:, start:stop], max)
        # combine_coarse_DA(da.iloc[:,start:stop],max)
        # if(elem.strip()=='686-0'):
        #     dict['file'].append('S172')
        # else:
        dict['file2'].append(elem)
        

    df_new['DA_refined']=dict['DA_refined']
    df_new['file2'] = dict['file2']

    # df_long=pd.DataFrame(dict)
    with open('DA_annotation_long.pickle', 'wb') as f:
        pickle.dump(df_new, f)
    df_long=pd.read_pickle('DA_annotation_long.pickle')
    # print(df_long['file'])
    # # print(df_long['DA'].head(2))
    # # df_long.at['DA', '686-0'] = 'S172'
    # # with open('data/address_DA_long_refined.pickle', 'wb') as f:
    # #     pickle.dump(df_long, f)
    for idx,row in df_long.iterrows():
        print(row['file'])
        print(len(row['DA_refined']))
        for elem in row['DA_refined']:
            print(elem)
        break
    return df_new

def show_stat_group():
    dic_dist = {
        'Answer:t3': 0,
        'Answer:t2': 0,
        'Answer:t1': 0,
        'Answer:t4': 0,
        'Answer:t5': 0,
        'Answer:t6': 0,
        'Answer:t7': 0,
        'Answer:t8': 0,
        'Question:General': 0,
        'Question:Reflexive': 0,
        "Answer:Yes": 0,
        "Answer:No": 0,
        "Answer:General": 0,
        "Instruction": 0,
        "Suggestion": 0,
        "Request": 0,
        "Offer": 0,
        "Acknowledgment": 0,
        "Request:Clarification": 0,
        "Feedback:Reflexive": 0,
        "Stalling": 0,
        "Correction": 0,
        "Farewell": 0,
        "Apology": 0,

        "Other": 0

    }
    df_agreement = pd.read_pickle('data/DA_label_wide_final.pickle')
    df_con = df_agreement.loc[df_agreement['label'] == 0]
    df_dem = df_agreement.loc[df_agreement['label'] == 1]
    keys = dic_dist.keys()
    for key in keys:

        dic_dist[key]=((df_agreement.loc[(df_agreement['label'] == 0) & (df_agreement['speaker'] == '*INV') , key].sum())/len(df_con))*100
        # dic_dist[key]=df_agreement.query("a == 1 and c == 2")['b'].sum()
    print("Control")
    print(dic_dist)
    print("clearing")

    for key in dic_dist.keys():
        dic_dist[key] = 0
    print(dic_dist)
    for key in keys:

        dic_dist[key]=((df_agreement.loc[(df_agreement['label'] == 1) & (df_agreement['speaker']== '*INV'), key].sum())/len(df_dem))*100
    print("Dementia")
    print(dic_dist)

    # for i, v in df_con.items():
    #     print('index: ', i, 'value: ', v)
    # for index, row in df_agreement.iterrows():


def contains_letters(phrase):
    return bool(re.search('[a-zA-Z]', phrase))
def make_wide_dataframe():
    import ast
    df_agreement=pd.read_pickle('data/DA_label_full.pickle')
    # print(df_agreement[1170:]['DA_Label'])
    length=len(df_agreement)
    dic_dist = {
        'Answer:t3': [0]*length,
        'Answer:t2': [0]*length,
        'Answer:t1': [0]*length,
        'Answer:t4': [0]*length,
        'Answer:t5':[0]*length,
        'Answer:t6': [0]*length,
        'Answer:t7': [0]*length,
        'Answer:t8': [0]*length,
        'Question:General': [0]*length,
        'Question:Reflexive': [0]*length,
        "Answer:Yes": [0]*length,
        "Answer:No": [0]*length,
        "Answer:General": [0]*length,
        "Instruction": [0]*length,
        "Suggestion": [0]*length,
        "Request": [0]*length,
        "Offer": [0]*length,
        "Acknowledgment": [0]*length,
        "Request:Clarification": [0]*length,
        "Feedback:Reflexive": [0]*length,
        "Stalling": [0]*length,
        "Correction": [0]*length,
        "Farewell":[0]*length,
        "Apology": [0]*length,

        "Other": [0]*length

    }
    track=0
    flag=0
    keys=dic_dist.keys()
    # for index, row in df_agreement.iterrows():
    #     try:
    #         # print("index %d" % (index))
    #         # print(row)
    #
    #         # for item in row['predicted_ovr']:
    #         #     print("index")
    #         for key in keys:
    #
    #             if key in row['DA_Label']:
    #                 print(row['DA_Label'])
    #                 print(type(row['DA_Label']))
    #                 print(index)
    #                 dic_dist[key][index]=1
    #
    #
    #     except KeyError:
    #         print("error")
    #
    #         continue
    for index, row in df_agreement.iterrows():
        try:
            # print("index %d" % (index))
            # print(row)

            # for item in row['predicted_ovr']:
            #     print("index")
            print(row['chat_file_name'])
            print(row['utterance_id'])
            if row['chat_file_name']=='002-0' and row['utterance_id']==0:
                dic_dist['Answer:t1'][index] = 1
                dic_dist['Answer:t2'][index] = 1
                dic_dist['Answer:t3'][index] = 1
                dic_dist['Answer:t5'][index] = 1
                dic_dist['Answer:t6'][index] = 1
                continue
            if row['chat_file_name']=='001-2' and row['utterance_id']==10:
                dic_dist['Answer:t3'][index] = 1
                dic_dist['Answer:t4'][index] = 1
                continue
            for key in keys:
                x = ast.literal_eval(row['DA_Label'])
                print(x)
                print(type(x))

                for term in x:
                    str=term.replace(" ","")
                    if str.startswith('Acknow'):
                        dic_dist['Acknowledgment'][index] = 1
                        continue

                    print(str)
                    if str ==key:
                        print(index)
                        dic_dist[key][index]=1


        except KeyError:


            continue
    df_wide = pd.DataFrame.from_dict(dic_dist )
    df_wide['file']=df_agreement.iloc[:]['chat_file_name']
    df_wide['speaker'] = df_agreement.iloc[:]['speaker']

    df_wide['utterance_id'] = df_agreement.iloc[:]['utterance_id']
    df_wide['label'] = df_agreement.iloc[:]['label']
    # df_wide['utterance'] = df_agreement.iloc[:]['utterance']
    print(len(df_wide))

    with open('data/DA_label_wide_final.pickle', 'wb') as f:
        pickle.dump(df_wide, f)
    return df_wide
def create_frequency(df):
    frequency=[]
    keys=[]
    files = df['file'].unique()

    # print(cols[1:26])
    for key in files:
        rows=[]
        print(key)
        df_slice=df.loc[df['file']==key]
        df_slice_par=df_slice.loc[df_slice['PAR']==1]
        df_slice_col=df_slice_par.iloc[:,1:26]
        for col in df_slice_col.columns.tolist():
            rows.append(df_slice_col[col].sum()/len(df_slice_col))
            print(col)

            print(df_slice_col[col].sum())
            print(len(df_slice_col))
            print(rows)
        frequency.append(rows)
        keys.append(key)
        # print(df_slice_col.columns.tolist())
        # print(df_slice_col)
        # print(type(df_slice_col))

    dict={'file':keys,'feature_DA':frequency}
    df_da=pd.DataFrame(dict)
    with open('data/full_address_DA_freq.pickle', 'wb') as f:
        pickle.dump(df_da, f)
    writer = pd.ExcelWriter('data/full_address_DA_freq.xlsx', engine='xlsxwriter')
    # # # # # #

    df_da.to_excel(writer, sheet_name='Sheet1',columns=df_da.columns.tolist())
    writer.save()

def create_dict(cols):
    dict={}
    for col in cols:
        dict[col]=[]
    return dict
def significance_test():
    from statsmodels.stats.proportion import proportions_ztest
    import numpy as np
    dic_dist = {
        'Answer:t3': 0,
        'Answer:t2': 0,
        'Answer:t1': 0,
        'Answer:t4': 0,
        'Answer:t5': 0,
        'Answer:t6': 0,
        'Answer:t7': 0,
        'Answer:t8': 0,
        'Question:General': 0,
        'Question:Reflexive': 0,
        "Answer:Yes": 0,
        "Answer:No": 0,
        "Answer:General": 0,
        "Instruction": 0,
        "Suggestion": 0,
        "Request": 0,
        "Offer": 0,
        "Acknowledgment": 0,
        "Request:Clarification": 0,
        "Feedback:Reflexive": 0,
        "Stalling": 0,
        "Correction": 0,
        "Farewell": 0,
        "Apology": 0,

        "Other": 0

    }
    # coarse_tags={
    #     'task:question':['Question:General','Question:Reflexive',"Instruction"],
    #     'task:answer': ['Answer:t1', 'Answer:t6',"Answer:Yes","Answer:No","Answer:General"],
    #     'feedback':["Acknowledgment","Request:Clarification","Feedback:Reflexive"],
    #     'time_manage':["Stalling"],
    #     'other':["Other"],
    #     'multi':['Answer:t1', 'Answer:t2','Answer:t3', 'Answer:t4','Answer:t5', 'Answer:t6','Answer:t7', 'Answer:t8']
    #
    # }
    coarse_tags = {
        'task': ['Question:General', 'Question:Reflexive', "Instruction",'Answer:t1', 'Answer:t6', "Answer:Yes", "Answer:No", "Answer:General",'Answer:t2', 'Answer:t3', 'Answer:t4', 'Answer:t5','Answer:t7',
                  'Answer:t8'],

        'feedback': ["Acknowledgment", "Request:Clarification", "Feedback:Reflexive"],
        'time_manage': ["Stalling"],
        'som': ["Farewell"],
        'multi': ['Answer:t1', 'Answer:t2', 'Answer:t3', 'Answer:t4', 'Answer:t5', 'Answer:t6', 'Answer:t7',
                  'Answer:t8']

    }
    df_agreement=pd.read_pickle('data/DA_label_wide_final.pickle')
    start=df_agreement.columns.get_loc('Answer:t3')
    stop=df_agreement.columns.get_loc('Other')
    # df_con = df_agreement.loc[(df_agreement['label'] == 0) & (df_agreement['speaker'] == '*PAR')]
    # df_dem = df_agreement.loc[(df_agreement['label'] == 1 )& (df_agreement['speaker'] == '*PAR')]
    df_con = df_agreement.loc[df_agreement['label'] == 0 ]
    df_dem = df_agreement.loc[df_agreement['label'] == 1 ]
    sample_size_a = len(df_con)
    sample_size_b = len(df_dem)
    keys=coarse_tags.keys()
    count_a=0
    count_b = 0
    # can we assume anything from our sample
    significance = 0.05
    # our samples - 82% are good in one, and ~79% are good in the other
    # note - the samples do not need to be the same size
    print(start)
    print(stop)
    # for idx, row in df_agreement.iterrows():
    #     sum=0
    #     for i in range(start,stop+1):
    #         # sum+=df_agreement.at(idx,i)
    #         sum+=row[i]
    #     if sum>4:
    #         if row['label']==0 and row['speaker']=='*PAR':
    #             count_a+=1
    #         elif row['label']==1 and row['speaker']=='*PAR':
    #             count_b+=1


    for key in keys:

        print(key)
        if key=='multi':
            break
        # sample_success_a, sample_size_a = (((df_agreement.loc[(df_agreement['label'] == 0)  , key].sum())), len(df_con))
        # sample_success_b, sample_size_b = (((df_agreement.loc[(df_agreement['label'] == 1)  , key].sum())), len(df_dem))
        sample_success_a=0
        sample_success_b=0
        for value in coarse_tags[key]:
            # sample_success_a  =sample_success_a+ ((df_agreement.loc[(df_agreement['label'] == 0) & (df_agreement['speaker'] == '*PAR' )  , value].sum()))
            # sample_success_b  =sample_success_b+ ((df_agreement.loc[(df_agreement['label'] == 1) & (df_agreement['speaker'] == '*PAR')   , value].sum()))
            sample_success_a = sample_success_a + ((df_agreement.loc[(df_agreement['label'] == 0) , value].sum()))
            sample_success_b = sample_success_b + ((df_agreement.loc[(df_agreement['label'] == 1) , value].sum()))

        # check our sample against Ho for Ha != Ho
        # sample_success_a, sample_size_a=(count_a,len(df_con))
        # sample_success_b, sample_size_b = (count_b, len(df_dem))
        print("all speaker")
        # print(sample_success_a)
        # print(sample_size_a)
        print('sample sum and size of control %d %d' % (sample_success_a, sample_size_a))
        print('sample sum and size of dementia %d %d' % (sample_success_b, sample_size_b))
        if(sample_success_a>=10 and sample_size_a-sample_success_a>=10 and sample_success_b>=10 and sample_size_a-sample_success_b>=10 ):
            successes = np.array([sample_success_a, sample_success_b])
            samples = np.array([sample_size_a, sample_size_b])
            print(successes)
            print(samples)
            # note, no need for a Ho value here - it's derived from the other parameters
            stat, p_value = proportions_ztest(count=successes, nobs=samples, alternative='two-sided')
            # report
            print('z_stat: %0.3f, p_value: %0.3f' % (stat, p_value))
            if p_value > significance:
                print("Fail to reject the null hypothesis - we have nothing else to say")
            else:
                print("Reject the null hypothesis - suggest the alternative hypothesis is true")
def make_progression_data():
    print('progression')

    data=pd.read_excel("../data/Pitt-data.xlsx",sheetname='data')
    for idx,row in data.iterrows():
        if row.mms is not None and row.mmse2 is not None:
            if abs(row.mms-row.mmse2)>=5:
                with open ('../data/progression/decline.txt','a') as f:
                    f.write(row.id+'\n')
                    f.close()



    print('done saving progression')

def select_fluency_data(fluency,result):
    folders = ['Control', 'Dementia']

    while len(result)<15:
        r1 = random.randint(0, len(fluency))
        result.add(fluency[r1])

    for folder in folders:

        PATH = "../data/Pitt/fluency/" + folder

        for path, dirs, files in os.walk(PATH):


            for filename in result:

                fullpath = os.path.join(path, filename)
                filename = int(filename.split('.')[0])

                with open(fullpath, 'r', encoding="utf8",errors='ignore')as input_file:
                    file_tokenization(input_file, "../data/Pitt/selected/",filename+'.txt')
                        # count+=1

        print('done saving')
def found(match,file):
    for elem in file:
        if elem.split('.')[0].split('-')[1]==match:
            return False
    return True
def make_fluency_data():
    data=pd.read_pickle('../data/DA_annotation_long_all.pickle')
    file=data.file.unique()
    fluency=[]
    folders = ['Control', 'Dementia']
    result = set()

    for folder in folders:

        PATH = "../data/Pitt/fluency/" + folder
        # file_to_write = open("data/segment.txt", 'a')
        for path, dirs, files in os.walk(PATH):
            for filename in files:
                if folder=='Control' and filename.split('.')[0] not in file:
                    result.add(filename)
                elif filename.split('.')[0] not in file and not found(filename.split('.')[0].split('-')[1],file) and  \
                        filename.split('.')[0].split('-')[1]=='0':
                    fluency.append(filename)
    select_fluency_data(fluency,result)

def print_result(true, test):
    f1 = f1_score(true, test)
    acc = accuracy_score(true, test)
    precision = precision_score(true, test)

    recall = recall_score(true, test)

    fpr, tpr, _ = roc_curve(true, test)

    print("acc: %f" % (acc))
    print("precision")
    print(precision)
    print("recall")
    print(recall)
    print("f1")
    print(f1)
    print("false pos rate, true pos rate")
    print(fpr)
    print(tpr)
    # plot_ROC(fpr, tpr)
def main():
    option=[1,2]
    for opt in option:
        if opt==1:
            make_fluency_data()
        if opt==2:
            make_progression_data()
if __name__=="__main__":
    main()
# df=pd.read_pickle()
# generate_single_utterances_dataframe()
# df_long=make_wide_dataframe()
# show_stat_group()
# significance_test()
# df_long=wide_to_long_DA()
# wide_to_long_DA()
# df_train = pd.read_pickle('data/all_address_DA_com.pickle')
# print(df_train['label'])
# df_train = pd.read_pickle('data/adress_test_DA.pickle')
# df_long=pd.read_pickle('data/address_DA_PAR_long.pickle')
# df_train = pd.read_pickle('data/address_DA_long_train_PAR.pickle')
# df_test = pd.read_pickle('data/address_DA_long_test_PAR.pickle')

# import ast
# df=pd.read_excel('data/full_address_DA_freq.xlsx',sheet_name="Sheet1")
# print(df['feature_DA'])
# df_train_da_1 = pd.read_pickle('data/adress_test_DA.pickle')
#
#
#
# print(df_train_da_1.columns.tolist())
# print(df_train_da_1['feature_DA'])
# dictionary={'feature_DA_PAR':[]}
# for idx,row in df_train.iterrows():
#     print(row['file'])
#     elem=df_long.loc[df_long['file']==row['file'].strip()]
   
#     for e in elem['DA']:
#         print(e)
#         print(len(e))
#         dictionary['feature_DA_PAR'].append(e)
    
# df_train['DA_PAR']=dictionary['feature_DA_PAR']
# print(df_train['DA_PAR'])
# print(df_train.columns.tolist())
# print(df_train.columns.get_loc('DA_PAR'))
# print(df_train.columns.get_loc('mmse'))
# print(df_train.columns.get_loc('label'))
# print(df_test.columns.get_loc('DA_PAR'))
# print(df_test.columns.get_loc('mmse'))
# print(df_test.columns.get_loc('class'))
#     pickle.dump(df_train, f)
# df_long=pd.read_pickle('data/address_DA_long_test_PAR.pickle')
# print(df_long.head(1))
# with open('data/address_DA_long_test_PAR.pickle', 'wb') as f:
# print(df_long.iloc[0]['DA_PAR'])
# print(df_long.columns.get_loc("label"))
# df=df_long.loc[0]
# print(df['label'])
#
#
# print(len(df_long['feature_DA'][1][0]))
# print(df_long['feature_DA'].head(10))
# print(len(df_long))

# for idx, row in df_long.iterrows():
#     print(row['file'])
#     print(len(row['feature_DA']))
#     for elem in row['feature_DA']:
#         print(elem)
#     break

# df1=pd.read_pickle('data/adress_test_DA.pickle')
# df1=df1.drop(['class'],axis=1)
# for idx, row in df1.iterrows():
#     if row['label'].strip()=='Control':
#         dict['class'].append(0)
#     elif row['label'].strip()=='Dementia':
#         dict['class'].append(1)

# df1['class']=dict['class']
# df2=pd.read_pickle('data/adress_test_DA.pickle')
# df_train_da_1 = pd.read_pickle('data/adress_full_interview_features_DA_train.pickle')
# df1['feature_DA']=df_train_da_1['feature_DA']
# print('train')
# print(df1.columns.tolist())
# print(df1.columns.tolist().index('Unnamed: 0'))
# print('test')
# print(df2.columns.tolist())
# print(df2.columns.tolist().index('feature_DA'))
# print(df1.head(1))


# df_wide=pd.read_excel('data/full_agreement_wide_extended.xlsx',sheet_name="Sheet1")
#
# # print(df_wide.columns.tolist().index('Other'))
# dict=create_dict(df_wide.columns.tolist()[1:26])
# cols=df_wide.columns.tolist()[1:26]
#
# # print(len(cols))
# for idx, row in df1.iterrows():
#     count=0
#     x = ast.literal_eval(row['feature_DA'])
#     for item in x:
#         print(item)
#         dict[cols[count]].append(item)
#         count=count+1
#
#
# df_new=pd.DataFrame(dict)
# # Place the DataFrames side by side
# horizontal_stack = pd.concat([df1, df_new], axis=1)
# horizontal_stack.drop(['feature_DA'], axis=1)
# print(horizontal_stack.columns.tolist().index('Answer:t3'))
# print(horizontal_stack.columns.tolist().index('Other'))
# print(horizontal_stack.head(1))
# with open('data/adress_test_DA.pickle', 'wb') as f:
#     pickle.dump(df1, f)
# df_train_da_2 = pd.read_pickle('data/adress_test_DA.pickle')
# print(df_train_da_2['class'])
#
# DA=[]
# for idx, row in df_train_da_1.iterrows():
#     num=[]
#     print(row['file'])
#     df_slice=df_train_da_2.loc[df_train_da_2['file']==row['file'].strip()]
#
#     for n in df_slice['feature_DA']:
#         print(n)
#
#         # if isinstance(n, int) or isinstance(n, float):
#         #     num.append(n)
#         #     print(n)
#
#         DA.append(n)
# print(DA)
# print(len(DA))
# print(len(df_train_da_1))
# df_train_da_1['feature_DA']=DA
# # print(df_train_da_1.columns.tolist())
# print(df_train_da_1)
# with open('data/adress_test_DA.pickle', 'wb') as f:
#     pickle.dump(df_train_da_1, f)
# create_frequency(df_train_da_1)
# df_train_da_1.drop(['chat_file_name'], axis=1)
# df_wide=pd.read_excel('data/all_address_DA.xlsx',sheet_name="Sheet1")
# df_train_da_1['chat_file_name']=df_wide['chat_file_name']
# with open('data/adress_full_interview_features_DA_train.pickle', 'wb') as f:
#     pickle.dump(df_train_da_1, f)
# df_train_da_1 = pd.read_pickle('data/full_agreement_wide_extended.pickle')
# df_train_da_2 = pd.read_pickle('data/full_agreement_wide_extended.pickle')
# df2=pd.read_excel('data/matched_test_wid_DA.xlsx',sheet_name="Sheet1")
# df3=pd.read_excel('data/matched_training_wid_DA.xlsx',sheet_name="Sheet1")
# df4=pd.concat([df_train_da_1,df_wide],ignore_index=True)
# df5=df_train_da_2.loc[df_train_da_2['chat_file_name'].isin(df4['fname'] )]
# name=[]
# for idx, row in df5.iterrows():
#     if row['chat_file_name'] in df4['fname']:
#         print(row['chat_file_name'])
#         slice=df4.loc[df4['fname']==row['chat_file_name']]
#         # df5.loc[idx,'chat_file_name']=slice['file']
#         name.append(slice['file'])
#
#
# df5.drop(['utterance_id'],axis=1)
# df5['file_name']=name
# df_train_da_1.drop(['label'],axis=1)
# df5.rename(columns={'chat_file_name': 'file'}, inplace=True)
# print(len(df4))
# print(df4['file'])
# print(df_train_da_1.columns.tolist())
# print(len(df5.columns.tolist()))
# print(len(df_train_da_1.columns.tolist()))
# df6=pd.concat([df5,df_train_da_1],ignore_index=True)
#
# print(df6.columns.tolist())
# print(df6['file'])
# df_train_full=pd.concat([df_train_da_1,df_train_da_2],ignore_index=True)
# print(df_train_da_1.columns.tolist())
# print(df_train_da_2.columns.tolist())
# df_test = pd.read_pickle('data/adress_test.pickle')
# # print("train")
# print(len(df_train))
# for indx,row in df_train.iterrows():
#
#     if row['file'].startswith('S')==False:
#         print(row['file'])
# # print(df_train.iloc[0:2]['utterances'])
# print("test")
# print(df_test.iloc[0:2]['file'])
# print(df_train_da_1.columns.tolist())
# print(len(df_train_da_1))
# generate_test_df()
# with open('data/file_to_text_train.pickle', 'rb') as f:
#     df = pickle.load(f)
#
# # with open('data/full_agreement_wide_extended.pickle', 'rb') as f:
# #     df_wide = pickle.load(f)
# df=pd.read_excel('data/raw_result_model_hidden_after_DA.xlsx',sheet_name='Sheet1')


# df1=pd.read_pickle('data/raw_result_model_fidden_after_DA.pickle')
# print("test")
# print(len(df1['file'].unique()))
# print(df1['file'].unique())
# df1=pd.read_pickle('data/file_to_text_train.pickle')
# print("train")
# print(len(df1['file'].unique()))
# print(df1['file'].unique())
# print(df.iloc[10]['utterance'])
# print(df.iloc[:50]['INV'])

# # df1=df1.drop(df1.index[0])
# df2=pd.read_excel('data/DA_files.xlsx',sheet_name='Sheet1')
# # # df2_new=df2.rename(columns={'file': 'fname'})
# df3=merge(df1, df2,on='fname')[['file', 'fname']]

# print(df_wide.head(2))
# print(df_wide.iloc[0:5]['utterance'])

# df4=pd.read_pickle('data/file_to_pid_merged.pickle')
# # # df_new=df.rename(columns={'file': 'fname'})
# print(len(df))
# print(df.head(-15))
# with open('data/file_to_pid_merged_train.pickle', 'wb') as f:
#     pickle.dump(df, f)
# df = pd.read_excel('data/Pitt_DA.xlsx',sheet_name='Sheet1')
# print_result(df['true'],df['result'])
# df = pd.read_pickle('data/DA_vector_refined.pickle')
# print(df)
# print(len(df))
# print(df.columns.values.tolist())
# print(df['embedding'])
# print((df['label'].value_counts()/df['label'].count())*100)

#
# dementia_dataframe = pd.DataFrame(dict)
#
# with open('data/DA_vector_refined.pickle', 'wb') as f:
#       pickle.dump(dementia_dataframe, f)
# max=0
# for index, row in df1.iterrows():
#
#
#
#     if len(row['utterance'])>max:
#         max=len(row['utterance'])
#
# print(max)


# # print(df1)
# print(df["gender"])
# print(df1["gender"])
# print(list(df.columns))
# print(list(df1.columns))
# transcript_to_file()
# dementia_dataframe=generate_full_interview_dataframe()
# dementia_dataframe=generate_test_df()
# print(list(dementia_dataframe.columns))
# print(dementia_dataframe.head(5))
# generate_full_interview_dataframe()
# voting()
# from cmath import sqrt
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import scipy.stats
# df1 = pd.read_excel('data/adress_acoustic_GP_final.xlsx',sheet_name='Sheet1')
# y_test=df1.loc[:,"test"]
# y_pred=df1.loc[:,"result"]
# test_rmse = sqrt(mean_squared_error(y_test , y_pred ))
# test_MAE = mean_absolute_error(y_test , y_pred )
# pearson = scipy.stats.pearsonr(y_test , y_pred )
# r2 = r2_score(y_test , y_pred )
# print("rmse " )
# print(test_rmse)
# print("mae  " )
# print(test_MAE)
# print("pearson")
# print(pearson)
# print("r2 %f"%(r2))


# df['word']=df1['word']
# columns={'short_pause_count':ft_1,'long_pause_count':ft_2,'very_long_pause_count':ft_3,
#                             'word_repetition_count':ft_4,
#                             'retracing_count':ft_5,
#                             'filled_pause_count':ft_6,
#                             'incomplete_utterance_count':ft_7,
#                             'label':labels,
#                             'filename':fname,
#                             'utterance_count':utterances_count
#                             }

# df['short_pause_count']=[int(b) / int(m) for b,m in zip(ft_1, word_count)]
# df['long_pause_count']=[int(b) / int(m) for b,m in zip(ft_2, word_count)]
# df['very_long_pause_count']=[int(b) / int(m) for b,m in zip(ft_3, word_count)]
# df['word_repetition_count']=[int(b) / int(m) for b,m in zip(ft_4, word_count)]
# df['retracing_count']=[int(b) / int(m) for b,m in zip(ft_5, word_count)]
# df['filled_pause_count']=[int(b) / int(m) for b,m in zip(ft_6, word_count)]
# df['incomplete_utterance_count']=[int(b) / int(m) for b,m in zip(ft_7, word_count)]
# df['utterance_count']=utterances_count
# df['word']=word_count
# print(df['word_repetition_count'])
# print(df['filled_pause_count'])
# print(len(fname))
# print(len(labels))
# print(len(utterances))
# df1=plot_hist()
# # columns={'file':fname,'label':labels,'utterances':utterances}
# # dementia_dataframe=pd.DataFrame(columns)
# # # # generate_single_utterances_dataframe()
# # #
# # # # dementia_dataframe = pd.DataFrame(dementia_list)
# # # # # print(dataframe)
# with open('data/adress_final_interview.pickle', 'wb') as f:
#      pickle.dump(df, f)
#
# df=pd.read_pickle('data/adress_final_interview.pickle')
# print(list(df.columns))
# print(df.shape)
# print(list(df.columns))
# print(df.head(3))

# writer = pd.ExcelWriter('DA_label_wide_final.xlsx', engine='xlsxwriter')
# # # # # # # # # # # #
# # # # # # df_train_da_1.to_excel(writer, sheet_name='Sheet1',columns=df_train_da_1.columns.tolist())
# # # # # # ['short_pause_count','long_pause_count',
# # # # # # # # # # # # # 'very_long_pause_count','word_repetition_count','retracing_count','filled_pause_count','incomplete_utterance_count',
# # # # # # # # # # # # # 'label','filename','utterance_count'])
# df_long.to_excel(writer, sheet_name='Sheet1',columns=df_long.columns.tolist())
# writer.save()

# plot_test()