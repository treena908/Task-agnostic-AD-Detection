from pickle import DEFAULT_PROTOCOL
import pandas as pd
import re
import sys
import os
def create_unigram_tag(row1,da1):
    multi_tag = re.compile(r"Answer:t\d")
    DA1=""
    # print('da')
    # print(da1)
    
    if len(da1)==1 and re.search(multi_tag,da1[0]):
        DA1=str(row1['speaker'])[1]+"_"+str(da1[0])   
    elif len(da1)>1 and re.search(multi_tag,da1[0]):
        DA1 = str(row1['speaker'])[1] + "_" + 'ans_Mtopic'
    elif len(da1)==1 and multi_tag not in da1:
        DA1=str(row1['speaker'])[1]+"_"+str(da1[0])
    return DA1
def create_tag(row1,da1):
    multi_tag = re.compile(r"Answer:t\d")
    DA1=""
    # print('da')
    # print(da1)
    
    if len(da1)==1 and re.search(multi_tag,da1[0]):
        DA1=str(row1['speaker'])[1]+"_"+'ans_topic'
    elif len(da1)>=3 and re.search(multi_tag,da1[0]):
        DA1 = str(row1['speaker'])[1] + "_" + 'Mtopic'
    elif len(da1)>1 and re.search(multi_tag,da1[0]):
        DA1 = str(row1['speaker'])[1] + "_" + 'ans_Mtopic'
    elif len(da1)==1 and multi_tag not in da1:
        DA1=str(row1['speaker'])[1]+"_"+str(da1[0])
    return DA1
def acccumulate_tags(DA_columns,row1):
  da1=[]
  for tag in DA_columns:
      if row1[tag]==1:
        da1.append(tag)
  return da1
def acccumulate_topic_tags(DA_columns,row1):
  da1=[]
  for tag in DA_columns:
      if row1[tag]==1:
        da1.append(tag)
  return da1  
def extract_trigram_DA(row1,row2,row3,DA_columns,option):
    
    
    
   
    da1=acccumulate_tags(DA_columns,row1)
    da2=acccumulate_tags(DA_columns,row2)
    da3=acccumulate_tags(DA_columns,row3)
  
    DA1=create_tag(row1,da1)
    DA2 = create_tag(row2, da2)
    DA3 = create_tag(row3, da3)
    return DA1,DA2,DA3
def extract_unigram_DA(row1,DA_columns,option):
    
    
    
    
    da1=acccumulate_tags(DA_columns,row1)
    
  
    DA1=create_unigram_tag(row1,da1)
    
    return DA1
def extract_bigram_DA(row1,row2,DA_columns,option):
    
    
    
    
    da1=acccumulate_tags(DA_columns,row1)
    da2=acccumulate_tags(DA_columns,row2)
  
    DA1=create_tag(row1,da1)
    DA2 = create_tag(row2, da2)
    return DA1,DA2
    # elif option==2:
    #   da1=acccumulate_coarse_tags(DA_columns,row1)
    #   da2=acccumulate_coarse_tags(DA_columns,row2)
    
    #   DA1=create_tag(row1,da1)
    #   DA2 = create_tag(row2, da2)
    #   return DA1,DA2
   
# def extract_bigram_DA(row1,row2,DA_columns,option):
    
    
    
    
#     da1=acccumulate_topic_tags(DA_columns,row1)
    
  
#     DA1=create_topic_tag(row1,da1)
    
#     return DA1
def print_topic_concentration(bigram,df,name,ct,ad,delete):
    
    save=0
    
    if not os.path.isfile('data/'+name+'_'+'feature_list.pickle'):
      save=1
      feature_list={name:[]}
    elif os.path.isfile('data/'+name+'_'+'feature_list.pickle'):
      f_list=pd.read_pickle('data/'+name+'_'+'feature_list.pickle')
      if name not in f_list.columns.tolist() or delete==1:
        save=1
        feature_list={name:[]}
        
    if save==0:
      return
    for key in bigram.keys():
      if bigram[key]['ct']+bigram[key]['ad']>5:
        result=[]
        
        # print(bigram[key])
        for k in bigram[key].keys():
          if k=='ct' and ct>0:
            
            
            result.append((bigram[key][k]*100)/ct)
            # print("%.3f"%result)
          elif  k=='ad' and ad>0:
            
            result.append((bigram[key][k]*100)/ad)
            # print("%.3f"%result)
        if len(result)>1 and abs(result[0]-result[1])>=.5 and save:
          feature_list[name].append(key)
          # print(key)
          # print(" ct %.3f"%result[0])
          # print(" ad %.3f"%result[1])
    if save:
      save_feature(feature_list,name)      

def compute_number(df,name):
  if name=='uigram' or name=='topic':
    sub=0
  elif name=='bigram':
    sub=1
  else:
    sub=2
  return len(df.loc[df['label']]==1)-sub,len(df.loc[df['label']]==0)-sub

def print_bigram(bigram,df,name,delete):
    
    
    ad,ct=compute_number(df,name)
    label=[]
    file=[]
    save=0
    if not os.path.isfile('data/'+name+'_'+'feature_list.pickle'):
      save=1
      feature_list={name:[]}
    elif os.path.isfile('data/'+name+'_'+'feature_list.pickle'):
      f_list=pd.read_pickle('data/'+name+'_'+'feature_list.pickle')
      if name not in f_list.columns.tolist() or delete==1:
        save=1
        feature_list={name:[]}
        
    if save==0:
      return


    for key in bigram.keys():
      if bigram[key]['ct']+bigram[key]['ad']>5:
        result=[]
        
        # print(bigram[key])
        for k in bigram[key].keys():
          if k=='ct':
            
            
            result.append((bigram[key][k]*100)/ct)
            # print("%.3f"%result)
          else:
            
            result.append((bigram[key][k]*100)/ad)
            # print("%.3f"%result)
        if abs(result[0]-result[1])>=.3 and save:
          # print(key)
          if save:
            feature_list[name].append(key)
          # print(" ct %.3f"%result[0])
          # print(" ad %.3f"%result[1])
    if save:
      save_feature(feature_list,name)


def bigram_DA(option,df,DA_columns,unique_file,delete,final_df):
    bigram={}
    
    df=pd.read_pickle('data/DA_label_wide_final.pickle')
    
    DA_columns=set(df.columns.tolist())-set(["",'file','speaker','utterance_id','label'])
    unique_file=set(df.file)
    if option:
      df_feature=pd.read_pickle('data/bigram_feature_list.pickle')
      feature_list=df_feature['bigram'].tolist()
    for name in unique_file:
        # print(name)
        row=df[df['file']==name]
        row_length=len(df.loc[df['file']==name])
        # print(len(row))
        for id in range(0,len(row)-1):
            
            # print(id)
            if row.iloc[id]['speaker']=='*INV' and row.iloc[id+1]['speaker']=='*PAR' or\
            row.iloc[id]['speaker']=='*PAR' and row.iloc[id+1]['speaker']=='*INV' or\
            row.iloc[id]['speaker']=='*PAR' and row.iloc[id+1]['speaker']=='*PAR':
                  
              da1,da2=extract_bigram_DA(row.iloc[id],row.iloc[id+1],DA_columns,option)
        
              if da1+'_'+da2 not in bigram:
                  bigram[da1+'_'+da2]=dict()
                  bigram[da1+'_'+da2]['ct']=0
                  bigram[da1+'_'+da2]['ad']=0
                  if row.iloc[id]['label']==0:
                      bigram[da1+'_'+da2]['ct']=1
                  else:
                      bigram[da1 + '_' + da2]['ad'] = 1
              else:
                  if row.iloc[id]['label']==0 :
                    if 'ct' in bigram[da1+'_'+da2]:
                      bigram[da1+'_'+da2]['ct']+=1
                    else:
                      bigram[da1+'_'+da2]['ct']=1

                  else:
                      if 'ad' in bigram[da1+'_'+da2]:
                        bigram[da1+'_'+da2]['ad']+=1
                      else:
                        bigram[da1+'_'+da2]['ad']=1
        #for feature
        if option:
          if generate_feature(bigram,feature_list,row_length,final_df,row.iloc[id]['label'])==0:
            print('error generating bigram feature %s'%name)
            sys.exit()
          bigram={}

    if option:
      print('generated bigram')
      return 

    
    print_bigram(bigram,df,'bigram',delete)
def update_dict(label,da1,trigram):
  if da1 not in trigram:
                  
    trigram[da1]=dict()
    trigram[da1]['ct']=0
    trigram[da1]['ad']=0
    if label==0:
        trigram[da1]['ct']=1
    else:
        trigram[da1]['ad'] = 1
  else:
    if label==0 :
      if 'ct' in trigram[da1]:
        trigram[da1]['ct']+=1
      else:
        trigram[da1]['ct']=1

    else:
        if 'ad' in trigram[da1]:
          trigram[da1]['ad']+=1
        else:
          trigram[da1]['ad']=1
def extract_topic_DA(cols,trigram,row):
  
  left_tag = re.compile(r"Answer:t[5-8]")
  right_tag=re.compile(r"Answer:t[1-4]")
  far_left=re.compile(r"Answer:t[6-7]")
  centre_left=re.compile(r"Answer:t[5|8]")
  
  far_right=re.compile(r"Answer:t[1|3]")
  centre_right=re.compile(r"Answer:t[2|4]")
  q_1=re.compile(r"Answer:t[1]")
  q_2=re.compile(r"Answer:t[2]")
  q_3=re.compile(r"Answer:t[3]")
  q_4=re.compile(r"Answer:t[4]")
  q_5=re.compile(r"Answer:t[5]")
  q_6=re.compile(r"Answer:t[6]")
  q_7=re.compile(r"Answer:t[7]")
  q_8=re.compile(r"Answer:t[8]")

  if re.search(far_left,cols):
    da1='far_left'
    update_dict(row,da1,trigram)
    da1='left_tag'
    update_dict(row,da1,trigram)
    if re.search(q_6,cols):
      da1='q_6'
      update_dict(row,da1,trigram)
    if re.search(q_7,cols):
      da1='q_7'
      update_dict(row,da1,trigram)
  elif re.search(far_right,cols):
    da1='far_right'
    update_dict(row,da1,trigram)
    da1='right_tag'
    update_dict(row,da1,trigram)
    if re.search(q_1,cols):
      da1='q_1'
      update_dict(row,da1,trigram)
    if re.search(q_3,cols):
      da1='q_3'
      update_dict(row,da1,trigram)
  elif re.search(right_tag,cols):
    
    da1='right_tag'
    update_dict(row,da1,trigram)
    if re.search(q_2,cols):
      da1='q_2'
      update_dict(row,da1,trigram)
    if re.search(q_4,cols):
      da1='q_4'
      update_dict(row,da1,trigram)
  elif re.search(left_tag,cols):
    
    da1='left_tag'
    update_dict(row,da1,trigram)
    if re.search(q_2,cols):
      da1='q_5'
      update_dict(row,da1,trigram)
    if re.search(q_4,cols):
      da1='q_8'
      update_dict(row,da1,trigram)
    
                  
def topic_concentration(option,df,DA_columns,unique_file,delete,final_df):
    multi_tag = re.compile(r"Answer:t\d")
    trigram={}
    
    ct=0
    ad=0
    
    if option:
      
      df_feature=pd.read_pickle('data/topic_feature_list.pickle')
      feature_list=df_feature['topic'].tolist()
    for name in unique_file:
      # print(name)
      row=df.loc[df['file']==name]
     
      # print(len(row))
     
      for id in range(0,len(row)):
          
          if row.iloc[id]['speaker']=='*PAR': 
            for cols in DA_columns:
              if re.search(multi_tag,cols) and row.iloc[id][cols]==1:
                

                if row.iloc[id]['label']==0:
                  ct+=1
                else:
                  ad+=1
                
                extract_topic_DA(cols,trigram,row.iloc[id]['label'])
                
     
      #for feature
      if option:
        # if name=='527-0':
        #   print('label %d'%row.iloc[id]['label'])
        #   print(trigram)
        #   print(ct)
        #   print(ad)
        #   sys.exit()
        if generate_topic_feature(trigram,feature_list,ct,ad,final_df,row.iloc[id]['label'])==0:
          print('error generating topic feature %s'%name)
          sys.exit()
        ct=0
        ad=0
        trigram={}

    if option:
      print('generated topic concentration')
      return 
                
                
      

              
          
              
    print_topic_concentration(trigram,df,'topic',ct,ad,delete)    
def generate_topic_feature(bigram,feature_list,ct,ad,final_df,label):

  
  for key in feature_list:
      if key in bigram.keys() :
        if label==0 and 'ct' in bigram[key] and bigram[key]['ct']>0  and ct>0:
          if key in final_df:
            final_df[key].append(bigram[key]['ct']/ct)
          else:
            final_df[key]=[]
            final_df[key].append(bigram[key]['ct']/ct)

          # feature.append(bigram[key]['ct']/ct)
        elif label==1 and'ad' in bigram[key] and bigram[key]['ad']>0 and ad>0:
          if key in final_df:
            final_df[key].append(bigram[key]['ad']/ad)
          else:
            final_df[key]=[]
            final_df[key].append(bigram[key]['ad']/ad)
          # feature.append(bigram[key]['ad']/ad)
        else:
         
          print('errors')
          print(key)
          print(ct)
          print(ad)
          print(bigram[key])
          return 0
          
      else:
        if key in final_df:
            final_df[key].append(0.0)
        else:
          final_df[key]=[]
          final_df[key].append(0.0)
  
  return 1
def generate_feature(bigram,feature_list,row_length,final_df,label):
  
  for key in feature_list:
      if key in bigram.keys() :
        
        if label==0 and 'ct' in bigram[key] and bigram[key]['ct']>0 and row_length>0 :
          # feature.append(bigram[key]['ct']/row_length)
          if key in final_df:
            final_df[key].append(bigram[key]['ct']/row_length)
            
          else:
            final_df[key]=[]
            final_df[key].append(bigram[key]['ct']/row_length)

        elif label==1 and 'ad' in bigram[key] and bigram[key]['ad']>0 and row_length>0:
          if key in final_df:
            final_df[key].append(bigram[key]['ad']/row_length)
           
          else:
            final_df[key]=[]
            final_df[key].append(bigram[key]['ad']/row_length)
            
        else:
          print('error generate  feature')
          return 0
      else:
        if key in final_df:
            final_df[key].append(0.0)
        else:
          final_df[key]=[]
          final_df[key].append(0.0)
  
  return 1

        
        
       
def save_feature(feature_set,name):
  if len(feature_set[name])>0:
    df=pd.DataFrame(feature_set)
    df.to_pickle('data/'+name+'_feature_list.pickle')
    print('%s saved'%(name+'_feature_list.pickle'))
    print(len(feature_set[name]))
  else:
    print('no feature to save %s'%(name+'_feature_list.pickle'))
    sys.exit()
  
 
  


def unigram_DA(option,df,DA_columns,unique_file,delete,final_df):

    trigram={}
    
   
    if option:
      df_feature=pd.read_pickle('data/unigram_feature_list.pickle')
      feature_list=df_feature['unigram'].tolist()
    
    for name in unique_file:
        # print('name: %s'%name)
        row=df[df['file']==name]
        row_length=len(df.loc[df['file']==name])
        # print(len(row))
        for id in range(0,len(row)):
            
            # print(id)
            if row.iloc[id]['speaker']=='*INV' or\
            row.iloc[id]['speaker']=='*PAR':
            
                  
              da1=extract_unigram_DA(row.iloc[id],DA_columns,option)
        
              if da1 not in trigram:
                  trigram[da1]=dict()
                  trigram[da1]['ct']=0
                  trigram[da1]['ad']=0
                  if row.iloc[id]['label']==0:
                      trigram[da1]['ct']=1
                  else:
                      trigram[da1]['ad'] = 1
              else:
                  if row.iloc[id]['label']==0 :
                    if 'ct' in trigram[da1]:
                      trigram[da1]['ct']+=1
                    else:
                      trigram[da1]['ct']=1

                  else:
                      if 'ad' in trigram[da1]:
                        trigram[da1]['ad']+=1
                      else:
                        trigram[da1]['ad']=1
        #for feature
        if option:
          
          if generate_feature(trigram,feature_list,row_length,final_df,row.iloc[id]['label'])==0:
            print(name)
            sys.exit()
          
          trigram={}
          
          
          

         

    if option:
      print('generated unigrams')
      return 
    print_bigram(trigram,df,'unigram',delete)    

def trigram_DA(option,df,DA_columns,unique_file,delete,final_df):
    trigram={}
    
   
    if option:
      df_feature=pd.read_pickle('data/trigram_feature_list.pickle')
      feature_list=df_feature['trigram'].tolist()
    
    for name in unique_file:
        # print(name)
        row=df[df['file']==name]
        row_length=len(df.loc[df['file']==name])
        # print(len(row))
        for id in range(0,len(row)-2):
            
            # print(id)
            if row.iloc[id]['speaker']=='*INV' and row.iloc[id+1]['speaker']=='*PAR'and row.iloc[id+2]['speaker']=='*PAR' or\
            row.iloc[id]['speaker']=='*INV' and row.iloc[id+1]['speaker']=='*INV'and row.iloc[id+2]['speaker']=='*PAR' or\
            row.iloc[id]['speaker']=='*INV' and row.iloc[id+1]['speaker']=='*PAR'and row.iloc[id+2]['speaker']=='*INV' or\
            row.iloc[id]['speaker']=='*PAR' and row.iloc[id+1]['speaker']=='*INV' and row.iloc[id+1]['speaker']=='*INV' or\
            row.iloc[id]['speaker']=='*PAR' and row.iloc[id+1]['speaker']=='*PAR'and row.iloc[id+1]['speaker']=='*INV' or\
            row.iloc[id]['speaker']=='*PAR' and row.iloc[id+1]['speaker']=='*INV'and row.iloc[id+1]['speaker']=='*PAR': 
                  
              da1,da2,da3=extract_trigram_DA(row.iloc[id],row.iloc[id+1],row.iloc[id+2],DA_columns,option)
        
              if da1+'_'+da2+'_'+da3 not in trigram:
                  trigram[da1+'_'+da2+'_'+da3]=dict()
                  trigram[da1+'_'+da2+'_'+da3]['ct']=0
                  trigram[da1+'_'+da2+'_'+da3]['ad']=0
                  if row.iloc[id]['label']==0:
                      trigram[da1+'_'+da2+'_'+da3]['ct']=1
                  else:
                      trigram[da1+'_'+da2+'_'+da3]['ad'] = 1
              else:
                  if row.iloc[id]['label']==0 :
                    if 'ct' in trigram[da1+'_'+da2+'_'+da3]:
                      trigram[da1+'_'+da2+'_'+da3]['ct']+=1
                    else:
                      trigram[da1+'_'+da2+'_'+da3]['ct']=1

                  else:
                      if 'ad' in trigram[da1+'_'+da2+'_'+da3]:
                        trigram[da1+'_'+da2+'_'+da3]['ad']+=1
                      else:
                        trigram[da1+'_'+da2+'_'+da3]['ad']=1
        #for feature
        if option:
          if generate_feature(trigram,feature_list,row_length,final_df,row.iloc[id]['label'])==0:
            print(name)
            sys.exit()
          trigram={}

    if option:
      print('generated trigrams')
      return   


    
    print_bigram(trigram,df,'trigram',delete)   
def initialize_df(final_df,unique_file,df):
  
  label=[]
  files=[]
  for name in unique_file:
    row=df.loc[df['file']==name]
    if len(row)>0:
      
      label.append(row.iloc[0]['label'])
      files.append(name)
    else:
      print('error in %s'%name)
      sys.exit()
  final_df['label']=label
  final_df['file']=files
  
  
def update_df(final_df,trigram,name,df):
  if name not in final_df.keys() and df==0:
    if len(trigram)==len(final_df['file']):
      final_df[name]=trigram
      print('%s saved'%name)
    else:
      print(len(trigram))
      print(len(final_df['file']))
      print('error saving %s'%name)
      sys.exit()
  elif name in final_df.keys() and df==1:
    
    print('drop column')
    if len(trigram)==len(final_df):
      final_df[name]=trigram
    else:
      print('error saving %s'%name)
      sys.exit()
def check_df(final_df):
  length=len(final_df['file'])
  for col in final_df.keys():
    if len(final_df[col])!=length:
      print('len mismatch %s'%col)
      return 0
  return 1
def save_df(final_df,name):
  if check_df(final_df):
    df=pd.DataFrame(final_df)
    df.to_pickle('data/'+name+'.pickle')
    print('%s .pickle saved'%name)
    pickle_to_xlsx(name)
  else:
    sys.exit()
def get_file_w_null(mmse):
  null_file={'file':[]}
  for id,row in mmse.iterrows():
    if row['mmse'] is None or row['mmse']==-1:
      null_file['file'].append(row['file'])
  df=pd.DataFrame(null_file)
  if len(df)>0:
    save_df(df,'no_mmse_file')
  else:
    print('noting to save')

def get_mmse():
  label =pd.read_pickle('data/data_w_feature.pickle')
  mmse=pd.read_pickle('data/DA_annotation_long_all.pickle')
  # print(type(mmse))
  get_file_w_null(mmse)
  mmse=pd.read_excel('data/DA_annotation_long_all.xlsx')

  score=[]
  for name in label['file'].unique():
    if mmse.loc[mmse['file']==name].mmse.values[0]!=None or mmse.loc[mmse['file']==name].mmse.values[0]!=-1:
      score.append(mmse.loc[mmse['file']==name].mmse.values[0])
    
  if len(score)==len(label):
    label['mmse']=score
    save_df(label,'data_w_feature')
  else:
    print('len mismatch')


def pickle_to_xlsx(name):
  df=pd.read_pickle('data/'+name+'.pickle')
  writer = pd.ExcelWriter('data/'+name+'.xlsx')
  # write dataframe to excel
  df.to_excel(writer)
  # save the excel
  writer.save()
  print('DataFrame is written successfully to Excel File.')

def create_feature():
  final_df={}
  da=1 # 1 when create feature for each conversation, otherwie create feature list based on all cov.
  delete=0 # 1 means delete the existing featue list and create new
  save=1 # 1 when want to save final_df with feature
  ngrams=[1,2,3,4]
  df=pd.read_pickle('data/DA_label_wide_final.pickle')
  DA_columns=set(df.columns.tolist())-set(["",'file','speaker','utterance_id','label'])
  unique_file=df['file'].unique()
  
  initialize_df(final_df,unique_file,df)
  
  for ngram in ngrams:
    if ngram==2:
      bigram_DA(da,df,DA_columns,unique_file,delete,final_df)
      # if da and save:

      #   update_df(final_df,bigram,'bigram',delete)
    elif ngram==3:
      trigram_DA(da,df,DA_columns,unique_file,delete,final_df)
      # if da and save:
      #   update_df(final_df,bigram,'trigram',delete)
    elif ngram==1:
      unigram_DA(da,df,DA_columns,unique_file,delete,final_df)
      
      # if da and save:
      #   update_df(final_df,bigram,'unigram',delete)

    elif ngram==4:
      topic_concentration(da,df,DA_columns,unique_file,delete,final_df)
      # if da and save:
      #   update_df(final_df,bigram,'topic',delete)
  if da and save:
    save_df(final_df,'data_w_feature')
# create_feature()
get_mmse()

