import psycopg2
import csv
import re
import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
import math
from collections import Counter
import datetime
import pandas as pd
import numpy as np
from math import acos, cos, asin, sin, atan2, tan, radians

#First_Filter_list=[['11410',3118047,169.7,169.9,'Lithology','GR'],['11410',3118060,22,23,'Lithology','CL']]
First_Filter_list=[]
Attr_col_list=[]
Litho_dico=[]
cleanup_dic_list=[]
Att_col_List_copy_tuple=[]
Attr_val_Dic=[]
Attr_val_fuzzy=[]


print("------------------start Dic_Attr_Col------------")
def Attr_COl():
    query = """SELECT * FROM public.dic_att_col_lithology"""
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    for record in cur:
        #print(record)
        Attr_col_list.append(record)
    outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
   
    with open('Dic_attr_col.csv', 'w') as f:
        cur.copy_expert(outputquery, f)
    

    cur.close()
    conn.close()

    print("------------------end Dic_Attr_Col------------")




print("------------------start Dic_Attr_val------------")
def Attr_Val_Dic():
    query = """SELECT * FROM public.dic_attr_val_lithology_filter"""
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    for record in cur:
        #print(record)
        Attr_val_Dic.append(record)
    outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
   
    with open('Dic_attr_val.csv', 'w') as f:
        cur.copy_expert(outputquery, f)
    

    cur.close()
    conn.close()

    print("------------------end Dic_Attr_val------------")


   



def Litho_Dico():
    print("------------------Start Litho_Dico------------")
    query = """SELECT litho_dic_1.clean  FROM litho_dic_1"""
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    #print(cur)
    for record in cur:
        #print(record)
        Litho_dico.append(record)
        #print(Litho_dico)
    #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
       
    #with open('Dic_litho.csv', 'w') as f:
        #cur.copy_expert(outputquery, f)
        
    #print(Litho_dico)
    cur.close()
    conn.close()
    print("------------------end Litho_Dico------------")


    
    

def Clean_Up():
    print("------------------start Clean_Up_Dico------------")


    query = """SELECT cleanup_lithology.clean  FROM cleanup_lithology"""
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    for record in cur:
        #print(record)
        cleanup_dic_list.append(record)
    #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
       
    #with open('cleanup_dic.csv', 'w',encoding="utf-8") as f:
        #cur.copy_expert(outputquery, f)
        

    cur.close()
    conn.close()

    print("------------------End Clean_Up_Dico------------")

  



def First_Filter():
    print("------------------start First_Filter------------")
    start = time.time()
    #out= open("DB_lithology_First1.csv", "w",encoding ="utf-8")
    query = """select t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn, t1.attributevalue 
    from public.dhgeologyattr t1 
    inner join public.dhgeology t2 
    on t1.dhgeologyid = t2.id 
    inner join collar t3 
    on t3.id = t2.collarid 
    inner join clbody t4 
    on t4.companyid = t3.companyid 
    WHERE(t3.longitude BETWEEN 115.5 AND 118) AND(t3.latitude BETWEEN - 30.5 AND - 27.5) 
    ORDER BY t3.companyid ASC"""


    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    a_list = [list(elem) for elem in cur]
    for row in a_list:
        att_val=row[4]
        for att_col_ele in Attr_col_list:
            dic_att_col=str(att_col_ele).replace('(','').replace(')','').replace(',','').replace('\'','')
            
            if att_val == dic_att_col :
                from_depth = row[2]                
                to_depth = row[3]
                if from_depth is not None and to_depth is not None:
                    if to_depth>from_depth:
                        First_Filter_list.append(row)
                        #print(row)
                    elif from_depth == to_depth:
                        to_depth = to_depth+0.01
                        row[3]=to_depth
                        First_Filter_list.append(row)
                        #print(row)
                    elif from_depth >to_depth:   
                        row[2]=to_depth       
                        row[3]=from_depth
                        First_Filter_list.append(row)
                        #print(row)
                 
                    #for column in row:
                        #out.write('%s,' %column)
                    #out.write('\n')
                   
                    
   

    cur.close()
    conn.close()
    out.close() 
    end = time.time()
    print(end - start)
    print("------------------End First_Filter------------")




def clean_text(text):
    text=text.lower().replace('unnamed','').replace('meta','').replace('meta-','').replace('undifferentiated ','').replace('unclassified ','')
    text=text.replace('differentiated','').replace('undiff','').replace('undiferentiated','').replace('undifferntiates','')
    text=(re.sub('\(.*\)', '', text)) # removes text in parentheses
    text=(re.sub('\[.*\]', '', text)) # removes text in parentheses
    text=text.replace('>','').replace('?','').replace('/',' ') 
    text = text.replace('>' , ' ')
    text = text.replace('<', ' ')
    text = text.replace('/', ' ')
    text = text.replace(' \' ', ' ')
    text = text.replace(',', ' ')
    text = text.replace('%', ' ')
    text = text.replace('-', ' ')
    text = text.replace('_', ' ')
    #text = text.replace('', ' ')
    #text = text.replace('+', '')
    text = text.replace('\'', ' ') 
    if text.isnumeric():
        text = re.sub('\d', ' ', text) #replace numbers
    text = text.replace('&' , ' ')
    text = text.replace(',', ' ')
    text = text.replace('.', ' ')
    text = text.replace(':', ' ')
    text = text.replace(';', ' ')
    text = text.replace('$', ' ')
    text = text.replace('@', ' ')
	
    for cleanup_dic_ele in cleanup_dic_list:
        cleaned_item =str(cleanup_dic_ele).replace('(','').replace(')','').replace(',','').replace('\'','')
        text = text.replace('cleaned_item','')
    return text








#Final File
def Final_Lithology_old():
    print("--------start of Final -----------")
    bestmatch=-1
    bestlitho=''
    top=[]
    p = re.compile(r'[- _]')
    fieldnames=['Company_ID','CollarID','Fromdepth','Todepth','Comapny_Lithocode','Company_Lithology','CET_Lithology','Score']
    out= open("DB_lithology_Final.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    query = '''SELECT dic_attr_val_lithology_filter.company_id,dic_attr_val_lithology_filter.company_code,replace(dic_attr_val_lithology_filter.comapany_litho, ',' , '_') as comapany_litho  FROM dic_attr_val_lithology_filter'''
    conn = psycopg2.connect(host='130.95.198.59', port = 5432, database='gswa_dh', user='postgres', password='loopie123pgpw')
    cur = conn.cursor()
    cur.execute(query)
    a_list = [list(elem) for elem in cur]
    for row in a_list:    
        for First_filter_ele in First_Filter_list:
            #ele_0 = str(First_filter_ele[0]).replace('(','').replace(')','').replace(',','').replace('\'','')    
            #ele_5 = str(First_filter_ele[5]).replace('(','').replace(')','').replace(',','').replace('\'','')
            
            company_code = row[1]
            company_litho = row[2]
            #print(row[0])
            #print( First_filter_ele[0])
            #print(row[1])
            #print( First_filter_ele[5])
            if int(row[0]) == First_filter_ele[0] and  row[1] == First_filter_ele[5]:
                #del First_filter_ele[4]
                #del First_filter_ele[4]
                cleaned_text=clean_text(row[2])
                #print(cleaned_text)
                words=(re.sub('\(.*\)', '', cleaned_text)).strip() 
                words=words.split(" ")
                last=len(words)-1 #position of last word in phrase
                
                for Litho_dico_ele in Litho_dico:              
                    #litho_words=str(Litho_dico_ele).lower().rstrip('\n\r').split(" ")
                    litho_words=re.split(p, str(Litho_dico_ele))
                    scores=process.extract(cleaned_text, litho_words, scorer=fuzz.token_set_ratio)
                    for sc in scores:                        
                        if(sc[1]>bestmatch): #better than previous best match
                            bestmatch =  sc[1]
                            bestlitho=litho_words[0]
                            top=sc
                            if(sc[0]==words[last]): #bonus for being last word in phrase
                                bestmatch=bestmatch*1.01
                        elif (sc[1]==bestmatch): #equal to previous best match
                            if(sc[0]==words[last]): #bonus for being last word in phrase
                                bestlitho=litho_words[0]
                                bestmatch=bestmatch*1.01
                            else:
                                top=top+sc

                #top = [list(elem) for elem in top]
                top_new = list(top)
                if top_new[1] >80:
                    #del First_filter_ele[4]
                    #del First_filter_ele[4]
                    #for column in First_filter_ele:
                    out.write('%s,' %First_filter_ele[0])
                    out.write('%s,' %First_filter_ele[1])
                    out.write('%s,' %(First_filter_ele[2]).replace(',' ,' '))
                    out.write('%s,' %First_filter_ele[3])
                    out.write('%s,' %row[1])
                    out.write('%s,' %row[2])
                    CET_Litho = str(top_new[0]).replace('(','').replace(')','').replace('\'','').replace(',','')
                    CET_Litho = CET_Litho.replace(',', ' ')
                    out.write('%s,' %CET_Litho)
                    out.write('%d,' %top_new[1])
                    out.write('\n')
                    #top.clear()
                    top_new[:] =[]
                    CET_Litho=''
                    bestmatch=-1
                    bestlitho=''
                else:
                    #del First_filter_ele[4]
                    #del First_filter_ele[4]
                    #for column in First_filter_ele:
                    out.write('%s,' %First_filter_ele[0])
                    out.write('%s,' %First_filter_ele[1])
                    out.write('%s,' %(First_filter_ele[2]).replace(',' ,' '))
                    out.write('%s,' %First_filter_ele[3])
                    out.write('%s,' %row[1])
                    out.write('%s,' %row[2])
                    out.write('Other,')
                    out.write('%d,' %top_new[1])
                    out.write('\n')
                    #top.clear()
                    top_new[:] =[]
                    CET_Litho=''
                    bestmatch=-1
                    bestlitho=''

    cur.close()
    conn.close()
    out.close()
    print("--------End of Final-----------")



def Attr_val_With_fuzzy():
    print("--------start of Attr_val_fuzzy-----------")
    bestmatch=-1
    bestlitho=''
    top=[]
    i=0
    attr_val_sub_list=[]
    #p = re.compile(r'[' ']')
    fieldnames=['CollarID','code','Attr_val','cleaned_text','Fuzzy_wuzzy','Score']
    out= open("Attr_val_fuzzy.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    Attr_val_Dic_new = [list(elem) for elem in Attr_val_Dic]
    for Attr_val_Dic_ele in Attr_val_Dic_new:
        

        cleaned_text=clean_text(Attr_val_Dic_ele[2])
        #if(cleaned_text =='granite'):
            #print(cleaned_text)
        words=(re.sub('\(.*\)', '', cleaned_text)).strip() 
        words=words.rstrip('\n\r').split(" ")
        last=len(words)-1 #position of last word in phrase
        for Litho_dico_ele in Litho_dico:
            #print(Litho_dico)
        #litho_words=str(Litho_dico_ele).lower().rstrip('\n\r').split(" ")
            #litho_words=re.split(" ", str(Litho_dico_ele))
            #litho_words=str(Litho_dico_ele).split(" ")
            litho_words=str(Litho_dico_ele).lower().rstrip('\n\r').replace('(','').replace(')','').replace('\'','').replace(',','').split(" ")
            #print(litho_words)
            #if(litho_words == "alkali-feldspar-granite"):
                #print("Alkali-feldspar-granite")


            scores=process.extract(cleaned_text, litho_words, scorer=fuzz.token_set_ratio)
            for sc in scores:                        
                if(sc[1]>bestmatch): #better than previous best match
                    bestmatch =  sc[1]
                    bestlitho=litho_words[0]
                    #print(bestmatch)
                    #print(bestlitho)
                    #top=sc
                    top.append([sc[0],sc[1]])
                    if(sc[0]==words[last]): #bonus for being last word in phrase
                        bestmatch=bestmatch*1.01
                        #print("inside 1")
                        #print(sc[0])
                        #print(words[last])
                elif (sc[1]==bestmatch): #equal to previous best match
                    if(sc[0]==words[last]): #bonus for being last word in phrase
                        bestlitho=litho_words[0]
                        bestmatch=bestmatch*1.01
                        #print(bestlitho)
                        #print(bestmatch)
                        #print(words[last])
                    else:
                        #top=top+sc
                        top.append([sc[0],sc[1]])
        
        #print(top)
        #top_new = list(top)
        #top_new=[list(elem) for elem in top]
        #for i in range(len(top)):
            
        #print(top_new)
        i=0
        #print(" %s %d " %(top_new[0], top_new[1] ))

        
        #for top_new_ele in top:
            #if(top_new_ele[0].replace('(','').replace(')','').replace('\'','').replace(',','') == cleaned_text):
               # bestlitho = cleaned_text
               # bestmatch = 100
                
            
                
        
           
            
            
        if bestmatch >80:
            #CET_Litho = str(top_new[0]).replace('(','').replace(')','').replace('\'','').replace(',','')
            #print(CET_Litho)
            
            #attr_val_sub_list.append(Attr_val_Dic_ele[0])
            #attr_val_sub_list.append(Attr_val_Dic_ele[1])
            #attr_val_sub_list.append(Attr_val_Dic_ele[2])
            #attr_val_sub_list.append(bestlitho)
            #attr_val_sub_list.append(top_new[1])
            #Attr_val_fuzzy.append(attr_val_sub_list)

            Attr_val_fuzzy.append([Attr_val_Dic_ele[0],Attr_val_Dic_ele[1],Attr_val_Dic_ele[2],cleaned_text,bestlitho,bestmatch]) #top_new[1]])  or top[0][1]
            
            #attr_val_sub_list.clear()
            
            out.write('%d,' %int(Attr_val_Dic_ele[0]))
            out.write('%s,' %Attr_val_Dic_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%s,' %Attr_val_Dic_ele[2].replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))     #.replace(',' , '').replace('\n' , ''))
            out.write('%s,' %cleaned_text)   #.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%s,' %bestlitho.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            #out.write('%d,' %top_new[1])
            out.write('%d,' %bestmatch)
            out.write('\n')
            #top_new[:] =[]
            top.clear()
            CET_Litho=''
            bestmatch=-1
            bestlitho=''
           
            
        else:
            #attr_val_sub_list.append(Attr_val_Dic_ele[0])
            #attr_val_sub_list.append(Attr_val_Dic_ele[1])
            #attr_val_sub_list.append(Attr_val_Dic_ele[2])
            #attr_val_sub_list.append('Other')
            #attr_val_sub_list.append(top_new[1])
            #Attr_val_fuzzy.append(attr_val_sub_list)
            #attr_val_sub_list.clear()


            Attr_val_fuzzy.append([Attr_val_Dic_ele[0],Attr_val_Dic_ele[1],Attr_val_Dic_ele[2],cleaned_text,'Other',bestmatch])  #top_new[1]])
            
            out.write('%d,' %int(Attr_val_Dic_ele[0]))
            out.write('%s,' %Attr_val_Dic_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','').replace(',' , '').replace('\n',''))
            out.write('%s,' %Attr_val_Dic_ele[2].replace('(','').replace(')','').replace('\'','').replace(',','').replace(',' , '').replace('\n',''))     #.replace(',' , '').replace('\n' , ''))
            out.write('%s,' %cleaned_text)   #.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('Other,')
            #out.write('%d,' %top_new[1])
            out.write('%d,' %bestmatch)
            out.write('\n')
            #top_new[:] =[]
            top.clear()
            CET_Litho=''
            bestmatch=-1
            bestlitho=''
            
            







    print("--------End of Attr_val_fuzzy-----------")



def Final_Lithology():
    print("--------start of Final -----------")
    query = """select t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn, t1.attributevalue 
		 from public.dhgeologyattr t1 
		 inner join public.dhgeology t2 
		 on t1.dhgeologyid = t2.id 
		 inner join collar t3 
		 on t3.id = t2.collarid 
		 inner join clbody t4 
		 on t4.companyid = t3.companyid
		 inner join public.dic_att_col_lithology t5
		 on t1.attributecolumn = t5.lithological
		 WHERE(t3.longitude BETWEEN 115.5 AND 118) AND(t3.latitude BETWEEN - 30.5 AND - 27.5) 
		 ORDER BY t3.companyid ASC"""


    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    First_Filter_list = [list(elem) for elem in cur]
    print("First Filter ready")
    fieldnames=['Company_ID','CollarID','Fromdepth','Todepth','Comapny_Lithocode','Company_Lithology','CET_Lithology','Score']
    out= open("DB_lithology_Final.csv", "w",encoding ="utf-8")
    #out_first_filter= open("DB_lithology_First.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    #Attr_val_Dic_new = [list(elem) for elem in Attr_val_Dic]
    for First_filter_ele in First_Filter_list:
        for Attr_val_fuzzy_ele in Attr_val_fuzzy:
            if int(Attr_val_fuzzy_ele[0].replace('\'' , '')) == First_filter_ele[0] and  Attr_val_fuzzy_ele[1].replace('\'' , '') == First_filter_ele[5]:
                out.write('%d,' %First_filter_ele[0])
                out.write('%d,' %First_filter_ele[1])
                out.write('%d,' %First_filter_ele[2])
                out.write('%s,' %First_filter_ele[3])
                out.write('%s,' %Attr_val_fuzzy_ele[1])
                out.write('%s,' %Attr_val_fuzzy_ele[2].replace('(','').replace(')','').replace('\'','').replace(',',''))
                out.write('%s,' %Attr_val_fuzzy_ele[4].replace('(','').replace(')','').replace('\'','').replace(',',''))   #.replace(',' , ''))
                out.write('%d,' %int(Attr_val_fuzzy_ele[5]))
                out.write('\n')

    
        #for column in First_filter_ele:
            #out_first_filter.write('%s,' %column)
        #out_first_filter.write('\n')
        	
	
    print("--------End of Final -----------")


def Upscale_lithology():
    print("--------start of Upsacle -----------")
    Hierarchy_litho_dico_List =[]
    query = """ select * from public.hierarchy_dico """
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    Hierarchy_litho_dico_List  = [list(elem) for elem in cur]
    CET_hierarchy_dico = pd.DataFrame(Hierarchy_litho_dico_List,columns=['Level_3','Level_2','Level_1'])
    #CET_hierarchy_dico.to_csv ('CET_hierarchy_dico.csv', index = False, header=True)
    #print (CET_hierarchy_dico)
    DB_Lithology= pd.read_csv('DB_Lithology_Final.csv',encoding = "ISO-8859-1", dtype='object')
    Upscaled_Litho=pd.merge(DB_Lithology, CET_hierarchy_dico, left_on='CET_Lithology', right_on='Level_3')
    Upscaled_Litho.sort_values("Company_ID", ascending = True, inplace = True)
    #Upscaled_Litho.drop(['Unnamed: 8'], axis=1)
    #del Upscaled_Litho['Unnamed: 8']
    Upscaled_Litho.to_csv ('Upscaled_Litho.csv', index = False, header=True)
    
    #Upscaled_Litho= Upscaled_Litho.loc[:, ~Upscaled_Litho.columns.str.contains('^Unnamed')]
    #Upscaled_Litho.reset_index(level=0, inplace=True)
    #Upscaled_Litho['CET_Litho']=Upscaled_Litho['index']
    #del Upscaled_Litho['index']
    #Upscaled_Litho.to_csv(DB_Lithology_Upscaled)
    print("--------End of Upsacle -----------")



def dsmincurb (len12,azm1,dip1,azm2,dip2):
    DEG2RAD = 3.141592654/180.0
    i1 = (90 - float(dip1)) * DEG2RAD
    a1 = float(azm1) * DEG2RAD
	
    i2 = (90 - float(dip2)) * DEG2RAD
    a2 = float(azm2) * DEG2RAD
	
    #Beta = acos(cos(I2 - I1) - (sin(I1)*sin(I2)*(1-cos(Az2-Az1))))
    dl = acos(cos(float(i2)-float(i1))-(sin(float(i1))*sin(float(i2))*(1-cos(float(a2)-float(a1)))))
    if dl!=0.:
        rf = 2*tan(dl/2)/dl  # minimum curvature
    else:
        rf=1				 # balanced tangential

    dz = 0.5*len12*(cos(float(i1))+cos(float(i2)))*rf
    dn = 0.5*len12*(sin(float(i1))*cos(float(a1))+sin(float(i2))*cos(float(a2)))*rf
    de = 0.5*len12*(sin(float(i1))*sin(float(a1))+sin(float(i2))*sin(float(a2)))*rf
    return dz,dn,de






def interp_ang1D(azm1,dip1,azm2,dip2,len12,d1):
    # convert angles to coordinates
    x1,y1,z1 = ang2cart(azm1,dip1)
    x2,y2,z2 = ang2cart(azm2,dip2)
    # interpolate x,y,z
    x = x2*d1/len12 + x1*(len12-d1)/len12
    y = y2*d1/len12 + y1*(len12-d1)/len12
    z = z2*d1/len12 + z1*(len12-d1)/len12
    # get back the results as angles
    azm,dip = cart2ang(x,y,z)
    return azm, dip
    #modified from pygslib
	
def ang2cart(azm, dip):
    DEG2RAD=3.141592654/180.0
    # convert degree to rad and correct sign of dip
    razm = float(azm) * float(DEG2RAD)
    rdip = -(float(dip)) * float(DEG2RAD)
    # do the conversion
    x = sin(razm) * cos(rdip)
    y = cos(razm) * cos(rdip)
    z = sin(rdip)
    return x,y,z
    #modified from pygslib
	
def cart2ang(x,y,z):
    if x>1.: x=1.
    if x<-1.: x=-1.
    if y>1.: y=1.
    if y<-1.: y=-1.
    if z>1.: z=1.
    if z<-1.: z=-1.
    RAD2DEG=180.0/3.141592654
    pi = 3.141592654
    azm= float(atan2(x,y))
    if azm<0.:
        azm= azm + pi*2
    azm = float(azm) * float(RAD2DEG)
    dip = -(float(asin(z))) * float(RAD2DEG)
    return azm, dip
    #modified from pygslib
	
def angleson1dh(indbs,indes,ats,azs,dips,lpt):
    for i in range (indbs,indes):
        a=ats[i]
        b=ats[i+1]
        azm1 = azs[i]
        dip1 = dips[i]
        azm2 = azs[i+1]
        dip2 = dips[i+1]
        len12 = ats[i+1]-ats[i]
        if lpt>=a and lpt<b:
            d1= lpt- a
            azt,dipt = interp_ang1D(azm1,dip1,azm2,dip2,len12,d1)
            return azt, dipt
    a=ats[indes]
    
    azt = azs[indes]
    
    dipt = dips[indes]
    print(a,"\t",azt,"\t",dipt)
    if float(lpt)>=float(a):
        return   azt, dipt
    else:
        return   np.nan, np.nan
    #modified from pygslib
	
def convert_lithology():
    print("--------start of convert Lithology -----------")
    collar= pd.read_csv('DB_Collar_Export.csv',encoding = "ISO-8859-1", dtype='object')
    survey= pd.read_csv('DB_Survey_Export.csv',encoding = "ISO-8859-1", dtype='object')
    litho= pd.read_csv('Upscaled_Litho.csv',encoding = "ISO-8859-1", dtype='object')
    
	
    collar.CollarID = collar.CollarID.astype(int)
    survey.CollarID = survey.CollarID.astype(int)
    survey.Depth = survey.Depth.astype(float)
    litho.CollarID = litho.CollarID.astype(int)
    litho.Fromdepth = litho.Fromdepth.astype(float)
	
    collar.sort_values(['CollarID'], inplace=True)
    #print(collar(['CollarID']))
    survey.sort_values(['CollarID', 'Depth'], inplace=True)
    #print(survey(['CollarID']))
    litho.sort_values(['CollarID', 'Fromdepth'], inplace=True)
    #print(litho(['CollarID']))
                      

    
    idc =collar['CollarID'].values
    #print(idc)
    xc = collar['X'].values
    yc = collar['Y'].values
    zc = collar['RL'].values
    ids = survey['CollarID'].values
    #print(ids)
    ats = survey['Depth'].values
    azs = survey['Azimuth'].values
    dips = survey['Dip'].values
    idt =litho['CollarID'].values
    #print(idt)
    fromt = litho['Fromdepth'].values
    tot = litho['Todepth'].values
    compid=litho['Company_ID'].values
    complc=litho['Comapny_Lithocode'].values
    compl=litho['Company_Lithology'].values
    cetlit=litho['CET_Lithology'].values
    score=litho['Score'].values
    lvl1=litho['Level_1'].values
    lvl2=litho['Level_2'].values
    lvl3=litho['Level_3'].values
	
    nc= idc.shape[0]
    ns= ids.shape[0]
    nt= idt.shape[0]
    
	
    azmt = np.empty([nt], dtype=float)
    dipmt = np.empty([nt], dtype=float)
    xmt = np.empty([nt], dtype=float)
    ymt = np.empty([nt], dtype=float)
    zmt = np.empty([nt], dtype=float)
    azbt = np.empty([nt], dtype=float)
    dipbt = np.empty([nt], dtype=float)
    xbt = np.empty([nt], dtype=float)
    ybt = np.empty([nt], dtype=float)
    zbt = np.empty([nt], dtype=float)
    azet = np.empty([nt], dtype=float)
    dipet = np.empty([nt], dtype=float)
    xet = np.empty([nt], dtype=float)
    yet = np.empty([nt], dtype=float)
    zet = np.empty([nt], dtype=float)

    azmt[:] = np.nan
    dipmt[:] = np.nan
    azbt [:]= np.nan
    dipbt [:]= np.nan
    azet[:] = np.nan
    dipet[:] = np.nan
    xmt[:] = np.nan
    ymt [:]= np.nan
    zmt [:]= np.nan
    xbt[:] = np.nan
    ybt[:] = np.nan
    zbt[:] = np.nan
    xet[:] = np.nan
    yet[:] = np.nan
    zet [:]= np.nan

    #print("1")
    fieldnames=['Company_ID','CollarID','FromDepth','ToDepth','Company_LithoCode','Company_Litho', 'CET_Litho','Score','Level_3', 'Level_2','Level_1',
			   'xbt','ybt','zbt','xmt','ymt', 'zmt', 'xet','yet','zet']

    
    out= open('DB_Lithology_Export_Calc.csv', "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    out.flush()

    indbt = 0
    indet = 0
    inds = 0
    indt = 0
    ii =0
    
    for jc in range(nc):
        #print("loop1")
        #print(jc)
        indbs = -1
        indes = -1
        inds = 0
        indt =0
        for js in range(inds, ns):
            #print("loop2")
            #print(idc[jc])
            #print(ids[js]
            #print(js)
            #if(idc[jc] == 124897):
                #print("124897 found in collar")

            #if(ids[js] == 124897):
                #print("124897 found in servey")
            if(idc[jc] != ids[js]):
                #print("break")
                continue;
            elif(idc[jc] == ids[js]):
                #print(" No break")
                if(idc[jc] == 124897):
                    ii =ii + 1
                    #print("124897 in both C S found ")
                inds = js
                indbs = js
                indes = js
                azm1  = azs[indbs]
                dip1 = dips[indbs]
                at = 0.
                x =  xc[jc]
                y =  yc[jc]
                z =  zc[jc]
                
                
                for jt in range(indt, nt):
                    #print("in litho loop")
                    #print(idc[jc])
                    #print(idt[jt])
                    #print(jt)
                    #if(idt[jt] == 124897):
                       # print("found 124897")
                    if(idc[jc] != idt[jt]):
                        continue;
                    if idc[jc] == idt[jt]:
                        if(idt[jt] == 124897):
                            print("found 124897")
                        try:
                            #print("3 loop inside")
                            indt = jt
                            #from
                            #print("2")
                            
                            azm2,dip2 = angleson1dh(indbs,indes,ats,azs,dips,fromt[jt])
                            #print(indbs)
                            #print(indes)
                            #print(ats)
                            #print(azs)
                            #print(dips)
                            #print(fromt[jt])
                            azbt[jt] = azm2
                            dipbt[jt] = dip2
                            len12 = float(fromt[jt]) - at
                            dz,dn,de = dsmincurb(len12,azm1,dip1,azm2,dip2)
                            xbt[jt] = de
                            ybt[jt] = dn
                            zbt[jt] = dz

                            #update
                            azm1 = azm2
                            dip1 = dip2
                            at   = float(fromt[jt])


                            #midpoint
                            #print("4")
                            if( tot[jt] == 'None'):  #for Empty todepth
                                print("None")
                                tot[jt]= float(fromt[jt])+0.1
                            
                            mid = float(fromt[jt]) + float((float(tot[jt])-float(fromt[jt]))/2)
                            azm2, dip2 = angleson1dh(indbs,indes,ats,azs,dips,mid)
                        
                            azmt[jt] = azm2
                            dipmt[jt]= dip2
                            len12 = mid - at
                            dz,dn,de = dsmincurb(len12,azm1,dip1,azm2,dip2)
                            xmt[jt] = de + xbt[jt]
                            ymt[jt] = dn + ybt[jt]
                            zmt[jt] = dz + zbt[jt]

                            #update
                            azm1 = azm2
                            dip1 = dip2
                            at   = mid

                            #to
                            #print("5")
                            #if( tot[jt] == None):  //for Empty todepth
                            #tot[jt]= float(fromt[jt])+0.1
                            azm2, dip2 = angleson1dh(indbs,indes,ats,azs,dips,float(tot[jt]))
                            azet[jt] = azm2
                            dipet[jt] = dip2
                            len12 = float(tot[jt]) - at
                            dz,dn,de = dsmincurb(len12,azm1,dip1,azm2,dip2)
                            xet[jt] = de + xmt[jt]
                            yet[jt] = dn + ymt[jt]
                            zet[jt] = dz + zmt[jt]

                            #update
                            azm1 = azm2
                            dip1 = dip2
                            at   = float(tot[jt])


                            #calculate coordinates
                            xbt[jt] = float(x)+float(xbt[jt])
                            ybt[jt] = float(y)+float(ybt[jt])
                            zbt[jt] = float(z)+float(zbt[jt])
                            xmt[jt] = float(x)+float(xmt[jt])
                            ymt[jt] = float(y)+float(ymt[jt])
                            zmt[jt] = float(z)+float(zmt[jt])
                            xet[jt] = float(x)+float(xet[jt])
                            yet[jt] = float(y)+float(yet[jt])
                            zet[jt] = float(z)+float(zet[jt])

                            # update for next interval
                            x = xet[jt]
                            y = yet[jt]
                            z = zet[jt]
                        except ValueError as e:
                            print ("type error" + str(e))

                        out.write('%s,' %compid[jt])
                        out.write('%s,' %idt[jt])
                        out.write('%s,' %fromt[jt])
                        out.write('%s,' %tot[jt])
                        out.write('%s,' %complc[jt])
                        out.write('%s,' %compl[jt])
                        out.write('%s,' %cetlit[jt])
                        out.write('%s,' %score[jt])
                        out.write('%s,' %lvl3[jt])
                        out.write('%s,' %lvl2[jt])
                        out.write('%s,' %lvl1[jt])
                        out.write('%s,' %xbt[jt])
                        out.write('%s,' %ybt[jt])
                        out.write('%s,' %zbt[jt])
                        out.write('%s,' %xmt[jt])
                        out.write('%s,' %ymt[jt])
                        out.write('%s,' %zmt[jt])
                        out.write('%s,' %xet[jt])
                        out.write('%s,' %yet[jt])
                        out.write('%s,' %zet[jt])
                        out.write('\n')
                        out.flush()

        #print(idc[jc])
        
                
           
                   
    out.close()

    print("--------End of convert Lithology -----------")



Start_Time = datetime.datetime.now()
#Attr_Val_Dic()
#Litho_Dico()
#Clean_Up()
#Attr_val_With_fuzzy()
#Attr_COl() not required
#First_Filter() not required
#Final_Lithology()
#Upscale_lithology()
convert_lithology()
End_Time = datetime.datetime.now()
print("Time taken in Hour_Min_Sec_MilliSec is:", End_Time-Start_Time)


