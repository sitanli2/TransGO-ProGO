from pymongo import MongoClient
from bson import ObjectId
import tqdm
#tqdm库用来生成进度条
import urllib.request as request #urllib3版本要<=1.25否则会与requests版本冲突
#urllin库用于操作网页 URL，并对网页的内容进行抓取处理。
from progressbar import ProgressBar
from jsonpath import jsonpath
from tkinter import _flatten #将二维列表转成一维列表
import requests
from bs4 import BeautifulSoup
import copy
mongoClient = MongoClient('mongodb://localhost:27017/')
dtmr_dev = mongoClient.get_database('Chembl_DOC')
dev = dtmr_dev.get_collection('Compound_molecule')
dev1 = dtmr_dev.get_collection('Assay')
dev2 = dtmr_dev.get_collection('Target')
dev3 = dtmr_dev.get_collection('Activity')
dev4 = dtmr_dev.get_collection('Drugs')

#需要存以下几张表：
#Target = {} #靶点信息表
#Compound_molecule = {}# 小分子化合物信息表,要注意带molecule（小分子）的接口。
#Assay ={}
#Activity = {} #活性物信息表
#Drugs = {} #药物信息表

html_url = "https://www.ebi.ac.uk/chembl/api/data/chembl_id_lookup?limit=300"
#接口名称ChEMBL ID Lookup，该接口用来得到所有的chembl ID，可以通过修改后面的limit来控制获取的chembl ID 数量，网页默认都是20个这和chembl网页一张只放20个数据有关，下二十个需要翻页。
response = request.urlopen(html_url)
XML = response.read()
#print("XML type:",type(XML))
#print("chembl id:",XML)
'''
经此验证后可知 XML 的内容就是上述接口的内容，且是一个XML网页文件不再是JSON了无法用JSONpath遍历它，想要遍历他或许要用beautifulsoup进行解析XML?
且可以看到改XML文件的末尾是有一个<page_meta>的板块内<limit>20</limit>，<total_count>4547276</total_count>说明一共有4547276个chembl ID但这里设置最多返回20个
那我就先拿20个呗 =_=
【注意】这20条ID里面不全是化合物小分子，还有Assay,必须将他们区分开，否则之后接口会调用报错。

之后我发现这个20条的限制是可以自己修改的比如
随便进去一个接口看他的xml文档可以发现，<page_meta>这个tag下面的<next>里面设置了limit为20这也是应为Chembl网页的一页就存放20个数据，再想要跟多的数据就得翻页
而在Xml里面我们知道了如何修改获得更多的数据如下：  
https://www.ebi.ac.uk/chembl/api/data/target + 后缀： ?limit=100&offset=200 这样的话limit就会变成100 返回100条数据。
'''
chembl_id_list = ['CHEMBL2368526']#'CHEMBL2368526' 作为一个属性比较全的小分子化合物用它来做test
assay_id_list = []
soup = BeautifulSoup(XML, 'xml')
#具体的soup.find以及findall方法详情可见daily_note 第十天：5/24 周三
all_id = soup.find_all('chembl_id_lookup')
#print(type(all_id))
for id in all_id:
    if(id.entity_type.string == 'COMPOUND' and id.status.string == 'ACTIVE'):
        if (len(chembl_id_list) < 20):#控制爬取不多不少20个chembl id
            chembl_id_list.append(id.chembl_id.get_text())

    elif(id.entity_type.string == 'ASSAY' and id.status.string == 'ACTIVE'):
        if (len(assay_id_list) < 10):#控制爬取不多不少10个assay id
            assay_id_list.append(id.chembl_id.get_text())
    # if (len(assay_id_list) == 10 and len(chembl_id_list) == 20):
    #     break
    elif(id.entity_type.string != 'COMPOUND' and id.entity_type.string != 'ASSAY'):#想找到除chembl id 和assay id之外的id
        print('此id的类型为：',id.entity_type.string)

print(chembl_id_list,'chembl id个数:',len(chembl_id_list))
print(assay_id_list,'assay id个数:',len(assay_id_list))
#经过这一切转换，现在的chembl_id_list[]是一个含有20个chembl_id的列表。

# molecule_list = []
# html_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/" + 'CHEMBL10'
# response = request.urlopen(html_url)
# XML = response.read()
# soup = BeautifulSoup(XML,'xml')
# molecule_inner_dict={}
# molecule_inner_dict['chembl_id'] = 'CHEMBL10'
# molecule_inner_dict['availability_type'] = soup.find('availability_type').get_text()
# molecule_inner_dict['chebi_par_id'] = soup.find('chebi_par_id').get_text()
# temp_dict = copy.deepcopy(molecule_inner_dict)
# molecule_list.append(temp_dict)
# print("outcome:",molecule_list)

'''
Compound_molecule 化合物小分子接口爬取
现在还差和每个化合物小分子对应的target进行关联得到targets信息，再插入molecule_dict中
'''
molecule_list = []
count = 0
for i in chembl_id_list:
    print('此时的chembl id=',i)
    html_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/" + str(i)
    response = request.urlopen(html_url)
    XML = response.read()
    soup = BeautifulSoup(XML,'xml')
    molecule_dict={}
    molecule_inner_hierarchy = {}
    molecule_inner_properties = {}
    molecule_inner_structures = {}
    molecule_dict['chembl_id'] = 'CHEMBL2368526'
    if (bool(soup.find('pref_name').string)):
        molecule_dict['name'] = soup.find('pref_name').string
    else:
        molecule_dict['name'] = 'null'
    print('this molecule have synonym',bool(soup.molecule_synonyms.synonym))
    if(bool(soup.molecule_synonyms.synonym) and bool(soup.molecule_synonyms.synonym.molecule_synonym.string)):
        molecule_dict['synonym'] =soup.molecule_synonyms.synonym.molecule_synonym.string
    else:
        molecule_dict['synonym'] = 'null'
    molecule_dict['type'] = soup.find('molecule_type').get_text()
    if (bool(soup.find('max_phase').string)):
        molecule_dict['Max_phase'] = soup.find('max_phase').string
    else:
        molecule_dict['Max_phase'] = 'null'
    if(bool(soup.molecule_properties.full_mwt.string)):
        molecule_dict['Molecular Weight'] = soup.molecule_properties.full_mwt.string
    else:
        molecule_dict['Molecular Weight'] = 'null'
    molecule_dict['active_chembl_id'] = soup.molecule_hierarchy.active_chembl_id.string #根据这个activate id可以将此分子化合物和与之有关的活性物相关联,从而得到需要的Bioactivities
    molecule_dict['Alogp'] = soup.molecule_properties.alogp.string
    molecule_dict['Polar Surface Area'] = soup.molecule_properties.psa.string
    molecule_dict['HBA'] = soup.molecule_properties.hba.string
    molecule_dict['HBD'] = soup.molecule_properties.hbd.string
    molecule_dict['Passes Ro3'] = soup.molecule_properties.ro3_pass.string
    molecule_dict['QED Weighted'] = soup.molecule_properties.qed_weighted.string
    molecule_dict['Target'] = []#这个需要关联target 靶点蛋白表 之后再说

    molecule_dict['chebi_par_id'] = soup.find('chebi_par_id').get_text()
    molecule_dict['chirality'] = soup.find('chirality').get_text()
    molecule_inner_hierarchy['active_chembl_id'] = soup.find('molecule_hierarchy').active_chembl_id.string
    molecule_inner_hierarchy['molecule_chembl_id'] = soup.find('molecule_hierarchy').molecule_chembl_id.string
    molecule_inner_hierarchy['parent_chembl_id'] = soup.find('molecule_hierarchy').parent_chembl_id.string
    molecule_dict['hierarchy'] = molecule_inner_hierarchy
    molecule_inner_properties['aromatic_rings'] = soup.find('molecule_properties').aromatic_rings.string
    molecule_inner_properties['full_molformula'] = soup.find('molecule_properties').full_molformula.string
    molecule_inner_properties['heavy_atoms'] = soup.find('molecule_properties').heavy_atoms.string
    molecule_inner_properties['molecular_species'] = soup.find('molecule_properties').molecular_species.string
    molecule_dict['properties'] = molecule_inner_properties
    molecule_inner_structures['canonical_smiles'] = soup.find('molecule_structures').canonical_smiles.string
    molecule_inner_structures['molfile'] = soup.find('molecule_structures').molfile.string
    molecule_inner_structures['standard_inchi'] = soup.find('molecule_structures').standard_inchi.string
    molecule_inner_structures['standard_inchi_key'] = soup.find('molecule_structures').standard_inchi_key.string
    molecule_dict['structures'] = molecule_inner_structures

    temp_dict = copy.deepcopy(molecule_dict)
    molecule_list.append(temp_dict)
    count += 1
result = dev.insert_many(molecule_list)
print(len(result.inserted_ids), 'records inserted!')



'''assay 论文，文章接口爬取'''
count = 0
assay_list = []
for i in assay_id_list:
    print('Assay count:', count, "id:", i)
    html_url = "https://www.ebi.ac.uk/chembl/api/data/assay/" + str(i)
    response = request.urlopen(html_url)
    XML = response.read()
    soup = BeautifulSoup(XML, 'xml')
    assay_dict = {}
    assay_dict['assay_chembl_id'] = i
    assay_dict['document_chembl_id'] = soup.find('assay').document_chembl_id.string
    assay_dict['target_chembl_id'] = soup.find('assay').target_chembl_id.string
    assay_dict['assay_organism'] = soup.find('assay').assay_organism.string
    assay_dict['assay_tax_id'] = soup.find('assay').assay_tax_id.string
    assay_dict['assay_type'] = soup.find('assay').assay_type.string
    assay_dict['assay_type_description'] = soup.find('assay').assay_type_description.string
    assay_dict['bao_format'] = soup.find('assay').bao_format.string
    assay_dict['bao_label'] = soup.find('assay').bao_label.string
    assay_dict['confidence_description'] = soup.find('assay').confidence_description.string
    assay_dict['confidence_score'] = soup.find('assay').confidence_score.string
    assay_dict['description'] = soup.find('assay').description.string
    # assay_dict['document_chembl_id'] = soup.find('assay').document_chembl_id.string
    # assay_dict['target_chembl_id'] = soup.find('assay').target_chembl_id.string
    assay_dict['relationship_description'] = soup.find('assay').relationship_description.string
    assay_dict['relationship_type'] = soup.find('assay').relationship_type.string
    temp_dict = copy.deepcopy(assay_dict)
    assay_list.append(temp_dict)
    count += 1
dev1.insert_many(assay_list)
'''
要得到target靶点信息首先要是像之前assay和molecule一样先获得target_id_list 然后再一个一个找
之前是从chembl_id接口获得的chembl_id 再将其分为 chembl_id_list = [] 和 assay_id_list = [] 然后再根据这两个一个一个找，但这样太麻烦了，但是好处是能找到他们之间的关系

而这回我则直接去target接口里面找了 https://www.ebi.ac.uk/chembl/api/data/target?limit=5 记得这回limit设置小点来个5条
这样之后则么找相关的接口也就明了了。
现在迫切需要解决的问题是如何将这几个接口关联起来？？？

'''
'''target 靶点蛋白接口爬取'''
html_url = "https://www.ebi.ac.uk/chembl/api/data/target?limit=5"#直接拿它给的前5条target
response = request.urlopen(html_url)
XML = response.read()
soup = BeautifulSoup(XML,'xml')
all_target = soup.targets.find_all('target',recursive=False)
#all_target 是一个长度为五的列表，列表元素类型为 <class 'bs4.element.Tag'> 所以可以对列表中的每个元素进行bs解析。
#print(all_target[0].target_type.string)   =SINGLE PROTEIN

i=0
target_list = []
while(i <= len(all_target)-1 ):
    target_dict={}
    target_inner_crossreference = {}
    target_inner_components = {}
    # print(all_target[0].target_chembl_id.string)  #可见target_chembl_id不在cross_references下面 我可以通过这个来获取target_chembl_id ！！！
    target_dict['target_chembl_id'] = all_target[i].target_chembl_id.string
    target_dict['organism'] = all_target[i].organism.string
    target_dict['pref_name'] = all_target[i].pref_name.string
    target_dict['target_type'] = all_target[i].target_type.string
    target_dict['tax_id'] = all_target[i].tax_id.string
    dict1={}#用来当做crossreferences 里面的内嵌字典target
    dict1['xref_id'] = all_target[i].cross_references.find_all('target',recursive=False)[0].xref_id.string
    dict1['xref_src'] = all_target[i].cross_references.find_all('target',recursive=False)[0].xref_src.string
    target_inner_crossreference['target1'] = dict1
    target_dict['cross_reference'] = target_inner_crossreference
    dict2={}#用来当作target_components 里面的内嵌字典target_component
    dict2['accession'] = all_target[i].target_components.target_component.accession.string
    dict2['component_description'] = all_target[i].target_components.target_component.component_description.string
    dict2['component_id'] = all_target[i].target_components.target_component.component_id.string
    dict2['component_type'] = all_target[i].target_components.target_component.component_type.string
    dict2['relationship'] = all_target[i].target_components.target_component.relationship.string
    target_dict['target_components'] = dict2

    temp_dict = copy.deepcopy(target_dict)
    target_list.append(temp_dict)
    i += 1
dev2.insert_many(target_list)

'''activities 活性物接口爬取'''
html_url = "https://www.ebi.ac.uk/chembl/api/data/activity?limit=5"#直接拿它给的前5条activity
response = request.urlopen(html_url)
XML = response.read()
soup = BeautifulSoup(XML,'xml')
all_activities = soup.activities.find_all('activity',recursive=False)
#all_activities 是一个长度为五的列表，列表元素类型为 <class 'bs4.element.Tag'> 所以可以对列表中的每个元素进行bs解析。

i=0
activitise_list = []
while(i <= len(all_activities)-1 ):
    activitise_dict = {}
    ligand={}#作为活性物字典activitise_dict 中的ligand嵌套字典。
    activitise_dict['activity_id'] = all_activities[i].activity_id.string
    activitise_dict['target_chembl_id'] = all_activities[i].target_chembl_id.string
    activitise_dict['target_tax_id'] = all_activities[i].target_tax_id.string
    activitise_dict['assay_chembl_id'] = all_activities[i].assay_chembl_id.string

    activitise_dict['document_chembl_id'] = all_activities[i].document_chembl_id.string
    activitise_dict['molecule_chembl_id'] = all_activities[i].molecule_chembl_id.string
    activitise_dict['parent_molecule_chembl_id'] = all_activities[i].parent_molecule_chembl_id.string
    activitise_dict['record_id'] = all_activities[i].record_id.string

    activitise_dict['target_organism'] = all_activities[i].target_organism.string
    activitise_dict['target_pref_name'] = all_activities[i].target_pref_name.string
    activitise_dict['assay_description'] = all_activities[i].assay_description.string
    activitise_dict['assay_type'] = all_activities[i].assay_type.string
    activitise_dict['bao_endpoint'] = all_activities[i].bao_endpoint.string
    activitise_dict['bao_format'] = all_activities[i].bao_format.string
    activitise_dict['bao_label'] = all_activities[i].bao_label.string
    activitise_dict['type'] = all_activities[i].type.string
    if (all_activities[i].ligand_efficiency.bei != None or all_activities[i].ligand_efficiency.le != None):
        print('ligand output:',all_activities[i].ligand_efficiency.bei)
        ligand['bei'] = all_activities[i].ligand_efficiency.bei.string
        ligand['le'] = all_activities[i].ligand_efficiency.le.string
        ligand['lle'] = all_activities[i].ligand_efficiency.lle.string
        ligand['sei'] = all_activities[i].ligand_efficiency.sei.string

    activitise_dict['ligand_effieiency'] = ligand
    temp_dict = copy.deepcopy(activitise_dict)
    activitise_list.append(temp_dict)
    i += 1
dev3.insert_many(activitise_list)
'''drug 关联药物接口爬取'''
html_url = "https://www.ebi.ac.uk/chembl/api/data/drug?limit=5"#直接拿它给的前5条target
response = request.urlopen(html_url)
XML = response.read()
soup = BeautifulSoup(XML,'xml')
all_drug = soup.drugs.find_all('drug',recursive=False)

i=0
drug_list = []
while(i <= len(all_drug)-1 ):
    drug_dict = {}
    applicants = {} #作为 drug_dict的内嵌字典
    reserch_codes = {} #作为 drug_dict的内嵌字典
    synonyms = {} #作为 drug_dict的内嵌字典
    print(i,'dont have drug error?:',bool(all_drug[i].atc_classification and all_drug[i].atc_classification.drug.code.string))
    if(bool(all_drug[i].atc_classification and all_drug[i].atc_classification.drug.code.string)):
        #没有的就要判个错,并将其置空,这个例子的第五个drug由于连atc_classification都没有，所以还要一起判断这个tag是否存在否则会直接报错。
        print('error find:',all_drug[i].atc_classification.drug.code.string)
        drug_dict['drug_code'] = all_drug[i].atc_classification.drug.code.string
        drug_dict['drug_description'] = all_drug[i].atc_classification.drug.description.string
    else:
        drug_dict['drug_code'] = 'null'
        drug_dict['drug_description'] = 'null'

    drug_dict['development_phase'] = all_drug[i].development_phase.string
    drug_dict['first_approval'] = all_drug[i].first_approval.string
    drug_dict['indication_class'] = all_drug[i].indication_class.string
    drug_dict['molecule_chembl_id'] = all_drug[i].molecule_chembl_id.string
    drug_dict['ob_patent'] = all_drug[i].ob_patent.string
    drug_dict['sc_patent'] = all_drug[i].sc_patent.string
    drug_dict['usan_year'] = all_drug[i].usan_year.string
    applicants_value = []
    all_value = all_drug[i].applicants.find_all('value')
    j = 0
    while (j <= len(all_value)-1):
        applicants_value.append(all_value[j].string)
        j += 1
    temp_list = copy.deepcopy(applicants_value)
    applicants['value'] = temp_list
    drug_dict['applicants'] = applicants

    research_codes_value = []
    all_value = all_drug[i].research_codes.find_all('value')
    j = 0
    while(j <= len(all_value)-1):
        research_codes_value.append(all_value[j].string)
        j += 1
    temp_list = copy.deepcopy(research_codes_value)
    reserch_codes['value'] = temp_list
    drug_dict['reserch_codes'] = reserch_codes

    synonyms_value = []
    all_value = all_drug[i].find_all('synonyms',recursive=False)[0].find_all('value')#这里把递归参数设为false因为子表中也有synonyms，因此不需要递归查询子表
    #print('test',bool(all_value[1].string))
    j = 0
    while (j <= len(all_value) - 1 and bool(all_value[j].string) == True):
        synonyms_value.append(all_value[j].string)
        j += 1
    temp_list = copy.deepcopy(synonyms_value)
    synonyms['value'] = temp_list
    drug_dict['synonyms'] = synonyms

    temp_dict = copy.deepcopy(drug_dict)
    drug_list.append(temp_dict)
    i += 1
dev4.insert_many(drug_list)












'''
1. 开始新的任务chembl网页爬虫 url: https://www.ebi.ac.uk/chembl/
1.1 项目任务简介：
一共需要四张表
第一张：target即靶点，这也是在chembl首页的最下方target点进去查看里面的内容和herb差不多就是一张列表第一行是key，下面是数据。
第二张：compound即小分子，这也是在首页下方Coumpounds里面点进去就行
第三张：activity 即活性物，也可以在网页最下方有一个2000多万条的activities
第四张：drugs 即药物，这可以在首页的泡泡图里面找到信息。
【附】：还需要给这三张表关联ASSAY，也可以通过首页的泡泡图找到Assay查看。
1.2 相关联网页数据接口：
其实上面的一大段只是说明了需要爬取的内容是什么样子的，具体在哪里找到这些数据不再像之前F12 后ctrl+L获取接口，而是直接从官网的web services里面找到所有接口，通过接口获得需要的json文件。
chembl数据库网站接口：
首页 -> web Services -> Documentation下方的Data Web Services 里面的Resources板块 可以找到所有的数据API接口，
Resources API url: https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
在Resources下面的 ChEMBL ID Lookup 可以索引所有的chembl ID
我已经将target，coumpound，activity对应的三个接口get访问得到的json复制下来了可在pycharm里面查看，也可以到上面的网址里面直接get访问接口得到完整的json。

'''